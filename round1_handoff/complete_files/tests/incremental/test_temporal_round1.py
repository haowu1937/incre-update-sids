"""CPU-only Round 1 tests. All observations use synthetic fixtures."""

import argparse
import ast
import gzip
import hashlib
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from genrec.temporal import (  # noqa: E402
    STAGES, build_temporal_protocol, iter_gzip_records,
    normalize_reviews, parse_record, reviews_from_sequences, temporal_boundaries,
)


def fixture():
    events = []
    for i in range(100):
        item = ("A", "B", "C", "D")[i % 4]
        if i == 65:
            item = "NEW"
        if i in (85, 86):
            item = "FUTURE_ONLY"
        events.append((f"u{i % 3}", item, 1000 + i // 2, 5))
    return events


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.protocol = build_temporal_protocol(fixture(), max_history=5)

    def test_strict_causality_and_history_cap(self):
        for rows in self.protocol["examples"].values():
            for row in rows:
                self.assertGreaterEqual(len(row["item_seq"]), 2)
                self.assertLessEqual(len(row["item_seq"]), 6)
                self.assertTrue(all(t < row["target_time"] for t in row["time_seq"][:-1]))
                self.assertEqual(row["target_time"], row["time_seq"][-1])
                self.assertEqual(len(row["item_seq"]), len(row["rating_seq"]))

    def test_timestamp_windows_do_not_overlap(self):
        windows = self.protocol["manifest"]["windows"]
        for previous, current in zip(STAGES, STAGES[1:]):
            self.assertLess(windows[previous]["last_timestamp"], windows[current]["first_timestamp"])
        self.assertEqual(sum(w["event_count"] for w in windows.values()), 100)

    def test_expected_cutoffs_and_validation_sizes(self):
        self.assertEqual(self.protocol["manifest"]["boundaries"], {
            "base_train_end": 54, "base_end": 60,
            "update_train_end": 78, "update_end": 80, "total": 100,
        })
        self.assertFalse(self.protocol["manifest"]["validation_refit"])

    def test_simultaneous_events_are_not_history(self):
        events = fixture() + [("same_user", "X", 1010, 5),
                              ("same_user", "Y", 1010, 5),
                              ("same_user", "Z", 1011, 5)]
        result = build_temporal_protocol(events)
        rows = [row for values in result["examples"].values() for row in values
                if row["user"] == "same_user"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_seq"], ["X", "Y", "Z"])
        self.assertEqual(rows[0]["time_seq"], [1010, 1010, 1011])

    def test_each_retained_event_is_accounted_for_once(self):
        manifest = self.protocol["manifest"]
        total_rows = 0
        sample_ids = []
        for stage in STAGES:
            window = manifest["windows"][stage]
            rows = self.protocol["examples"][stage]
            self.assertEqual(len(rows) + window["excluded_empty_history"], window["event_count"])
            total_rows += len(rows)
            sample_ids.extend(row["sample_id"] for row in rows)
        self.assertEqual(len(set(sample_ids)), total_rows)

    def test_stage_registries_exclude_future_ids(self):
        registries = self.protocol["registries"]
        self.assertNotIn("NEW", registries["base"]["item2id"])
        self.assertIn("NEW", registries["update"]["item2id"])
        for registry in registries.values():
            self.assertNotIn("FUTURE_ONLY", registry["item2id"])
            self.assertEqual(registry["item2id"]["[PAD]"], 0)

    def test_registry_ids_and_raw_round_trips_are_stable(self):
        registries = self.protocol["registries"]
        for old_name, new_name in zip(("base_train", "base", "update_train"),
                                      ("base", "update_train", "update")):
            for kind in ("user", "item"):
                old = registries[old_name][f"{kind}2id"]
                new = registries[new_name][f"{kind}2id"]
                for raw_id, old_id in old.items():
                    self.assertEqual(new[raw_id], old_id)
        for registry in registries.values():
            for raw_id, gr_id in registry["item2id"].items():
                self.assertEqual(registry["id2item"][gr_id], raw_id)
                if gr_id:
                    global_id = gr_id - 1
                    self.assertEqual(registry["id2item"][global_id + 1], raw_id)

    def test_unservable_targets_and_history_are_retained_and_counted(self):
        rows = self.protocol["examples"]["future_test"]
        self.assertEqual(sum(not row["target_servable"] for row in rows), 2)
        self.assertEqual(self.protocol["manifest"]["windows"]["future_test"]["unservable_targets"], 2)
        self.assertTrue(any(not all(row["history_available"]) for row in rows))

    def test_exact_duplicates_and_repeats(self):
        reviews = [("u", "A", 1, 5), ("u", "A", 1, 5), ("u", "A", 2, 5)]
        events, stats = normalize_reviews(reviews)
        self.assertEqual(len(events), 2)
        self.assertEqual(stats["exact_duplicates_removed"], 1)
        events, stats = normalize_reviews(reviews, "keep_exact")
        self.assertEqual(len(events), 3)
        self.assertEqual(len({event["sample_id"] for event in events}), 3)
        self.assertEqual(stats["exact_duplicates_removed"], 0)

    def test_deterministic_input_order(self):
        reviews = fixture()
        random.Random(42).shuffle(reviews)
        self.assertEqual(self.protocol, build_temporal_protocol(reviews, max_history=5))

    def test_reject_empty_coarse_and_invalid_inputs(self):
        for reviews in ([], [("u", "A", i, 5) for i in range(4)]):
            with self.assertRaises(ValueError):
                build_temporal_protocol(reviews)
        for timestamp in (True, 1.5, "123"):
            with self.assertRaises(ValueError):
                normalize_reviews([("u", "A", timestamp, 5)])
        with self.assertRaises(ValueError):
            normalize_reviews([("u", "A", 1, float("nan"))])
        with self.assertRaises(ValueError):
            build_temporal_protocol(fixture(), max_history=0)

    def test_reject_unsorted_boundaries(self):
        events, _ = normalize_reviews(fixture())
        with self.assertRaises(ValueError):
            temporal_boundaries(list(reversed(events)))

    def test_reject_empty_causal_splits(self):
        events = [(f"unique_user_{i}", "A", 1000 + i, 5) for i in range(100)]
        with self.assertRaisesRegex(ValueError, "No eligible causal examples"):
            build_temporal_protocol(events)

    def test_boundaries_move_to_timestamp_edges(self):
        events = fixture() + [("extra_user", f"extra_{i}", 1030, 5) for i in range(11)]
        protocol = build_temporal_protocol(events)
        windows = protocol["manifest"]["windows"]
        for left, right in zip(STAGES, STAGES[1:]):
            self.assertLess(windows[left]["last_timestamp"], windows[right]["first_timestamp"])
        self.assertNotEqual(protocol["manifest"]["fraction_deviation"]["base"], 0)

    def test_existing_sequence_adapter_preserves_times(self):
        self.assertEqual(reviews_from_sequences({"u": {
            "item": ["A", "B"], "time": [10, 20], "rating": [4, 5]
        }}), [("u", "A", 10, 4), ("u", "B", 20, 5)])
        with self.assertRaises(ValueError):
            reviews_from_sequences({"u": {"item": ["A"], "time": []}})


class ParserTests(unittest.TestCase):
    def test_json_and_legacy_literals_preserve_text(self):
        self.assertEqual(parse_record('{"title":"true false", "flag":true}'),
                         {"title": "true false", "flag": True})
        self.assertEqual(parse_record("{'asin':'A', 'flag':False, 'value':None}"),
                         {"asin": "A", "flag": False, "value": None})
        with self.assertRaises(ValueError):
            parse_record("[1, 2]")

    def test_parser_rejects_executable_expression(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = pathlib.Path(directory) / "executed"
            payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
            with self.assertRaises(ValueError):
                parse_record(payload)
            self.assertFalse(marker.exists())

    def test_gzip_records_include_line_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "reviews.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write('{"asin":"A"}\n\ninvalid_record\n')
            reader = iter_gzip_records(path)
            self.assertEqual(next(reader), {"asin": "A"})
            with self.assertRaisesRegex(ValueError, r"reviews\.json\.gz:3"):
                next(reader)

    def test_actual_amazon_parser_method_uses_safe_reader(self):
        # Compile the actual patched method without importing optional project
        # dependencies. This exercises its body, not a copied implementation.
        source = ROOT / "genrec/datasets/AmazonReviews2014/dataset.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "AmazonReviews2014")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_parse_gz")
        imported = any(isinstance(node, ast.ImportFrom) and node.module == "genrec.temporal"
                       and any(alias.name == "iter_gzip_records" for alias in node.names)
                       for node in tree.body)
        self.assertTrue(imported)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == "eval" for node in ast.walk(method)))
        namespace = {"iter_gzip_records": iter_gzip_records}
        code = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                       str(source), "exec")
        exec(code, namespace)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "legacy.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("{'asin': 'A', 'title': 'true false'}\n")
            self.assertEqual(list(namespace["_parse_gz"](None, str(path))),
                             [{"asin": "A", "title": "true false"}])


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"test": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"test": test.id(), "status": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"test": test.id(), "status": "error"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New evidence directory; never overwrite an existing run")
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    config = {"fixture": "round1_fixture_v1", "seed": 42, "max_history": 5,
              "duplicate_policy": "drop_exact"}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True, check=True,
                                timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "not_available"
    source_paths = ("genrec/temporal.py", "genrec/datasets/AmazonReviews2014/dataset.py",
                    "tests/incremental/test_temporal_round1.py")
    source_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                     for path in source_paths}
    summary = {"run_id": output.name, "round": "round1", "stage": "E0_subset",
               "inputs": "synthetic_only", "environment": sys.version,
               "seed": 42, "source_sha": commit, "source_files_sha256": source_hashes,
               "resolved_config": config,
               "resolved_config_sha256": hashlib.sha256(
                   json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
               "execution_status": "passed" if result.wasSuccessful() else "failed",
               "tests_run": result.testsRun, "tests": result.records,
               "beauty_status": "not_run", "E0_full_status": "not_run",
               "E1_status": "not_run", "runtime": "not_measured",
               "memory": "not_measured"}
    (output / "checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if result.wasSuccessful():
        artifact = build_temporal_protocol(fixture(), max_history=5)
        (output / "synthetic_protocol.json").write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Evidence: {output / 'checks.json'}", flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
