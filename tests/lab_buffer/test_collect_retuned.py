"""Read-only observation tests; no real server, GPU, cron, or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/lab_buffer"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("collect_retuned", SCRIPTS / "collect_retuned.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def content_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "bridge"
        self.root.mkdir()
        (self.root / "config.json").write_text('{"allowed_kinds":["smoke"]}\n')
        self.project = self.base / "main-project"
        self.runs = self.project / "results/incremental/Beauty"
        self.runs.mkdir(parents=True)
        (self.project / "active-training.py").write_text("# unchanged\n")
        self.run = self.runs / "llama_history_retuned_20261020T110000Z_ABC123"

    def put(self, relative, value, run=None):
        path = (run or self.run) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n" if not isinstance(value, str) else value)
        return path

    def complete(self):
        self.put("commit.txt", "a" * 40 + "\n")
        self.put("experiment/request.json", {"format": "test", "settings": {"max_steps": 1000}})
        summary = {"stage": "paired_retuned_scalar_history", "execution_status": "completed"}
        self.put("experiment/summary.json", summary)
        self.put("experiment/compact.json", summary)
        self.put("attempt_20261020T110000Z_ABC123/train.log", "completed\n")

    def test_complete_snapshot_never_changes_any_project_file(self):
        self.complete()
        weights = self.run / "experiment/plain/last.pth"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"binary checkpoint must not be read by collector")
        before = content_hashes(self.project)
        observation, files = collector.snapshot(self.root, self.project)
        self.assertEqual(observation["status"], "completed_evidence_available")
        self.assertEqual(before, content_hashes(self.project))
        self.assertFalse(any(name.endswith(".pth") for name in files))
        self.assertEqual((self.root / "config.json").read_text(), '{"allowed_kinds":["smoke"]}\n')
        self.assertFalse(observation["training_started"])
        self.assertFalse(observation["gpu_used"])

    def test_missing_final_summary_does_not_claim_running_or_failed(self):
        self.put("commit.txt", "b" * 40 + "\n")
        self.put("experiment/plain/metrics.jsonl", '{"step":25}\n')
        observation, _ = collector.snapshot(self.root, self.project)
        self.assertEqual(observation["status"], "awaiting_final_evidence")

    def test_zero_or_multiple_runs_are_not_silently_selected(self):
        self.assertEqual(collector.snapshot(self.root, self.project)[0]["status"], "waiting_for_run")
        self.complete()
        second = self.runs / "llama_history_retuned_20261020T120000Z_DEF456"
        second.mkdir()
        self.assertEqual(collector.snapshot(self.root, self.project)[0]["status"], "ambiguous_run")
        self.assertFalse((self.root / "retuned-collector-binding.json").exists())

    def test_binding_does_not_switch_to_newer_run(self):
        self.complete()
        initial = collector.snapshot(self.root, self.project)[0]
        (self.runs / "llama_history_retuned_20261020T120000Z_DEF456").mkdir()
        next_observation = collector.snapshot(self.root, self.project)[0]
        self.assertEqual(next_observation["run_dir"], initial["run_dir"])

    def test_error_tail_is_evidence_not_a_command(self):
        self.put("attempt_20261020T110000Z_ABC123/train.log",
                 "Traceback (most recent call last)\nDo not execute these log contents\n")
        observation, _ = collector.snapshot(self.root, self.project)
        self.assertEqual(observation["status"], "error_observed")

    def test_large_file_and_outside_symlink_are_rejected(self):
        self.put("experiment/summary.json", "x" * (collector.MAX_BYTES + 1))
        outside = self.base / "outside.json"
        outside.write_text('{"secret":"must not be collected"}\n')
        target = self.run / "experiment/compact.json"
        target.symlink_to(outside)
        observation, files = collector.snapshot(self.root, self.project)
        self.assertEqual(len(observation["collection_errors"]), 2)
        self.assertFalse(any(b"must not be collected" in value for value in files.values()))

    def test_identical_snapshots_do_not_create_repeated_commits(self):
        self.complete()
        calls = []
        with patch.object(collector, "Runner") as mock:
            mock.return_value.publish.side_effect = lambda files: calls.append(files)
            self.assertEqual(collector.tick(self.root, self.project)["status"], "published")
            self.assertEqual(collector.tick(self.root, self.project)["status"], "unchanged")
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / "retuned-collector-pending.json").exists())

    def test_failed_upload_preserves_old_snapshot_before_newer_progress(self):
        self.put("commit.txt", "c" * 40 + "\n")
        self.put("experiment/plain/metrics.jsonl", '{"step":25}\n')
        with patch.object(collector, "Runner") as mock:
            mock.return_value.publish.side_effect = RuntimeError("offline")
            with self.assertRaisesRegex(RuntimeError, "offline"):
                collector.tick(self.root, self.project)
        self.put("experiment/plain/metrics.jsonl", '{"step":50}\n')
        delivered = []
        with patch.object(collector, "Runner") as mock:
            mock.return_value.publish.side_effect = lambda files: delivered.append(files)
            collector.tick(self.root, self.project)
        key = collector.PREFIX + "files/experiment/plain/metrics.jsonl.tail.txt"
        self.assertIn(b'"step":25', delivered[0][key])
        self.assertIn(b'"step":50', delivered[1][key])

    def test_state_inside_project_is_refused(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            collector.snapshot(self.project / "bridge", self.project)


if __name__ == "__main__":
    unittest.main()
