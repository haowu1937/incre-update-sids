"""Temporal helpers for the existing genrec dataset pipeline.

This module has no model, graph or text-encoder dependency. It consumes the
four-column tuples returned by AmazonReviews2014._load_reviews and emits
causal rows compatible with the repository's history-plus-target convention.
Training integration is deliberately a later handoff round.
"""

import ast
import copy
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from numbers import Integral, Real


STAGES = ("base_train", "base_val", "update_train", "update_val", "future_test")
CATALOG_REFERENCE = {
    "base_train": "base_train",
    "base_val": "base_train",
    "update_train": "update_train",
    "update_val": "update_train",
    "future_test": "update",
}


def parse_record(line):
    """Parse JSON or a legacy Python literal, without executing source."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        try:
            record = ast.literal_eval(line)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Record must be JSON or a Python literal") from exc
    if not isinstance(record, dict):
        raise ValueError("Record must be an object/dict")
    return record


def iter_gzip_records(path):
    """Read UTF-8 gzip records; errors identify the file and line number."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield parse_record(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reviews_from_sequences(all_seqs):
    """Adapt existing all_seqs raw IDs and actual timestamps to review tuples."""
    result = []
    for user, sequence in all_seqs.items():
        items = sequence["item"]
        times = sequence["time"]
        ratings = sequence.get("rating", [None] * len(items))
        if len(items) != len(times) or len(items) != len(ratings):
            raise ValueError(f"Sequence columns have different lengths for {user!r}")
        result.extend(zip([user] * len(items), items, times, ratings))
    return result


def normalize_reviews(reviews, duplicate_policy="drop_exact"):
    """Preserve repeats; exact duplicates mean equal user/item/time/rating.

    Review text is not used to define an implicit interaction. With
    keep_exact, duplicate events get distinct deterministic occurrence IDs.
    """
    if duplicate_policy not in ("drop_exact", "keep_exact"):
        raise ValueError("duplicate_policy must be drop_exact or keep_exact")
    checked = []
    for row in reviews:
        if len(row) != 4:
            raise ValueError("Each review must contain user, item, timestamp, rating")
        user, item, timestamp, rating = row
        if any(not isinstance(value, str) or not value or value == "[PAD]"
               for value in (user, item)):
            raise ValueError("Raw user/item IDs must be nonempty strings other than [PAD]")
        if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
            raise ValueError("Timestamps must be integer UNIX seconds")
        timestamp = int(timestamp)
        try:
            datetime.fromtimestamp(timestamp, timezone.utc)
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError("Timestamp is outside the supported UTC range") from exc
        if rating is not None:
            if isinstance(rating, bool) or not isinstance(rating, Real):
                raise ValueError("Ratings must be numeric or None")
            rating = float(rating)
            if not math.isfinite(rating):
                raise ValueError("Ratings must be finite")
        checked.append((user, item, timestamp, rating))
    checked.sort(key=lambda row: (row[2], row[0], row[1],
                                 (row[3] is not None, row[3] or 0.0)))
    occurrences = defaultdict(int)
    events = []
    duplicates = 0
    for row in checked:
        occurrence = occurrences[row]
        occurrences[row] += 1
        if occurrence:
            duplicates += 1
            if duplicate_policy == "drop_exact":
                continue
        user, item, timestamp, rating = row
        events.append({"user": user, "item": item, "timestamp": timestamp,
                       "rating": rating, "sample_id": _digest([*row, occurrence])})
    if not events:
        raise ValueError("No interactions remain")
    return events, {"input_events": len(checked), "retained_events": len(events),
                    "exact_duplicates_observed": duplicates,
                    "exact_duplicates_removed": duplicates if duplicate_policy == "drop_exact" else 0,
                    "duplicate_policy": duplicate_policy,
                    "repeat_item_policy": "preserve",
                    "rating_threshold": None}


def _nearest_edge(edges, desired):
    if not edges:
        raise ValueError("No timestamp boundary can produce nonempty windows")
    return min(edges, key=lambda index: (abs(index - desired), index))


def temporal_boundaries(events):
    """Choose 60/80% edges, then late-10% validation within each window.

    Ties select the earlier edge. Reserve at least two timestamp groups for
    each base/update window and one for future test, so every stage is nonempty.
    """
    n_events = len(events)
    if not n_events:
        raise ValueError("No interactions supplied")
    if any(events[i]["timestamp"] > events[i + 1]["timestamp"]
           for i in range(n_events - 1)):
        raise ValueError("Events must be sorted by timestamp")
    starts = [0] + [i for i in range(1, n_events)
                    if events[i - 1]["timestamp"] != events[i]["timestamp"]] + [n_events]
    groups = len(starts) - 1
    if groups < 5:
        raise ValueError("At least five distinct timestamps are required for five nonempty stages")
    base_end = _nearest_edge(starts[2:groups - 2], 0.6 * n_events)
    base_group = starts.index(base_end)
    update_end = _nearest_edge(starts[base_group + 2:groups], 0.8 * n_events)
    base_train_end = _nearest_edge([i for i in starts if 0 < i < base_end],
                                   0.9 * base_end)
    update_train_end = _nearest_edge([i for i in starts if base_end < i < update_end],
                                     base_end + 0.9 * (update_end - base_end))
    return {"base_train_end": base_train_end, "base_end": base_end,
            "update_train_end": update_train_end, "update_end": update_end,
            "total": n_events}


def registry_snapshots(events, boundaries):
    """Append observed identities; never include future-only identities."""
    registry = {"user2id": {"[PAD]": 0}, "item2id": {"[PAD]": 0},
                "id2user": ["[PAD]"], "id2item": ["[PAD]"]}
    endpoints = {
        boundaries["base_train_end"]: "base_train",
        boundaries["base_end"]: "base",
        boundaries["update_train_end"]: "update_train",
        boundaries["update_end"]: "update",
    }
    snapshots = {}
    for count, event in enumerate(events[:boundaries["update_end"]], 1):
        for kind in ("user", "item"):
            raw_id = event[kind]
            forward, reverse = registry[f"{kind}2id"], registry[f"id2{kind}"]
            if raw_id not in forward:
                forward[raw_id] = len(reverse)
                reverse.append(raw_id)
        if count in endpoints:
            snapshots[endpoints[count]] = copy.deepcopy(registry)
    return snapshots


def build_temporal_protocol(reviews, max_history=50, duplicate_policy="drop_exact"):
    """Prepare raw causal rows and catalog flags, without running any model.

    item_seq is history followed by exactly one target, matching the existing
    _tokenize_once convention. The current train tokenizer expands prefixes;
    these rows MUST NOT be fed to it until explicit-example mode is delivered.
    Catalog flags keep unservable targets/history visible rather than dropping
    them silently. Their model-input policy is wired in the next round.
    """
    if isinstance(max_history, bool) or not isinstance(max_history, Integral) or max_history < 1:
        raise ValueError("max_history must be a positive integer")
    events, preprocessing = normalize_reviews(reviews, duplicate_policy)
    boundaries = temporal_boundaries(events)
    registries = registry_snapshots(events, boundaries)
    edges = [0, boundaries["base_train_end"], boundaries["base_end"],
             boundaries["update_train_end"], boundaries["update_end"], len(events)]
    stage_at = [stage for stage, start, end in zip(STAGES, edges, edges[1:])
                for _ in range(start, end)]
    rows = {stage: [] for stage in STAGES}
    excluded = {stage: 0 for stage in STAGES}
    histories = defaultdict(list)
    indexed = enumerate(events)
    for _, same_time in groupby(indexed, key=lambda pair: pair[1]["timestamp"]):
        group = list(same_time)
        for index, event in group:
            stage = stage_at[index]
            history = histories[event["user"]][-max_history:]
            if not history:
                excluded[stage] += 1
                continue
            catalog_version = CATALOG_REFERENCE[stage]
            catalog = registries[catalog_version]["item2id"]
            sequence = [*history, event]
            rows[stage].append({
                "user": event["user"],
                "item_seq": [entry["item"] for entry in sequence],
                "time_seq": [entry["timestamp"] for entry in sequence],
                "rating_seq": [entry["rating"] for entry in sequence],
                "sample_id": event["sample_id"], "target_time": event["timestamp"],
                "catalog_version": catalog_version,
                "target_servable": event["item"] in catalog,
                "history_available": [entry["item"] in catalog for entry in history],
            })
        # Simultaneous interactions are appended only after every target at this
        # timestamp has been formed. Equal-time history order is deterministic.
        for _, event in group:
            history = histories[event["user"]]
            history.append(event)
            if len(history) > max_history:
                del history[:-max_history]
    empty_stages = [stage for stage in STAGES if not rows[stage]]
    if empty_stages:
        raise ValueError(f"No eligible causal examples in stages: {empty_stages}")
    windows = {}
    for stage, start, end in zip(STAGES, edges, edges[1:]):
        first = events[start]["timestamp"]
        last = events[end - 1]["timestamp"]
        targets = rows[stage]
        windows[stage] = {
            "event_count": end - start,
            "first_timestamp": first, "last_timestamp": last,
            "first_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(last, timezone.utc).isoformat(),
            "example_count": len(targets), "excluded_empty_history": excluded[stage],
            "unservable_targets": sum(not row["target_servable"] for row in targets),
            "examples_with_unavailable_history": sum(not all(row["history_available"]) for row in targets),
        }
    manifest = {
        "schema_version": 1, "protocol": "global_temporal_60_20_20_no_refit",
        "normalized_interactions_sha256": _digest(events),
        "max_history": int(max_history), "preprocessing": preprocessing,
        "boundaries": boundaries, "windows": windows,
        "cutoffs_exclusive": {
            key: events[boundaries[f"{key}_end"]]["timestamp"]
            for key in ("base_train", "base", "update_train", "update")
        },
        "fraction_deviation": {
            "base": boundaries["base_end"] / len(events) - 0.6,
            "update_end": boundaries["update_end"] / len(events) - 0.8,
            "base_validation": (boundaries["base_end"] - boundaries["base_train_end"]) / boundaries["base_end"] - 0.1,
            "update_validation": (boundaries["update_end"] - boundaries["update_train_end"]) / (boundaries["update_end"] - boundaries["base_end"]) - 0.1,
        },
        "catalog_reference": CATALOG_REFERENCE.copy(),
        "catalog_sizes": {key: len(value["item2id"]) - 1 for key, value in registries.items()},
        "same_timestamp_policy": "exclude_all_simultaneous_events_from_history",
        "history_protocol": "offline_prequential_raw_history_with_availability_flags",
        "validation_refit": False,
        "training_integration": "not_provided_in_round1",
    }
    manifest["protocol_sha256"] = _digest(manifest)
    return {"manifest": manifest, "registries": registries, "examples": rows}
