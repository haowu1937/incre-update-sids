#!/usr/bin/env python3
"""Read bounded text evidence from one retuned run; never execute its commands.

Uses the installed Runner's transport and lock. Does not change its smoke-only
gate, touch checkpoints, load ML, launch experiments, or write to the project.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from runner import Runner, atomic_json, now


PREFIX = "observations/history-retuned/"
RUN_NAME = re.compile(r"llama_history_retuned_[A-Za-z0-9_]+\Z")
MAX_BYTES = 64 * 1024
TAIL_BYTES = 8 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")


def safe_path(path, project):
    if path.is_symlink() or project not in path.resolve().parents:
        raise ValueError("Evidence path escapes the configured project or is a symlink")
    return path


def read_text(path, project, tail=False):
    safe_path(path, project)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not tail and before.st_size > MAX_BYTES:
            raise ValueError("Evidence exceeds 64 KiB")
        offset = max(0, before.st_size - TAIL_BYTES) if tail else 0
        handle.seek(offset)
        raw = handle.read(TAIL_BYTES if tail else MAX_BYTES + 1)
        after = os.fstat(handle.fileno())
    stable = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    return raw, dict(source_size=after.st_size, source_mtime_ns=after.st_mtime_ns,
                     read_offset=offset, captured_sha256=hashlib.sha256(raw).hexdigest(),
                     stable_during_read=stable, tail=tail)


def snapshot(root, project):
    root, project = Path(root).resolve(), Path(project).resolve()
    if root == project or project in root.parents or root in project.parents:
        raise ValueError("Collector state must be outside the research project")
    base = project / "results/incremental/Beauty"
    binding_path = root / "retuned-collector-binding.json"
    files, metadata, errors, parsed = {}, {}, {}, {}
    version_path = root / "collector-code-sha.txt"
    observation = dict(schema_version=1, observer="history-retuned", project=str(project),
                       collector_code_sha=version_path.read_text().strip() if version_path.is_file() else None,
                       collector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       read_only=True, training_started=False, gpu_used=False,
                       execution_evidence="bounded text snapshots; no model/checkpoint inspection")
    if binding_path.exists():
        bound = json.loads(binding_path.read_text())
        if bound["project"] != str(project) or not RUN_NAME.fullmatch(bound["run_name"]):
            raise ValueError("Collector binding/project mismatch")
        run = base / bound["run_name"]
        if not run.is_dir():
            observation.update(status="bound_run_missing", run_dir=str(run))
            return observation, files
    else:
        candidates = sorted(path for path in base.glob("llama_history_retuned_*")
                            if path.is_dir() and not path.is_symlink() and RUN_NAME.fullmatch(path.name))
        if len(candidates) != 1:
            observation.update(status="waiting_for_run" if not candidates else "ambiguous_run",
                               candidate_runs=[path.name for path in candidates])
            return observation, files
        run = candidates[0]
        # Binding happens once; later runs never silently replace the observed run.
        atomic_json(binding_path, dict(project=str(project), run_name=run.name, bound_at=now()))
    safe_path(run, project)
    observation["run_dir"] = str(run)
    full = ["commit.txt", "paths.txt", "experiment/request.json", "experiment/summary.json",
            "experiment/compact.json", "experiment/pair_checks.json"]
    tails = ["experiment/error.txt"]
    for arm in ("plain", "historical"):
        full += ["experiment/" + arm + "/summary.json", "experiment/" + arm + "/failure.json"]
        tails += ["experiment/" + arm + "/metrics.jsonl"]
    attempts = sorted(path for path in run.glob("attempt_*") if path.is_dir() and not path.is_symlink())
    if attempts:
        observation["latest_attempt"] = attempts[-1].name
        tails += [str((attempts[-1] / name).relative_to(run)) for name in ("train.log", "checks.log")]
    for relative, tail in [(p, False) for p in full] + [(p, True) for p in tails]:
        path = run / relative
        if not path.exists():
            continue
        try:
            raw, info = read_text(path, project, tail=tail)
            text = raw.decode("utf-8")
            remote = PREFIX + "files/" + relative + (".tail.txt" if tail else "")
            files[remote] = raw
            metadata[relative] = dict(info, remote_path=remote)
            if not tail and relative.endswith(".json"):
                parsed[relative] = json.loads(text)
        except (OSError, ValueError, UnicodeError) as error:
            errors[relative] = type(error).__name__ + ": " + str(error)
    observation.update(files=metadata, collection_errors=errors)
    raw_commit = files.get(PREFIX + "files/commit.txt", b"").decode().strip()
    observation["run_code_sha"] = raw_commit if SHA.fullmatch(raw_commit) else None
    summary = parsed.get("experiment/summary.json", {})
    observation["reported_execution_status"] = summary.get("execution_status")
    required = ("commit.txt", "experiment/request.json", "experiment/summary.json",
                "experiment/compact.json")
    complete_evidence = all(name in metadata and metadata[name]["stable_during_read"]
                            and name not in errors for name in required)
    if summary.get("execution_status") == "completed" and complete_evidence and observation["run_code_sha"]:
        observation["status"] = "completed_evidence_available"
    elif summary.get("execution_status") == "failed" and complete_evidence:
        observation["status"] = "failed_evidence_available"
    elif any(name.endswith("failure.json") for name in parsed):
        observation["status"] = "failure_evidence_available"
    elif any(b"Traceback (most recent call last)" in raw for name, raw in files.items()
             if name.endswith(".tail.txt")):
        observation["status"] = "error_observed"
    else:
        # Missing final files do not prove a live process or a failed experiment.
        observation["status"] = "awaiting_final_evidence"
    return observation, files


def tick(root, project):
    root = Path(root).resolve()
    if not (root / "config.json").is_file():
        raise ValueError("Install the smoke-only Runner first")
    with (root / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="busy", read_only=True)
        state_path = root / "retuned-collector-state.json"
        pending_path = root / "retuned-collector-pending.json"
        # Deliver the preserved snapshot first; newer output must not discard
        # an older error/result that was captured during a network outage.
        if pending_path.exists():
            pending = json.loads(pending_path.read_text())
            Runner(root).publish({name: value.encode("utf-8") for name, value in pending["files"].items()})
            previous_observation = pending["observation"]
            atomic_json(state_path, dict(published_fingerprint=previous_observation["evidence_fingerprint"],
                        published_at=now(), run_dir=previous_observation.get("run_dir"),
                        status=previous_observation["status"]))
            pending_path.unlink()
        observation, files = snapshot(root, project)
        content = json.dumps(observation, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
        fingerprint = hashlib.sha256(content).hexdigest()
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        if previous.get("published_fingerprint") == fingerprint:
            return dict(status="unchanged", observed_status=observation["status"], read_only=True)
        observation.update(observed_at=now(), evidence_fingerprint=fingerprint)
        files[PREFIX + "observation.json"] = (json.dumps(observation, ensure_ascii=False, indent=2) + "\n").encode()
        # Spool the actual snapshot before delivery. A failed push is not evidence
        # of successful collection; never advance published_fingerprint on failure.
        atomic_json(pending_path, dict(observation=observation,
                    files={name: raw.decode("utf-8") for name, raw in files.items()}))
        Runner(root).publish(files)
        atomic_json(state_path, dict(published_fingerprint=fingerprint, published_at=now(),
                                    run_dir=observation.get("run_dir"), status=observation["status"]))
        pending_path.unlink()
        return dict(status="published", observed_status=observation["status"],
                    run_dir=observation.get("run_dir"), read_only=True, gpu_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--delay-seconds", type=int, choices=range(0, 31), default=0,
                        help="Offset cron from the Runner's minute boundary")
    args = parser.parse_args()
    try:
        if args.delay_seconds:
            time.sleep(args.delay_seconds)
        print(json.dumps(tick(args.root, args.project), ensure_ascii=False))
        return 0
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
