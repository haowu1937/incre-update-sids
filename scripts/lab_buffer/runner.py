#!/usr/bin/env python3
"""Small, single-host Git mailbox runner. Python standard library only.

All Git writes and checkouts live under --root; the research checkout is never
opened. Start in smoke-only mode. State is durable before process creation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
FINAL = {"completed", "failed", "timed_out", "blocked", "interrupted"}
MAX_MESSAGE = 64 * 1024
TAIL_BYTES = 12 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_task(raw, filename):
    if len(raw) > MAX_MESSAGE:
        raise ValueError("Task exceeds 64 KiB")
    task = json.loads(raw)
    if not isinstance(task, dict):
        raise ValueError("Task must be an object")
    allowed = {"schema_version", "job_id", "worker_id", "kind", "code_sha",
               "timeout_seconds", "created_at", "argv", "artifacts"}
    if set(task) - allowed:
        raise ValueError("Unknown task fields")
    if type(task.get("schema_version")) is not int or task["schema_version"] != 1:
        raise ValueError("Expected schema_version=1")
    for key in ("job_id", "worker_id"):
        if not isinstance(task.get(key), str) or not NAME.fullmatch(task[key]):
            raise ValueError("Invalid " + key)
    if filename != "jobs/" + task["job_id"] + ".json":
        raise ValueError("Filename/job_id mismatch")
    if not isinstance(task.get("code_sha"), str) or not SHA.fullmatch(task["code_sha"]):
        raise ValueError("Expected full code SHA")
    if task.get("kind") not in ("smoke", "experiment"):
        raise ValueError("Unknown job kind")
    timeout = task.get("timeout_seconds")
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("Timeout must be 1..86400 seconds")
    if not isinstance(task.get("created_at"), str) or not task["created_at"]:
        raise ValueError("Missing created_at")
    if task["kind"] == "smoke" and ("argv" in task or "artifacts" in task):
        raise ValueError("Smoke jobs have a fixed command and fixed output")
    if task["kind"] == "experiment":
        argv = task.get("argv")
        if (not isinstance(argv, list) or len(argv) < 2
                or not all(isinstance(arg, str) and "\0" not in arg for arg in argv)
                or argv[0] not in ("python", "bash")):
            raise ValueError("Expected argv beginning with python or bash; no shell eval")
        artifacts = task.get("artifacts", [])
        if not isinstance(artifacts, list) or len(artifacts) > 8:
            raise ValueError("Expected at most 8 explicit artifacts")
        for item in artifacts:
            path = Path(item) if isinstance(item, str) else Path("/")
            if path.is_absolute() or ".." in path.parts or path.suffix not in (".json", ".txt", ".md"):
                raise ValueError("Artifact must be a relative json/txt/md path")
    return task


class Runner:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        self.repo = self.root / "transport.git"
        self.env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
        self.env.update({"GIT_AUTHOR_NAME": "Lab Buffer Runner",
                         "GIT_AUTHOR_EMAIL": "lab-buffer@localhost",
                         "GIT_COMMITTER_NAME": "Lab Buffer Runner",
                         "GIT_COMMITTER_EMAIL": "lab-buffer@localhost"})
        self.env.pop("GIT_DIR", None)
        self.env.pop("GIT_WORK_TREE", None)
        self.env.pop("GIT_INDEX_FILE", None)
        for key in ("worker_id",):
            if not NAME.fullmatch(self.config[key]):
                raise ValueError("Invalid worker id")
        if self.config.get("allowed_kinds") not in (["smoke"], ["smoke", "experiment"]):
            raise ValueError("Invalid allowed_kinds")
        for key in ("requests_branch", "results_branch", "code_branch"):
            if not NAME.fullmatch(self.config[key]):
                raise ValueError("Invalid branch")

    def git(self, *args, data=None, check=True, timeout=45, extra_env=None):
        env = dict(self.env, **(extra_env or {}))
        result = subprocess.run(["git", "--git-dir=" + str(self.repo), *args],
                                input=data, capture_output=True, env=env, timeout=timeout)
        if check and result.returncode:
            # Avoid echoing remote URLs (which may contain credentials).
            raise RuntimeError("Git operation failed: " + args[0] + " (see connection/credentials)")
        return result

    def ref(self, key):
        return "refs/remotes/origin/" + self.config[key]

    def fetch(self, keys=("requests_branch", "results_branch", "code_branch")):
        self.git("fetch", "--quiet", "origin", *[
            "+refs/heads/" + self.config[key] + ":" + self.ref(key) for key in keys])

    def read(self, ref, path):
        result = self.git("show", ref + ":" + path, check=False)
        return result.stdout if result.returncode == 0 else None

    def publish(self, files, branch_key="results_branch", expected_parent=None):
        """Construct a commit atop the current results head; retry FF races.

        Retry only delivery. No training subprocess is ever created here.
        """
        branch = self.config[branch_key]
        for _ in range(3):
            self.fetch((branch_key,))
            parent = self.git("rev-parse", self.ref(branch_key)).stdout.decode().strip()
            if expected_parent is not None and parent != expected_parent:
                raise RuntimeError("Concurrent branch change; reread before updating control state")
            index = self.root / ("publish-index-" + uuid.uuid4().hex)
            env = {"GIT_INDEX_FILE": str(index)}
            try:
                self.git("read-tree", parent, extra_env=env)
                for path, data in sorted(files.items()):
                    blob = self.git("hash-object", "-w", "--stdin", data=data).stdout.decode().strip()
                    self.git("update-index", "--add", "--cacheinfo", "100644", blob, path, extra_env=env)
                tree = self.git("write-tree", extra_env=env).stdout.decode().strip()
                old_tree = self.git("rev-parse", parent + "^{tree}").stdout.decode().strip()
                if tree == old_tree:
                    return
                commit = self.git("commit-tree", tree, "-p", parent,
                                  data=b"Record lab buffer state\n").stdout.decode().strip()
                result = self.git("push", "--quiet", "origin", commit + ":refs/heads/" + branch, check=False)
                if result.returncode == 0:
                    return
                if expected_parent is not None:
                    raise RuntimeError("Concurrent branch change; reread before updating control state")
            finally:
                index.unlink(missing_ok=True)
        raise RuntimeError("Results delivery failed; local result retained for retry")

    def state_path(self, job_id):
        return self.root / "state" / (job_id + ".json")

    def save(self, state):
        state["pending_upload"] = True
        state["result"]["updated_at"] = now()
        atomic_json(self.state_path(state["result"]["job_id"]), state)

    def deliver(self, state):
        result = state["result"]
        job_id = result["job_id"]
        files = {"jobs/" + job_id + ".json":
                 (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()}
        log = self.root / "logs" / (job_id + ".log")
        if log.is_file():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - TAIL_BYTES))
                files["logs/" + job_id + ".tail.txt"] = handle.read().decode("utf-8", "replace").encode()
        for name, value in state.get("artifact_text", {}).items():
            files["artifacts/" + job_id + "/" + name] = value.encode()
        self.publish(files)
        state["pending_upload"] = False
        atomic_json(self.state_path(job_id), state)

    def pause(self, reason):
        atomic_json(self.root / "PAUSED.json", {"reason": reason, "time": now()})

    def recover_and_flush(self):
        uncertain = False
        for path in sorted((self.root / "state").glob("*.json")):
            state = json.loads(path.read_text())
            result = state["result"]
            if result["status"] in ("starting", "running"):
                result["status"] = "interrupted"
                result["reason"] = "Runner stopped during launch/run; inspect recorded PID before any retry"
                result["finished_at"] = now()
                self.pause(result["reason"] + ": " + result["job_id"])
                self.save(state)
                uncertain = True
            if state["pending_upload"]:
                self.deliver(state)
        return uncertain

    def new_state(self, task, raw):
        version_file = self.root / "runner-code-sha.txt"
        return {"pending_upload": True, "result": {
            "schema_version": 1, "job_id": task["job_id"],
            "worker_id": self.config["worker_id"], "kind": task["kind"],
            "request_sha256": digest(raw), "code_sha": task["code_sha"],
            "runner_code_sha": version_file.read_text().strip() if version_file.is_file() else None,
            "status": "starting", "started_at": now(), "updated_at": now(),
            "exit_code": None, "pid": None,
            "workspace": str(self.root / "workspaces" / task["job_id"]),
            "log_path": str(self.root / "logs" / (task["job_id"] + ".log")),
            "log_tail": "logs/" + task["job_id"] + ".tail.txt",
            "artifacts": [], "training_started": False}}

    @staticmethod
    def supervise(process, deadline, timed_out):
        """Enforce the child deadline even while Git delivery is blocked."""
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out.set()
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def collect_artifacts(self, task, state, workspace):
        names = (["outputs/lab-buffer-smoke.json"] if task["kind"] == "smoke"
                 else task.get("artifacts", []))
        texts = {}
        for name in names:
            path = workspace / name
            if (path.is_symlink() or not path.is_file()
                    or workspace not in path.resolve().parents
                    or path.stat().st_size > MAX_MESSAGE):
                raise ValueError("Missing/unsafe/oversized expected artifact: " + name)
            raw = path.read_bytes()
            value = raw.decode("utf-8")
            if path.suffix == ".json":
                json.loads(value)
            texts[name] = value
        state["artifact_text"] = texts
        state["result"]["artifacts"] = ["artifacts/" + task["job_id"] + "/" + name for name in texts]
        state["result"]["artifacts_sha256"] = {
            "artifacts/" + task["job_id"] + "/" + name: digest(value.encode())
            for name, value in texts.items()}

    def authorize_main(self, task, raw):
        """An opt-in main deployment accepts only the controller's exact active job."""
        if not self.config.get("main_integration"):
            return True
        if task["kind"] != "experiment":
            raise ValueError("New smoke jobs are outside the deployed main acceptance plan")
        from main_control import authorize_task
        state_raw = self.read(self.ref("requests_branch"), "control/main.json")
        return authorize_task(self.root, task, raw, json.loads(state_raw) if state_raw else None)

    def execute(self, task, raw):
        state = self.new_state(task, raw)
        result = state["result"]
        if task["kind"] not in self.config["allowed_kinds"]:
            result.update(status="blocked", reason="Smoke-only deployment: experiment execution is disabled",
                          training_started=False, finished_at=now())
            self.save(state)
            self.deliver(state)
            return result
        workspace = Path(result["workspace"])
        if workspace.exists():
            raise RuntimeError("Workspace already exists without state; inspect manually")
        if self.git("merge-base", "--is-ancestor", task["code_sha"], self.ref("code_branch"), check=False).returncode:
            raise ValueError("Task SHA is not reachable from the configured code branch")
        # Durable claim BEFORE checkout or child launch. Even a crash in the next
        # statement cannot cause an automatic second launch on the next tick.
        result["training_started"] = False
        self.save(state)
        self.deliver(state)
        process = None
        watchdog = None
        timed_out = threading.Event()
        started = time.monotonic()
        try:
            self.git("worktree", "add", "--detach", str(workspace), task["code_sha"])
            actual = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], env=self.env).decode().strip()
            if actual != task["code_sha"]:
                raise RuntimeError("Checkout SHA mismatch")
            result["actual_code_sha"] = actual
            env = dict(self.env, PYTHONDONTWRITEBYTECODE="1")
            env["PATH"] = str(Path(self.config["python"]).parent) + os.pathsep + env.get("PATH", "/usr/bin:/bin")
            env["LAB_BUFFER_JOB_ID"] = task["job_id"]
            env["LAB_BUFFER_CODE_SHA"] = task["code_sha"]
            if task["kind"] == "smoke":
                argv = [self.config["python"], "-u", "-B", "scripts/lab_buffer/smoke.py"]
                env["CUDA_VISIBLE_DEVICES"] = ""
            else:
                argv = list(task["argv"])
                if argv[0] == "python":
                    argv[0] = self.config["python"]
                env["CUDA_VISIBLE_DEVICES"] = str(self.config.get("gpu", "0"))
                if self.config.get("main_integration"):
                    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0",
                               CUBLAS_WORKSPACE_CONFIG=":4096:8")
            result["argv"] = argv
            log = Path(result["log_path"])
            with log.open("xb") as handle:
                process = subprocess.Popen(argv, cwd=workspace, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                watchdog = threading.Thread(target=self.supervise, args=(
                    process, started + task["timeout_seconds"], timed_out), daemon=True) if not self.config.get("main_integration") else None
                if watchdog:
                    watchdog.start()
                result.update(status="running", pid=process.pid,
                              training_started=task["kind"] == "experiment" and not self.config.get("main_integration"),
                              process_started=True)
                self.save(state)
                next_heartbeat = 0.0
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if (self.config.get("main_integration") and elapsed > task["timeout_seconds"]
                            and not timed_out.is_set()):
                        timed_out.set()
                        result["deadline_exceeded"] = True
                        result["reason"] = "Main task deadline exceeded; child continues, no termination authorized"
                        self.pause("Main task exceeded deadline; child NOT terminated: " + task["job_id"])
                    if elapsed >= next_heartbeat:
                        result["elapsed_seconds"] = round(elapsed, 3)
                        self.save(state)
                        try:
                            self.deliver(state)
                        except Exception as error:
                            print("Heartbeat upload deferred: " + str(error), file=sys.stderr, flush=True)
                        next_heartbeat = elapsed + self.config.get("heartbeat_seconds", 60)
                    time.sleep(0.1)
            if watchdog:
                watchdog.join()
            result["exit_code"] = process.returncode
            result["status"] = ("timed_out" if timed_out.is_set() else
                                "completed" if process.returncode == 0 else "failed")
            if result["status"] == "completed":
                self.collect_artifacts(task, state, workspace)
        except BaseException as error:
            result["reason"] = type(error).__name__ + ": " + str(error)
            if process is not None and process.poll() is None:
                result["status"] = "interrupted"
                self.pause("Child may still be running: " + task["job_id"])
            else:
                result["status"] = "failed"
                result["exit_code"] = process.returncode if process is not None else None
        finally:
            if result["status"] != "completed" and workspace.is_dir():
                # Return any completed evidence even on failure; never require all
                # success artifacts in order to report the error itself.
                for name in task.get("artifacts", []):
                    try:
                        partial = dict(task, artifacts=[name])
                        saved = dict(state.get("artifact_text", {}))
                        self.collect_artifacts(partial, state, workspace)
                        saved.update(state["artifact_text"])
                        state["artifact_text"] = saved
                    except (ValueError, OSError, UnicodeError):
                        pass
                state["result"]["artifacts"] = ["artifacts/" + task["job_id"] + "/" + name
                                                  for name in state.get("artifact_text", {})]
                state["result"]["artifacts_sha256"] = {
                    "artifacts/" + task["job_id"] + "/" + name: digest(value.encode())
                    for name, value in state.get("artifact_text", {}).items()}
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["finished_at"] = now()
            self.save(state)
        self.deliver(state)
        return result

    def tick(self):
        (self.root / "state").mkdir(exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        (self.root / "workspaces").mkdir(exist_ok=True)
        with (self.root / "runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"status": "busy"}
            self.recover_and_flush()
            if (self.root / "PAUSED.json").exists():
                return {"status": "paused", "reason": "Inspect PAUSED.json; no jobs launched"}
            self.fetch()
            paths = self.git("ls-tree", "-r", "--name-only", self.ref("requests_branch"), "--", "jobs/").stdout.decode().splitlines()
            for path in sorted(paths):
                if not path.endswith(".json"):
                    continue
                job_id = Path(path).stem
                if path != "jobs/" + job_id + ".json" or not NAME.fullmatch(job_id):
                    self.pause("Invalid task filename: " + path)
                    raise ValueError("Invalid task filename; mailbox paused for inspection")
                raw = self.read(self.ref("requests_branch"), path)
                previous_raw = self.read(self.ref("results_branch"), path)
                local = self.state_path(job_id)
                if local.exists() or previous_raw is not None:
                    previous = (json.loads(local.read_text())["result"] if local.exists()
                                else json.loads(previous_raw))
                    if previous["request_sha256"] != digest(raw):
                        self.pause("Immutable task changed: " + job_id)
                        raise RuntimeError("Task mutation detected; no jobs launched")
                    if previous["status"] not in FINAL:
                        self.pause("Remote task may be active: " + job_id)
                        return {"status": "paused"}
                    continue
                try:
                    task = validate_task(raw, path)
                except (ValueError, TypeError) as error:
                    rejected = {"job_id": job_id, "kind": "invalid", "code_sha": None}
                    state = self.new_state(rejected, raw)
                    state["result"].update(status="failed", finished_at=now(),
                                           reason="Invalid request: " + str(error))
                    self.save(state)
                    self.deliver(state)
                    return state["result"]
                if task["worker_id"] != self.config["worker_id"]:
                    continue
                if not self.authorize_main(task, raw):
                    return {"status": "control_wait", "training_started": False}
                # Experimental jobs are never started while a separate GPU job
                # is visible. This is advisory; smoke mode is the hard rollout gate.
                if task["kind"] == "experiment" and "experiment" in self.config["allowed_kinds"]:
                    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                         capture_output=True, text=True, timeout=10)
                    if gpu.returncode or gpu.stdout.strip():
                        return {"status": "gpu_busy", "training_started": False}
                return self.execute(task, raw)
            return {"status": "idle"}


def initialize(root, remote, python, worker):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError("Root already exists; installation refuses to overwrite it")
    if not NAME.fullmatch(worker) or not Path(python).is_absolute() or not os.access(python, os.X_OK):
        raise ValueError("Expected valid worker id and absolute executable Python")
    root.mkdir(parents=True)
    config = {"worker_id": worker, "python": python,
              "allowed_kinds": ["smoke"], "requests_branch": "lab-requests",
              "results_branch": "lab-results", "code_branch": "vibe-dev",
              "heartbeat_seconds": 60, "gpu": "0"}
    atomic_json(root / "config.json", config)
    subprocess.run(["git", "init", "--bare", "--quiet", str(root / "transport.git")], check=True)
    subprocess.run(["git", "--git-dir=" + str(root / "transport.git"), "remote", "add", "origin", remote], check=True)
    runner = Runner(root)
    runner.fetch()
    # Validate non-interactive write credentials before installing the timer.
    runner.publish({"workers/" + worker + ".json": (json.dumps({
        "schema_version": 1, "worker_id": worker, "status": "installed_smoke_only",
        "allowed_kinds": ["smoke"], "time": now(), "training_started": False}) + "\n").encode()})
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tick = commands.add_parser("tick")
    tick.add_argument("--root", required=True)
    init = commands.add_parser("init")
    init.add_argument("--root", required=True)
    init.add_argument("--remote", required=True)
    init.add_argument("--python", required=True)
    init.add_argument("--worker", default="server2")
    args = parser.parse_args()
    try:
        if args.command == "init":
            root = initialize(args.root, args.remote, args.python, args.worker)
            print(json.dumps({"status": "initialized", "root": str(root), "allowed_kinds": ["smoke"]}))
        else:
            print(json.dumps(Runner(args.root).tick(), ensure_ascii=False))
    except Exception as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
