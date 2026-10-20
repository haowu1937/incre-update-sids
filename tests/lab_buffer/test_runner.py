"""Real local-Git integration tests. No GPU, network, cron modification or ML."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lab_runner", REPO / "scripts/lab_buffer/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def git(path, *args, data=None):
    return subprocess.run(["git", "-C", str(path), *args], input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout.decode().strip()


class MailboxTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.remote = self.base / "remote.git"
        self.author = self.base / "author"
        subprocess.run(["git", "init", "--bare", "-q", str(self.remote)], check=True)
        subprocess.run(["git", "init", "-q", str(self.author)], check=True)
        git(self.author, "config", "user.name", "Test")
        git(self.author, "config", "user.email", "test@localhost")
        git(self.author, "checkout", "-b", "vibe-dev")
        for name in ("scripts/lab_buffer/smoke.py", "scripts/lab_buffer/runner.py", "scripts/lab_buffer/install.sh"):
            target = self.author / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / name).read_bytes())
        git(self.author, "add", ".")
        git(self.author, "commit", "-qm", "code")
        self.sha = git(self.author, "rev-parse", "HEAD")
        git(self.author, "remote", "add", "origin", str(self.remote))
        git(self.author, "push", "-q", "origin", "vibe-dev")
        for branch in ("lab-requests", "lab-results"):
            git(self.author, "branch", branch, self.sha)
            git(self.author, "push", "-q", "origin", branch)
        self.root = self.base / "worker"
        runner.initialize(self.root, str(self.remote), sys.executable, "server2")
        self.worker = runner.Runner(self.root)

    def task(self, job="smoke-001", **kwargs):
        value = dict(schema_version=1, job_id=job, worker_id="server2", kind="smoke",
                     code_sha=self.sha, timeout_seconds=30, created_at=runner.now())
        value.update(kwargs)
        return value

    def submit(self, value):
        git(self.author, "checkout", "lab-requests")
        path = self.author / "jobs" / (value["job_id"] + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        git(self.author, "add", "jobs")
        git(self.author, "commit", "-qm", "task")
        git(self.author, "push", "-q", "origin", "lab-requests")

    def remote_result(self, job):
        self.worker.fetch(("results_branch",))
        return json.loads(self.worker.read(self.worker.ref("results_branch"), "jobs/" + job + ".json"))

    def experiment_config(self):
        self.worker.config["allowed_kinds"] = ["smoke", "experiment"]

    def test_smoke_roundtrip_restart_and_no_main_changes(self):
        sentinel = self.base / "main-training" / "still-running.txt"
        sentinel.parent.mkdir()
        sentinel.write_text("untouched")
        self.submit(self.task())
        result = self.worker.tick()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["training_started"])
        report = self.remote_result("smoke-001")
        self.assertEqual(report["actual_code_sha"], self.sha)
        self.assertEqual(report["exit_code"], 0)
        artifact = self.worker.read(self.worker.ref("results_branch"), report["artifacts"][0])
        self.assertEqual(json.loads(artifact)["cuda_visible_devices"], "")
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertEqual(runner.Runner(self.root).tick()["status"], "idle")
        # Even if the local job record is lost, the remote claim prevents rerun.
        self.worker.state_path("smoke-001").unlink()
        self.assertEqual(self.worker.tick()["status"], "idle")

    def test_upload_failure_retries_delivery_without_execution(self):
        self.submit(self.task())
        original = self.worker.deliver
        def delivery(state):
            if state["result"]["status"] == "completed":
                raise RuntimeError("simulated connection loss")
            return original(state)
        with patch.object(self.worker, "deliver", side_effect=delivery):
            with self.assertRaisesRegex(RuntimeError, "connection loss"):
                self.worker.tick()
        state = json.loads(self.worker.state_path("smoke-001").read_text())
        self.assertEqual(state["result"]["status"], "completed")
        self.assertTrue(state["pending_upload"])
        with patch.object(self.worker, "execute", side_effect=AssertionError("must not rerun")):
            self.assertEqual(self.worker.tick()["status"], "idle")
        self.assertEqual(self.remote_result("smoke-001")["status"], "completed")

    def test_launch_crash_pauses_without_starting_another_job(self):
        task = self.task()
        self.submit(task)
        raw = (self.author / "jobs/smoke-001.json").read_bytes()
        state = self.worker.new_state(task, raw)
        self.worker.save(state)
        self.submit(self.task("smoke-002"))
        with patch.object(self.worker, "execute", side_effect=AssertionError("must not execute")):
            self.assertEqual(self.worker.tick()["status"], "paused")
        self.assertEqual(self.remote_result("smoke-001")["status"], "interrupted")
        self.assertTrue((self.root / "PAUSED.json").is_file())

    def test_smoke_gate_blocks_experiment(self):
        self.submit(self.task("exp-001", kind="experiment", argv=["python", "-c", "raise Exception('not allowed')"]))
        result = self.worker.tick()
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["training_started"])
        self.assertFalse(Path(result["workspace"]).exists())

    def test_nonzero_exit_is_returned(self):
        self.experiment_config()
        self.worker.config["gpu"] = ""
        self.submit(self.task("failure-001", kind="experiment", argv=["python", "-c", "print('failure detail'); raise SystemExit(7)"]))
        idle = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        real_run = subprocess.run
        def route(args, **kwargs):
            return idle if args[0] == "nvidia-smi" else real_run(args, **kwargs)
        with patch.object(subprocess, "run", side_effect=route):
            result = self.worker.tick()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertIn(b"failure detail", self.worker.read(self.worker.ref("results_branch"), result["log_tail"]))

    def test_changed_task_pauses_without_second_execution(self):
        original = self.task()
        self.submit(original)
        self.worker.tick()
        original["timeout_seconds"] = 29
        self.submit(original)
        with self.assertRaisesRegex(RuntimeError, "mutation"):
            self.worker.tick()
        self.assertTrue((self.root / "PAUSED.json").exists())

    def test_wrong_worker_is_ignored(self):
        self.submit(self.task(worker_id="other-server"))
        self.assertEqual(self.worker.tick()["status"], "idle")

    def test_invalid_request_is_returned_and_not_repeated(self):
        self.submit(self.task(timeout_seconds=True))
        result = self.worker.tick()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Invalid request", result["reason"])
        self.assertFalse(result["training_started"])
        self.assertEqual(self.worker.tick()["status"], "idle")

    def test_live_worker_lock_prevents_second_launch(self):
        with (self.root / "runner.lock").open("a") as handle:
            runner.fcntl.flock(handle, runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB)
            self.assertEqual(self.worker.tick()["status"], "busy")

    def test_gpu_busy_defers_opted_in_experiment(self):
        self.experiment_config()
        self.submit(self.task("exp-001", kind="experiment", argv=["python", "-c", "raise SystemExit(0)"]))
        original = subprocess.run
        def route(args, **kwargs):
            if args[0] == "nvidia-smi":
                return subprocess.CompletedProcess([], 0, stdout="12345\n", stderr="")
            return original(args, **kwargs)
        with patch.object(subprocess, "run", side_effect=route):
            self.assertEqual(self.worker.tick()["status"], "gpu_busy")
        self.assertFalse(self.worker.state_path("exp-001").exists())

    def test_deadline_applies_during_slow_heartbeat_delivery(self):
        self.experiment_config()
        self.submit(self.task("timeout-001", kind="experiment", timeout_seconds=1,
                              argv=["python", "-c", "import time; time.sleep(30)"]))
        original_run = subprocess.run
        original_delivery = self.worker.deliver
        def route(args, **kwargs):
            if args[0] == "nvidia-smi":
                return subprocess.CompletedProcess([], 0, stdout="", stderr="")
            return original_run(args, **kwargs)
        def slow_delivery(state):
            if state["result"]["status"] == "running":
                time.sleep(1.5)
            return original_delivery(state)
        with patch.object(subprocess, "run", side_effect=route), patch.object(self.worker, "deliver", side_effect=slow_delivery):
            result = self.worker.tick()
        self.assertEqual(result["status"], "timed_out")
        self.assertLess(result["elapsed_seconds"], 5)
        self.assertIsNotNone(result["exit_code"])

    def test_read_write_credentials_checked_before_scheduler(self):
        manifest = json.loads(self.worker.read(self.worker.ref("results_branch"), "workers/server2.json"))
        self.assertEqual(manifest["allowed_kinds"], ["smoke"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.initialize(self.root, str(self.remote), sys.executable, "server2")

    def test_installer_preserves_existing_cron_and_rejects_overwrite(self):
        fake_bin = self.base / "fake-bin"
        fake_bin.mkdir()
        cron_store = self.base / "crontab.txt"
        existing = "17 4 * * * /usr/bin/true # existing-main-task\n"
        cron_store.write_text(existing)
        crontab = fake_bin / "crontab"
        crontab.write_text("#!" + sys.executable + "\n" + '''import os,sys
from pathlib import Path
store=Path(os.environ["TEST_CRON_STORE"])
if sys.argv[1:]==["-l"]:
    if store.exists(): print(store.read_text(),end="")
    else:
        print("no crontab for test",file=sys.stderr)
        sys.exit(1)
else: store.write_bytes(Path(sys.argv[1]).read_bytes())
''')
        crontab.chmod(0o755)
        env = dict(os.environ, TEST_CRON_STORE=str(cron_store),
                   PATH=str(fake_bin) + os.pathsep + os.environ["PATH"])
        root = self.base / "installed-worker"
        command = ["bash", str(self.author / "scripts/lab_buffer/install.sh"), str(root), sys.executable]
        completed = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        installed = cron_store.read_text()
        self.assertTrue(installed.startswith(existing))
        self.assertEqual(installed.count("# git-lab-buffer-server2"), 1)
        self.assertEqual(json.loads((root / "config.json").read_text())["allowed_kinds"], ["smoke"])
        again = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(cron_store.read_text(), installed)


class ValidationTest(unittest.TestCase):
    def test_smoke_cannot_carry_arbitrary_command(self):
        task = dict(schema_version=1, job_id="x", worker_id="server2", kind="smoke",
                    code_sha="a" * 40, timeout_seconds=1, created_at="today", argv=["bash", "bad.sh"])
        with self.assertRaisesRegex(ValueError, "fixed command"):
            runner.validate_task(json.dumps(task).encode(), "jobs/x.json")

    def test_path_traversal_and_boolean_timeout_rejected(self):
        task = dict(schema_version=1, job_id="x", worker_id="server2", kind="experiment",
                    code_sha="a" * 40, timeout_seconds=1, created_at="today",
                    argv=["python", "x.py"], artifacts=["../secret.txt"])
        with self.assertRaisesRegex(ValueError, "relative"):
            runner.validate_task(json.dumps(task).encode(), "jobs/x.json")
        task.update(artifacts=[], timeout_seconds=True)
        with self.assertRaisesRegex(ValueError, "Timeout"):
            runner.validate_task(json.dumps(task).encode(), "jobs/x.json")


if __name__ == "__main__":
    unittest.main()
