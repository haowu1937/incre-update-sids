#!/usr/bin/env python3
"""Mailbox communication check. No torch import, data read, or GPU access."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


output = Path("outputs/lab-buffer-smoke.json")
output.parent.mkdir(exist_ok=True)
versions = {}
for package in ("torch", "transformers", "numpy"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
report = {
    "stage": "git_lab_buffer_smoke", "execution_status": "completed",
    "job_id": os.environ["LAB_BUFFER_JOB_ID"], "marker": "git-buffer-ok-v1",
    "python": platform.python_version(), "package_metadata_only": versions,
    "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "training_started": False, "gpu_used": False, "main_checkout_modified": False,
}
with output.open("x") as handle:
    json.dump(report, handle, indent=2)
    handle.write("\n")
print(json.dumps(report, ensure_ascii=False), flush=True)
