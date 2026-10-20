#!/usr/bin/env bash
# Run once from the pinned independent bootstrap clone. Existing root is reused.
set -euo pipefail
LAB_PYTHON=/home/hao/miniconda3/envs/frec-gpu/bin/python
LAB_ROOT=/home/hao/frec/git-lab-buffer
LAB_HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_CHECKOUT=$(git -C "$LAB_HERE" rev-parse --show-toplevel)
test -x "$LAB_PYTHON"
test -f "$LAB_ROOT/config.json"
test -f "$LAB_ROOT/bin/runner.py"
command -v crontab >/dev/null
if ! "$LAB_PYTHON" -B -m unittest discover -s "$LAB_CHECKOUT/tests/lab_buffer" -v > "$LAB_CHECKOUT/main-checks.log" 2>&1; then
  tail -n 100 "$LAB_CHECKOUT/main-checks.log"
  exit 1
fi
"$LAB_PYTHON" -u -B "$LAB_HERE/attach_main.py" --root "$LAB_ROOT"
nohup "$LAB_PYTHON" -u -B "$LAB_ROOT/bin/main_service.py" --root "$LAB_ROOT" \
  >> "$LAB_ROOT/runner.log" 2>&1 < /dev/null &
printf 'SERVER_SERVICE_PID=%s\nRUNNER_LOG=%s\n' "$!" "$LAB_ROOT/runner.log"
echo 'Attached to existing Runner; queued real artifact acceptance. No new training.'
echo 'Results and receipt return through Git; no output copying is needed.'
