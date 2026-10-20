#!/usr/bin/env bash
# Invoke from an independent bootstrap clone, never from the training checkout.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: bash install.sh ROOT ABSOLUTE_PYTHON'; exit 1; }
LAB_ROOT=$1
LAB_PYTHON=$2
[[ $LAB_ROOT = /* && $LAB_ROOT != *[[:space:]\%\']* ]] || {
  echo 'Use an absolute ROOT without whitespace, percent signs or quotes.'; exit 1;
}
command -v crontab >/dev/null
command -v git >/dev/null
[[ -x $LAB_PYTHON ]] || { echo 'Python executable does not exist.'; exit 1; }
[[ $LAB_PYTHON = /* && $LAB_PYTHON != *[[:space:]\%\']* ]] || {
  echo 'Use an absolute Python path without whitespace, percent signs or quotes.'; exit 1;
}
LAB_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_BOOTSTRAP=$(git -C "$LAB_SCRIPT_DIR" rev-parse --show-toplevel)
LAB_REMOTE=$(git -C "$LAB_BOOTSTRAP" remote get-url origin)
LAB_VERSION=$(git -C "$LAB_BOOTSTRAP" rev-parse HEAD)
"$LAB_PYTHON" -B "$LAB_SCRIPT_DIR/runner.py" init \
  --root "$LAB_ROOT" --remote "$LAB_REMOTE" --python "$LAB_PYTHON" --worker server2
mkdir -p "$LAB_ROOT/bin"
cp -- "$LAB_SCRIPT_DIR/runner.py" "$LAB_ROOT/bin/runner.py"
printf '%s\n' "$LAB_VERSION" > "$LAB_ROOT/runner-code-sha.txt"
# Merge one labelled entry; preserve all pre-existing cron entries verbatim.
LAB_CRON_TMP=$(mktemp)
trap 'rm -f -- "$LAB_CRON_TMP"' EXIT
if LC_ALL=C crontab -l > "$LAB_CRON_TMP" 2> "$LAB_ROOT/crontab-read.stderr"; then
  :
else
  if ! "$LAB_PYTHON" -B - "$LAB_ROOT/crontab-read.stderr" <<'PY'
from pathlib import Path
import sys
sys.exit(0 if 'no crontab for' in Path(sys.argv[1]).read_text().lower() else 1)
PY
  then echo 'Unable to read crontab; stopped before installing schedule.'; exit 1; fi
fi
if "$LAB_PYTHON" -B - "$LAB_CRON_TMP" <<'PY'
from pathlib import Path
import sys
sys.exit(0 if '# git-lab-buffer-server2' in Path(sys.argv[1]).read_text() else 1)
PY
then echo 'A labelled Runner entry already exists; stopped without replacing it.'; exit 1; fi
cp -- "$LAB_CRON_TMP" "$LAB_ROOT/crontab-before.txt"
printf '\n* * * * * %q -B %q tick --root %q >> %q 2>&1 # git-lab-buffer-server2\n' \
  "$LAB_PYTHON" "$LAB_ROOT/bin/runner.py" "$LAB_ROOT" "$LAB_ROOT/runner.log" >> "$LAB_CRON_TMP"
crontab "$LAB_CRON_TMP"
echo "Runner installed: $LAB_ROOT"
echo 'Mode: smoke only; experiments disabled. Polling: every minute.'
"$LAB_PYTHON" -B "$LAB_ROOT/bin/runner.py" tick --root "$LAB_ROOT"
