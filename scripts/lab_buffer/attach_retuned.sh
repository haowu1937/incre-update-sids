#!/usr/bin/env bash
# Add only a text collector; do not change the installed Runner or experiment gate.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: bash attach_retuned.sh RUNNER_ROOT PROJECT_ROOT'; exit 1; }
LAB_ROOT=$1
LAB_PROJECT=$2
for LAB_PATH in "$LAB_ROOT" "$LAB_PROJECT"; do
  [[ $LAB_PATH = /* && $LAB_PATH != *[[:space:]\%\']* ]] || {
    echo 'Use absolute paths without whitespace, percent signs or quotes.'; exit 1;
  }
done
test -f "$LAB_ROOT/config.json"
test -f "$LAB_ROOT/bin/runner.py"
test -d "$LAB_PROJECT/results/incremental/Beauty"
command -v crontab >/dev/null
LAB_HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_PYTHON=$(python3 -B - "$LAB_ROOT/config.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['python'])
PY
)
[[ -x $LAB_PYTHON && $LAB_PYTHON = /* && $LAB_PYTHON != *[[:space:]\%\']* ]]
LAB_CRON=$(mktemp)
trap 'rm -f -- "$LAB_CRON"' EXIT
LC_ALL=C crontab -l > "$LAB_CRON"
if "$LAB_PYTHON" -B - "$LAB_CRON" <<'PY'
from pathlib import Path
import sys
sys.exit(0 if '# git-lab-retuned-collector' in Path(sys.argv[1]).read_text() else 1)
PY
then echo 'Collector schedule already exists; preserved current setup.'; exit 1; fi
if [[ -e $LAB_ROOT/bin/collect_retuned.py ]]; then
  cmp -- "$LAB_HERE/collect_retuned.py" "$LAB_ROOT/bin/collect_retuned.py" || {
    echo 'A different collector is installed; preserved it.'; exit 1;
  }
else
  cp -- "$LAB_HERE/collect_retuned.py" "$LAB_ROOT/bin/collect_retuned.py"
fi
git -C "$LAB_HERE" rev-parse HEAD > "$LAB_ROOT/collector-code-sha.txt"
# Try an initial capture. A busy Runner is retried by the scheduled collector.
"$LAB_PYTHON" -B "$LAB_ROOT/bin/collect_retuned.py" --root "$LAB_ROOT" --project "$LAB_PROJECT"
cp -- "$LAB_CRON" "$LAB_ROOT/crontab-before-collector.txt"
printf '\n* * * * * %q -B %q --root %q --project %q --delay-seconds 20 >> %q 2>&1 # git-lab-retuned-collector\n' \
  "$LAB_PYTHON" "$LAB_ROOT/bin/collect_retuned.py" "$LAB_ROOT" "$LAB_PROJECT" \
  "$LAB_ROOT/collector.log" >> "$LAB_CRON"
crontab "$LAB_CRON"
echo 'Read-only retuned collector attached. Experiments remain disabled.'
