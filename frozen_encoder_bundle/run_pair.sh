#!/usr/bin/env bash
set -euo pipefail
FX_E1_DIR="${1:?Usage: bash frozen_encoder_bundle/run_pair.sh E1_DIR}"
FX_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FX_RUN_ID="frozen_encoder_$(date -u +%Y%m%dT%H%M%SZ)"
FX_PLAIN_DIR="ckpt/incremental/Beauty/${FX_RUN_ID}_plain_100"
FX_HISTORY_DIR="ckpt/incremental/Beauty/${FX_RUN_ID}_history_100"
FX_COMPARE_DIR="results/incremental/Beauty/${FX_RUN_ID}_comparison"
FX_PATHS_FILE="${FX_RUN_ID}_paths.sh"
if [[ -e "$FX_PATHS_FILE" ]]; then echo "Paths file already exists: $FX_PATHS_FILE" >&2; exit 1; fi
printf 'FX_E1_DIR=%q\nFX_RUN_ID=%q\nFX_PLAIN_DIR=%q\nFX_HISTORY_DIR=%q\nFX_COMPARE_DIR=%q\n' \
  "$FX_E1_DIR" "$FX_RUN_ID" "$FX_PLAIN_DIR" "$FX_HISTORY_DIR" "$FX_COMPARE_DIR" > "$FX_PATHS_FILE"
printf 'Run paths saved: %s\n' "$FX_PATHS_FILE"
python -u -B history_aware_frozen_encoder.py train \
  --mode plain --e1-dir "$FX_E1_DIR" --output "$FX_PLAIN_DIR" \
  --device cuda --max-steps 100 --anchor-batch-size 32 \
  --log-every 10 --save-every 10 --history-progress-every 0 \
  2>&1 | tee "${FX_RUN_ID}_plain.log"
python -u -B history_aware_frozen_encoder.py train \
  --mode history-aware --e1-dir "$FX_E1_DIR" --output "$FX_HISTORY_DIR" \
  --device cuda --max-steps 100 --history-gradient-ratio 0.1 \
  --anchor-batch-size 32 --log-every 10 --save-every 10 --history-progress-every 0 \
  2>&1 | tee "${FX_RUN_ID}_history.log"
python -u -B history_aware_frozen_encoder.py compare \
  --plain-dir "$FX_PLAIN_DIR" --history-dir "$FX_HISTORY_DIR" \
  --output "$FX_COMPARE_DIR" 2>&1 | tee "${FX_RUN_ID}_comparison.log"
python -B "$FX_BUNDLE_DIR/show_frozen.py" "$FX_COMPARE_DIR" \
  | tee "${FX_RUN_ID}_compact.txt"
printf 'Share only: %s\n' "${FX_RUN_ID}_compact.txt"
