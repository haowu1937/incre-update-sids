#!/usr/bin/env bash
set -euo pipefail
WS_PREVIOUS_ROOT="${1:?Usage: bash weight_sweep_bundle/run_weight_sweep.sh PREVIOUS_RESULTS_ROOT E1_DIR [WORKERS] [GPU_INDEX]}"
WS_E1_DIR="${2:?E1_DIR required}"
WS_WORKERS="${3:-3}"
WS_DEVICE="${4:-0}"
WS_RUN_ID="weight_sweep_$(date -u +%Y%m%dT%H%M%SZ)"
WS_OUTPUT="results/incremental/Beauty/${WS_RUN_ID}"
if [[ -e "${WS_RUN_ID}_paths.sh" ]]; then echo 'Run paths already exist' >&2; exit 1; fi
for WS_KEY in WS_PREVIOUS_ROOT WS_E1_DIR WS_WORKERS WS_DEVICE WS_RUN_ID WS_OUTPUT; do
 printf '%s=%q\n' "$WS_KEY" "${!WS_KEY}"
done > "${WS_RUN_ID}_paths.sh"
python -u -B run_history_weight_sweep.py \
 --previous-root "$WS_PREVIOUS_ROOT" --e1-dir "$WS_E1_DIR" \
 --output "$WS_OUTPUT" --workers "$WS_WORKERS" --devices "$WS_DEVICE" \
 2>&1 | tee "${WS_RUN_ID}.log"
printf 'Share only: %s/compact.txt\n' "$WS_OUTPUT"
