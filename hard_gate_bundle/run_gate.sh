#!/usr/bin/env bash
set -euo pipefail
HG_E1_DIR="${1:?Usage: bash hard_gate_bundle/run_gate.sh E1_DIR FROZEN_PLAIN_DIR [UNGATED_HISTORY_DIR]}"
HG_PLAIN_DIR="${2:?Frozen encoder Plain output is required}"
HG_UNGATED_DIR="${3:-}"
HG_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HG_RUN_ID="hard_gated_$(date -u +%Y%m%dT%H%M%SZ)"
HG_RQVAE_DIR="ckpt/incremental/Beauty/${HG_RUN_ID}_100"
HG_COMPARE_DIR="results/incremental/Beauty/${HG_RUN_ID}_comparison"
if [[ ! -f "$HG_PLAIN_DIR/summary.json" ]]; then echo "Missing frozen Plain summary: $HG_PLAIN_DIR" >&2; exit 1; fi
if [[ -n "$HG_UNGATED_DIR" && ! -f "$HG_UNGATED_DIR/summary.json" ]]; then echo "Missing optional ungated summary: $HG_UNGATED_DIR" >&2; exit 1; fi
if [[ -e "${HG_RUN_ID}_paths.sh" ]]; then echo 'Run paths already exist' >&2; exit 1; fi
printf 'HG_E1_DIR=%q\nHG_PLAIN_DIR=%q\nHG_UNGATED_DIR=%q\nHG_RUN_ID=%q\nHG_RQVAE_DIR=%q\nHG_COMPARE_DIR=%q\n' \
 "$HG_E1_DIR" "$HG_PLAIN_DIR" "$HG_UNGATED_DIR" "$HG_RUN_ID" "$HG_RQVAE_DIR" "$HG_COMPARE_DIR" > "${HG_RUN_ID}_paths.sh"
printf 'Run paths saved: %s\n' "${HG_RUN_ID}_paths.sh"
python -u -B history_aware_hard_gate.py train \
 --e1-dir "$HG_E1_DIR" --plain-dir "$HG_PLAIN_DIR" --output "$HG_RQVAE_DIR" \
 --device cuda --max-steps 100 --history-gradient-ratio 0.1 \
 --anchor-batch-size 32 --temperature 0.1 --min-h-improvement 0.000001 \
 --rq-mse-ratio-cap 1.01 --log-every 10 --save-every 10 \
 2>&1 | tee "${HG_RUN_ID}_train.log"
HG_OPTIONAL_ARGS=()
if [[ -n "$HG_UNGATED_DIR" ]]; then HG_OPTIONAL_ARGS=(--ungated-dir "$HG_UNGATED_DIR"); fi
python -u -B history_aware_hard_gate.py compare \
 --rqvae-dir "$HG_RQVAE_DIR" --output "$HG_COMPARE_DIR" "${HG_OPTIONAL_ARGS[@]}" \
 2>&1 | tee "${HG_RUN_ID}_comparison.log"
python -B "$HG_BUNDLE_DIR/show_gate.py" "$HG_COMPARE_DIR" | tee "${HG_RUN_ID}_compact.txt"
printf 'Share only: %s\n' "${HG_RUN_ID}_compact.txt"
