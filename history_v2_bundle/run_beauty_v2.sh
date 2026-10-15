#!/usr/bin/env bash
# Run manually in frec-gpu from the existing repository root.
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f history_aware_update_v2.py ]]; then
  printf 'Run from the repository root after installing apply_v2.py.\n' >&2
  exit 1
fi

E1_DIR="${1:-results/incremental/Beauty/historical_damage_20261009T142825Z}"
HA_RUN_ID="historical_joint_v2_$(date -u +%Y%m%dT%H%M%SZ)"
HA_SMOKE_DIR="ckpt/incremental/Beauty/${HA_RUN_ID}_smoke"
HA_RQVAE_DIR="ckpt/incremental/Beauty/${HA_RUN_ID}_1000"
HA_EVAL_DIR="results/incremental/Beauty/${HA_RUN_ID}_evaluation"
paths_file="${HA_RUN_ID}_paths.sh"
if [[ -e "$paths_file" ]]; then
  printf 'Run ID already exists; start again with a fresh timestamp.\n' >&2
  exit 1
fi
{
  printf 'E1_DIR=%q\n' "$E1_DIR"
  printf 'HA_RUN_ID=%q\n' "$HA_RUN_ID"
  printf 'HA_SMOKE_DIR=%q\n' "$HA_SMOKE_DIR"
  printf 'HA_RQVAE_DIR=%q\n' "$HA_RQVAE_DIR"
  printf 'HA_EVAL_DIR=%q\n' "$HA_EVAL_DIR"
} > "$paths_file"
printf 'Run paths saved to %s\n' "$paths_file"

python -u -B history_aware_update_v2.py train \
  --e1-dir "$E1_DIR" --output "$HA_SMOKE_DIR" --device cuda \
  --max-steps 50 --history-gradient-ratio 0.1 --anchor-batch-size 32 \
  --log-every 10 --save-every 10 \
  2>&1 | tee "${HA_RUN_ID}_smoke.log"
python -B "$bundle_dir/show_v2.py" "$HA_SMOKE_DIR" --require-steps 50

python -u -B history_aware_update_v2.py train \
  --e1-dir "$E1_DIR" --output "$HA_RQVAE_DIR" --device cuda \
  --max-steps 1000 --history-gradient-ratio 0.1 --anchor-batch-size 32 \
  --resume-from "$HA_SMOKE_DIR/last.pth" --save-every 10 \
  2>&1 | tee "${HA_RUN_ID}_1000.log"
python -B "$bundle_dir/show_v2.py" "$HA_RQVAE_DIR" --require-steps 1000

python -u -B history_aware_update_v2.py evaluate \
  --e1-dir "$E1_DIR" --rqvae-dir "$HA_RQVAE_DIR" --output "$HA_EVAL_DIR" \
  --device cuda --batch-size 32 --ranking \
  2>&1 | tee "${HA_RUN_ID}_evaluation.log"
python -B "$bundle_dir/show_v2.py" "$HA_RQVAE_DIR" "$HA_EVAL_DIR" \
  | tee "${HA_RUN_ID}_compact.txt"
printf 'Share the compact result in %s\n' "${HA_RUN_ID}_compact.txt"
