#!/usr/bin/env bash
# Fresh Base pilot: three hidden layers and a 64-dimensional quantization space.
# Usage: bash scripts/run_llama_base_3hidden_e64.sh FEATURES_DIR [TOTAL_STEPS]
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
  echo 'Usage: bash scripts/run_llama_base_3hidden_e64.sh FEATURES_DIR [TOTAL_STEPS]'
  exit 1
}
FEATURES_DIR=$1
TOTAL_STEPS=${2:-50}
[[ $TOTAL_STEPS =~ ^[1-9][0-9]*$ ]] || {
  echo 'TOTAL_STEPS must be a positive integer.'
  exit 1
}
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting training.'
  exit 1
}
test -f "$FEATURES_DIR/summary.json"
test -f "$FEATURES_DIR/base_train/embeddings.npy"
git diff --check

export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B train_incremental_rqvae.py --help >/dev/null
mkdir -p results/incremental/Beauty ckpt/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_base_3hidden_e64_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
BASE_DIR="ckpt/incremental/Beauty/$(basename "$RUN")_${TOTAL_STEPS}"
git rev-parse HEAD > "$RUN/commit.txt"
git branch --show-current > "$RUN/branch.txt"
cp -- scripts/run_llama_base_3hidden_e64.sh "$RUN/run_command.sh"
printf 'FEATURES_DIR=%s\nBASE_DIR=%s\nTOTAL_STEPS=%s\nSEED=2026\n' \
  "$FEATURES_DIR" "$BASE_DIR" "$TOTAL_STEPS" > "$RUN/paths.txt"

TRAIN=(python -u -B train_incremental_rqvae.py
  --features-dir "$FEATURES_DIR" --output "$BASE_DIR"
  --device cuda:0 --max-steps "$TOTAL_STEPS"
  --batch-size 256 --lr 0.001 --weight-decay 0 --seed 2026
  --num-emb-list 256 256 256 --e-dim 64 --layers 512 256 128
  --quant-loss-weight 0.25 --kmeans-iters 50 --init-samples 4096
  --eval-batch-size 256 --save-every 25 --log-every 10)
printf '%q ' "${TRAIN[@]}" > "$RUN/train_command.sh"
printf '\n' >> "$RUN/train_command.sh"
printf 'Training log: %s/train.log\n' "$RUN"
if ! "${TRAIN[@]}" > "$RUN/train.log" 2>&1; then
  tail -n 100 "$RUN/train.log"
  printf 'RESULT_DIR=%s\n' "$RUN"
  exit 1
fi

python -B report_content_rqvae.py --features-dir "$FEATURES_DIR" \
  --rqvae-dir "$BASE_DIR" --output "$RUN/report" \
  2>&1 | tee "$RUN/report.log"
printf 'RESULT_DIR=%s\n' "$RUN"
