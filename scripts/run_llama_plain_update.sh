#!/usr/bin/env bash
# Plain content update from the selected Llama Base RQ-VAE.
# UPDATE_STEPS excludes parent Base steps and includes any resumed Update steps.
set -euo pipefail

[[ $# -ge 4 && $# -le 6 ]] || {
  echo 'Usage: bash scripts/run_llama_plain_update.sh PROTOCOL_DIR FEATURES_DIR BASE_RQVAE_DIR BASE_SHA256 [UPDATE_STEPS [RESUME_FROM]]'
  exit 1
}
PROTOCOL_DIR=$1
FEATURES_DIR=$2
BASE_RQVAE_DIR=$3
BASE_SHA256=$4
UPDATE_STEPS=${5:-50}
RESUME_FROM=${6:-}
[[ $BASE_SHA256 =~ ^[0-9a-f]{64}$ && $UPDATE_STEPS =~ ^[1-9][0-9]*$ ]] || {
  echo 'Expected a full Base checkpoint SHA256 and a positive Update step budget.'
  exit 1
}
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting training.'
  exit 1
}
test -f "$PROTOCOL_DIR/summary.json"
test -f "$FEATURES_DIR/summary.json"
test -f "$BASE_RQVAE_DIR/summary.json"
test -f "$BASE_RQVAE_DIR/last.pth"
if [[ -n $RESUME_FROM ]]; then
  test -f "$RESUME_FROM"
fi
git diff --check

export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B train_plain_incremental_rqvae.py --help >/dev/null
mkdir -p results/incremental/Beauty ckpt/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_plain_update_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
UPDATE_DIR="ckpt/incremental/Beauty/$(basename "$RUN")_${UPDATE_STEPS}"
git rev-parse HEAD > "$RUN/commit.txt"
git branch --show-current > "$RUN/branch.txt"
cp -- scripts/run_llama_plain_update.sh "$RUN/run_command.sh"
printf 'PROTOCOL_DIR=%s\nFEATURES_DIR=%s\nBASE_RQVAE_DIR=%s\nBASE_SHA256=%s\nUPDATE_DIR=%s\nUPDATE_STEPS=%s\nRESUME_FROM=%s\nSEED=2026\n' \
  "$PROTOCOL_DIR" "$FEATURES_DIR" "$BASE_RQVAE_DIR" "$BASE_SHA256" \
  "$UPDATE_DIR" "$UPDATE_STEPS" "$RESUME_FROM" > "$RUN/paths.txt"

# Initialize CUDA before the source-bound engine resets its memory statistics.
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
TRAIN=(python -u -B -c "$BOOTSTRAP" train_plain_incremental_rqvae.py
  --protocol-dir "$PROTOCOL_DIR" --features-dir "$FEATURES_DIR"
  --base-rqvae-dir "$BASE_RQVAE_DIR" --expected-base-sha256 "$BASE_SHA256"
  --output "$UPDATE_DIR" --device cuda:0 --max-steps "$UPDATE_STEPS"
  --batch-size 256 --lr 0.001 --weight-decay 0 --seed 2026
  --eval-batch-size 256 --save-every 25 --log-every 10)
if [[ -n $RESUME_FROM ]]; then
  TRAIN+=(--resume-from "$RESUME_FROM")
fi
printf '%q ' "${TRAIN[@]}" > "$RUN/train_command.sh"
printf '\n' >> "$RUN/train_command.sh"
printf 'Training log: %s/train.log\n' "$RUN"
if ! "${TRAIN[@]}" > "$RUN/train.log" 2>&1; then
  tail -n 100 "$RUN/train.log"
  printf 'RESULT_DIR=%s\nUPDATE_DIR=%s\n' "$RUN" "$UPDATE_DIR"
  exit 1
fi

python -B - "$UPDATE_DIR/summary.json" "$RUN/compact.txt" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
keys = ('execution_status', 'parent_base_checkpoint_sha256', 'parent_base_optimizer_steps',
        'training_catalog', 'training_items', 'feature_dim', 'item_groups',
        'update_optimizer_steps', 'steps_this_run', 'processed_examples', 'resume_from',
        'warm_start_all_weights_equal', 'original_base_assignments_equal_before_update',
        'parent_base_checkpoint_unchanged', 'checkpoint_reload_assignments_equal',
        'initial_train_catalog_metrics', 'final_train_catalog_metrics',
        'initial_group_metrics', 'final_group_metrics', 'base_semantic_churn',
        'collision_statistics', 'trained_components', 'optimizer_reset_at_update_start',
        'codebook_reinitialization', 'codeword_alignment', 'historical_prediction_penalty_used',
        'collision_suffix_status', 'historical_prediction_damage_status',
        'recommendation_training_status', 'future_test_status', 'runtime_seconds',
        'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')
compact = {key: summary[key] for key in keys}
compact['config'] = summary['config']
text = json.dumps(compact, indent=2, allow_nan=False) + '\n'
Path(sys.argv[2]).write_text(text)
print(text, end='')
PY
printf 'RESULT_DIR=%s\nUPDATE_DIR=%s\n' "$RUN" "$UPDATE_DIR"
