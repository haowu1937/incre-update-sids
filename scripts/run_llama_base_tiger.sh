#!/usr/bin/env bash
# Independent TIGER training on an already verified frozen SID export.
# TOTAL_STEPS is cumulative; MODE=smoke defers full Base validation.
set -euo pipefail

[[ $# -ge 2 && $# -le 5 ]] || {
  echo 'Usage: bash scripts/run_llama_base_tiger.sh SID_DIR RQVAE_SHA256 [TOTAL_STEPS [RESUME_FROM [smoke|full]]]'
  exit 1
}
SID_DIR=$1
RQVAE_SHA256=$2
TOTAL_STEPS=${3:-50}
RESUME_FROM=${4:-}
MODE=${5:-smoke}
[[ $RQVAE_SHA256 =~ ^[0-9a-f]{64}$ && $TOTAL_STEPS =~ ^[1-9][0-9]*$ ]] || {
  echo 'Expected a full checkpoint SHA256 and a positive cumulative step budget.'
  exit 1
}
[[ $MODE == smoke || $MODE == full ]] || { echo 'MODE must be smoke or full.'; exit 1; }
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting training.'
  exit 1
}
test -f "$SID_DIR/summary.json"
if [[ -n $RESUME_FROM ]]; then
  test -f "$RESUME_FROM"
fi
git diff --check

export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B train_incremental_tiger.py --help >/dev/null
mkdir -p results/incremental/Beauty ckpt/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_base_tiger_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
TIGER_DIR="ckpt/incremental/Beauty/$(basename "$RUN")_${TOTAL_STEPS}"
git rev-parse HEAD > "$RUN/commit.txt"
git branch --show-current > "$RUN/branch.txt"
cp -- scripts/run_llama_base_tiger.sh "$RUN/run_command.sh"
printf 'SID_DIR=%s\nRQVAE_SHA256=%s\nTIGER_DIR=%s\nTOTAL_STEPS=%s\nRESUME_FROM=%s\nMODE=%s\nSEED=2026\n' \
  "$SID_DIR" "$RQVAE_SHA256" "$TIGER_DIR" "$TOTAL_STEPS" "$RESUME_FROM" "$MODE" > "$RUN/paths.txt"

# The bound engine resets CUDA memory statistics before constructing its model.
# Initialize CUDA in this same child process without changing bound source files.
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
TRAIN=(python -u -B -c "$BOOTSTRAP" train_incremental_tiger.py
  --sid-dir "$SID_DIR" --expected-rqvae-sha256 "$RQVAE_SHA256"
  --output "$TIGER_DIR" --device cuda:0 --seed 2026
  --batch-size 128 --eval-batch-size 16 --max-steps "$TOTAL_STEPS"
  --lr 0.003 --weight-decay 0.05 --warmup-steps 200
  --probe-size 512 --eval-every 500 --log-every 25 --save-every 250
  --num-beams 20 --topk 5 10)
if [[ -n $RESUME_FROM ]]; then
  TRAIN+=(--resume-from "$RESUME_FROM")
fi
if [[ $MODE == smoke ]]; then
  TRAIN+=(--smoke)
fi
printf '%q ' "${TRAIN[@]}" > "$RUN/train_command.sh"
printf '\n' >> "$RUN/train_command.sh"
printf 'Training log: %s/train.log\n' "$RUN"
if ! "${TRAIN[@]}" > "$RUN/train.log" 2>&1; then
  tail -n 100 "$RUN/train.log"
  printf 'RESULT_DIR=%s\nTIGER_DIR=%s\n' "$RUN" "$TIGER_DIR"
  exit 1
fi

python -B - "$TIGER_DIR/summary.json" "$RUN/compact.txt" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
keys = ('execution_status', 'execution_mode', 'parameters', 'optimizer_steps',
        'steps_this_run', 'processed_examples', 'resume_from', 'model_initialization',
        'rqvae_checkpoint_sha256', 'best_optimizer_step', 'validation_probe_queries',
        'best_validation_probe', 'full_base_validation', 'selected_checkpoint_reload_equal',
        'training_catalog', 'ranking_catalog', 'eligible_items', 'rqvae_frozen',
        'sid_refit', 'update_training_status', 'future_test_status',
        'runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')
compact = {key: summary[key] for key in keys}
compact['config'] = summary['config']
compact['sid_summary_sha256'] = summary['binding']['sid_summary_sha256']
text = json.dumps(compact, indent=2, allow_nan=False) + '\n'
Path(sys.argv[2]).write_text(text)
print(text, end='')
PY
printf 'RESULT_DIR=%s\nTIGER_DIR=%s\n' "$RUN" "$TIGER_DIR"
