#!/usr/bin/env bash
# Evaluate the completed retuned History-aware RQ with frozen and fine-tuned TIGER.
set -euo pipefail
[[ $# == 2 || $# == 3 ]] || {
  echo 'Usage: bash scripts/run_llama_historical_future.sh E1_DIR RETUNED_EXPERIMENT_DIR [EXISTING_FUTURE_RUN_TO_RESUME]'
  exit 1
}
cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet || ! git diff --cached --quiet ||
   [[ -n $(git ls-files --others --exclude-standard -- . ':(exclude)verify_incremental_embeddings.py') ]]; then
  git status --short
  echo '工作区存在其他修改，已停止并保留现场。'; exit 1
fi
E1_DIR=$1
RETUNED_DIR=$2
test -f "$E1_DIR/summary.json"
test -f "$RETUNED_DIR/summary.json"
test -f "$RETUNED_DIR/historical/last.pth"
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESUME=()
if [[ $# == 3 ]]; then
  RUN=$3
  test -f "$RUN/evaluation/config.json"
  RESUME=(--resume)
else
  mkdir -p results/incremental/Beauty
  RUN=$(mktemp -d "results/incremental/Beauty/llama_historical_future_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  git rev-parse HEAD > "$RUN/commit.txt"
  git branch --show-current > "$RUN/branch.txt"
  cp -- scripts/run_llama_historical_future.sh "$RUN/run_command.sh"
  printf 'E1_DIR=%s\nRETUNED_DIR=%s\nTIGER_UPDATE_STEPS=2000\n' "$E1_DIR" "$RETUNED_DIR" > "$RUN/paths.txt"
fi
exec 9>"$RUN/run.lock"
flock -n 9 || { echo 'This output already has an active launcher; stopped.'; exit 1; }
ATTEMPT=$(mktemp -d "$RUN/attempt_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
git rev-parse HEAD > "$ATTEMPT/commit.txt"
finish_paths() {
  printf 'RESULT_DIR=%s\nFUTURE_DIR=%s/evaluation\nATTEMPT_DIR=%s\n' "$RUN" "$RUN" "$ATTEMPT"
}
finish_paths
printf 'Protocol and real RQ-VAE/TIGER checks: %s/checks.log\n' "$ATTEMPT"
if ! { python -B tests/incremental/test_future_protocol.py &&
       python -B tests/incremental/test_historical_future.py --require-torch; } > "$ATTEMPT/checks.log" 2>&1; then
  tail -n 100 "$ATTEMPT/checks.log"; finish_paths; exit 1
fi
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
CMD=(python -u -B -c "$BOOTSTRAP" evaluate_historical_future.py
  --e1-dir "$E1_DIR" --retuned-dir "$RETUNED_DIR" --output "$RUN/evaluation" --device cuda:0
  --max-steps 2000 --lr 0.003 --warmup-steps 200 --eval-batch-size 16 --probe-size 512
  --selection-steps 100 250 500 1000 2000 --topk 5 10 20 --num-beams 20
  --log-every 100 --save-every 50 "${RESUME[@]}")
printf '%q ' "${CMD[@]}" > "$ATTEMPT/evaluate_command.sh"
printf '\n' >> "$ATTEMPT/evaluate_command.sh"
printf 'TIGER fine-tuning and Future log: %s/evaluate.log\n' "$ATTEMPT"
if ! "${CMD[@]}" > "$ATTEMPT/evaluate.log" 2>&1; then
  tail -n 100 "$ATTEMPT/evaluate.log"; finish_paths; exit 1
fi
python -B - "$RUN" <<'PY'
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
r = json.loads((run/'evaluation/compact.json').read_text())
t = r['training']
short = {k:r[k] for k in ('execution_status', 'condition_checks', 'future_used_for_selection',
    'selection_locked_before_future', 'original_inputs_unchanged', 'rqvae_training_performed',
    'runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
short['training'] = {k:t[k] for k in ('update_optimizer_steps', 'processed_examples',
    'best_update_optimizer_step', 'exact_base_warm_start', 'selected_checkpoint_reload_equal')}
short['selected_update_validation'] = t['full_update_validation']['full_denominator']
short['future'] = {s:{k:m[k] for k in ('all_queries', 'servable_queries', 'unavailable_targets',
    'coverage', 'eligible_items', 'invalid_predictions', 'update_optimizer_steps',
    'full_denominator', 'conditional_on_servable', 'runtime_seconds')} for s,m in r['future'].items()}
short['finetuned_minus_frozen'] = r['finetuned_minus_frozen']
(run/'key_results.json').write_text(json.dumps(short, indent=2, allow_nan=False)+'\n')
for key, value in short.items():
    print(key+': '+json.dumps(value, allow_nan=False))
PY
finish_paths
printf 'KEY_RESULTS=%s/key_results.json\n' "$RUN"
