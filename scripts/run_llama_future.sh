#!/usr/bin/env bash
# Both RQ snapshots: frozen TIGER Future and paired Update-finetuned TIGER Future.
set -euo pipefail
[[ $# -ge 1 && $# -le 2 ]] || {
  echo 'Usage: bash scripts/run_llama_future.sh E1_DIR [EXISTING_RUN_TO_RESUME]'
  exit 1
}
E1_DIR=$1
RESUME_RUN=${2:-}
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting Future evaluation.'
  exit 1
}
test -f "$E1_DIR/summary.json"
git diff --check
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -n $RESUME_RUN ]]; then
  RUN=$RESUME_RUN
  test -d "$RUN/evaluation"
  test -f "$RUN/evaluation/config.json"
else
  mkdir -p results/incremental/Beauty
  RUN=$(mktemp -d "results/incremental/Beauty/llama_future_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  git rev-parse HEAD > "$RUN/commit.txt"
  git branch --show-current > "$RUN/branch.txt"
  cp -- scripts/run_llama_future.sh "$RUN/run_command.sh"
  printf 'E1_DIR=%s\nUPDATE_TIGER_STEPS=2000\n' "$E1_DIR" > "$RUN/paths.txt"
fi
ATTEMPT=$(mktemp -d "$RUN/attempt_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
git rev-parse HEAD > "$ATTEMPT/commit.txt"
finish_paths() {
  printf 'RESULT_DIR=%s\nFUTURE_DIR=%s/evaluation\nATTEMPT_DIR=%s\n' "$RUN" "$RUN" "$ATTEMPT"
}
printf 'Protocol and real tiny TIGER checks: %s/checks.log\n' "$ATTEMPT"
if ! { python -B tests/incremental/test_future_protocol.py &&
       python -B tests/incremental/test_future_tiger.py; } > "$ATTEMPT/checks.log" 2>&1; then
  tail -n 100 "$ATTEMPT/checks.log"
  finish_paths
  exit 1
fi
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
EVAL=(python -u -B -c "$BOOTSTRAP" evaluate_incremental_future.py
  --e1-dir "$E1_DIR" --output "$RUN/evaluation" --device cuda:0
  --max-steps 2000 --lr 0.003 --warmup-steps 200 --eval-batch-size 16
  --probe-size 512 --selection-steps 100 250 500 1000 2000
  --topk 5 10 20 --num-beams 20 --log-every 100 --save-every 50)
if [[ -n $RESUME_RUN ]]; then EVAL+=(--resume); fi
printf '%q ' "${EVAL[@]}" > "$ATTEMPT/evaluate_command.sh"
printf '\n' >> "$ATTEMPT/evaluate_command.sh"
printf 'Future pipeline log: %s/evaluate.log\n' "$ATTEMPT"
if ! "${EVAL[@]}" > "$ATTEMPT/evaluate.log" 2>&1; then
  tail -n 100 "$ATTEMPT/evaluate.log"
  finish_paths
  exit 1
fi
python -B - "$RUN/evaluation/summary.json" "$RUN/compact.txt" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text())
compact = {key: report[key] for key in ('execution_status', 'config', 'pair_checks', 'plain_minus_base',
    'catalog_policy', 'evaluation_target_policy', 'future_used_for_selection',
    'selection_locked_before_future', 'original_checkpoints_unchanged', 'runtime_seconds',
    'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')}
compact['training'] = {arm: {key: training[key] for key in ('update_optimizer_steps',
    'processed_examples', 'best_update_optimizer_step', 'best_update_validation', 'full_update_validation',
    'exact_base_warm_start', 'selected_checkpoint_reload_equal', 'runtime_seconds')}
    for arm, training in report['training'].items()}
compact['future'] = {arm: {state: {key: value for key, value in metrics.items() if key != 'slices'}
    for state, metrics in states.items()} for arm, states in report['future'].items()}
text = json.dumps(compact, indent=2, allow_nan=False)+'\n'
Path(sys.argv[2]).write_text(text)
print(text, end='')
PY
finish_paths
printf 'REPORT_FRAGMENT=%s/evaluation/experiment_report_fragment.md\n' "$RUN"
