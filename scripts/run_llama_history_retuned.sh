#!/usr/bin/env bash
# New paired Plain/History-aware recipe; original Base, full SIDs and frozen teacher.
set -euo pipefail
[[ $# == 1 || $# == 2 ]] || {
  echo 'Usage: bash scripts/run_llama_history_retuned.sh E1_DIR [EXISTING_RUN_TO_RESUME]'; exit 1;
}
cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet || ! git diff --cached --quiet ||
   [[ -n $(git ls-files --others --exclude-standard -- . ':(exclude)verify_incremental_embeddings.py') ]]; then
  git status --short
  echo '工作区存在其他修改，已停止并保留现场。'; exit 1
fi
E1=$1
test -f "$E1/summary.json"
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RESUME=()
if [[ $# == 2 ]]; then
  RUN=$2
  test -f "$RUN/experiment/request.json"
  test "$(cat "$RUN/commit.txt")" = "$(git rev-parse HEAD)" || {
    echo 'Resume requires the identical Git commit; preserved existing run.'; exit 1;
  }
  RESUME=(--resume)
else
  mkdir -p results/incremental/Beauty
  RUN=$(mktemp -d "results/incremental/Beauty/llama_history_retuned_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  git rev-parse HEAD > "$RUN/commit.txt"
  cp -- scripts/run_llama_history_retuned.sh "$RUN/run_command.sh"
  printf 'E1_DIR=%s\nEXPERIMENT_DIR=%s/experiment\n' "$E1" "$RUN" > "$RUN/paths.txt"
fi
ATTEMPT=$(mktemp -d "$RUN/attempt_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
finish_paths() {
  printf 'RESULT_DIR=%s\nEXPERIMENT_DIR=%s/experiment\nATTEMPT_DIR=%s\n' "$RUN" "$RUN" "$ATTEMPT"
}
finish_paths
printf 'Protocol and real RQ-VAE/TIGER checks: %s/checks.log\n' "$ATTEMPT"
if ! python -B tests/incremental/test_historical_update_round12.py > "$ATTEMPT/checks.log" 2>&1; then
  tail -n 80 "$ATTEMPT/checks.log"; finish_paths; exit 1
fi
if ! python -B tests/incremental/test_retuned_historical.py --require-torch >> "$ATTEMPT/checks.log" 2>&1; then
  tail -n 100 "$ATTEMPT/checks.log"; finish_paths; exit 1
fi
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
CMD=(python -u -B -c "$BOOTSTRAP" train_retuned_historical.py
  --e1-dir "$E1" --output "$RUN/experiment" --device cuda:0
  --max-steps 1000 --lr 0.0001 --history-weight 0.0001 --temperature 0.5
  --anchor-batch-size 128 --max-grad-norm 0.1 --eval-batch-size 128
  --log-every 25 --save-every 50 --ranking "${RESUME[@]}")
printf '%q ' "${CMD[@]}" > "$ATTEMPT/train_command.sh"
printf '\n' >> "$ATTEMPT/train_command.sh"
printf 'Paired retuned training and evaluation log: %s/train.log\n' "$ATTEMPT"
if ! "${CMD[@]}" > "$ATTEMPT/train.log" 2>&1; then
  tail -n 100 "$ATTEMPT/train.log"; finish_paths; exit 1
fi
cat "$RUN/experiment/compact.json"
finish_paths
