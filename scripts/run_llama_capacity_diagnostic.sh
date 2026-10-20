#!/usr/bin/env bash
# Reconstruct only the short gap between the saved checkpoint and allocation failure.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: bash scripts/run_llama_capacity_diagnostic.sh FAILED_RUN_DIR'; exit 1; }
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without replay.'; exit 1;
}
test -f "$1/paths.txt"
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_capacity_diag_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
git rev-parse HEAD > "$RUN/commit.txt"
cp -- scripts/run_llama_capacity_diagnostic.sh "$RUN/run_command.sh"
printf 'FAILED_RUN=%s\n' "$1" > "$RUN/paths.txt"
printf 'Diagnostic checks: %s/checks.log\n' "$RUN"
if ! python -B tests/incremental/test_historical_capacity_diagnostic.py --require-torch > "$RUN/checks.log" 2>&1; then
  tail -n 80 "$RUN/checks.log"
  printf 'RESULT_DIR=%s\n' "$RUN"
  exit 1
fi
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
CMD=(python -u -B -c "$BOOTSTRAP" diagnose_historical_capacity.py
  --run-dir "$1" --output "$RUN/diagnostic" --device cuda:0 --max-replay-steps 50)
printf '%q ' "${CMD[@]}" > "$RUN/diagnose_command.sh"
printf '\n' >> "$RUN/diagnose_command.sh"
printf 'Capacity replay log: %s/replay.log\n' "$RUN"
if ! "${CMD[@]}" > "$RUN/replay.log" 2>&1; then
  tail -n 100 "$RUN/replay.log"
  printf 'RESULT_DIR=%s\n' "$RUN"
  exit 1
fi
cat "$RUN/diagnostic/compact.json"
printf 'RESULT_DIR=%s\n' "$RUN"
