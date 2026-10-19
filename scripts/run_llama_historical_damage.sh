#!/usr/bin/env bash
# Paired history-only evaluation of a completed Plain Update; no model training.
set -euo pipefail

[[ $# -ge 5 && $# -le 6 ]] || {
  echo 'Usage: bash scripts/run_llama_historical_damage.sh PROTOCOL_DIR SID_DIR TIGER_DIR UPDATE_RQVAE_DIR BASE_RQVAE_SHA256 [loss|ranking]'
  exit 1
}
PROTOCOL_DIR=$1
SID_DIR=$2
TIGER_DIR=$3
UPDATE_RQVAE_DIR=$4
BASE_RQVAE_SHA256=$5
MODE=${6:-loss}
[[ $BASE_RQVAE_SHA256 =~ ^[0-9a-f]{64}$ ]] || { echo 'Expected a full Base RQ-VAE SHA256.'; exit 1; }
[[ $MODE == loss || $MODE == ranking ]] || { echo 'MODE must be loss or ranking.'; exit 1; }
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting evaluation.'
  exit 1
}
test -f "$PROTOCOL_DIR/summary.json"
test -f "$SID_DIR/summary.json"
test -f "$TIGER_DIR/summary.json"
test -f "$TIGER_DIR/best.pth"
test -f "$UPDATE_RQVAE_DIR/summary.json"
test -f "$UPDATE_RQVAE_DIR/last.pth"
git diff --check

# Check the selected lineage and teacher checksum before importing the GPU stack.
# The engine additionally verifies source/package/data bindings and all artifacts.
TEACHER_SHA=$(python -B - "$SID_DIR" "$TIGER_DIR" "$UPDATE_RQVAE_DIR" "$BASE_RQVAE_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

sid_dir, tiger_dir, update_dir = map(Path, sys.argv[1:4])
sid, tiger, update = [json.loads((path/'summary.json').read_text())
                      for path in (sid_dir, tiger_dir, update_dir)]
if any(report.get('execution_status') != 'completed' for report in (sid, tiger, update)):
    raise ValueError('Expected completed SID, Base TIGER and Plain Update artifacts')
if (sid['rqvae_checkpoint_sha256'] != sys.argv[4]
        or tiger['rqvae_checkpoint_sha256'] != sys.argv[4]
        or update['parent_base_checkpoint_sha256'] != sys.argv[4]):
    raise ValueError('SID, teacher and Update must share the selected Base RQ-VAE')
digest = hashlib.sha256()
with (tiger_dir/'best.pth').open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024*1024), b''):
        digest.update(chunk)
teacher_sha = digest.hexdigest()
if teacher_sha != tiger['artifact_files_sha256'].get('best.pth'):
    raise ValueError('Selected Base TIGER checksum mismatch')
print(teacher_sha)
PY
)

export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -B measure_historical_damage.py --help >/dev/null
mkdir -p results/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_plain_history_damage_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
E1_DIR="$RUN/evaluation"
git rev-parse HEAD > "$RUN/commit.txt"
git branch --show-current > "$RUN/branch.txt"
cp -- scripts/run_llama_historical_damage.sh "$RUN/run_command.sh"
printf 'PROTOCOL_DIR=%s\nSID_DIR=%s\nTIGER_DIR=%s\nUPDATE_RQVAE_DIR=%s\nBASE_RQVAE_SHA256=%s\nTEACHER_SHA=%s\nMODE=%s\nE1_DIR=%s\nSEED=2026\n' \
  "$PROTOCOL_DIR" "$SID_DIR" "$TIGER_DIR" "$UPDATE_RQVAE_DIR" \
  "$BASE_RQVAE_SHA256" "$TEACHER_SHA" "$MODE" "$E1_DIR" > "$RUN/paths.txt"

# Initialize CUDA before the unchanged, source-bound engine resets memory stats.
BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
EVAL=(python -u -B -c "$BOOTSTRAP" measure_historical_damage.py
  --protocol-dir "$PROTOCOL_DIR" --sid-dir "$SID_DIR" --tiger-dir "$TIGER_DIR"
  --update-rqvae-dir "$UPDATE_RQVAE_DIR" --expected-tiger-sha256 "$TEACHER_SHA"
  --output "$E1_DIR" --device cuda:0 --batch-size 128
  --anchor-size 4096 --bootstrap-reps 500 --seed 2026)
if [[ $MODE == ranking ]]; then
  EVAL+=(--ranking)
fi
printf '%q ' "${EVAL[@]}" > "$RUN/evaluate_command.sh"
printf '\n' >> "$RUN/evaluate_command.sh"
printf 'Evaluation log: %s/evaluate.log\n' "$RUN"
if ! "${EVAL[@]}" > "$RUN/evaluate.log" 2>&1; then
  tail -n 100 "$RUN/evaluate.log"
  printf 'RESULT_DIR=%s\nE1_DIR=%s\n' "$RUN" "$E1_DIR"
  exit 1
fi

python -B - "$E1_DIR/summary.json" "$RUN/compact.txt" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
keys = ('execution_status', 'selected_tiger_sha256', 'selected_tiger_optimizer_step',
        'base_rqvae_sha256', 'update_rqvae_sha256', 'groups', 'base_validation_ranking',
        'historical_damage_detected', 'detection_rule', 'labels', 'primary_condition',
        'diagnostic_condition', 'diagnostic_condition_serving_sid', 'ranking_catalog',
        'base_validation_all_queries', 'base_validation_nll_queries',
        'base_validation_targets_without_old_catalog_labels',
        'cohort_selection', 'cohort_seed', 'requested_queries_per_anchor_cohort',
        'new_full_sids_unique', 'old_suffix_slots_reserved', 'tokenizer_vocab_size',
        'tiger_frozen', 'frozen_teacher_parameters_equal', 'old_validation_nll_reproduced',
        'historical_aware_training_status', 'codebook_training_performed',
        'recommendation_training_performed', 'future_test_status',
        'runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')
compact = {key: summary[key] for key in keys}
text = json.dumps(compact, indent=2, allow_nan=False) + '\n'
Path(sys.argv[2]).write_text(text)
print(text, end='')
PY
printf 'RESULT_DIR=%s\nE1_DIR=%s\n' "$RUN" "$E1_DIR"
