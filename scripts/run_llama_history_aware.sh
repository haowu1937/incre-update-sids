#!/usr/bin/env bash
# Train from the bound Base, then evaluate against the already measured Plain arm.
set -euo pipefail

[[ $# -ge 1 && $# -le 4 ]] || {
  echo 'Usage: bash scripts/run_llama_history_aware.sh E1_DIR [UPDATE_STEPS [LAMBDA [RESUME_FROM]]]'
  exit 1
}
E1_DIR=$1
UPDATE_STEPS=${2:-1000}
LAMBDA=${3:-0.001}
RESUME_FROM=${4:-}
[[ $UPDATE_STEPS =~ ^[1-9][0-9]*$ ]] || { echo 'Expected a positive Update step budget.'; exit 1; }
cd "$(git rev-parse --show-toplevel)"
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || {
  echo 'Local changes exist; stopped without starting training.'
  exit 1
}
test -f "$E1_DIR/summary.json"
if [[ -n $RESUME_FROM ]]; then test -f "$RESUME_FROM"; fi
git diff --check
PLAIN_STEPS=$(python -B - "$E1_DIR/summary.json" "$UPDATE_STEPS" "$LAMBDA" <<'PY'
import json
import math
from pathlib import Path
import sys

e1 = json.loads(Path(sys.argv[1]).read_text())
if e1.get('execution_status') != 'completed' or e1.get('E1_status') != 'completed':
    raise ValueError('Expected completed Plain E1 measurement')
if e1.get('base_validation_ranking') is None:
    raise ValueError('Complete the Plain ranking measurement before History-aware training')
plain = json.loads((Path(e1['update_rqvae_dir'])/'summary.json').read_text())
if (plain.get('execution_status') != 'completed'
        or plain['parent_base_checkpoint_sha256'] != e1['base_rqvae_sha256']):
    raise ValueError('Plain measurement lineage mismatch')
if not 0 < int(sys.argv[2]) <= plain['update_optimizer_steps']:
    raise ValueError('Historical budget must not exceed the completed Plain budget')
weight = float(sys.argv[3])
if not math.isfinite(weight) or weight <= 0:
    raise ValueError('Expected a finite positive historical weight')
print(plain['update_optimizer_steps'])
PY
)

export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results/incremental/Beauty ckpt/incremental/Beauty
RUN=$(mktemp -d "results/incremental/Beauty/llama_history_aware_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
HISTORY_DIR="ckpt/incremental/Beauty/$(basename "$RUN")_${UPDATE_STEPS}"
EVALUATION_DIR="$RUN/evaluation"
git rev-parse HEAD > "$RUN/commit.txt"
git branch --show-current > "$RUN/branch.txt"
cp -- scripts/run_llama_history_aware.sh "$RUN/run_command.sh"
printf 'E1_DIR=%s\nUPDATE_STEPS=%s\nLAMBDA=%s\nRESUME_FROM=%s\nHISTORY_DIR=%s\nEVALUATION_DIR=%s\nHISTORY_INPUT_CONDITION=new_full_sid\n' \
  "$E1_DIR" "$UPDATE_STEPS" "$LAMBDA" "$RESUME_FROM" "$HISTORY_DIR" "$EVALUATION_DIR" > "$RUN/paths.txt"
finish_paths() {
  printf 'RESULT_DIR=%s\nHISTORY_DIR=%s\nEVALUATION_DIR=%s\n' "$RUN" "$HISTORY_DIR" "$EVALUATION_DIR"
}
printf 'Synthetic model checks: %s/checks.log\n' "$RUN"
if ! python -B tests/incremental/test_historical_update_round12.py > "$RUN/checks.log" 2>&1; then
  tail -n 100 "$RUN/checks.log"
  finish_paths
  exit 1
fi

BOOTSTRAP='import runpy,sys,torch; torch.cuda.init(); torch.cuda.set_device(0); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
TRAIN=(python -u -B -c "$BOOTSTRAP" train_historical_incremental_rqvae.py
  --e1-dir "$E1_DIR" --output "$HISTORY_DIR" --device cuda:0
  --max-steps "$UPDATE_STEPS" --compatibility-weight "$LAMBDA"
  --temperature 0.1 --anchor-batch-size 32 --history-input-condition new_full_sid
  --log-every 25 --save-every 50)
if [[ -n $RESUME_FROM ]]; then TRAIN+=(--resume-from "$RESUME_FROM"); fi
printf '%q ' "${TRAIN[@]}" > "$RUN/train_command.sh"
printf '\n' >> "$RUN/train_command.sh"
printf 'Training log: %s/train.log\n' "$RUN"
if ! "${TRAIN[@]}" > "$RUN/train.log" 2>&1; then
  tail -n 100 "$RUN/train.log"
  finish_paths
  exit 1
fi

# Compare actual checkpoint sampler endpoints and optimizer settings before ranking.
if [[ $UPDATE_STEPS == "$PLAIN_STEPS" ]]; then
  if ! python -B - "$E1_DIR/summary.json" "$HISTORY_DIR" "$RUN/pair_checks.json" > "$RUN/pair_checks.log" 2>&1 <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()

e1 = json.loads(Path(sys.argv[1]).read_text())
plain_dir, history_dir = Path(e1['update_rqvae_dir']), Path(sys.argv[2])
plain, history = [json.loads((directory/'summary.json').read_text()) for directory in (plain_dir, history_dir)]
require(history.get('execution_status') == 'completed', 'Historical training is incomplete')
require(history.get('full_sid_export_status') == 'completed', 'Historical full SID export is unavailable')
checkpoints = []
for directory, report, expected_format in ((plain_dir, plain, 'plain_update_rqvae_v1'),
                                          (history_dir, history, 'historical_update_rqvae_v1')):
    require(digest(directory/'last.pth') == report['artifact_files_sha256']['last.pth'], 'Checkpoint checksum mismatch')
    checkpoint = torch.load(directory/'last.pth', map_location='cpu', weights_only=False)
    require(checkpoint.get('format') == expected_format, 'Unexpected checkpoint format')
    require(checkpoint['config'] == report['config'] and checkpoint['binding'] == report['binding'], 'Checkpoint/report mismatch')
    require(checkpoint['sampler_state']['step'] == report['update_optimizer_steps'], 'Checkpoint step mismatch')
    checkpoints.append(checkpoint)
pc, hc = checkpoints
require(digest(plain_dir/'last.pth') == e1['update_rqvae_sha256'], 'E1 Plain checkpoint changed')
require(hc['binding']['plain_update_checkpoint_sha256'] == e1['update_rqvae_sha256'], 'Wrong paired Plain checkpoint')
require(hc['binding']['parent_base_checkpoint_sha256'] == e1['base_rqvae_sha256'], 'Wrong starting Base checkpoint')
require(hc['binding']['selected_tiger_sha256'] == e1['selected_tiger_sha256'], 'Wrong frozen teacher')
require(all(hc['config'].get(key) == value for key, value in pc['config'].items()), 'Content configuration differs')
require(pc['initial_metrics'] == hc['initial_metrics'] and pc['initial_group_metrics'] == hc['initial_group_metrics'], 'Starting content metrics differ')
require(pc['optimizer']['param_groups'] == hc['optimizer']['param_groups'], 'Content optimizer settings differ')
for key in ('step', 'cursor', 'epoch', 'processed_examples'):
    require(pc['sampler_state'][key] == hc['sampler_state'][key], 'Sampler endpoint differs: '+key)
for key in ('order', 'generator_state'):
    require(torch.equal(pc['sampler_state'][key], hc['sampler_state'][key]), 'Sampler endpoint differs: '+key)
require(history['teacher_parameters_unchanged'] and history['old_target_labels_fixed'], 'Teacher or old labels changed')
require(history['anchor_holdout_used_for_training'] is False and history['base_validation_used_for_training'] is False,
        'Held-out data entered training')
result = dict(execution_status='passed', same_base_and_teacher=True, same_content_config=True,
    same_content_optimizer_settings=True, same_content_sampler_endpoint=True,
    same_initial_content_metrics=True, same_update_step_budget=True,
    update_optimizer_steps=history['update_optimizer_steps'], processed_examples=history['processed_examples'],
    same_compute=False, historical_penalty_training_cohort='anchor_train only')
Path(sys.argv[3]).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
  then
    tail -n 100 "$RUN/pair_checks.log"
    finish_paths
    exit 1
  fi
  EVAL=(python -u -B -c "$BOOTSTRAP" evaluate_historical_update.py
    --e1-dir "$E1_DIR" --historical-rqvae-dir "$HISTORY_DIR"
    --output "$EVALUATION_DIR" --device cuda:0 --batch-size 128 --ranking)
  printf '%q ' "${EVAL[@]}" > "$RUN/evaluate_command.sh"
  printf '\n' >> "$RUN/evaluate_command.sh"
  printf 'Evaluation log: %s/evaluate.log\n' "$RUN"
  if ! "${EVAL[@]}" > "$RUN/evaluate.log" 2>&1; then
    tail -n 100 "$RUN/evaluate.log"
    finish_paths
    exit 1
  fi
else
  echo 'Partial training checkpoint saved; matched evaluation waits for the full Plain step budget.'
fi

python -B - "$HISTORY_DIR/summary.json" "$EVALUATION_DIR/summary.json" "$RUN" <<'PY'
import json
from pathlib import Path
import sys

training = json.loads(Path(sys.argv[1]).read_text())
evaluation_path, run = Path(sys.argv[2]), Path(sys.argv[3])
keys = ('execution_status', 'update_optimizer_steps', 'steps_this_run', 'processed_examples', 'resume_from',
        'initial_train_catalog_metrics', 'final_train_catalog_metrics', 'final_group_metrics',
        'base_semantic_churn', 'collision_statistics', 'history_input_condition',
        'anchor_train_queries', 'anchor_queries_processed', 'teacher_parameters_unchanged',
        'anchor_holdout_used_for_training', 'base_validation_used_for_training',
        'full_sid_export_status', 'parent_base_checkpoint_unchanged', 'checkpoint_reload_assignments_equal',
        'runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes')
compact = dict(training={key: training[key] for key in keys}, config=training['config'], evaluation=None)
if evaluation_path.exists():
    evaluation = json.loads(evaluation_path.read_text())
    compact['pair_checks'] = json.loads((run/'pair_checks.json').read_text())
    compact['evaluation'] = {key: evaluation[key] for key in ('execution_status', 'groups',
        'paired_compatible_vs_plain', 'base_validation_ranking', 'rqvae_reconstruction', 'churn',
        'primary_cohort', 'teacher_frozen', 'recommendation_training_status', 'future_test_status', 'runtime_seconds')}
text = json.dumps(compact, indent=2, allow_nan=False)+'\n'
(run/'compact.txt').write_text(text)
print(text, end='')
PY
finish_paths
