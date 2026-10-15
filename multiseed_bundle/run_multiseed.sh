#!/usr/bin/env bash
set -euo pipefail
MS_E1_DIR="${1:?Usage: bash multiseed_bundle/run_multiseed.sh E1_DIR COMPLETED_2026_PLAIN_DIR COMPLETED_2026_HISTORY_DIR}"
MS_PLAIN_2026="${2:?Completed frozen encoder Plain 2026 directory required}"
MS_HISTORY_2026="${3:?Completed ungated frozen encoder History-aware 2026 directory required}"
MS_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MS_RUN_ID="paired_seeds_$(date -u +%Y%m%dT%H%M%SZ)"
MS_RESULTS_ROOT="results/incremental/Beauty/${MS_RUN_ID}"
MS_CKPT_ROOT="ckpt/incremental/Beauty/${MS_RUN_ID}"
MS_PREFLIGHT_DIR="${MS_RESULTS_ROOT}/preflight_2026"
MS_PLAIN_2027="${MS_CKPT_ROOT}/seed2027_plain"
MS_HISTORY_2027="${MS_CKPT_ROOT}/seed2027_history"
MS_PLAIN_2028="${MS_CKPT_ROOT}/seed2028_plain"
MS_HISTORY_2028="${MS_CKPT_ROOT}/seed2028_history"
MS_EVAL_2026="${MS_RESULTS_ROOT}/seed2026_evaluation"
MS_EVAL_2027="${MS_RESULTS_ROOT}/seed2027_evaluation"
MS_EVAL_2028="${MS_RESULTS_ROOT}/seed2028_evaluation"
MS_AGGREGATE_DIR="${MS_RESULTS_ROOT}/three_seed_summary"
if [[ -e "${MS_RUN_ID}_paths.sh" ]]; then echo 'Run paths already exist' >&2; exit 1; fi
MS_PATH_KEYS=(MS_E1_DIR MS_PLAIN_2026 MS_HISTORY_2026 MS_RUN_ID MS_RESULTS_ROOT MS_CKPT_ROOT MS_PREFLIGHT_DIR MS_PLAIN_2027 MS_HISTORY_2027 MS_PLAIN_2028 MS_HISTORY_2028 MS_EVAL_2026 MS_EVAL_2027 MS_EVAL_2028 MS_AGGREGATE_DIR)
for MS_KEY in "${MS_PATH_KEYS[@]}"; do printf '%s=%q\n' "$MS_KEY" "${!MS_KEY}"; done > "${MS_RUN_ID}_paths.sh"
mkdir -p "$MS_RESULTS_ROOT"
python -B - "$MS_RESULTS_ROOT/predeclared_plan.json" "$MS_E1_DIR" "$MS_PLAIN_2026" "$MS_HISTORY_2026" <<'PY'
import json,sys
plan=dict(update_seeds=[2026,2027,2028],new_training_seeds=[2027,2028],reuse_completed_2026=True,
    committed_steps_per_arm=100,rho=.1,temperature=.1,anchor_batch_size=32,encoder_frozen=True,
    fixed=['Base checkpoint','old TIGER','content embeddings','training catalog','anchor_train','anchor_holdout','other hyperparameters'],
    seed_selection='include all three predeclared seeds',e1_dir=sys.argv[2],plain_2026=sys.argv[3],history_2026=sys.argv[4])
with open(sys.argv[1],'x') as f:json.dump(plan,f,indent=2);f.write('\n')
PY
python -u -B audit_history_pair.py preflight \
 --e1-dir "$MS_E1_DIR" --plain-dir "$MS_PLAIN_2026" --history-dir "$MS_HISTORY_2026" \
 --output "$MS_PREFLIGHT_DIR" --device cuda --batch-size 32 --progress \
 2>&1 | tee "${MS_RUN_ID}_preflight.log"
# Evaluation only for 2026: no training call is made for that seed.
python -u -B history_update_multiseed.py evaluate \
 --update-seed 2026 --expected-steps 100 --e1-dir "$MS_E1_DIR" \
 --plain-dir "$MS_PLAIN_2026" --history-dir "$MS_HISTORY_2026" --output "$MS_EVAL_2026" \
 --device cuda --batch-size 32 --progress 2>&1 | tee "${MS_RUN_ID}_seed2026_evaluation.log"
for MS_SEED in 2027 2028; do
 MS_PLAIN_KEY="MS_PLAIN_${MS_SEED}";MS_HISTORY_KEY="MS_HISTORY_${MS_SEED}";MS_EVAL_KEY="MS_EVAL_${MS_SEED}"
 MS_PLAIN_CURRENT="${!MS_PLAIN_KEY}";MS_HISTORY_CURRENT="${!MS_HISTORY_KEY}";MS_EVAL_CURRENT="${!MS_EVAL_KEY}"
 python -u -B history_update_multiseed.py train \
  --mode plain --update-seed "$MS_SEED" --e1-dir "$MS_E1_DIR" --output "$MS_PLAIN_CURRENT" \
  --device cuda --max-steps 100 --anchor-batch-size 32 --temperature 0.1 \
  --log-every 10 --save-every 10 --history-progress-every 0 \
  2>&1 | tee "${MS_RUN_ID}_seed${MS_SEED}_plain.log"
 python -u -B history_update_multiseed.py train \
  --mode history-aware --update-seed "$MS_SEED" --e1-dir "$MS_E1_DIR" --output "$MS_HISTORY_CURRENT" \
  --device cuda --max-steps 100 --history-gradient-ratio 0.1 --anchor-batch-size 32 --temperature 0.1 \
  --log-every 10 --save-every 10 --history-progress-every 0 \
  2>&1 | tee "${MS_RUN_ID}_seed${MS_SEED}_history.log"
 python -u -B history_update_multiseed.py evaluate \
  --update-seed "$MS_SEED" --expected-steps 100 --e1-dir "$MS_E1_DIR" \
  --plain-dir "$MS_PLAIN_CURRENT" --history-dir "$MS_HISTORY_CURRENT" --output "$MS_EVAL_CURRENT" \
  --device cuda --batch-size 32 --progress 2>&1 | tee "${MS_RUN_ID}_seed${MS_SEED}_evaluation.log"
done
python -u -B history_update_multiseed.py aggregate \
 --evaluation-dirs "$MS_EVAL_2026" "$MS_EVAL_2027" "$MS_EVAL_2028" --output "$MS_AGGREGATE_DIR" \
 2>&1 | tee "${MS_RUN_ID}_aggregate.log"
python -B "$MS_BUNDLE_DIR/show_multiseed.py" "$MS_AGGREGATE_DIR" | tee "${MS_RUN_ID}_compact.txt"
printf 'Share only: %s\n' "${MS_RUN_ID}_compact.txt"
