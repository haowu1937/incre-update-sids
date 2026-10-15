# Bounded History-aware weight sweep

Unzip into the project root. This package adds separate files and verifies the
prior audited source hashes; it does not modify previous training engines or
their checkpoint bindings. No package upgrades or model downloads are needed
in the existing frec-gpu environment.

```bash
conda activate frec-gpu
cd ~/frec/graph-informed-sids-web
unzip history_weight_sweep.zip -d weight_sweep_bundle
python -B weight_sweep_bundle/apply_weight_sweep.py --repo .
source paired_seeds_20261015T135505Z_paths.sh
bash weight_sweep_bundle/run_weight_sweep.sh "$MS_RESULTS_ROOT" "$MS_E1_DIR" 3 0
```

The last two arguments mean 3 concurrent seed processes sharing physical GPU 0.
Individual jobs write complete logs; the controller prints a heartbeat every
30 seconds. A seed job includes training followed by independent training-anchor
evaluation. Weights proceed sequentially, with a barrier between levels.

## Fixed design

- Update seeds: 2026, 2027, 2028; Base, teacher, features and cohorts remain fixed.
- Reuse each seed's completed frozen-encoder Plain 100-step control and rho=0.1
  endpoint from the prior three-seed run. Read-only preflight checks each pair.
- New rho values: 0.3, 1, 3, 10, in order. Each starts afresh from the identical
  Base checkpoint, resets AdamW, and uses exactly the matched Plain content order.
- Every new arm has a fixed 100 optimizer-step budget. This is a comparison of
  weight settings at equal budgets, with no warm start from another rho level.
- Encoder remains frozen; decoder and three codebooks receive the original
  content objective. Historical gradients enter only the three codebooks.
- The per-layer mixture is g_RQ + alpha*g_H, where
  alpha = rho*norm(g_RQ)/(norm(g_H)+1e-12), detached from autograd. Either zero
  norm gives zero added history gradient. Increasing rho does not guarantee a
  proportionate Adam parameter update or a reduction in discrete hard-SID H.
- All 4096 training anchors are used per history gradient step; hard full-SID
  forward, fixed old labels, detached suffix, original temperature 0.1 and fixed
  Base distance scales. Complete SID overflow rolls back and stops that arm;
  the incomplete arm is explicitly ineligible. No replacement seeds are drawn.

## Predeclared selection and stopping

H is mean(max(NLL_new_history - NLL_old_history, 0)) with fixed old targets.
At a weight level, every one of the 3 seeds must satisfy all of:

1. Complete 100 steps and export valid complete SIDs.
2. Training-anchor H decreases by at least 1% relative to its matched Plain.
3. The upper bound of its paired, user-cluster bootstrap CI95 for training H
   difference is strictly below zero (500 replicates, same cohort seed).
4. Full-catalog reconstruction MSE is no more than 1.01 times matched Plain.

Two adjacent predeclared weight levels must pass all conditions. The second of
those two levels is selected. This is an operational *training stability*
criterion; bootstrap intervals on the repeatedly inspected training set are
exploratory and are not multiple-testing-corrected evidence of generalization.

The controller freezes selection.json before evaluating the selected weight on
anchor_holdout and Base validation for all three seeds. Their results cannot
change selection or trigger further weight escalation. These cohorts have been
examined in earlier exploratory work; they are not a new untouched test set.
The report separately states whether all three holdout differences and their
paired user CI upper bounds are below zero. No ranking or future test is run.

If no pair of adjacent levels passes by rho=10, the run stops and reports that
training stability was not obtained in this range. It performs no new held-out
evaluation in that case. All levels, including failures, are retained. There is
no assumption that a large enough weight must work. At most 12 new training
jobs are launched. Parallel jobs on one GPU need not run three times faster.

## Results and resume

The final console line gives the path to compact.txt. Share only that file;
full metrics and individual seed logs stay in the output directory. The
predeclared_plan.json, intermediate summary.json and frozen selection.json
record settings and decisions.

If a process fails (including OOM), the controller stops the other active jobs
and retains checkpoints. To reuse that same sweep and reduce GPU concurrency:

```bash
# Replace this filename with the paths file printed/created for your run.
source weight_sweep_YYYYMMDDTHHMMSSZ_paths.sh
set -o pipefail
python -u -B run_history_weight_sweep.py \
  --previous-root "$WS_PREVIOUS_ROOT" --e1-dir "$WS_E1_DIR" \
  --output "$WS_OUTPUT" --workers 1 --devices "$WS_DEVICE" \
  2>&1 | tee "${WS_RUN_ID}_resume.log"
```

Completed matching jobs/evaluations are reused. Interrupted training resumes
from its latest checkpoint into a new attempt directory. A SID-overflow arm
is a recorded stopped result, not a transient failure to retry. A partially
written independent evaluation directory triggers a descriptive error instead
of silent overwrite. Source/data/plan changes also fail before reuse.

For separate GPUs, call run_history_weight_sweep.py directly with
`--workers 3 --devices 0 1 2`, keeping the other arguments above. Physical
placement may differ from the old Plain config; numerical settings are fixed.
Resume each arm on the same device index used to create its checkpoint.

Optional synthetic CPU regression checks (real data are never used):

```bash
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B tests/incremental/test_history_weight_sweep.py
```
