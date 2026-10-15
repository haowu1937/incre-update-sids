# History-aware joint RQ-VAE Update v2

This returns to the main joint-update experiment. It starts from the bound
Base RQ-VAE (C_old, 5000 steps), rather than Plain Update or the repair endpoint.
Real Beauty training is run manually on your Linux machine. Supplied validation
uses synthetic CPU data in a separate environment.

## Protocol fixed before this experiment

- Content: the same eligible 11707-item catalog, uniform shuffled sampler,
  seed 2026, content batch 256, repository AdamW lr .001 and weight decay 0,
  and a target of 1000 Update optimizer steps as the E1-bound Plain run.
- Content RQ loss updates encoder, decoder, and three codebooks. No graph,
  CF loss, Sinkhorn, content-encoder fine-tuning, or codebook reinitialization.
- Historical loss is mean positive-part old-target sequence NLL increase.
  The teacher is the selected frozen Base TIGER; original labels, history
  availability, masks, EOS and padding positions remain fixed.
- History gradient reaches ONLY three codebook tables. Historical encoder
  output is detached. Soft-distance STE is used for semantic tokens and
  residuals; its forward equals the current hard semantic embeddings.
- History forward uses COMPLETE new SIDs, including the deterministic current
  collision suffix. Suffix allocation has no gradient.
- ALL 4096 fixed anchor_train queries contribute to EVERY history gradient,
  accumulated in batches of 32. The query sum is divided by the full cohort
  size, including any partial final batch. No holdout rows enter training.
- Per codebook: alpha = rho * ||g_RQ|| / (||g_H|| + 1e-12), with rho=.1 fixed
  before evaluation. Alpha is detached. Final gradient = g_RQ + alpha*g_H.
  If either norm is zero, the added history gradient is zero. This is gradient
  mixing, not a single global scalar-loss weight. No higher-order derivatives.
- Log per-layer gradient norms, alpha, effective ratios, directional cosines,
  actual relative AdamW parameter updates, full SID capacity and code usage.
- Select the final fixed-budget checkpoint. Do not select by holdout, Base
  validation, a descending training damage curve, or reconstruction score.

The four modifications are tested together in this first v2 run. Later ablations
are needed to attribute any effect to an individual modification. Matching
optimizer steps does not match compute: this run adds substantial teacher work.

## Install

From your existing repository root, in frec-gpu:

```bash
unzip ~/Downloads/history_aware_update_v2.zip -d history_v2_bundle
python -B history_v2_bundle/apply_v2.py --repo .
```

The installer checks required prior source hashes and adds two new files.
Existing preparation, old historical training, repair scripts, and artifacts
are unchanged. No new dependency installation or model download is required
on the environment used for your earlier E1 and repair runs.

To execute the entire fixed protocol (50-step check, continue to 1000, then
evaluate), use this single manual command instead of the separate commands:

```bash
bash history_v2_bundle/run_beauty_v2.sh
```

The runner stops on any error or incomplete budget. It records all run paths in
`historical_joint_v2_TIMESTAMP_paths.sh` and the final compact result in
`historical_joint_v2_TIMESTAMP_compact.txt`. For recovery, source the actual
paths file in your terminal and follow the resume instructions below. The
runner never starts an automatic parameter sweep or adjusts rho.

## Run 50 steps first

```bash
set -o pipefail

E1_DIR="results/incremental/Beauty/historical_damage_20261009T142825Z"
HA_RUN_ID="historical_joint_v2_$(date -u +%Y%m%dT%H%M%SZ)"
HA_SMOKE_DIR="ckpt/incremental/Beauty/${HA_RUN_ID}_smoke"
HA_RQVAE_DIR="ckpt/incremental/Beauty/${HA_RUN_ID}_1000"

python -u -B history_aware_update_v2.py train \
  --e1-dir "$E1_DIR" \
  --output "$HA_SMOKE_DIR" \
  --device cuda \
  --max-steps 50 \
  --history-gradient-ratio 0.1 \
  --anchor-batch-size 32 \
  --log-every 10 \
  --save-every 10 \
  2>&1 | tee "${HA_RUN_ID}_smoke.log"

python -B history_v2_bundle/show_v2.py "$HA_SMOKE_DIR" --require-steps 50
```

The cached old reference uses 4096 queries once per process. Each nonzero-rho
optimizer step then performs 128 teacher-gradient batches. A 50-step check
processes 204800 history-gradient queries; 1000 cumulative steps process 4096000
if no candidate is rolled back. The history-query count per step is 128 times
the old 32-anchor setting; this is not a prediction of total runtime scaling.
Use the measured median step time and printed rough remaining training hours.
The estimate excludes startup, final content evaluation and ranking, and can
change with server load.

Progress appears every 32 history batches. Expect the initial identical Base
history to have zero positive damage and zero history gradient. Later steps can
increase or decrease damage: there is no monotonic damage acceptance rule.

`--require-steps` fails if the requested run did not finish. In that case share
the compact output and stop; do not launch formal evaluation or change rho
based on holdout. Nonfinite losses fail instead of silently skipping training.

## Continue to 1000 cumulative steps

In the SAME terminal, using the same model/data/normalization configuration:

```bash
python -B history_v2_bundle/show_v2.py "$HA_SMOKE_DIR" --require-steps 50

python -u -B history_aware_update_v2.py train \
  --e1-dir "$E1_DIR" \
  --output "$HA_RQVAE_DIR" \
  --device cuda \
  --max-steps 1000 \
  --history-gradient-ratio 0.1 \
  --anchor-batch-size 32 \
  --resume-from "$HA_SMOKE_DIR/last.pth" \
  --save-every 10 \
  2>&1 | tee "${HA_RUN_ID}_1000.log"

python -B history_v2_bundle/show_v2.py "$HA_RQVAE_DIR" --require-steps 1000
```

1000 is the TOTAL target, so this executes 950 additional steps. Checkpoint
resume restores model, AdamW, content order/cursor, sampler generator and RNG.
Changing rho, temperature, anchor batch size, data/source/package bindings,
or device is rejected. Log/save/progress intervals and target budget may change.

## Evaluate once the 1000-step run completes

```bash
HA_EVAL_DIR="results/incremental/Beauty/${HA_RUN_ID}_evaluation"

python -u -B history_aware_update_v2.py evaluate \
  --e1-dir "$E1_DIR" \
  --rqvae-dir "$HA_RQVAE_DIR" \
  --output "$HA_EVAL_DIR" \
  --device cuda \
  --batch-size 32 \
  --ranking \
  2>&1 | tee "${HA_RUN_ID}_evaluation.log"

python -B history_v2_bundle/show_v2.py "$HA_RQVAE_DIR" "$HA_EVAL_DIR"
```

Share only the compact output (roughly 35 lines), not the full logs or JSON.
The primary cohort is anchor_holdout: held out from history training and user
disjoint from anchor_train, but seen by the original Base TIGER training.
Primary condition is complete new SID. Old-suffix results are a diagnostic.
Report paired history-aware-minus-Plain positive damage and sequence NLL with
the E1 user-cluster bootstrap protocol. Negative differences favor history-aware.

Base ranking keeps original Base-train candidates and original full SID target
labels. All 9192 queries enter the full denominator, including 503 unavailable
targets as misses. Old and Plain rankings are reused from hash-bound E1; the
new historical inputs are decoded with the existing prefix-constrained beam
search. Ranking deltas are point estimates, without ranking confidence intervals.

## Capacity and interruption recovery

Before committing a candidate model state, complete SID allocation is checked
for all 11707 items using the original reserved-slot policy and suffix capacity.
If infeasible, the proposed optimizer update is rolled back, including model,
AdamW, content sampling and RNG. Training stops and exports the last feasible
checkpoint with execution_status=stopped. The rejected capacity audit is saved.
No suffix wraparound, vocabulary expansion, or old-suffix fallback is used.
This feasibility safeguard is recorded as a protocol difference from Plain;
a stopped run cannot be called a completed 1000-step comparison.

For an interrupted process, last.pth is saved before the first step and every
10 successful steps. It can be resumed into a NEW output directory with the
same configuration. An interrupted directory may have no summary.json; that
does not prevent resume from its checkpoint. At most 9 completed steps may
need repeating. To resume, set --resume-from to the actual last.pth path and
--max-steps to 1000. Never reuse an existing output directory.

This run does not train the recommendation model, evaluate the future test
split, or claim an improvement before the real results are available.

## Optional local synthetic checks

```bash
USE_TF=0 USE_FLAX=0 python -B tests/incremental/test_history_aware_update_v2.py
```

See validation.json and validation.log for separate CPU sandbox checks.
