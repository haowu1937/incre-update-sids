# Hard-checked codebook repair

This is an exploratory second stage after the existing E1-bound Plain Update.
Run all real commands on your Linux machine; the supplied validation used only
synthetic CPU data in a separate environment.

## Install

From the repository root, with the previous Round12 suffix fix installed:

```bash
unzip ~/Downloads/codebook_hard_repair.zip -d codebook_repair_bundle
python -B codebook_repair_bundle/apply_repair.py --repo .
```

The installer checks previous source versions before writing and adds only new
files. The existing direction diagnostic is reused if identical. No pip install,
model download, or overwrite of previous experiment artifacts is needed.

## Repair

```bash
set -o pipefail
E1_DIR="results/incremental/Beauty/historical_damage_20261009T142825Z"
REPAIR_RUN_ID="codebook_repair_$(date -u +%Y%m%dT%H%M%SZ)"
REPAIR_DIR="ckpt/incremental/Beauty/$REPAIR_RUN_ID"

python -u -B repair_historical_codebooks.py repair \
  --e1-dir "$E1_DIR" \
  --output "$REPAIR_DIR" \
  --device cuda \
  --max-rounds 10 \
  --gradient-queries 512 \
  --batch-size 32 \
  2>&1 | tee "${REPAIR_RUN_ID}.log"

python -B codebook_repair_bundle/show_repair.py "$REPAIR_DIR"
```

The starting checkpoint is resolved from E1: the Plain Update, rather than the
failed historical run or the later calibrated-lambda historical run. Encoder,
decoder, and teacher are fixed. All three codebooks may change.

At each round, the existing hard-forward STE gives a pure history gradient on a
fixed deterministic 512-row subset of anchor_train. Proposed descent steps have
joint codebook relative L2 sizes 1e-3, 3e-4, 1e-4, 3e-5, and 1e-5, tried in that
order. The first candidate satisfying all conditions is accepted:

1. Exact hard positive-part old-target sequence NLL damage decreases by more
   than 1e-6 on ALL anchor_train rows (4096 for this experiment).
2. Full eligible-catalog reconstruction MSE is at most 1.01 times the INITIAL
   Plain Update MSE, with the reference fixed across all repair rounds.
3. Original reserved-slot full SID allocation remains feasible.

Hard verification changes three semantic tokens and keeps old history suffixes,
labels, masks, and EOS. Unique full SID evaluation is separate. This distinction
is retained from the earlier experiment and must be reported.

Rejected candidates are restored. No holdout or Base validation loss selects a
candidate. A round with no acceptable tested proposal stops the repair. This is
a finite grid search along one surrogate direction, not a proof that no useful
codebook direction exists. A zero gradient stops with a separate reason.

If `accepted_repair_steps = 0`, the saved checkpoint has the original Plain
weights. Share the nine-line compact output; another full ranking evaluation
is unnecessary. If at least one repair step was accepted, run evaluation below.
There is no resume option: at most ten short repair rounds are run in this
first experiment. An interrupted output directory is preserved; restart with
a fresh run ID. Do not manually delete existing checkpoints to bypass checks.

## Evaluate

```bash
REPAIR_EVAL_DIR="results/incremental/Beauty/${REPAIR_RUN_ID}_evaluation"

python -u -B repair_historical_codebooks.py evaluate \
  --e1-dir "$E1_DIR" \
  --repair-dir "$REPAIR_DIR" \
  --output "$REPAIR_EVAL_DIR" \
  --device cuda \
  --batch-size 32 \
  --ranking \
  2>&1 | tee "${REPAIR_RUN_ID}_evaluation.log"

python -B codebook_repair_bundle/show_repair.py "$REPAIR_DIR" "$REPAIR_EVAL_DIR"
```

Share only the compact output. Evaluation covers anchor_train, anchor_holdout,
and the servable Base validation tensors. Both complete new SIDs and the old
suffix control are reported. Paired user-cluster bootstrap confidence intervals
are computed for repair-minus-Plain sequence NLL and positive damage. Negative
differences favor repair. The primary generalization cohort is anchor_holdout;
it is held out from repair, but was seen by the original Base teacher training.

Ranking uses the original Base-train item candidates and original target full
SIDs, with unavailable targets counted as misses in all 9192 queries. Old and
Plain ranking references come from the hash-bound E1 result. New ranking is
computed for the repaired historical inputs. Ranking differences are point
estimates; no ranking significance claim is made.

Training-anchor improvement is guaranteed only by acceptance checks; holdout,
full-SID, and ranking improvement are not guaranteed. This adds work beyond
Plain Update's 1000 optimizer steps, records query/full-catalog work counters,
and cannot establish a matched-compute superiority claim. A later comparison
needs an additional-budget Plain control. No recommendation model update or
future-test evaluation is performed here.

Optional synthetic regression tests (not Beauty or GPU validation):

```bash
USE_TF=0 USE_FLAX=0 python -B tests/incremental/test_codebook_repair.py
```

See validation.json and validation.log for the separate CPU sandbox checks.
