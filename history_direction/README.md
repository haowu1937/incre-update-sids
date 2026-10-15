# Pure historical-gradient direction diagnostic

This local diagnostic is manually executed on your server. It does not train a new model or overwrite a checkpoint. Local package tests used synthetic data and CPU only.

## Why this diagnostic

The calibrated joint training ended with higher actual old-suffix positive damage than Plain Update on both anchor_train and anchor_holdout. The training-group increase is 0.14463411 -> 0.15820330 (+9.38%). That does not by itself prove the surrogate direction is wrong: the optimizer also followed the RQ objective, and the gradient weighting was small. It motivates isolating the history gradient before another full training run.

The default probe point is the existing, completed, E1-bound Plain Update RQ-VAE. A deterministic subsample of up to 512 anchor_train queries is selected. No anchor_holdout or recommendation validation query is used to compute a gradient, choose a direction or score a candidate. The old teacher, old target labels, masks and suffixes remain fixed.

## Measurements

1. Compute the current exact hard-input positive damage and pure historical STE gradient on the same selected queries. No content loss contributes to this direction.
2. Construct a continuous *local tangent model* whose hard offsets are frozen at this point and whose soft token/codeword functions vary continuously. At the base point its forward and first derivative must match the current STE implementation. This is not a new serving SID model or a claim about the globally fully-soft loss.
3. Perturb parameters in both negative-gradient and reverse-gradient directions at relative L2 sizes 1e-5, 1e-4 and 1e-3. Test jointly, encoder only and codebooks only. Each perturbation starts from the same original parameters.
4. Recompute real hard assignments and real old-suffix positive damage, along with tangent loss, assignment changes, code usage, largest bucket and complete content reconstruction MSE. No optimizer is involved. All parameters are restored afterward; no checkpoint is written.

Relative steps are defined as `norm(parameter perturbation) / norm(parameters in the chosen scope)`. These are normalized local directional probes, not AdamW learning rates. The same step magnitude is used for the two signs in each pair.

## Interpretation limits

- A tiny step with no assignment changes has a flat hard loss; this is expected inside an argmin cell and does not prove the gradient is incorrect.
- A negative tangent change but positive hard change once assignments change indicates local disagreement between the proxy and real discrete objective at that step size.
- A negative hard change along descent provides local evidence of a usable direction at this checkpoint/subset. It does not establish full-run or holdout improvement.
- If both signs worsen the hard loss, the current discrete assignment may be locally favorable, or the tested radius may not expose a useful transition. Do not infer a universal sign bug from one point.
- A zero gradient is reported explicitly, without being mislabeled as a successful descent test.

The test reports the condition actually trained: three new semantic tokens plus each item's old suffix. It does not allocate serving SIDs during the probe; candidate collision counts remain visible.

## Manual installation and execution

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip history_gradient_direction.zip -d history_direction
python -B history_direction/apply_direction.py --repo .

set -o pipefail
DIR_RUN_ID="history_direction_$(date -u +%Y%m%dT%H%M%SZ)"
DIR_RESULT="results/incremental/Beauty/$DIR_RUN_ID"
python -u -B check_history_gradient_direction.py \
  --e1-dir results/incremental/Beauty/historical_damage_20261009T142825Z \
  --output "$DIR_RESULT" --device cuda \
  --queries 512 --batch-size 32 \
  2>&1 | tee "${DIR_RUN_ID}.log"

python -B history_direction/show_direction.py "$DIR_RESULT"
```

The compact output contains 18 candidate rows plus a short header. `hard_delta` is the change in actual hard-input positive damage relative to the probe point; negative is better. `tangent_delta` is the change in the matched continuous local loss. `changed_history` is relative to the probe point, not C_old. `RQ_MSE` evaluates the complete eligible content catalog and is not part of the direction gradient.

Two new source files are installed: the diagnostic and its synthetic test. Existing bound model/training/evaluation source files remain unchanged, so completed experiments remain readable. The installer checks source hashes and is idempotent. It does not require dependency changes.

Optional CPU synthetic test:

```bash
python -B tests/incremental/test_history_direction.py
```
