# Historical Update collapse diagnostic

This is a short diagnostic that you manually run on your server. It reads existing E1 and checkpoint files and adds two new source files. It does not modify bound RQ-VAE/TIGER/history training source files or existing checkpoint contents. The package was tested on synthetic data and CPU only.

## Evidence so far

At the logged 1000th training step, all 11707 eligible items share one semantic triple. The history penalty is 1.03742 on 32 training anchors, and its weighted value is approximately 3.89 times the batch RQ loss. This loss ratio is NOT a gradient ratio. The current batch reconstruction MSE is 0.000231119; do not treat this as a full-catalog reconstruction evaluation. The optional read-only checkpoint evaluation computes the full eligible-catalog result and its own matching mean-embedding baseline.

The earlier suffix fix allows training with old suffix tokens, but does not prevent semantic collapse. Expanding the suffix vocabulary or continuing from the collapsed checkpoint would not recover learned item distinctions.

## Matched diagnosis

Two trajectories start from the same selected Base RQ-VAE with reset AdamW and the same content sampling, LR, seed, anchor order and batch size. Defaults: 30 steps each, lambda=0 and lambda=0.001. They do not continue the collapsed run. No recommendation validation or held-out anchors are used in the diagnostic objective.

The zero-weight branch computes historical derivatives for observation, but excludes them exactly from the optimizer loss. A synthetic check requires final weights to match the actual original repository Trainer after the same steps. The diagnostic still uses that Trainer's optimizer and fit_steps.

Before each optimizer update, it records separate RQ and raw history gradients per encoder, decoder and each codebook, their cosine and the norm ratio after multiplying history gradients by lambda. After the update, it records full-catalog hard assignment counts, maximum bucket size, per-codebook usage and entropy, fixed content-sample encoder variance and actual parameter changes. Soft assignment entropy/max probability and normalized nearest-codeword margins help identify a saturated surrogate. Hard local-batch argmin agreement with authoritative catalog assignments is also recorded.

`trace.jsonl` includes both pre-update and post-update observations, avoiding the ambiguity of the original logger, whose historical assignments were evaluated before the optimizer update. No additional clipping, regularizer or optimizer change is introduced.

## Manual installation and run

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip historical_collapse_diagnostic.zip -d collapse_diag
python -B collapse_diag/apply_diagnostic.py --repo .

set -o pipefail
FAILED_RQVAE_DIR="$(python -B collapse_diag/find_failed_run.py)"
printf 'FAILED_RQVAE_DIR=%s\n' "$FAILED_RQVAE_DIR"
DIAG_RUN_ID="collapse_diag_$(date -u +%Y%m%dT%H%M%SZ)"
DIAG_DIR="results/incremental/Beauty/$DIAG_RUN_ID"

python -u -B diagnose_historical_collapse.py \
  --e1-dir results/incremental/Beauty/historical_damage_20261009T142825Z \
  --failed-rqvae-dir "$FAILED_RQVAE_DIR" \
  --output "$DIAG_DIR" --device cuda \
  --steps 30 --anchor-batch-size 32 --weights 0 0.001 \
  2>&1 | tee "${DIAG_RUN_ID}.log"

python -B collapse_diag/show_diagnostic.py "$DIAG_DIR"
```

The latest completed, capacity-exhausted, fixed-old-suffix Beauty run with 1000 steps is selected; the diagnostic validates all checkpoint bindings. If you intentionally want a different run, set FAILED_RQVAE_DIR manually. Optional checkpoint inspection can also be omitted by removing --failed-rqvae-dir.

## Interpreting the traces

- If both branches collapse at step 1, the first collapse comes from the ordinary RQ update: the historical hinge is zero at the identical starting assignment. Inspect reset-AdamW update size and hard codeword margins before selecting a smaller or constrained update.
- If lambda=0 recovers but lambda=0.001 stays collapsed, the additional compatibility gradient changes recovery. Inspect its norm ratio, direction conflict and soft saturation rather than attributing the issue only to suffix allocation.
- If only the nonzero branch collapses later, inspect the first divergent update and component gradients.
- If both short branches are stable, locate the first collapse in the existing longer log before extending the diagnostic. A stable 30-step run does not prove long-run stability.

The current evidence does not establish which explanation occurred on Beauty. No fixed learning rate, lambda or surrogate replacement is claimed to solve the problem until these traces are available.

Optional synthetic CPU verification:

```bash
python -B tests/incremental/test_collapse_diagnostic.py
```
