# Single-step diagnostic for History-aware RQ-VAE Update v2

This follows the completed v2 trajectory: full-training damage recovered to
about .10 in the first 25-100 steps, then increased to .24254 at the final
checkpoint. The first step had zero history gradient, so its large parameter
change was a content-only update. We do not infer a cause from the trajectory
or select an early checkpoint as a successful model.

This diagnostic examines the available 50-step and 1000-step checkpoints.
Each checkpoint supplies its exact next content minibatch, saved AdamW moments,
RNG, normalized gradient configuration, full SID allocation and old teacher.
Both paired AdamW proposals start from identical states and minibatches.

The diagnostic commits no training steps and writes no model checkpoints.
No ranking or holdout loss is evaluated. The source checkpoint files are
hash checked before and after. All candidate parameter changes are in memory.

## Install from the project root

The downloaded ZIP belongs in the project root, as in your usual workflow:

```bash
unzip history_update_step_diagnostic.zip -d history_step_bundle
python -B history_step_bundle/apply_step.py --repo .
```

The installer adds only two new files after checking existing v2 dependencies.
It does not edit the hash-bound v2 training source or old experiment artifacts.
No package installation or model download is required on your frec-gpu machine.

## Run on the existing two checkpoints

```bash
set -o pipefail
source historical_joint_v2_20261014T124630Z_paths.sh
STEP_DIAG_RUN_ID="history_step_$(date -u +%Y%m%dT%H%M%SZ)"
STEP_DIAG_DIR="results/incremental/Beauty/$STEP_DIAG_RUN_ID"

python -u -B diagnose_history_update_step.py \
  --e1-dir "$E1_DIR" \
  --rqvae-dirs "$HA_SMOKE_DIR" "$HA_RQVAE_DIR" \
  --output "$STEP_DIAG_DIR" \
  --device cuda \
  2>&1 | tee "${STEP_DIAG_RUN_ID}.log"

python -B history_step_bundle/show_step.py "$STEP_DIAG_DIR"
```

Share only the compact output (about 30 lines). The test performs two full
training-anchor gradients and several full-anchor forward evaluations, rather
than another 1000-step training run. It uses the saved anchor batch size 32.
The 1000-step checkpoint's proposal is a hypothetical next step in memory,
not an extension of the formal experiment's budget.

## Candidate meanings

| Candidate | In-memory change |
|---|---|
| content_adam | One AdamW update with the content gradient only |
| joint_adam | Same AdamW state and batch, with the original v2 normalized history gradient added to books |
| joint_encoder_only | Only the encoder change from joint_adam; all other parameters at the checkpoint |
| joint_codebooks_only | Only the book change from joint_adam; encoder/decoder at the checkpoint |
| history_direct_descent | Direct negative history gradient on books, relative joint book L2 step .001; no AdamW/content update |
| history_direct_reverse | Same direct direction and magnitude, opposite sign |

The direct candidates are skipped when the history gradient is exactly zero.
Component transplants are diagnostic hybrid states, not training settings.
Their effects are not additive because assignments and suffix allocations
depend on encoder and codebook combinations.

All hard objectives use complete NEW historical input SIDs, original fixed
old full SID targets, fixed masks/EOS, and ALL 4096 anchor_train queries. The
complete suffix map is recomputed independently for every candidate using the
original Base reserved-slot allocator. No suffix fallback is used. Infeasible
candidates have no hard loss and are explicitly marked infeasible.

## Read the output

- `hard_delta` is candidate positive damage minus checkpoint positive damage;
  negative means the proposed change reduced the true hard-forward objective.
- `joint_minus_content_hard_damage` compares the two matched AdamW proposals;
  negative means adding the history gradient helped this one step relative to
  the content-only proposal. Positive means it hurt this local comparison.
- `proxy_book_delta` is history STE gradient dot book parameter change. This
  local linear estimate omits encoder assignment and suffix changes. It is
  not the derivative of discrete argmin/SID allocation and does not guarantee
  the sign of the hard objective change.
- `extra_Adam_update/content_Adam_update` measures the norm of the DIFFERENCE
  between the two proposed book updates relative to the content-only update.
  It need not equal the .1 raw-gradient ratio. A large ratio alone is not proof
  of a bad direction; inspect actual hard loss changes as well.
- Parameter/candidate restoration is checked, and there is no holdout-based
  direction or checkpoint selection. Findings apply locally to these two
  states and this one next content batch; they cannot establish a general
  optimizer cause or a new model's holdout performance.

`semantic_change` and `suffix_change` are catalog-item fractions relative to
the probed checkpoint. RQ_MSE is a full-catalog reconstruction evaluation.
Neither is a recommendation metric.

The 50-step checkpoint is the model AFTER step 50, so its reference should
match H_before of step 51 (about .102307), not H_before of step 50. The final
checkpoint reference should match the final .242539 training damage.

## Validation

Synthetic CPU tests use actual repository RQ-VAE/TIGER and verify that the
joint AdamW proposal exactly reproduces the next v2 training step, including
a sampler rollover; all source files remain unchanged. See validation.log
and validation.json. No real Beauty/GPU diagnostic has run locally.
