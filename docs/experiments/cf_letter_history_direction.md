# Proposed CF-codebook historical-gradient direction diagnostic

Status: **design_only_pending_explicit_authorization**. No server request, contract,
scope authorization or control-state change has been published. The active main
plan remains the completed CF refresh diagnostic. This proposal is isolated on a
development branch. Its submission builder and production command reject
`execution_authorized=false`; synthetic tests cannot grant server authorization.

Automatic approval review rejected the attempt to prepare a shared dispatch lease,
stating that a 60-step History-direction experiment exceeds the latest explicit
Base-only authorization. A separate explicit user confirmation is required before
activating this proposal. No lease or job was created by that rejected action.

The completed C0(E1) diagnostic measured anchor_train H=0.102947295 with frozen
codebooks and Base GPT. The new quantizer trains only three codebooks; it has no
encoder/decoder. The previous content-model gradient diagnostic therefore does
not establish the new model's gradient routes or hard-SID direction.

## Proposed bounded experiment

Use the exact accepted E1, Base codebooks, selected Base GPT and original old
four-digit reference likelihood. Twelve independent reset branches, five steps
each, have a **60-step total cap**:

| Optimizer | Learning rates | Objectives | Steps |
| --- | --- | --- | ---: |
| SGD, no momentum or weight decay | 1e-6, 1e-5, 1e-4 | H-only, RQ-only, Mixed | 45 |
| Reset original AdamW, weight decay .01 | 1e-4 | H-only, RQ-only, Mixed | 15 |

Fix the same 128 anchor_train queries and content-sampler sequence across branches.
Temperature=.5, gradient clip=.1, frozen Base distance scales. The declared
hard-forward/soft-backward semantic embedding and soft residual path is retained;
the discrete suffix has no surrogate gradient. Four positive-H training-only
gradient probes set one scalar lambda0=.1*median(||gQ||/||gH||), with zero norms
skipped. This is a fixed scale calibration, not a lambda or STE sweep.

Before updates, require original NLL and positive full hard H reproduction,
direct-ID/input-embedding hard parity, finite per-book routes and frozen teacher/
CF inputs. After every update regenerate hard complete old-I0 SIDs, preserve
unchanged-prefix suffixes and reserve all old slots before allocating movers.
Record actual fixed-batch H before/after, component norm ratios/cosine, clipping,
parameter displacement and gH dot actual update. At 0/1/5, also score complete
anchor_train and full E1 RQ. Preserve rejected/last-valid states and count committed
infeasible updates. A hard-SID plateau is inconclusive; five steps are not a
convergence experiment.

The proposed task reuses the installed generic v1 Runner and hash-bound read-only
Base/refresh artifacts in a fresh independent output. No installed helper, policy,
read root, cron or main-workspace change is needed. Mandatory actual CPU model
checks precede any server micro-update. The scientific verifier reaggregates raw
hard-H records, checks reset recipes/sampling/counters and every suffix allocation,
and reloads CPU likelihoods and canonical final reconstruction. Bounded return
artifacts include compact, calibration, two curve tables, audit and runtime status.

Excluded: full 800-step lambda pilots, 1000-step paired training, SASRec/GPT fitting,
ranking validation, holdout/Future scoring, alignment, dead-code resets, automatic
retry and budget recycling. Finish the proposed diagnostic/report and stop.

## Prepared implementation and evidence

- Source: `genrec/cf_letter_history_direction.py` and task-scoped protocol/driver/
  verifier. The training loader can exclude all validation/Future query files.
- Local: 17 protocol/regression tests passed across Base, refresh and direction
  suites; nine actual-model tests were unavailable because cloud Torch is absent.
- Proposed mandatory preflight: Base + direction suites, 17 tests including seven
  actual Torch/GPT/codebook tests. Their success is not yet established on server.
- Python 3.10 syntax and whitespace checks pass. Existing Base/Future report and
  all previous requests, receipts, checkpoints and experiment products are intact.
