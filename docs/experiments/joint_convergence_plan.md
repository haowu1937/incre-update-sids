# Train-only joint RQ-VAE / historical-damage convergence study

Status: **implementation delivered; server attachment and real-model checks pending; no jobs queued**.
Prepared on 2026-10-23 against vibe-dev `fad7f94ef1b7774b145b66ed77cb41e6879b69ff`.
Hao subsequently requested implementation/execution with “继续”. The new scope is
`history-joint-convergence-v1`; it preserves the completed `history-gradient-v1`
execution record and does not spend its exhausted 60-step budget.

## Objective and interpretation

Find a parameter setting that both improves content modeling and reaches a stable,
acceptably low true hard-SID historical damage H. Require lower H than a paired Plain
control at the same learning rate and training budget. Do not optimize reconstruction
MSE alone: the production content objective is

`L_RQ = reconstruction_MSE + 0.25 * quantization_loss`,
`L_mixed = L_RQ + lambda * H`,
`H = mean(relu(NLL_new_full_history_SID - NLL_old_history_SID))`.

H is zero at the unchanged Base mapping, so monotonic descent from Base is not a
meaningful requirement. A constant loss caused by negligible updates is not sufficient
either. Test stable late training, useful adaptation and the paired damage advantage.

The previous 1000-step pair reduced MSE but increased full L_RQ:
0.006417349 initially, 0.006498323 Plain and 0.006507471 Historical finally.
Historical anchor_train H was 0.138890274 versus Plain 0.121296058.
The sampled weighted historical/RQ gradient norm ratio at lambda=1e-4 had median
1.3148% in the encoder and 2.6478% in the codebooks. These observations motivate a
controlled learning-rate / lambda search, but do not prove that a larger lambda works.
The five-step reset-optimizer diagnostic is not evidence of long-run convergence.

## Fixed scientific conditions

- Fresh copy of the same original Base RQ step-1000 weights for every new arm.
  Base checkpoint SHA256:
  `23d4163c8cb458d7830cdca66900a2e15727d33456fc530b6b026c856fceab55`.
  No codebook reinitialization, alignment or suffix expansion.
- Features: original unmodified Llama 4096-dimensional float32 vectors under
  `/home/hao/frec/graph-informed-sids-web/cache/incremental/Beauty/llama7b_features_20261016T132127Z`.
  Use the existing source/hash-bound deployment inputs from the completed E1 and
  retuned runs; verify their exact paths and hashes before model execution.
- All 11707 Update-train catalog content items, with groups Base train=10924,
  Base-validation first seen=255, Update-train first seen=528. No new feature fitting.
- Encoder hidden dimensions [512,256,128], latent dimension 64, three 256-entry
  codebooks; encoder, decoder and all codebooks trainable. Keep existing RQ commitment
  settings, distance scales and loss weights. No graph/CF/SK additions.
- Original frozen Base TIGER, original 4096 anchor_train queries, rotating training
  anchor batches of 128, content batch size 256. Current hard full history SID;
  original target, mask and candidate catalog. Suffix capacity remains 256.
- Keep STE, temperature=0.5, max_grad_norm=0.1 and AdamW weight_decay=0 fixed.
  Keep the installed production beta/epsilon values and record them explicitly.
- Primary seed=2026. Every arm shares Base weights, content order, anchor draw order
  and deterministic settings. Plain measures H on the same anchor draws without
  including H in its optimized objective.
- No anchor_holdout, Base-validation ranking, Update-validation ranking or Future
  scoring for screening, selection or early stopping. Reading bound files for
  input-integrity checks does not authorize evaluating those cohorts.
- Every selected extension restores the complete optimizer, scheduler, samplers and
  RNG states. It is a faithful continuation, not a fresh AdamW start.

## Candidate matrix and common learning-rate schedule

| Peak learning rate | Plain lambda | Historical lambdas |
| --- | --- | --- |
| 3e-5 | 0 | 1e-4, 1e-3, 1e-2 |
| 1e-4 | 0 | 1e-4, 1e-3, 1e-2 |
| 3e-4 | 0 | 1e-4, 1e-3, 1e-2 |

Twelve arms form a full factorial matrix. Compare lambdas at a fixed peak LR and
learning rates at a fixed lambda. Do not attribute cross-cell differences to lambda
alone. The old lambda=1e-3 capacity failure used other settings and does not isolate
a lambda effect.

All candidates use the same declared 3000-step schedule: linear warmup over steps
1-50 to the peak LR, cosine decay to 0.1 times that peak by step 2500, then a constant
nonzero LR for steps 2501-3000. Record the actual LR at every step. The common schedule
is a deliberate new recipe; the paired lambda=0 arms control for that change.
Do not reset or rebase the schedule at the stage boundary.

## Stages and bounded new budget

| Stage | Work | Maximum new optimizer steps |
| --- | --- | ---: |
| Throughput calibration | Four independent 100-step runs: two sequential and the same two concurrently | 400 |
| A: screening | All 12 matrix arms through step 1500 | 18000 |
| B: convergence | At most two Historical candidates plus their matching Plain controls, at most four distinct arms, faithfully extended from 1500 to 3000 | 6000 |
| C: confirmation | Best feasible Historical candidate and matching Plain, fresh paired seed 2027, each through step 3000 | 6000 |
| Total | Includes calibration; excludes all previous experiments | **30400** |

These are the new study limits, separate from the completed 60-step diagnostic.
No task is queued until the pinned server attachment passes. Stage C is conditional on a feasible
candidate; unused budget is not reassigned to retries, new parameters, extra seeds or
TIGER/Future work. Calibration trials have unique outputs and are not research endpoints.

At step 1500, capacity/finite checks are mandatory. Use means over the latest eight
completed-epoch observations, including the matching Plain observations. First retain
up to two Historical candidates meeting the final quality constraints below, ordered
by lower H, then lower L_RQ. If fewer qualify, fill from the non-dominated finite,
capacity-feasible Historical RQ/H frontier, ordered by the following fixed violation
score (lower is better):

`V = relu(RQ/RQ_initial - 1) + relu(RQ/(1.02*RQ_plain) - 1)
   + relu((H - 0.95*H_plain)/max(0.95*H_plain,0.001))
   + relu(MSE_new/(0.9*MSE_new_initial) - 1)`.

Label these filled slots exploratory. If the frontier has insufficient candidates,
leave slots unused. Break score ties by lower H, then lower RQ, then smaller LR and
lambda. A Plain capacity failure makes its corresponding LR ineligible for selection.
The H=0 tie rule in the final criteria still applies; a zero violation score alone
does not demonstrate a historical benefit. Publish all inputs to this ranking.
This is an early screen, not a convergence finding: a discarded slow candidate might
have improved later, so conclusions apply only to the retained settings and budget.

Select the final configuration using the stage-B training metrics and the rules below.
Stage C confirms the same configuration; it cannot select a replacement using seed
2027 results. If it fails, report failure rather than searching that seed.

## Measurements that distinguish progress from minibatch noise

There are 46 content batches per full epoch (last batch size 187). Record actual epoch
boundaries; step 1500 is about 32.6 epochs and step 3000 about 65.2 epochs. Do not infer
epochs from processed_examples / batch_size when the last batch is shorter.

At step 0, every completed content epoch, the stage boundary and the final step:
1. Rebuild production hard full SIDs over the whole catalog and audit exact suffix
   feasibility; do not score only a differentiable proxy.
2. Evaluate full-catalog L_RQ, reconstruction and quantization, plus each content group.
3. Evaluate H, old/new NLL and signed delta NLL over the same full 4096 anchor_train
   queries, and over one fixed 128-query monitor subset. These are training monitors,
   not independent validation estimates.
4. Record semantic churn, full-SID churn, suffix-only changes, unique semantic codes,
   codebook utilization, maximum bucket and required suffix capacity.
5. Record teacher/inputs unchanged and all optimizer/sampler progress identities.

At every optimizer step record pre-update minibatch RQ/reconstruction/quantization,
unweighted actual hard H, lambda*H, combined objective, LR and clipping norm/coefficient.
RQ-only must measure actual H instead of logging a zero placeholder. Do not compare
combined objectives across different lambdas as if their scale were the same.

Every 25 steps and at epoch endpoints record raw RQ and H gradient norms, lambda-weighted
ratios and cosine per encoder/codebooks, plus actual update norms and dot products.
Keep those probes observational. A first-order STE direction is not a hard-H guarantee.
Full-catalog audits follow every update; save the last valid full state and a separate
failure record if capacity or numerical checks fail.

Persist machine-readable curves, all declared arms including failures, final summaries,
plots and complete resume state. Monitor evaluation must preserve model mode and RNG
and must not advance training samplers. Keep raw epoch snapshots needed to verify
plateau calculations. Do not smooth curves before storing the raw values.

## Predeclared engineering acceptance rules

These thresholds are practical study criteria, not proof of mathematical convergence.
Freeze them in the executable manifest before running; do not relax them to manufacture
a successful arm.

Use the last eight completed-epoch evaluations within the final 500-step constant-LR
tail. Compare their first four and last four means. Require:
- Absolute relative drift of L_RQ <=2%, and its coefficient of variation <=3%.
- Absolute H drift <=max(0.001, 0.02*mean H), and H's max-minus-min over the eight
  observations <=max(0.005, 0.10*mean H).
- No nonfinite metric or full-SID capacity violation.

Use the same eight-epoch means for final quality comparisons:
- Historical L_RQ below its unchanged-Base initial value, and <=1.02 times its paired
  Plain L_RQ.
- Historical H <=0.95 times its paired Plain H. If both H values are numerically zero,
  report a tie with no demonstrated historical-penalty benefit.
- Reconstruction MSE for the 528 Update-first-seen items improves by at least 10%
  against the frozen Base. Track overall and old-item reconstruction too.

Among candidates meeting all conditions, choose the lowest H, breaking a tie on L_RQ.
Use last-eight-epoch means, not a transient best checkpoint. Report the fixed
step-3000 endpoint separately. Require the chosen pair to satisfy the same criteria
at seed 2027 before describing the setting as reproducible across the two seeds.
The result remains train-side evidence, not a downstream recommendation claim.

Failure to meet these criteria is a useful negative result. Stop after this matrix
and confirmation; do not automatically change STE, temperature, trainable components,
gradient routing, loss weights or search bounds. A later mechanism ablation requires
its own scoped design.

## GPU concurrency and execution contract

Memory capacity alone does not establish a speedup. Calibrate two matching 100-step
trials sequentially and concurrently, including teacher/anchor caching and evaluation
overhead. Record aggregate steps/s, per-arm latency, GPU utilization, CPU load,
per-process allocated/reserved bytes and total nvidia-smi memory.

Start at concurrency 2 only if aggregate throughput improves by >=10%, there is no
OOM, and peak total GPU memory stays <=24 GiB, leaving at least 8 GiB on the 32 GiB V100.
Otherwise use 1. The delivered implementation caps concurrency at 2; the optional
four-worker pilot in the initial proposal is deferred. OOM, a violated memory guard
or uncertain partial progress pauses dispatch for diagnosis instead of repeating steps.

Run all simultaneous arms under a single Runner-owned bundle with a recorded child
PID set, independent job/arm IDs and output directories. Keep the external GPU-busy
gate: a process outside that owned set means wait. Do not run multiple independent
Runners or weaken the existing safety gate. Fix/record CPU thread and data-worker
settings across speed measurements and research arms. Never vary scientific batch
size to fill memory.

The old installed Runner is serial. The new exact gate and bundle are delivered through
the one-time existing-Runner attachment; do not claim they are deployed before evidence.
Every real task must bind an immutable
code SHA, exact argv/config, input hashes, fresh job_id/output root, deadline, expected
artifacts and result receipt. Parent and each arm need success/failure/interruption
records. A deadline pauses new dispatch and preserves live-process evidence; it does
not authorize killing the user's running experiments.

Latest Git state inspected for this design: completed 60-step diagnostic, no next job,
and worker snapshot `runner_status=paused`, observed 2026-10-22T16:47:13Z.
A later server read tied the pause to a Git-fetch failure at 2026-10-22T16:42:27Z.
Hao archived that pause after successful guarded fetch and terminal-state checks;
the Git worker snapshot at 2026-10-23T09:05:17Z reports idle, with no user pause.
Before execution, recheck the actual server state and deploy only the necessary
bundle/measurement support through the existing installation. Do not clear pause
records speculatively, switch the main experiment workspace, modify original inputs,
terminate live jobs or duplicate cron/Web tasks.

## Deliverables and stopping point

Per arm: immutable config/input/code identities, stdout/stderr, step and epoch curves,
archived stage/final checkpoints, resume state, capacity audit, resource measurements
and compact result. Batch summary: all 12 matrix rows (including failures), paired
training comparisons, declared selection arithmetic, seed-2027 confirmation if eligible,
and English experiment_report.md with the current downstream table preserved.

No TIGER fitting, holdout evaluation, Base-compatibility ranking or Future inference
is included in the bounded 30400-step study. Those remain a later decision after a
training-side candidate has passed the declared checks.
