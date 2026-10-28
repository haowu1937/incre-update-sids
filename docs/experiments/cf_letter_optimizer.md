# H-only optimizer ablation and conditional matched verification

Hao approved dispatch on 2026-10-28 with **下发这个任务** after reviewing the five-new-arm plan and conditional pair. Scope `cf-letter-optimizer-v1`; job `cf-letter-optimizer-001`; exact settings in `cf_letter_optimizer.json`. New actual optimizer-step cap **7,000**, including 5 x 1,000 H-only screening steps and at most 2 x 1,000 paired verification steps. No older unused budget is recycled. Reuse the accepted prior H-only control without rerunning its 1,000 steps.

## Matrix and fixed conditions

| LR | AdamW beta1=0.9 | AdamW beta1=0 |
| --- | --- | --- |
| 1e-4 | Accepted control from cf-letter-history-long-repair-001 | New h_lr1e4_b0, 1,000 steps |
| 3e-5 | New h_lr3e5_b09, 1,000 steps | New h_lr3e5_b0, 1,000 steps |
| 1e-5 | New h_lr1e5_b09, 1,000 steps | New h_lr1e5_b0, 1,000 steps |

Each new arm independently restores the same Base C0, accepted fixed E1 (11,691 x 64), and selected frozen Base GPT. Only three 512 x 64 codebooks train. AdamW moments reset; beta2=0.999, epsilon=1e-8, weight_decay=0, constant LR, clipping=0.1. Temperature=0.5; keep the accepted STE, residual path and native quantizer gradient routing unchanged. Beta1=0 removes first-moment history but retains second-moment scaling; this is not SGD.

Content batch=4,096; rotating anchor batch=128 over the exact original 4,096 anchor_train identities. Seed=2026, identical content and anchor sampler sequences, including partial batches. The original E0 four-digit reference NLL, target, user, mask, positions and I0 catalog remain fixed. Initial full H must reproduce 0.10294729462475516. No serving SID is allocated to added CF rows. Preserve the same reserved-slot suffix rule and check capacity after every update.

Retain every-step actual batch H/NLL, RQ loss, objective, component gradients, clipping, displacement and sampler identities. Fixed-128 hard H every 10 steps; full-4096 H and full-catalog native RQ/hard MSE every 10 completed content epochs (30 steps), plus endpoints. Store raw queries, every-step semantic codes, optimizer/sampler/RNG states, monitor checkpoints, last valid and rejected states. Do not select an intermediate minimum. Capacity failure stops that arm and is retained; no reset, alignment, expansion or silent retry.

## Predeclared selection and conditional pair

Assess the last eight **scheduled** full monitors (780, 810, ..., 990), excluding the incidental partial-epoch endpoint at step 1,000. Retain the previous joint stability criteria: native RQ trend span <=2% and CV <=3%; H trend span <=2% and range <=10% of its mean, with the same near-zero rule. A candidate must also complete 1,000 steps with feasible capacity, reduce tail-mean H by >=5% relative to the common initial H, and keep both tail-mean native RQ and hard MSE within +2% of their initial full-catalog values. Endpoint results are reported separately. This prevents no-update plateaus from qualifying.

Compare all six matrix entries, including the hash-bound accepted control. Among eligible entries choose lowest tail-mean H, then lowest tail-mean native RQ, then declared order (control, followed by manifest matrix order). Selection uses training-side H/reconstruction only. If no candidate qualifies, report the full matrix and stop after at most 5,000 new steps.

Only if a candidate qualifies, independently restore C0(E1) for `pair_rq_only` and `pair_mixed`, 1,000 steps each with that candidate's LR/beta1 and identical original sampling. The accepted scalar **lambda=0.0006386606341359484 stays fixed**; no recalibration or sweep. H-only is reused from screening. Apply the previous Mixed gate: >=5% lower tail-mean H than matched RQ-only, native RQ and hard MSE each within +2%, stable Mixed tail, and feasible capacity. Record negative results and stop. These finite-window criteria do not prove asymptotic convergence or held-out benefit.

## Dispatch, concurrency and evidence

Use the installed generic v1 Runner, with a complete code SHA, immutable request/contract/scope, exact deployment hash, 7,000-step reservation and fresh output `/home/hao/frec/git-lab-buffer/workspaces/cf-letter-optimizer-001/outputs/cf-optimizer`. Do not upgrade helpers/gates, change cron/policy/read roots or switch the main workspace. Base, CF-refresh and completed control dependencies are explicitly SHA-bound read-only predecessors, copied into this job before use and rechecked at completion. Per-arm hardlinks refer only to those new private copies, never the old source workspaces.

At most two GPU children are managed by this task. A durable launch journal reserves each case once before Popen; no repeated case name or sum over 7,000 is permitted. Each child has an independent 1,000-step durable intent counter and output. Wait while external GPU work is visible. If a foreign process appears during training, observations become uncertain, memory exceeds 24,576 MiB, manual pause occurs, or the four-hour guard expires, stop launching new arms while existing children finish. Never terminate a live child. A failed child stops future launches while its already-running sibling finishes; preserve both logs and evidence. The outer task timeout remains 86,400 seconds.

The unchanged Runner accepts at most eight explicit artifacts, each <=64 KiB. Return runtime, compact, data audit, selection and four lossless minified curve groups. All 100 fixed and 34 full monitor points per completed arm are retained; raw traces and weights remain local and hash-indexed. No Runner limit is raised.

Mandatory server preflight runs the existing Base and long-diagnostic suites plus the new optimizer suite. The new real-model checks execute all seven optimizer/objective recipes on tiny actual CPU models, independently reload native loss/hard MSE and original/new GPT likelihoods, verify per-step suffixes and the read-only scientific arm verifier, and confirm the accepted history gradient route. Cloud skips are not passes. The vectorized read-only suffix auditor is checked against the original allocator on randomized valid/overflow ledgers, including reserved-slot gaps; training still uses the original allocator.

The task-SHA verifier reaggregates raw batch/fixed/full H, sampler/optimizer states, per-step capacity, canonical monitor reconstruction and CPU likelihoods; recomputes selection and conditional paired gates; and checks exact new-step accounting and unchanged original dependencies. The generic observer commits an English report and receipt under the existing shared lease. Manual continue and the existing hourly monitor inspect that same state and never redispatch the bundle.

Only Base_train and Update_train are accessed. No validation ranking, anchor_holdout, Future, SASRec/GPT training, lambda/STE changes, extra seeds, budget extensions or automatic retry. Keep the existing Base/Future tables and all earlier negative evidence. Finish this matrix/conditional pair, verify/analyze/report, then stop with next_job_id=null.
