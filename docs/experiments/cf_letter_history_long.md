# Three matched CF codebook loss trajectories

Authorized by Hao on 2026-10-28: after reviewing H-only / RQ-only / Mixed with longer diagnostics, Hao said **开始下发实验任务**. This supersedes the unexecuted 60-step direction proposal. It does not authorize the remaining migration study, parameter sweeps or downstream training.

Scope `cf-letter-history-long-v1`; job `cf-letter-history-long-001`. The immutable manifest is `cf_letter_history_long.json`. One serial bundle reserves **3,000 optimizer steps: three independent arms of 1,000 steps**. Equal steps and sampling do not imply equal compute. Each arm's runtime is recorded.

## Fixed starting state and objectives

All arms independently restore the accepted Base C0 codebooks and the same accepted refreshed E1 table from `cf-letter-refresh-diag-001`. E1 contains 11,691 training-observed rows, dimension 64. There is no quantizer encoder/decoder. Only three 512 x 64 codebooks train. Base GPT, selected at step 1,192 / epoch 4, and SASRec/CF vectors stay frozen. Initial full anchor_train H must reproduce **0.10294729462475516** before any updates. The positive initial damage comes from CF refresh; the original E0 reference is not rebased to C0(E1).

| Arm | Minimized objective | Steps |
| --- | --- | ---: |
| H-only | Hard-forward historical H, STE backward | 1,000 |
| RQ-only | Native collaborative quantizer objective L_Q | 1,000 |
| Mixed | L_Q + lambda0 H | 1,000 |

All arms use reset AdamW (betas 0.9/0.999, epsilon 1e-8), constant LR 1e-4, **weight decay 0**, global gradient clipping 0.1, seed 2026 and STE temperature 0.5. Weight decay 0 is a declared change from Base quantizer fitting, applied equally to all three arms to isolate the losses. No optimizer moments are inherited. Distance scales are frozen Base-E0 median row-distance standard deviations, using the first 4,096 items.

Content batch=4,096; same reshuffled order, including partial final batches. Training anchor batch=128; the same rotating sampler over the original 4,096 anchor_train identities in all arms. Anchor sampler seed=2026+78. A separate fixed monitor uses 128 queries from randperm(seed=2026+77). Complete sampler states, optimizer, RNG and weights are saved. No automatic resume/retry is authorized.

Calibrate lambda0 once before the arms using four fixed positive-H training batches: **0.1 x median(||g_Q|| / ||g_H||)**. Require all four finite, nonzero component norms and positive H; otherwise preserve the diagnostic and perform zero updates. Lambda stays fixed. No lambda or temperature scan, gradient normalization or STE replacement is included.

The frozen retriever scores updated full history SIDs, but original target SID, user, mask, positions and old I0 catalog remain fixed. H is mean(relu(new four-digit sequence NLL - original Base sequence NLL)); no EOS. Hard suffix allocation has no surrogate gradient. Preserve unchanged old slots and reserve all old widths before appending movers. No new-item serving SID, codeword alignment, dead-code reset, suffix expansion or wraparound.

## Measurements and acceptance

- Every step: actual sampled content loss and training-batch H before update, minimized objective, content/anchor sampler hashes and counters, component gradient norms/weighted ratios/cosines, clipping and actual codeword displacement. Raw per-query batch NLL is retained. Recompute old-I0 canonical semantic codes and complete suffix feasibility after every committed update.
- Every 10 steps: hard H on the fixed 128-query monitor, regenerated from full discrete SIDs.
- Every 10 **completed content epochs**: full 4,096-query hard H, signed delta NLL, full-catalog native L_Q and hard reconstruction MSE, old-item MSE, semantic/full-SID churn, utilization, capacity and gradients. With 11,691 rows and batch 4,096, this is every 30 steps. Also evaluate steps 0 and 1,000. An incidental final partial epoch does not enter scheduled-tail stability.
- Last eight scheduled full monitors: native L_Q relative linear-trend span <=2% and CV <=3%; H relative linear-trend span <=2% and range <=10% of its mean. Relative trend span = abs(OLS slope) x (n-1) / abs(mean). When mean H <=1e-8, use absolute range <=1e-8 instead. Report insufficient tails explicitly.
- Mixed efficacy requires at least 5% lower tail-mean H than matched RQ-only, both native L_Q and hard MSE within +2%, a stable Mixed tail and feasible complete SIDs. Preserve negative results. A stable finite window does not guarantee asymptotic convergence.

Capacity failure preserves the rejected checkpoint, last valid checkpoint, codes and trace, and stops that arm; other predeclared reset arms may run. Numerical/implementation failures preserve evidence and pause the task. Never auto-extend the budget or retry the same fault.

## Execution and evidence boundary

Use the installed generic v1 Runner without changing installed helpers, policy/read roots, cron or the main project checkout. Bind complete code SHA, manifest, deployed main-deployment.json, accepted Base and refresh dependency hashes. Read-only accepted files are copied into this job's own output and rechecked at completion. Output: `/home/hao/frec/git-lab-buffer/workspaces/cf-letter-history-long-001/outputs/cf-history-long`.

Wait for external GPU work. The existing four-hour guard stops new arms while a live arm finishes; the outer task timeout is 86,400 seconds. No process is terminated. Maximum GPU memory guard=24,576 MiB; execution is serial to preserve the paired setup.

Mandatory server CPU preflight: Base model suite and the new long-diagnostic suite, including real GPT/codebook gradients, three independent resets, rotating sampling, monitor schedule, checkpoints and scientific-verifier acceptance on tiny real models. Cloud skips are not real-model passes. The task's read-only verifier reaggregates raw batch/fixed/full NLL, independently regenerates suffix capacity for every step, reloads monitoring checkpoints for native loss/MSE and checks CPU initial/final GPT NLL before the generic observer commits the English report and receipt.

Only Base_train and Update_train are accessed. No validation ranking, anchor_holdout, Future, SASRec fitting or recommender fitting. Existing Base/Future tables and all old negative evidence remain intact. Stop after this diagnostic is verified, analyzed and reported; next_job_id=null.
