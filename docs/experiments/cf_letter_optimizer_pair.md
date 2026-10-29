# Explicit two-setting RQ-only / Mixed verification

Hao requested the second stage for both lr=1e-5 settings on 2026-10-29. This is a
new explicit authorization, not a claim that either H-only setting passed the old
screening gate. Preserve the original five-arm results, 2% H-trend limit and
blocked scientific receipt. Do not spend the original conditional 2000 steps.

| Setting | LR | AdamW beta1 | Objective | New steps |
| --- | ---: | ---: | --- | ---: |
| lr1e5_b09 | 1e-5 | 0.9 | RQ-only | 1000 |
| lr1e5_b09 | 1e-5 | 0.9 | Mixed | 1000 |
| lr1e5_b0 | 1e-5 | 0 | RQ-only | 1000 |
| lr1e5_b0 | 1e-5 | 0 | Mixed | 1000 |

Each arm independently restores accepted Base C0 and fixed refreshed E1, not the
H-only endpoint. Only three 512x64 codebooks update. Mixed minimizes native RQ +
0.0006386606341359484 H; RQ-only minimizes native RQ and measures H with the same
anchors. Keep beta2=0.999, epsilon=1e-8, weight decay=0, clipping=0.1,
temperature=0.5, seed2026, accepted STE/residual/native-RQ gradient routing,
content batch4096, rotating anchor batch128 and identical sampler order.

The selected Base GPT, E1 and SASRec remain frozen. H uses the original four-digit
E0 target/reference, mask, positions and I0 candidate addresses. Only the original
4096 anchor_train queries and Base_train/Update_train data enter this diagnostic.
No H-only retraining, new seed, lambda calibration/sweep, STE change, validation
ranking, anchor_holdout, Future or recommender training.

Record actual pre-update batch H/gradient components/clipping every step, fixed128
hard H every10 steps, full4096 hard H and full-catalog native RQ/MSE every10
completed content epochs (30 steps), plus endpoints. Regenerate production hard
SIDs and complete reserved-slot suffix capacity after every update. Preserve
rejected and last-valid checkpoints on infeasibility; do not expand capacity.

For each setting, compare Mixed to its matched RQ-only arm using the last8
scheduled full monitors. Preserve the existing criteria: Mixed H reduction>=5%,
native RQ and hard MSE each<=+2%, joint tail stability including H trend<=2% and
H range<=10%, RQ trend<=2% and CV<=3%; require completed feasible arms. Report
negative or inconclusive results and curves without selecting intermediate minima.
Finite-window stability does not establish asymptotic convergence.

The completed zero-update index audit checked170 old monitors, with CPU/saved
differences at4 points/9 elements. All4 implicated GPU reloads exactly reproduced
saved full and old-item assignments. Original full versus old-item saved chunks
matched. First differing residual choices had float64 distance gaps of order
1e-9 to1e-7; later residual differences can propagate. These measured facts support
backend sensitivity at those points, without accepting the entire old study.

For this new task, scientific acceptance explicitly authorizes **read-only GPU
reloads**, on the same CUDA device and deterministic/TF32 settings as training.
Require exact full-catalog assignments and exact old-item assignments with their
original chunk shapes for every monitoring checkpoint and final state. Do not
replace hard addresses with CPU alternatives or relax their equality tolerance.
The task-local verifier sets CUDA visibility before importing Torch because the
generic verifier environment defaults to CPU. It waits for foreign GPU activity,
checks for foreign processes during replay, and changes no installed Runner code.
CPU still reaggregates original query records/H and checks initial/final GPT NLL.

Job `cf-letter-optimizer-pair-001`, scope `cf-letter-optimizer-pair-v1`; new cap4000,
parallel<=2, wall guard4h. A durable per-arm reservation precedes launch. Four
CPU preflight suites, including actual tiny four-arm training plus full acceptance,
must pass on the server before Beauty updates. Local Torch skips are not passes.
The six returned artifacts fit the existing generic-v1 interface; all raw metrics,
query records and checkpoints stay in the independent job output. Mandatory
read-only inputs bind accepted Base/refresh and index-audit evidence by SHA256.

Reuse the installed Runner/cron. GPU occupied before launch means wait; foreign
activity, memory>24GiB, manual pause or deadline stops new arms while active arms
finish. No process termination, main-workspace switch, input write, old-result
overwrite, infrastructure upgrade, automatic retry or extension. On completion,
verify, update the English report/receipt under the shared CAS lease, then stop.
