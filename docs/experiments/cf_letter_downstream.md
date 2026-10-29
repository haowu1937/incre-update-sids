# Authorized CF/GPT downstream comparison

Hao approved this five-arm plan by saying **下发这个任务** on2026-10-29.
Job `cf-letter-gpt-future-001`, scope `cf-letter-gpt-future-v1`, scientific manifest
`cf_letter_downstream.json`. New fitting cap10000 GPT optimizer updates and five
complete Future prediction passes. No RQ, CF or SASRec fitting or extra H-only run.

| Arm | Frozen full-SID mapping | GPT Update steps |
| --- | --- | ---: |
| ft_old | Original accepted C0(E0) SID table, unchanged | 2000 |
| plain_b09 | Accepted RQ-only,lr1e-5,beta1=.9,step1000 | 2000 |
| historical_b09 | Accepted Mixed,lr1e-5,beta1=.9,step1000 | 2000 |
| plain_b0 | Accepted RQ-only,lr1e-5,beta1=0,step1000 | 2000 |
| historical_b0 | Accepted Mixed,lr1e-5,beta1=0,step1000 | 2000 |

Reuse the exact accepted Base GPT (selected Base step1192) with reset AdamW in
each arm: lr1e-4,weight_decay0.01,beta=(.9,.999),epsilon1e-8,batch256,
clip1,FP32,dropout0.1,constant LR and seed2026. All GPT parameters train. These
GPT betas are identical across arms and distinct from the upstream codebook betas.
Target-only mean four-digit token CE; no EOS or H penalty in GPT fitting.

All arms use the same Base-train I0 identities (10924 items), the same eligible
Update training queries (target in I0), sampler order and coverage masks. New
items remain masked history blocks and unavailable targets in this comparison.
Truncate the most recent20 raw events before availability masking; retain interior
positions. Real recommendation history AND target use the current arm's same map.
Do not use the new-history/old-target historical-damage diagnostic as ranking input.

Every arm trains exactly2000 new updates. Selection candidates0/100/250/500/1000/
2000 use the same deterministic512-query Update-validation probe, full-denominator
NDCG@10, earliest tie. Selected Update step may be0. Report full3280-query Update
validation once at the selected checkpoint. Never select an RQ intermediate minimum.

All five selected model hashes, mapping hashes and probe identities must be locked
before the driver opens Future query content. Then evaluate all34189 original
prequential Future queries for each arm, with fixed I0,beam20,Recall/NDCG@5/10/20,
no seen-item filtering, parameter update or SID allocation. Outside-I0 targets are
explicit misses in the full denominator; also report common-served metrics,
coverage, unavailable history and NLL. The two primary comparisons are Historical
minus matched Plain; four additional comparisons use FT-old. Compute paired user-
cluster bootstrap95% intervals (500 draws). This is one engineering seed, not a
cross-seed confirmation. Future never changes configurations or model selection.

Source Base and paired-step1000 artifacts are explicitly SHA-bound read-only
dependencies. Copy only into the new task's private output tree; hardlinks are
only between those new copies. Retain complete checkpoint/query/trace evidence.
The old five-arm optimizer screening receipt remains blocked; the accepted later
paired study supplies the four mappings. Previous negative RQ/H findings remain.

Reuse generic Runner v1 without installed helper, policy, cron or main-checkout
changes. The same-architecture accepted Base GPT observed peak6440MiB; allow at
most two owned training children with aggregate24576MiB protection. Future runs
serially. Foreign GPU activity before launch means wait; activity/unknown state,
manual pause or resource/deadline guards stop new stages while active work finishes.
The task deadline is24h; a20h driver guard stops subsequent stages, never kills
processes. No automatic retry, extra seed, budget recycling or extension.

Server preflight requires the existing actual Base tests plus new actual five-arm
CPU fitting, constrained generation, Future-lock and full scientific-verifier
acceptance. Cloud Torch skips are not passes. Read-only scientific acceptance
rechecks hashes, immutable source maps, fitting order/counters/reset, all probe
record aggregation/selection, lock timing, complete Future records/metrics and
paired bootstrap, plus independent CPU selected-model NLL samples. Large records
stay local; seven bounded JSON artifacts use the existing mailbox interface.

The server appends accepted results under the shared CAS lease and deduplication
marker. The hourly/manual controller maintains the separate CF/GPT block near the
top of the English experiment report, preserving the earlier Llama/TIGER table
and its different catalog/protocol. Analyze, report and stop, next_job_id=null.
