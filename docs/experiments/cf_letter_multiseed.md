# Five new paired seeds: codebook beta1=0.9 and CF/LETTER Future

Hao approved dispatch on2026-10-30 after reviewing this fixed plan. Job
`cf-letter-multiseed-001`, scope `cf-letter-multiseed-v1`; machine manifest is
`cf_letter_multiseed.json`. This explicit authorization supersedes the previous
completed-study stop only for this study. No installed Runner/helper/policy/cron
upgrade, main-workspace change, input write or termination is needed.

Seeds2027,2028,2029,2030,2031 are all predeclared. Pilot2026 is shown separately;
beta1=0 results remain in the historical record and are excluded from the new
primary comparison. Restore the same accepted C0/Base GPT and fixed accepted E1
for every seed; SASRec, Base and E1 are not retrained. These intervals therefore
condition on the existing Base/E1. Future has already been viewed: this is fixed-
recipe repeatability, not a new blind test.

| Method per seed | Codebook training | GPT Update fitting |
| --- | --- | --- |
| FT-old | Original C0(E0) SID unchanged; zero steps |2000 steps |
| Plain | RQ-only from C0 on E1;1000 steps |2000 steps |
| Historical | RQ+0.0006386606341359484 H;1000 steps |2000 steps |

Only three512x64 codebooks update. Reset AdamW lr1e-5,beta1=.9,beta2=.999,
epsilon1e-8,wd0,clip.1,content batch4096,anchor batch128,temperature.5; retain
the existing STE/residual/native-RQ and suffix rules. Matched Plain/Historical
share each seed's content and anchor order. Keep original target/reference/mask/
positions/I0, the frozen Base GPT and the exact4096 anchor_train identities.
The data/cohort seed stays2026 while optimizer/sampler seeds vary. No lambda
recalibration, intermediate minimum selection, H-only experiment or stability
gate filtering of seeds. Record every-step batch H/gradients/capacity, fixed128
every10 steps and full4096/RQ every10 completed content epochs, plus endpoints.
Only final step1000 full SIDs enter GPT adaptation. Preserve an infeasible seed
and all rejected/last-valid outputs; stop the bundle without replacing that seed.

All fifteen GPT models restore the accepted selected Base checkpoint1192 with
fresh AdamW lr1e-4,wd.01,beta(.9,.999),epsilon1e-8,batch256,clip1,FP32,dropout.1.
Use each seed's identical fitting order across all three methods. Four target
token CE, no EOS/H term; current history and target use the same arm map. Fit
the same24039 target-in-I0 Update queries; fixed I0 has10924 items. Preserve20
raw events before availability masking and interior positions. New items are
unavailable in this comparison.

Actually fit2000 updates for every GPT. Select0/100/250/500/1000/2000 on the same
fixed512 Update-validation full-denominator NDCG@10 probe (cohort seed2026),
earliest tie. Prior selected250 does not fix the new selected step. Full3280
Update validation is reporting only. Lock all fifteen model/map/probe identities
before opening Future query content, then run fifteen serial full34189-query
passes: same prequential history,beam20,no seen filtering,Recall/NDCG5/10/20,
full and common-served denominators,coverage,NLL,unavailable history. Outside-I0
targets remain misses; coverage should reproduce20281/34189.

Compute each seed's score first, then equally average five seeds. Report mean,
sample SD and two-sided95% Student-t interval with df4 for each method. Primary
endpoint is full-denominator Future NDCG@10; Recall@10 is secondary. Compute
Historical-minus-Plain within seed, then its mean/SD/t interval; also paired
Plain-minus-FT-old and Historical-minus-FT-old. Keep all cutoffs and all seeds.
Per-seed user-cluster bootstrap95% intervals (500 draws, cohort seed2026 for
reproducible draws) are additional query-sampling evidence, distinct from seed
uncertainty. Do not treat five copies of the same Future queries as independent
test samples. No Future-driven tuning, seed selection, extension or extra seed.

New reservation40000:10000 codebook +30000 GPT steps, fifteen Future passes.
At most two owned training children, aggregate24576MiB GPU guard; external GPU
waits,20h stops new stages and24h timeout never kills running work. Unknown GPU,
foreign activity during owned work, pause, failures or infeasibility preserve
evidence and stop subsequent stages while active children finish. Fresh phase/
arm reservations and intent/completed counters are checked against the total.
Require40GiB free private output space. Old tasks/artifacts/budgets are preserved.

Eight bounded artifacts: runtime,compact,data_audit,rq_results,runs,selections,
future,statistics. Large local traces, checkpoint snapshots, raw query records,
full probe curves and all15 selection hashes remain SHA indexed. Server preflight
must pass real Base, paired-codebook, downstream and new two-seed end-to-end CPU
model tests; cloud skips are not passes. Scientific acceptance reuses exact
production-CUDA RQ monitor reloads and raw hard-H/suffix checks, all GPT sampler/
reset/selection/Future checks and CPU NLL samples, reaggregates bootstrap and
cross-seed statistics, and verifies source immutability. No criterion is relaxed.

The manual and existing hourly controller share CAS leases and receipts. Once
accepted, update the English report's top multi-seed table with scores and seed
intervals; retain the pilot, old beta1=0 rows and prior negative RQ/H findings.
Report per-seed H/RQ and selected steps. Mark completed and stop(next_job_id=null).
Failure is not an automatic repair/retry authorization.
