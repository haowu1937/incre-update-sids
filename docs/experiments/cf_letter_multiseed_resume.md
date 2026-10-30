# Fixed-budget multi-seed continuation

The original job stopped on task-control Git fetch exit128 after10000 RQ and24000 GPT updates. It has no global lock or Future result. The original request, result, needs_attention receipt and all source outputs stay unchanged.

## Execution

- Job `cf-letter-multiseed-resume-001`, scope `cf-letter-multiseed-resume-v1`; new6000 steps, family40000. Only `s2031_ft_old`, `s2031_plain`, `s2031_historical`,2000 GPT steps each. No RQ retraining.
- Validate the exact old terminal result, runtime and all eight artifact hashes, complete original file index, unchanged inputs, no uncertain step intent, exact10 RQ/12 GPT counters and no Future access. Reject unexpected seed2031 state.
- Copy indexed evidence into `workspaces/cf-letter-multiseed-resume-001/outputs/cf-multiseed-resume`; never hardlink to the source. Internal hardlinks only deduplicate freshly copied private files. Require enough disk for unique source bytes plus8GiB. Preserve original per-arm code SHA and immutable reused-file hashes.
- Keep fixed Base/E1, cohorts2026, seeds2027-2031, codebook beta1=.9 and all parameters. Three new GPT arms use the original reset recipe. Scheduler starts at the original12-arm journal and cannot exceed the full15-arm/30000 GPT allocation. New-only total must equal6000.
- Lock all fifteen selected checkpoints/maps before opening any Future content. Serial15 passes,34189 queries,I0=10924,beam20,topk5/10/20. The original five-seed means/sample SD/t95%CI(df4), within-seed paired contrasts and500-draw user bootstrap remain unchanged.
- Max2 owned training children,24576MiB guard, foreign GPU waits.20h guard prevents new stages;24h wrapper deadline does not kill. Unknown status/manual pause/child failure stops new work. No automatic retry.

## Communication and acceptance

Task-control fetch errors produce `waiting_for_git`, keep pending work and use60/120/300/900-second backoff. Already active work continues; a fresh successful control read is required before later stages. Redacted stderr is retained locally. Invalid control JSON is a real blocker, not a network retry. Installed Runner/core/cron are unchanged; separate host startup/delivery I/O behavior is not claimed fixed.

The existing mandatory Base/pair/downstream/multiseed real CPU checks plus a new actual private-copy/mixed-code/two-seed pipeline must pass before Beauty. Cloud has no Torch/Transformers; skips are not passes. New tests exercise no-alias copying, source counts, waiting/recovery and exact local pause reconciliation.

The final verifier checks immutable original and copied evidence, reloads every RQ monitor on original production CUDA, accepts original versus new GPT metadata by exact per-arm code SHA, checks raw fitting/selection/Future records, reaggregates all statistics and independently checks CPU likelihood samples. It verifies40000 family updates but reports6000 new steps to generic accounting. Failed original scope remains needs_attention. Eight bounded JSON artifacts retain the family counters and separate new step count.

## One-time local pause reconciliation

After code and immutable new task are queued, execute the pinned `scripts/lab_buffer/reconcile_multiseed_resume.py` under the existing Runner lock. It requires exact original failure evidence, no active/undelivered task, fresh remote task/contract and noninteractive fetch plus dry-run push. It archives only the matching local failure pause and error, preserves all state, does not install code or launch a process, and leaves the existing cron to claim the task. A mismatched pause remains untouched.

Update the English report and existing top multi-seed table after scientific acceptance, then stop. This continuation does not authorize extra parameters, seeds, Future passes or budget.
