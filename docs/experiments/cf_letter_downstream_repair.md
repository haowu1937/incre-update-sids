# Single zero-real-step report-formatting repair

Original job `cf-letter-gpt-future-001`, code
`e4e5476d3581eb28e3c0e5ef2d852d04f2f3c2ae`, failed the real CPU preflight.
The tiny end-to-end test uses Recall/NDCG at1/2/4; the final verifier report
formatter incorrectly requested hard-coded at5/10/20 keys. Its five synthetic
CPU adaptations, constrained predictions, global lock, record reaggregation,
checkpoint checks and paired arithmetic had completed before this report error.

The exact runtime and returned result hashes are captured in
`cf_letter_downstream_failure.json`. Runtime command_started=false,
checks_completed=1, status=failed: **zero Beauty GPT updates and zero real Future
passes**. Preserve the failed worktree, logs, immutable request/result and
needs_attention receipt. Never describe the synthetic test's Future messages
as evaluation of the actual34189-query dataset.

Fix only the formatter to use the configured metric cutoffs. Add a pure regression
for both tiny1/2/4 and production5/10/20. All production settings, five maps,
2000-step fitting caps, common I0, six selection points and Future protocol remain
unchanged. Existing mandatory real-model end-to-end preflight remains required.

Replacement job `cf-letter-gpt-future-repair-001`, scope
`cf-letter-gpt-future-repair-v1`, manifest `cf_letter_downstream_repair.json`.
This is a diagnosed software repair under Hao's standing authorization to fix
in-scope failures with a new job ID. The whole original/replacement family still
has only10000 actual GPT steps and five real Future passes. The replacement may
start only after checking exact original failure bytes, command_started=false
and absence of the original real output directory. Check that evidence again in
scientific acceptance; bind it as another read-only input.

No Runner/helper/gate/policy/cron upgrade. The installed main_service pauses local
scanning on any failed task before later Git fetches. A Git request cannot clear
that local PAUSED guard. Prepare the concrete replacement first, then reconcile
only this exact failed-preflight pause under runner.lock, preserving its bytes in
resolved-pauses. Do not clear another failure, resume uncertain work, kill a
process or repeat the original job. After reconciliation the existing cron
fetches and runs the replacement normally. If its checks fail, preserve evidence
and stop; do not automatically cycle through more repairs.
