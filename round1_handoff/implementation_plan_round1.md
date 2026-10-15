# Implementation plan — Round 1 update

- Scope: E0 and E1; E2–E5 remain not_run.
- Source reviewed: 5c2ca487f961e91e75e7d8f3ca78349faf3dc82e.
- Local SHA: pending automatic run metadata, not an audit prerequisite.
- Environment: user reports source activate frec-gpu runs upstream code;
  environment audit/setup skipped by user request. AGENTS.md absent by user confirmation.
- Round 1 status: code_provided; server execution not yet confirmed.

## Delivered
1. genrec/temporal.py: safe records, raw sequence adapter, global temporal
   windows, strictly causal rows, stable append-only registry snapshots,
   availability flags, counts and content hashes.
2. AmazonReviews2014/dataset.py: narrow parser import/body patch only.
3. tests/incremental/test_temporal_round1.py: 19 CPU synthetic checks;
   run evidence uses new result directories and never overwrites prior runs.

## Pending runnable rounds
1. After Round 1 server evidence: extend genrec/dataset.py and genrec/tokenizer.py,
   Amazon temporal cache path and prepare_dataset.py as needed; consume causal
   rows once, isolate train/validation/catalog state and retain metadata outside
   T5 kwargs. Keep legacy leave-one-out usable.
2. Fix gensid/datasets_module.py and graph_aware_sid/graph_utils.py:
   noncontiguous global IDs, zero truncation, graph-node mapping, cache state and
   isolated-node handling. Add stage-filtered graph/feature manifests.
3. Extend VQ/RQ/RQVAE and main_gensid.py: capacity-aware initialization,
   restored initialization state, fixed assignment policy and raw-hard export.
   Add fixed-capacity deterministic collision suffix allocation and SID handoff.
4. Add frozen TIGER loss/embedding probes and differentiable assignments;
   verify old labels, matching logits, finite per-level gradients and unchanged
   GR weights with returned synthetic evidence.
5. Extend existing trainers/Pipeline with exact steps, checkpoint/resume,
   deterministic item retrieval and thin incremental orchestration/configs.
6. After all synthetic E0 checks: single-seed reduced 2014 Beauty pipeline;
   inspect old-model learning and measured artifact manifests.
7. After E0 evidence: E1 frozen-model mapping combinations and update-LR
   sweep; raw-content first, then graph-informed with feature/parameter controls.

## Unresolved implementation choices
Text-encoder revision/local files, actual feasible budgets and precision,
suffix capacity before training, and model-input handling of unavailable
prequential histories will be explicitly implemented before their first run.

Next action: apply Round 1 with git apply --check, run the 19 CPU checks,
return the complete log and both JSON artifacts.
