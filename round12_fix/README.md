# Round12 suffix capacity fix

This patch is installed and run manually on your Linux server. Local validation used synthetic data and CPU only, not your Beauty dataset or V100.

## Failure and experimental change

The original training hook required a complete unique SID allocation at every intermediate optimizer step. The fixed suffix vocabulary has only 256 slots per semantic prefix, and all old identifiers remain reserved. A temporary collision concentration or a prefix with too many reserved-plus-appended slots can exhaust this capacity after even one ordinary RQ-VAE step. This is an allocation failure, not evidence of failed CUDA initialization or a completed compatibility experiment.

The new default compatibility training condition is explicitly `new_semantics_old_suffix`. It replaces the three semantic tokens with current hard assignments and preserves each historical item's original suffix token. Labels and masks stay old, and the teacher stays frozen. The code still uses a soft-distance surrogate gradient for semantic assignments. This condition was already measured as a diagnostic in E1 and showed positive historical damage. It is an input-only condition and is NOT asserted to be a unique deployment SID mapping. This is a deliberate change to the training objective, not a silent allocator fallback.

The original condition remains selectable with `--history-input-condition new_full_sid`, but retains its strict capacity requirement and can still fail during transient collapse. The recommended commands below select the fixed old-suffix condition consistently in both runs.

Every compatibility step audits the whole eligible catalog, logging both:

- `maximum_current_collision_bucket`: number of current items sharing a semantic prefix;
- `maximum_required_suffix_capacity`: largest required slot width after old reservations and appended movers/new items;
- `full_sid_allocation_feasible`: feasibility under the unchanged Round11 allocation policy.

No vocabulary is enlarged, no suffix is wrapped modulo 256, no collision is hidden as a unique full SID, and no old E1/teacher/SID source file is changed.

## End-of-run capacity status

The fixed step-budget checkpoint and semantic assignments are saved even if the end-of-run unique full SID allocation is infeasible. Summary `execution_status=completed` refers to training; check the separate `full_sid_export_status` and `deployment_sid_ready` fields. `capacity_exhausted` means full deployment SID export did not complete. This is useful for a 50-step smoke checkpoint: it remains resumable to 1000 steps.

Final capacity details are in `suffix_capacity_audit.json`, including overflow prefixes, current bucket sizes, reserved old slot counts, reserved widths, appended item counts and required capacities. A feasible endpoint writes the original complete SID map/ledger. An infeasible endpoint writes neither and is explicitly refused by the matched full-SID evaluation before any evaluation output is created. If the final 1000-step checkpoint is infeasible, it must not be presented as a deployment-ready or ranked result.

Formal matched evaluation still uses actual unique complete new SIDs plus the separate old-suffix diagnostic, compared to the same ordinary Update E1 at the same step budget and with the same queries, old labels and old candidate catalog. Its summary now records the training input condition as well.

## Manual installation

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip round12_suffix_capacity_fix.zip -d round12_fix
python -B round12_fix/apply_fix.py --repo .
```

Only `gensid/historical_incremental_content.py` is modified and backed up. One regression test file is added. The installer checks the original Round12 SHA before writing, and is idempotent. No dependency changes are required on your server.

Start a NEW run from the same Base checkpoint. Do not resume the failed old-condition run; its config and source binding differ. Existing failed directories and logs can be retained.

```bash
set -o pipefail
E1_DIR="results/incremental/Beauty/historical_damage_20261009T142825Z"
HIST_RUN_ID="historical_rqvae_fixed_$(date -u +%Y%m%dT%H%M%SZ)"
HIST_SMOKE_DIR="ckpt/incremental/Beauty/${HIST_RUN_ID}_smoke"
HIST_RQVAE_DIR="ckpt/incremental/Beauty/${HIST_RUN_ID}_1000"

python -u -B train_historical_incremental_rqvae.py \
  --e1-dir "$E1_DIR" --output "$HIST_SMOKE_DIR" \
  --device cuda --max-steps 50 \
  --compatibility-weight 0.001 --temperature 0.1 --anchor-batch-size 32 \
  --history-input-condition new_semantics_old_suffix \
  2>&1 | tee "${HIST_RUN_ID}_smoke.log"

cat "$HIST_SMOKE_DIR/suffix_capacity_audit.json"

python -u -B train_historical_incremental_rqvae.py \
  --e1-dir "$E1_DIR" --output "$HIST_RQVAE_DIR" \
  --device cuda --max-steps 1000 \
  --compatibility-weight 0.001 --temperature 0.1 --anchor-batch-size 32 \
  --history-input-condition new_semantics_old_suffix \
  --resume-from "$HIST_SMOKE_DIR/last.pth" \
  2>&1 | tee "${HIST_RUN_ID}_1000.log"
```

The formal run adds 950 steps to the 50-step checkpoint. Even a capacity-exhausted smoke endpoint can resume. Keep the same training condition and parameters across both runs.

If the formal summary reports `full_sid_export_status=completed`, evaluate using the existing command:

```bash
HIST_EVAL_DIR="results/incremental/Beauty/${HIST_RUN_ID}_evaluation"
python -u -B evaluate_historical_update.py \
  --e1-dir "$E1_DIR" --historical-rqvae-dir "$HIST_RQVAE_DIR" \
  --output "$HIST_EVAL_DIR" --device cuda --batch-size 128 --ranking \
  2>&1 | tee "${HIST_RUN_ID}_evaluation.log"
python -B round12_bundle/show_results.py "$HIST_EVAL_DIR"
```

## Validation

8 original CPU synthetic checks passed again, including exact lambda-zero reproduction, frozen teacher gradient checks, exact resume and matched ranking. Three new checks cover a real 300-item collision in a 256-slot bucket, overflow due to old reserved slots despite a small current bucket, old-suffix forward/gradient without calling the allocator, and end-of-smoke overflow preserving a resumable checkpoint while rejecting invalid full-SID evaluation. Python 3.10 syntax parsing also passed.

Optional manual test commands on your server:

```bash
python -B tests/incremental/test_historical_update_round12.py
python -B tests/incremental/test_suffix_capacity_fix12.py
```
