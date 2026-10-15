# Round 7: Base content RQ-VAE

This patch uses the repository's RQVAE, ResidualVectorQuantizer,
VectorQuantizer, MLPLayers, kmeans helper and Trainer optimizer factory.
It adds a fixed-step pure-content method to Trainer and an artifact-aware
entry point. No graph round is required. It does not rerun content encoding.

## Apply and check

Run from /home/hao/frec/graph-informed-sids-web after transferring the ZIP:

```bash
set -o pipefail
unzip round7_base_rqvae_bundle.zip -d round7_handoff
git apply --check round7_handoff/round7.patch
git apply round7_handoff/round7.patch
python -B tests/incremental/test_rqvae_round7.py 2>&1 | tee round7_tests.log
```

Expected: 9 synthetic-only CPU tests, passed. Existing torch, numpy,
scikit-learn and tqdm are used; no PyG, TensorBoard, model download or
k-means-constrained is imported by this entry point. No environment changes
are requested. If patch check fails, retain its error; do not overwrite the
complete files blindly. Complete files are included for inspection.

## Locate the completed feature run

If FEATURE_RUN_ID still identifies the successful Round 6 run, use:

```bash
FEATURE_DIR="cache/incremental/Beauty/$FEATURE_RUN_ID"
```

Otherwise this command selects the unique completed feature run with the
protocol and Base embedding SHA returned in your log. It ignores partial runs
and fails rather than guessing if more than one run matches:

```bash
FEATURE_DIR="$(python -B - <<'PY'
import json
from pathlib import Path
matches = []
for path in Path('cache/incremental/Beauty').glob('features_*/summary.json'):
    report = json.loads(path.read_text())
    if (report.get('stage') == 'E0_fixed_content_embeddings'
        and report.get('execution_status') == 'completed'
        and report.get('protocol_sha256') == '97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66'
        and report.get('artifact_files_sha256', {}).get('base_train/embeddings.npy')
            == '305dc6a7a73feb2ffd39077f2609413fb38987709ac34c41ba4745e59a91f09d'):
        matches.append(path.parent)
if len(matches) != 1:
    raise SystemExit('Expected one matching completed feature run; choose --features-dir explicitly: ' + str(matches))
print(matches[0])
PY
)" || exit 1
```

The training entry point independently verifies all three Base payload hashes,
float32 finite values, shape and raw/GR/global/row ID round trips. It never
opens Update matrices. Expected real training shape: [10924, 768].

## Smoke run, then continue to the first training budget

```bash
SMOKE_RUN_ID="base_rqvae_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
SMOKE_DIR="ckpt/incremental/Beauty/$SMOKE_RUN_ID"
OMP_NUM_THREADS=4 python -B train_incremental_rqvae.py \
  --features-dir "$FEATURE_DIR" \
  --output "$SMOKE_DIR" \
  --device cuda --max-steps 50 --batch-size 256 \
  --log-every 10 --save-every 10 \
  2>&1 | tee "${SMOKE_RUN_ID}.log"
```

After the smoke run exits successfully, continue:

```bash
RQVAE_RUN_ID="base_rqvae_$(date -u +%Y%m%dT%H%M%SZ)"
RQVAE_DIR="ckpt/incremental/Beauty/$RQVAE_RUN_ID"
OMP_NUM_THREADS=4 python -B train_incremental_rqvae.py \
  --features-dir "$FEATURE_DIR" \
  --output "$RQVAE_DIR" \
  --resume "$SMOKE_DIR/last.pth" \
  --device cuda --max-steps 1000 --batch-size 256 \
  2>&1 | tee "${RQVAE_RUN_ID}.log"
```

max-steps is the TOTAL optimizer-step budget, including resumed steps. The
second command runs 950 additional steps. These are initial diagnostic
budgets, not a convergence claim. Do not execute the second command if the
first failed. Existing output directories fail. Logs expose initialization,
initial Base-catalog metrics and periodic step/epoch/loss counts. Codebook
kmeans initialization uses CPU scikit-learn even with --device cuda; the
training model and quantizers use the selected device.

Default model: 768 -> 512 -> 256 -> 128 -> 64 -> 32 encoder; reversed decoder;
three 256-word codebooks, dropout=0, no batch norm, MSE reconstruction,
quantization weight=0.25. Quantizer commitment coefficient remains the
repository's mu=0.25. AdamW lr=0.001, weight decay=0, batch=256, seed=2026.
Initialize from a seeded sample of up to 4096 Base training items, three
sequential residual kmeans initializations, max_iter=50 using the existing
layers.kmeans function. This is an explicitly recorded initialization choice;
the legacy constrained initializer remains available to older entry points.
No Sinkhorn, CF/graph loss, PCA or added input normalization is used.

## Outputs and resume contract

- last.pth: model, optimizer, Base data/source/package bindings, sampler order,
  cursor, CPU sampler generator and Python/NumPy/Torch/CUDA RNG state.
- metrics.jsonl: one row per step performed in this run directory.
- semantic_assignments.jsonl: 10924 real Base item rows, each with raw ASIN,
  GR ID, zero-based gensid global ID, feature row and three zero-based codes.
- config.json and summary.json: identities, budgets, Base reconstruction and
  quantization metrics, codebook utilization, collisions and payload hashes.

Checkpoints are selected at the last fixed budget step; no validation data or
collision-best selection is used. Initial/final metrics are diagnostics on
the Base training catalog, not held-out quality. Checkpoint-reloaded
assignments must match before a completed summary is written. A failed or
interrupted run has no completed summary; the last atomic checkpoint can be
resumed into a NEW output directory. Only load trusted checkpoints created
by this entry point, since optimizer/RNG restoration uses full serialization.

Resume enforces identical feature-summary and Base payload hashes, source,
Torch/NumPy/scikit-learn versions, model, optimizer, initialization settings,
seed, batch size and device string. Budget/log/save intervals can change.
The synthetic resume check establishes exact continuation on the checked CPU
environment. Real V100 execution still needs the server run; no cross-device
or cross-version numerical identity is claimed.

Raw semantic collisions are intentionally retained and reported. These
three-code assignments are not yet a complete TIGER item-tokenizer artifact.
Next round will establish a persistent collision-suffix registry and map the
eligible Base evaluation items with the frozen RQ-VAE before TIGER training.
Do not create an ASIN-based token or variable suffix vocabulary manually.

Return the 9-test result and the smoke/final summary.json (or full error log).
Checkpoints, raw review files and embeddings need not be uploaded.
