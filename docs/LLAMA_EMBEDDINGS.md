# Beauty temporal Llama embeddings

Status: code provided; real Beauty/Llama GPU inference has **not** been run by
the developer. Hao runs it on his Linux GPU server. No RQ-VAE, SID or recommender
training is included in this change. Existing T5 features remain unchanged.

## Definition and scope

`prepare_incremental_llama.py` follows the public Mitigating notebook:
<https://github.com/iskbaga/semantic-id-alignment/blob/main/notebooks/AmazonBeautyDownload.ipynb>.

- Local Hugging Face Llama weights only, using `LlamaModel` and slow
  `LlamaTokenizer`. No Hub access or pretrained downloads.
- Text: `Title: {title}. Categories: {first category path joined by comma}. Description: {description}.`
  List-valued descriptions use their Python string representation, as in the
  notebook. Brand, reviews, ratings, related products and popularity are excluded.
  The previous T5 content text is **not** reused: it includes a different field
  format and brand. The original metadata file is reread and checksum-verified.
- Fixed padding/truncation to 1,024 tokens, right padding, tokenizer's saved BOS
  and EOS settings, with EOS used as PAD if PAD is absent.
- Last hidden layer, padding vectors zeroed, sum divided by **1,024**, not by the
  number of nonpadding tokens. No L2 normalization and no projection.
- Model inference defaults to FP16 and PyTorch SDPA, with float32 pooling and
  float32 `.npy` output. These are explicit engineering choices for V100 memory
  and efficiency; they do not promise bitwise reproduction of the notebook's
  default precision/attention. `--attention eager` and `--dtype float32` are
  available, but require a fresh run output when changed.
- Empty/absent category paths become `None`. Missing eligible metadata or
  malformed category types fail explicitly. Missing metadata is not silently
  replaced with a generic product vector.

The local directory name alone cannot establish that weights are exactly the
author's `huggyllama/llama-7b`. Config, tokenizer files and selected safetensor
shards are hashed; `published_checkpoint_identity_verified` stays false.

## Temporal/data contracts

The user's existing protocol is reused; the author's ten-block split is not.
Reviews are read only for checksum verification, not for text features or a new
split. The completed protocol and content manifests are verified before output
creation or model loading. The prior content directory locates the metadata
file and proves its identity/row maps. `--metadata` can override its path, but
the original recorded checksum must still match.

For the Beauty protocol previously reported by Hao:

| Feature directory | Expected items | Use |
|---|---:|---|
| `base_train` | 10,924 | Base training/validation candidate catalog |
| `update_train` | 11,707 | Update training/validation candidate catalog |
| `update` | 11,775 | Deployment catalog for Future evaluation |

Actual counts are read from protocol registries, not hardcoded. There are no
separate item embeddings per interaction split: the split uses its eligible
catalog. Future-only items are excluded. Encoding is item-independent, with a
frozen encoder and static metadata. Each eligible item is encoded once, then
identical prefix rows are exported for earlier catalogs. This does not fit an
encoder or normalization statistics using future items. Static metadata still
has the existing historical-publication-date limitation.

Each stage contains `embeddings.npy`, `feature_rows.json`, and `registry.json`.
Rows preserve GR ID 0 as PAD; feature row/global gensid ID = GR ID - 1.
Additional files are `item_texts.jsonl`, `shared_embeddings.npy`,
`encoding_plan.json`, `progress.json`, `summary.json`, and `compact.txt`.
The summary uses the existing `E0_fixed_content_embeddings` artifact contract;
this compatibility does not mean existing trained checkpoints can be reused.

## Run on Hao's server

From `/home/hao/frec/graph-informed-sids-web`, after checking out the delivered
Git branch/commit:

```bash
conda activate frec-gpu
set -o pipefail

# No changes to requirements.txt or automatic dependency upgrades.
python -B -c 'import torch, transformers, sentencepiece, safetensors, numpy; print("torch", torch.__version__, "transformers", transformers.__version__, "numpy", numpy.__version__, "CUDA", torch.cuda.is_available())'
python -B tests/incremental/test_llama_content.py

LLAMA_RUN_ID="llama7b_features_$(date -u +%Y%m%dT%H%M%SZ)"
LLAMA_FEATURE_DIR="cache/incremental/Beauty/$LLAMA_RUN_ID"

python -u -B prepare_incremental_llama.py \
  --protocol-dir cache/incremental/Beauty/beauty_20261007T104452Z \
  --content-dir cache/incremental/Beauty/content_20261008T123254Z \
  --reviews cache/AmazonReviews2014/Beauty/raw/reviews_Beauty_5.json.gz \
  --model-dir /home/shared/huggingface/llama-7b \
  --output "$LLAMA_FEATURE_DIR" \
  --device cuda:0 \
  --dtype float16 \
  --batch-size 1 \
  --max-length 1024 \
  2>&1 | tee "$LLAMA_RUN_ID.log"

cat "$LLAMA_FEATURE_DIR/compact.txt"
```

Batch size 1 is the conservative starting configuration; actual V100 memory and
runtime remain unmeasured. GPU memory statistics are recorded by the process.
The initial checksum scan reads all local model shards and can take time; it
prints each filename before hashing. Progress reports items/s and estimated
remaining encoding time after the model has loaded.

In the developer sandbox, the synthetic tests exercise pooling, input identity,
stage exports, manifest hashes, future exclusion, interruption/resume and
no-overwrite. The real backend test builds a tiny random Llama locally on CPU,
with no weights downloaded; it is skipped when torch/transformers are absent.
Passing that test does not establish full-size Llama/V100 correctness or speed.

Delivery checks: 10 new synthetic checks passed; the tiny real-backend check was
skipped because torch/transformers are unavailable in the developer sandbox.
The existing Round 6 content suite passed all 12 checks. The broader Round 3
suite could not import its dataset stack because `datasets` is unavailable;
no Round 3 pass is claimed. CLI help and whitespace checks also passed.

## Interruptions

Each successfully encoded batch is flushed and an atomic row cursor is saved
inside `OUTPUT.incomplete`. Completed outputs are never overwritten.

After interruption, rerun the **same encode command** with `--resume` added,
using exactly the same output path, device, batch size, length, precision, model
files, source files and environment. Resume verifies those bindings and the
saved rows before continuing. If opening a new shell, recover the original run
ID from its log; do not generate a new timestamp when resuming.

If CUDA OOM occurs and batch size is already 1, retain the failure log and
report it. Do not silently reduce sequence length: that changes the feature
definition. Config/model/code changes require a new output directory. No need
to delete failed or old outputs.

Return only `compact.txt` and any error traceback initially. Success means
completed status, expected stage counts/dimension, shared rows equal, finite
nonzero embeddings and valid artifact hashes. It makes no recommendation
quality or historical-damage improvement claim.
