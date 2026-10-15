# Round 8: frozen Base SID snapshots and causal TIGER inputs

Use the user's completed 5000-step checkpoint:
ckpt/incremental/Beauty/base_rqvae_5000_20261009T082337Z/last.pth
SHA256: b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d

No RQ-VAE training, graph processing or content re-encoding occurs in this
round. The repository's existing RQ-VAE and EncoderDecoderTokenizerMixin are
reused. The existing TIGER recommendation model will consume this adapter in
the next training handoff; it is not trained by this preparation command.

## Apply and validate

From /home/hao/frec/graph-informed-sids-web after transferring the ZIP:

```bash
set -o pipefail
unzip round8_frozen_sids_bundle.zip -d round8_handoff
git apply --check round8_handoff/round8.patch
git apply round8_handoff/round8.patch
OMP_NUM_THREADS=1 python -B tests/incremental/test_temporal_sids_round8.py \
  2>&1 | tee round8_tests.log
```

Expected: 11 synthetic-only CPU tests and OK. Tests train a tiny actual
repository RQ-VAE, then verify frozen export, persistent suffix slots, causal
flags, catalog boundaries, masking, prefix constraints, deterministic outputs,
checksums and no overwrite. No PyG, datasets, TensorBoard, SentenceTransformer
or model download is imported by this entry point. Existing Torch, NumPy,
scikit-learn and tqdm are sufficient. Do not change the server environment.

Patch modifies only genrec/tokenizer.py from Round 2. It postpones type-only
dataset imports and derives the encoder attention mask from non-padding IDs
inside the active sequence. Known-item tokenization retains its prior output.
Other code is new: two adapters, one CLI and one test file. If patch check
fails, retain its error; do not overwrite complete files blindly.

## Prepare real Beauty artifacts

```bash
SID_RUN_ID="base_sids_$(date -u +%Y%m%dT%H%M%SZ)"
SID_DIR="cache/incremental/Beauty/$SID_RUN_ID"

OMP_NUM_THREADS=4 python -u -B prepare_incremental_sids.py \
  --protocol-dir cache/incremental/Beauty/beauty_20261007T104452Z \
  --features-dir cache/incremental/Beauty/features_20261008T135919Z \
  --rqvae-dir ckpt/incremental/Beauty/base_rqvae_5000_20261009T082337Z \
  --output "$SID_DIR" \
  --device cuda \
  --suffix-capacity 256 \
  --expected-checkpoint-sha256 b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d \
  2>&1 | tee "${SID_RUN_ID}.log"
```

Frozen Base assignments are recomputed using the original export batch size
and must equal all 10924 original three-code assignments. Data, package,
source and checkpoint identities are checked against Round 7 bindings.
CUDA is used explicitly to retain the original execution device; a different
device can fail the assignment check due to floating point differences.
The model stays in eval/no-grad mode; initialized codebooks must remain so.

## Fixed token vocabulary and collision suffix

Three semantic codebooks remain [256,256,256]. Append a separate zero-based
collision suffix with fixed capacity 256. All items, including singleton
semantic codes, have four digits. Stable temporal registry ID order determines
allocation within each three-code bucket; suffixes start at zero and are
persisted in allocation_registry.json. Appending items never renumbers the
old items, and capacity overflow fails rather than expanding the vocabulary.

| Token interval | Meaning |
| --- | --- |
| 0 | Padding and T5 decoder start |
| 1–256 | First semantic digit |
| 257–512 | Second semantic digit |
| 513–768 | Third semantic digit |
| 769–1024 | Collision suffix |
| 1025 | EOS |

Vocabulary size is 1026; semantic length is three, full item-token length is
four. With history length 50, encoder length is 201 and labels have length 5.
No user tokens are added, allowing new users without a fitted user-ID table.
The fixed vocabulary is intended to remain unchanged in subsequent Update
experiments. Changing the RQ-VAE code assignment is a distinct versioned SID
operation; this append-only allocator rejects in-place semantic code changes.

## Catalog snapshots and validation denominators

- base_train: 10924 items, eligible for Base training AND Base validation.
- base: 11179 items, deployable after the Base validation window ends. Its
  extra 255 items are encoded with the frozen RQ-VAE and appended to the suffix
  registry, preserving every old full SID. This map is not a Base-validation
  candidate set.

The fixed-content update_train matrix is only a storage source for the Base
deployment prefix. Prefix identities and bytewise equality of existing Base
features are checked; only the 11179 Base-end items receive assignments.
No Update-only or Future-only item is assigned, and no model is refit.

| Split | Raw causal examples | Labeled tensor rows | Targets absent from Base training catalog |
| --- | ---: | ---: | ---: |
| Base training | 76252 | 76252 | 0 |
| Base validation | 9192 | 8689 | 503 |

Every raw example has an audit row. The 503 unavailable targets have no target
label and therefore no tensor row. Later ranking metrics must report both
the 8689 eligible-target conditional denominator and the complete 9192
denominator, counting unavailable targets as misses in the latter. They are
not silently dropped from experiment reporting.

Unavailable historical items remain in their raw temporal records. Their
four encoder token positions are padding IDs with attention zero; EOS always
remains active, including histories with no available items. The summary
reports unavailable-history counts and fully masked histories. No future
history is used. Repeated item interactions are retained, and candidate
decoding must not exclude seen items merely because they appear in history.

## Outputs and next training boundary

- base_train/ and base/: allocation_registry.json, item2sem_ids.json,
  item2tokens.json, registry.json. Full SIDs must be unique in each snapshot.
- tokenizer_contract.json: vocabulary, lengths, catalog/time references,
  padding and label-availability policy.
- tiger/base_train.npz and tiger/base_val.npz: input_ids, attention_mask,
  labels, target_global_item_ids, source_rows. Numeric arrays only; load with
  allow_pickle=False. Input IDs/labels are int32; the training loader must cast
  them to torch.long. Masks are uint8; source_rows point to original JSONL rows.
- tiger/*_audit.jsonl: sample identity and target/history availability for all
  raw rows, with tensor_row null for unavailable targets.
- summary.json: source bindings, payload hashes and counts. Only a successful
  atomic directory commit has the completed summary at the final output path.

FrozenSIDTokenizer exposes allowed_next_tokens(prefix) for later catalog-
constrained TIGER generation. Only eligible full SIDs and their EOS tokens
may be returned. Do not point legacy main_genrec.py at these artifacts yet:
its default pipeline would rebuild datasets/tokenizers and does not consume
this temporal tensor/audit contract. The next handoff connects the existing
TIGER model to these tensors, fixed-step training and the two denominators.

Return the 11-test result and the real preparation summary.json. Raw data,
embedding matrices and checkpoints need not be uploaded.
