# Round 6: pure-content SID features

Prerequisite: passing Round 1–3 temporal data path. Graph rounds 4/5 are optional
and are not imported. Apply round6.patch against user-uploaded prepare_dataset.py.
New code: genrec/temporal_content.py, prepare_incremental_content.py, and 12
synthetic tests. Existing encode_sent_emb is reused with optional revision and
lazy heavy imports. Its CLI imports remain in the __main__ entry point.

## Execute

1. git apply --check round6_handoff/round6.patch
2. git apply round6_handoff/round6.patch
3. python -B tests/incremental/test_temporal_round6.py --output NEW_RESULTS_DIR
4. python -B prepare_incremental_content.py metadata --metadata META_BEAUTY_GZIP
   --protocol-dir ROUND3_BEAUTY_DIR --output NEW_CONTENT_DIR
5. python -B prepare_incremental_content.py encode --content-dir NEW_CONTENT_DIR
   --output NEW_FEATURE_DIR --device cuda --batch-size 32

Expected synthetic checks: 12. They use a clearly labelled fake encoder and do
not establish real model/Beauty success. numpy and pandas are required; no model
or graph kernels are needed for these tests. Real encode requires the existing
sentence-transformers/transformers/torch environment and access to model files.

Default model: sentence-transformers/sentence-t5-base; expected dim 768;
max input length 256 encoder tokens; batch size 32; float32 saved output. Resolve
the selected Hub branch/tag to a full commit and load that commit. A full
--model-revision COMMIT avoids an API resolution call. --local-files-only needs
an explicit full Hub commit, or --model /absolute/local/model/directory whose
file tree hashes identify it. No silent substitute encoder. Use --device cpu
if needed; GPU execution is not mandatory for this preprocessing stage.

## Protocol and payloads

metadata uses only title, brand, categories, description. Flatten string/list
fields, remove HTML tags, decode entities, collapse whitespace, retain field
labels and delimit lists with semicolons. No raw ASIN or review text is inserted
into the encoder text. All behavioural/other fields are ignored. Conflicting
allowed-field metadata duplicates fail. Behaviour-only differences do not.
Absent records or empty text fail by default and report counts. Optional
--missing-policy placeholder explicitly fills them with the same generic
sentence and marks/counts these rows. Do not enable this without reporting it.

Static product metadata has no historical publication guarantee: disclose an
offline approximation. Scan metadata but retain only items observed by Update
end. No future-only item is encoded. A fixed pretrained encoder is evaluated
once on the Update deployment catalog; it is not fit or fine-tuned on that
catalog. Reuse identical prefix rows for earlier training catalogs. No added
PCA/projection/normalization; preserve the selected encoder's own output.

Stage directories:
- base_train: 10924 items expected for the user's current protocol.
- update_train: 11707 items expected; includes old and newly observed items.
- update: 11775 deployment items expected, including 68 first observed during
  Update validation. This directory must NOT become an RQ-VAE training view.

Metadata payload per stage: content.jsonl and registry.json.
Feature payload per stage: embeddings.npy, feature_rows.json, registry.json.
GR ID 0 is padding; no padding embedding row is written. Feature row equals
zero-based gensid global item ID for these append-only prefix snapshots, and
all mappings are also exported explicitly. Base features are bitwise identical
to the shared prefixes of the Update views. Summary includes shapes/norms,
metadata/data/model identities and payload checksums. Existing outputs fail;
partial outputs remain .incomplete. A legacy raw .sent_emb intermediate is
converted to explicit .npy stage views and is not retained downstream.

## Evidence and next boundary

Local 12 fake-encoder tests passed. Patch and syntax validation passed. No real
metadata/encoder/model was run locally. RQ-VAE/TIGER training and full E0/E1
remain not_run. Next stage trains old RQ-VAE using base_train features, then
exports fixed-capacity SID mappings and trains the existing TIGER.

Sources for encoder interface and dimensions:
https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
https://huggingface.co/sentence-transformers/sentence-t5-base
