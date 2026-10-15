# Round 2: explicit causal examples

Base snapshot: 5c2ca487f961e91e75e7d8f3ca78349faf3dc82e.
Requires the user-verified Round 1 temporal.py.

Changed code: genrec/dataset.py, genrec/tokenizer.py.
New test: tests/incremental/test_temporal_round2.py (8 synthetic checks).

prepare_temporal_stage accepts original four-column review tuples, returns
existing HF Dataset objects, and keeps diagnostic metadata separately on
self.temporal_metadata. It is an explicit API, not an automatic replacement
of the legacy dataset loader. Do not invoke legacy cached mappings after
preparing temporal data. Model-ready catalogs, SID mapping, graph training,
real Beauty preprocessing and the incremental CLI are not delivered yet.

Use explicit_causal_examples=True and n_user_tokens=0. The encoder-decoder
mixin then encodes each record once; legacy prefix expansion remains the
default. Attention masks now follow actual encoded length for both modes.
This also applies to TIGER through its existing mixin inheritance; no TIGER
file modification is needed. Decoder-only tokenizers are outside this round.

Apply round2.patch using git apply --check before git apply. Run:
python -B tests/incremental/test_temporal_round2.py --output NEW_OUTPUT_DIR
Return terminal output and checks.json. No training command is ready yet.
