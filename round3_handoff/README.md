# Round 3: raw Beauty reviews to temporal artifacts

Prerequisites: Round 1 and Round 2 applied and user-verified.
New files only: genrec/temporal_artifacts.py, prepare_incremental_data.py,
tests/incremental/test_temporal_round3.py.

Reuses AbstractDataset.prepare_temporal_stage and genrec.temporal. No model,
metadata encoder, graph or new recommendation framework is added. The gzip
reader preserves numeric ratings without coercing them to integer.

Run five synthetic tests before real data. Then locate the existing
reviews_Beauty_5.json.gz and run prepare_incremental_data.py --reviews PATH
--output NEW_DIRECTORY. --max-history defaults to 50. --source-url may record
the known download origin; identity is declared, not independently verified.
No downloads or automatic package installations occur.

Output: protocol_manifest.json, registries.json, five stage JSONL files,
summary.json with hashes, user cohorts, temporal/catalog statistics and costs.
.jsonl files retain raw IDs plus target/history availability flags. They are
not yet training-ready SID tensors. A successful preparation is not E0 full
completion or a research result. Metadata and graph construction remain next.

No processed legacy cache is read or changed. Existing/incomplete outputs
are rejected. Failed writes remain under OUTPUT.incomplete for inspection.
The whole 5-core release is used: future selection bias from full-period
5-core filtering is recorded. Users need not appear in every time window.
First events without earlier history are excluded and counted.

No Git initialization is required. Apply patches from the project root with
git apply --check then git apply; these commands work outside a Git worktree.
