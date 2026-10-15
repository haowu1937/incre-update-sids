# Round 5: stage-specific graph construction

Adds genrec/temporal_graphs.py, prepare_incremental_graphs.py and 8 synthetic
checks; patches gensid/graph_aware_sid/graph_utils.py against user round5_source.zip.
No changes to datasets_module.py or prepare_dataset.py in this round.

Prerequisites: Round 1–4 code and completed Round 4 graph-input artifacts.
Apply round5.patch with git apply --check, then git apply. A Git repository is
not required. If --check fails, do not overwrite files with complete_files.

Run tests/incremental/test_temporal_round5.py --output NEW_RESULT_DIRECTORY.
Requires the existing numpy/pandas/scipy/polars/torch/torch-geometric environment;
torch-cluster is now imported only for the optional random-walk path.
Then run prepare_incremental_graphs.py --graph-inputs-dir ROUND4_OUTPUT
--output NEW_GRAPH_DIRECTORY. Base and Update graphs are saved independently.

## Protocol

Use only each stage's raw observed training-window events, with no further
suffix truncation. Preserve full eligible catalog, stable global identities,
explicit local order and isolated nodes. Equal-time events have no invented
sequence order: adjacent distinct per-user timestamp groups produce a directed
Cartesian product of their events; no pairs inside one timestamp group. Repeat
event multiplicity contributes to counts. No-time-tie data reduces to ordinary
adjacent co-occurrence. A chronological repeat of the same item produces a
self-pair, as in the original adjacent path. Converting to undirected sums both
directions; the original converter also doubles a self-pair. This is recorded
explicitly and must remain fixed across compared methods.

Graph dump contains meta.json, graph_df.parquet, node_map.parquet, pyg_data.pt,
and graph_manifest.json. The manifest binds protocol, input event hash, catalog
hash and global order to checksummed payloads. load_verified_graph checks all
payload hashes before restoring graph state. Export checks exact restored
edge_index/edge_weight/x equality. Outputs must be new; partial writes remain
in a .incomplete sibling. The top-level summary measures actual isolated graph
nodes separately from catalog items with no window interaction.

Legacy behavior is preserved with the default strict_timestamp_groups=False.
The shared skip_n_last=0 bug is corrected; None and positive truncation remain
supported. Supplying a prefiltered dataframe rejects further truncation.

## Boundary

The PyG x field is the existing degree placeholder, NOT content embeddings.
No metadata, sentence encoder, APPNP, fused-feature cache, RQVAE or TIGER runs.
The existing APPNPEmbDataset identity/cache issues remain to be patched before
its incremental use; do not feed these files directly to the legacy loader.

## Local validation

Patch/compile checks and extracted real mask function tests passed. Full 8
checks were not run locally: torch/polars/PyG are unavailable in the sandbox.
Server verification and real graph statistics await user evidence.
