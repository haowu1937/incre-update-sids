# Round 4: observed training-window graph inputs

Prerequisite: Round 1–3 code, passing Round 3 checks and completed raw-review
preparation. This bundle adds three files and does not modify legacy code.

Apply `git apply --check round4_handoff/round4.patch`, then
`git apply round4_handoff/round4.patch` from the project root.
These commands work outside a Git repository. Do not use `git diff --check`
unless the directory is a Git repository.

Run `tests/incremental/test_temporal_round4.py --output NEW_RESULTS_DIR`.
Expected: 8 synthetic checks. Then run `prepare_incremental_graph_inputs.py`
with `--reviews`, `--protocol-dir` (Round 3 directory), and a new `--output`.

## Outputs and meaning

Each base/update directory contains:
- observed_events.jsonl: every retained raw event in the training window,
  with raw identity, timestamp, stable GR/global IDs and explicit local IDs.
- user_sequences.jsonl: those events grouped by user, preserving timestamps.
- item_nodes.json: all eligible training-snapshot catalog items in explicit
  local order; flags indicate which have a training-window interaction.
- registry.json: the corresponding stable training-snapshot registry.

Base uses timestamps strictly before base_train cutoff. Update uses the
update-training window only: base cutoff inclusive to update_train cutoff
exclusive. Base-validation, update-validation, and Future events are excluded
from graph interactions. Update catalog membership includes previously
observed items, including items first seen in Base validation. This is catalog
eligibility, not use of validation targets as graph edges or training labels.

Singleton and first events remain present. Sequence timestamps retain ties;
the deterministic order inside a timestamp group does not assert causality.
No edges or tie-edge policy are implemented here. A node with no window
interaction is flagged explicitly; actual graph degree is not yet measured.
Downstream builder must use skip_n_last=0 and explicit maps, not subtract a
held-out suffix again or assume global IDs equal local tensor rows.

Round 3 source data and all seven hashed artifacts are verified. Its protocol
and registries are reproduced from raw reviews before export. Output must be
new; interrupted writes remain in a .incomplete sibling for inspection.

No metadata, embeddings, APPNP, graph losses, codebooks, or GR model are run.
This is an input contract for the existing gensid path, not a new graph stack.
Dataset provenance remains the declared provenance from Round 3.
