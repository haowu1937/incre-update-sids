# Pure-content-first plan

User chose fixed content embedding -> existing RQ-VAE -> existing TIGER.
Graph-informed SIDs, LETTER and GRIS are later comparisons.

1. Round 6 fixed Sentence-T5 content features: server completion confirmed;
   encoder commit fc5d4628481afbbaaacd7af6bb07cf9d3865f781, 768 dimensions.
2. Round 7: train Base RQ-VAE on the 10924 Base training items only, export
   three raw semantic code indices, record collisions and checkpoint state.
3. Next: persistent collision suffix and frozen Base evaluation SID mapping;
   train the existing TIGER on Base explicit causal examples.
4. E1: Update RQ-VAE, freeze old TIGER and old target SIDs, compare old/new
   input-history encodings on user-disjoint historical probes.
5. Historical preservation loss and adaptation comparisons under matched
   content/tokenizer/capacity/budget controls; later alternative SID versions.

Round 7 modifies three existing source files (lazy imports, quantizer state,
Trainer fixed-step support), adds two entry/helper files and one test file.
No existing content/data/graph files are overwritten. Checkpoint selection
is fixed-budget last, and continuation binds artifacts/config/source.
Full E0 and E1 remain not_run until the recommendation and historical
comparison stages actually finish. Full-period 5-core selection bias and
static-metadata approximation still apply.
