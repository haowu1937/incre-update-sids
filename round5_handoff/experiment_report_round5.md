# Round 5 evidence

User Round 3: synthetic tests passed; Beauty preparation completed.
User Round 4: real graph input preparation completed, base 107108 events/10924
nodes; update 35689 events/11707 nodes, 2440 nodes without window interactions.
Round 4 user synthetic result was not supplied in this thread.

Round 5 local: syntax and extracted skip_n_last mask contracts passed.
Full local graph integration tests: not_run (missing torch/polars/PyG).
User Round 5: code_provided. Real graph measurements: not_run.
APPNP/content features/model training/full E0/E1: not_run.

Next: inspect measured graph isolation/timestamp ties, then implement fixed
content features and explicit APPNP dataset/global-to-node identity and cache
validation. Reuse uploaded prepare_dataset.encode_sent_emb; pin the content
encoder revision and exclude behavioural metadata fields.
