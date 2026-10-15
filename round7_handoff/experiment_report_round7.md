# Evidence boundary

Confirmed server results supplied by user:
- Beauty temporal preprocessing completed; 198502 events, no exact duplicates.
- Base training catalog: 10924 items; 76252 causal recommendation examples.
- Fixed content encoding completed: Base [10924,768], Update training
  [11707,768], Update deployment [11775,768], float32, fixed Sentence-T5.
- Content encoding full runtime reported 43.0316 seconds. Earlier network
  delay is not included in that successful run's runtime.

Local Round 7 uses real repository Torch RQVAE with SYNTHETIC data, not a
fake model. Nine CPU tests cover data hashes/ID mapping, absent graph imports,
exact optimizer-step/partial-batch accounting, collision export, reload flags,
finite gradients, reconstruction improvement, uninterrupted vs resumed
weights/optimizer/SID equality, no overwrite and resume-binding/budget rejection.

Local environment: Python 3.12, Torch 2.6.0+cpu in an isolated scratch test
dependency directory. The server's Python 3.10.18 / Torch 2.6.0+cu118 / V100
combination has not been executed locally. Test logs and local_checks.json
record actual checks; they do not establish Beauty RQ-VAE convergence.

Server Base RQ-VAE: code_provided, awaiting run.
TIGER training: not_run. E0_full_status: not_run. E1_status: not_run.
No recommendation metric, historical penalty, or alternative SID experiment
is claimed by this handoff. Raw three-code collisions are not resolved here.
