# Pure-content SID baseline progression

1. Fixed Sentence-T5 content features: completed on the server.
2. Base RQ-VAE: server 1000-step run completed, then continued to 5000 steps
   after training-catalog diagnostics. Freeze the 5000-step model for the first
   recommendation baseline. No held-out ranking result selected this model.
3. Round 8: persisted collision suffix, Base-training and Base-deployment
   snapshots, explicit causal TIGER tensors and complete availability audits.
4. Next: existing TIGER training/evaluation with frozen SIDs and catalog-
   constrained generation; report eligible-target and full denominators.
5. E1: update SID codebooks, freeze old TIGER/old target SIDs, measure prediction
   changes from old/new history encodings on disjoint historical probes.
6. Historical prediction penalty, matched adaptation controls, then optional
   graph-informed, LETTER and GRIS variants.

Round 8 scope is four new source/test files and one existing tokenizer change.
No Round 7 RQ-VAE source changes, new model implementation, retraining,
metadata processing or recommendation quality claim is included. The first
three semantic digits are distinct from the fourth collision suffix digit.
Old full SIDs are immutable during ordinary catalog extension. Base-validation
targets outside the Base training catalog stay unavailable, even though a
later Base-deployment SID snapshot can represent them.
