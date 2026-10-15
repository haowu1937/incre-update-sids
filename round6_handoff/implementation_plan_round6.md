# Plan revision: pure-content first

User decision 2026-10-08 supersedes earlier graph-first implementation ordering.
First validation: fixed content embedding -> existing RQ-VAE -> existing TIGER.
TIGER is chosen for reuse of the already adapted causal tokenizer path. HSTU
and LETTER/GRIS SID variants remain later experiments. Graph rounds are saved
optional work, not prerequisites and not evidence for first validation.

1. Round 6: metadata/features, server synthetic + real feature checks pending.
2. Train old RQ-VAE on Base training-catalog features with three semantic
   codebooks, fixed capacities and separately managed collision suffix.
3. Export old SID; train old TIGER on explicit Base causal examples.
4. E1: update RQ-VAE with Update data. Freeze old TIGER and old target SIDs;
   compare old/new input-history encodings on user-disjoint historical probes.
5. Add differentiable historical loss; compare historical damage and fresh
   adaptation under identical content, tokenizer settings and training budgets.
6. Later add LETTER/GRIS and other SID versions.

Keep all temporal/cold-catalog/no-refit contracts. Do not fit on validation or
Future targets. Keep replay and diagnostic users disjoint. No first-round
claim requires collaborative information to be injected into SID.
