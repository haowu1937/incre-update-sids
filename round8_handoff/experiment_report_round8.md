# Evidence and interpretation

User supplied two completed real Beauty RQ-VAE summaries:

| Metric | 1000 steps | 5000 steps |
| --- | ---: | ---: |
| Training-catalog reconstruction MSE | 0.0001423425414480207 | 0.00010876538503473197 |
| Reduction relative to mean-embedding MSE | 38.539% | 53.037% |
| Total training-catalog loss | 0.00015839561142255178 | 0.00015930112012508876 |
| Quantization loss | 0.00006421228075333581 | 0.00020214293888545483 |
| Unique three-code assignments | 6287 | 10090 |
| Codebooks used out of 256 | 10 / 127 / 62 | 38 / 224 / 192 |
| Items in semantic collision groups | 6910 | 1459 |
| Largest semantic collision group | 37 | 26 |

Mean-embedding baseline MSE on the same Base training matrix:
0.00023159830016084015. The first codebook remains underused (14.844%);
the second and third use 87.5% and 75%. Reconstruction and assignment
distinctiveness improved, but total loss did not decline and quantization
loss increased. These are training-catalog diagnostics, not recommendation
quality or convergence evidence. The 5000-step model is a practical first
downstream baseline; full utilization is not imposed as a quality guarantee.

Selected checkpoint SHA256:
b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d.
Both server runs reported checkpoint-reloaded assignment equality. Real
5000-step continuation runtime: 107.4858 seconds for 4000 added steps.

Local Round 8 tests use synthetic temporal interactions, synthetic fixed
features and the actual repository RQ-VAE. They do not use real Beauty data,
run a content encoder or train TIGER. Logs/local_checks.json record the actual
test, patch and CLI results. Unknown history masking is an explicit serving
projection while retaining the original raw causal/audit records.

Server SID preparation: code_provided, awaiting execution.
TIGER training/evaluation: not_run. E0_full_status: not_run. E1_status: not_run.
Static metadata timing approximation and full-period 5-core filtering bias
remain disclosed limitations of the original experiment inputs.
