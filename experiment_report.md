# History-aware SID Update Experiment Report

Last updated: 2026-10-20. Current pipeline: Amazon Beauty / frozen Llama7B content features / content-only RQ-VAE / TIGER.

## Primary downstream results

Future test, **full denominator: 34189 queries**. Recall is shown as a percentage; NDCG is shown in raw units.

| Tokenizer | TIGER state | Selected Update step | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Base RQ, extended catalog | Frozen Base TIGER | 0 | 0.476762% | 0.678581% | 0.933049% | 0.003373898224 | 0.004021264440 | 0.004668545582 |
| Plain RQ, 1000 Update steps | Frozen Base TIGER | 0 | 0.058498% | 0.099447% | 0.204744% | 0.000304899962 | 0.000440264698 | 0.000694146659 |
| Frozen Base RQ, extended catalog | Selected after Update fine-tuning | 500 | 0.699055% | 1.216766% | 1.813449% | 0.004427404386 | 0.006088073259 | 0.007579371337 |
| Plain RQ, 1000 Update steps | Selected after Update fine-tuning | 500 | 0.596683% | 0.979847% | 1.798824% | 0.003873351250 | 0.005105906826 | 0.007118554090 |

All four conditions use the same 11775-item Update deployment catalog, beam=20, seen_item_filter=false, and invalid_predictions=0. The 6648 unavailable targets count as misses; coverage is 80.555149%. The Base control keeps Base RQ-VAE weights frozen and adds observed Update items; it is not the original Base deployment snapshot.

This is the main table previously located in Section 5.3. Maintain it here as completed downstream results become available. History-aware and retuned experiments have no completed Future results yet and are not assigned placeholder scores. Evaluation details, the servable-query table, and checkpoint selection are in [Section 5](#5-completed-baseplain-future-evaluation).


### Latest Base validation compatibility ranking

Completed paired run `llama_history_retuned_20261020T123132Z_Edpqdn`, confirmed by user-returned key fields: 1000 steps per arm and passed pair checks. Frozen Base TIGER, fixed old targets and Base train candidate addresses; full denominator=9192, with 8689 servable queries. These are compatibility metrics, **not Future results**.

| Condition | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Original Base reference | 1.294604% | 1.936466% | 0.008709150212 | 0.010793847025 |
| Retuned Plain | 1.196693% | 1.642733% | 0.007767552735 | 0.009225761662 |
| Retuned History-aware | 1.196693% | 1.718886% | 0.007927259322 | 0.009605876138 |

The returned Plain ranking entry now confirms the values previously derived from differences. History-aware gains +0.076153 percentage points in Recall@10 (+4.64% relative; 151 → 158 hits) and +0.000380114476 in NDCG@10 (+4.12% relative). Recall@5 is unchanged; both arms remain below the original Base reference. Final largest collision buckets are 30 / 32 for Plain / History-aware. However, History-aware worsens anchor_train H by 14.51% and anchor_holdout H by 10.57%; both paired H-difference 95% intervals are entirely positive. The modest ranking gain does not establish that the historical-damage objective succeeded. See Section 6.6. No ranking confidence intervals or statistical-significance claim are available.

This report records results returned from Hao's GPU server. The cloud workspace has no Beauty data or model checkpoints. Training and evaluation numbers come from returned compact reports and have not been independently recomputed in the cloud. Code delivery, local checks, and server execution are recorded separately. Earlier sentence-T5 results remain in the existing reports; they are not combined with this Llama pipeline as a controlled encoder ablation.

## 1. Latest status and conclusions

| Stage | Status | Available evidence |
| --- | --- | --- |
| Llama content features | Completed; server result returned | 4096 dimensions, float32; shared-item features unchanged |
| Base RQ-VAE | Completed; server result returned | 1000 total steps, 3 hidden layers, quantization dimension 64 |
| Base SID export | Completed; server result returned | Original Base assignments unchanged; full SIDs unique |
| Base TIGER | Completed; server result returned | 5000 total steps; step 4500 selected; full Base validation evaluated |
| Plain RQ-VAE Update | Completed; server result returned | 1000 total Update steps; parent checkpoint preserved; reload checks passed |
| Plain anchors and frozen TIGER ranking | Completed; server result returned | Paired NLL on anchor_train, anchor_holdout and Base validation; Base ranking |
| Base/Plain Future ranking with frozen TIGER | Completed; server result returned | Same Update deployment catalog; Plain Recall@10 / NDCG@10 lower by 85.34% / 89.05% |
| Base/Plain Future ranking after Update TIGER fine-tuning | Completed; server result returned | 2000 training steps per arm; step 500 selected for both; Plain Recall@10 / NDCG@10 lower by 19.47% / 16.13% |
| History-aware RQ-VAE and matched evaluation | Failed before completing the budget; server result returned | 418 steps completed; suffix allocation failed on attempt 419; disk checkpoint at step 400; final anchor, ranking and Future evaluation not run |
| Bounded replay of History-aware capacity failure | Completed / overflow_reproduced; server result returned | Logged observables matched for steps 401–418; bucket [13,26,8] contained 266 items at attempt 419, exceeding capacity 256; no old slots reserved in that bucket |
| Retuned paired Plain / History-aware experiment | Completed; extracted server results returned | Both completed 1000 steps; pair checks passed; largest collision buckets 30 / 32. History-aware holdout H is 10.57% higher with a positive paired interval, despite Recall@10 / NDCG@10 gains of 4.64% / 4.12%. Future not_run; see Section 6.6. |

Current conclusions:

1. Plain Update reduced reconstruction MSE on the cumulative catalog by 23.68%, while changing the three-level semantic codes of 98.96% of old items. Better reconstruction does not establish compatibility with the old recommender.
2. With the old target fixed and old TIGER frozen, Plain increased anchor_holdout mean sequence NLL by +0.235389, with a user-cluster bootstrap 95% interval of [0.207473, 0.264948]. Positive-part damage H was 0.331016. Historical input re-encoding damage was observed.
3. Base validation Recall@10 fell from 1.936466% to 1.533943% (20.79% relative decrease), and NDCG@10 fell from 0.01079385 to 0.00779528 (27.78% relative decrease). These are compatibility rankings with frozen TIGER and old target addresses.
4. Preserving the old history suffix still gave anchor_holdout mean ΔNLL=+0.189471 and H=0.283439. This is an input diagnostic, not a deployable full-SID mapping. The difference between conditions is not an additive causal estimate of suffix damage.
5. The first History-aware run targeted 1000 steps with λ=0.001, completed 418 steps and stopped during allocation on attempt 419. Only step 400 was saved to disk; final full anchor/holdout evaluation was not run. Bounded replay reproduced bucket [13,26,8] containing 266 items against suffix capacity 256, with zero reserved old-slot width. The current bucket itself exceeded capacity; old-slot reservation was not the cause in this bucket. There were still 5981 distinct three-level codes, so this was not complete collapse. Section 6.3 gives the evidence and limits. Benefits remain unverified; 1000 steps is a paired budget, not an established convergence point or optimal limit.
6. The four Future conditions share coverage of 80.555149%. With frozen TIGER, Plain Recall@10 was 0.099447% versus Base 0.678581%; after matched fine-tuning it was 0.979847% versus Base 1.216766%. Fine-tuning narrowed the gap, but this run did not show a Future ranking benefit from Plain codebook updating. These end-to-end results are distinct from the fixed-old-target diagnostic in Section 4.
7. The retuned recipe reduces update size, smooths the historical STE and increases anchor batch size. A Plain arm with the same content settings is rerun so that stability from a lower learning rate is not attributed to the historical penalty. Parameters were set from training-failure evidence, not holdout/Future results. Both retuned arms now completed 1000 steps with passed pair checks and no reported suffix failure. Final largest buckets are 30 / 32, but History-aware increases H on both anchor cohorts, with paired user-cluster intervals above zero. Reconstruction is effectively unchanged versus retuned Plain. Its modest Base validation ranking gains do not demonstrate success on the damage objective or Future benefits; see Section 6.6.

## 2. Fixed settings and evaluation definitions

- Dataset: Amazon Reviews 2014 Beauty 5-core, using the existing global temporal split and samples without re-splitting.
- Catalog sizes: Base train 10924; Base deployment 11179; Update train 11707.
- Update training item groups: 10924 Base train items, 255 first seen in Base validation, and 528 first seen in Update train.
- Features: locally frozen Llama7B over title / categories / description; last-layer hidden states, sequence length 1024, padding zeroed before sum/1024 pooling. Output is float32 / 4096 dimensions, with no additional normalization, projection or fine-tuning.
- RQ-VAE: 4096 → 512 → 256 → 128 → 64, mirrored decoder, 3×256 semantic codebooks. Outer quantization loss weight=0.25; no BN, dropout, CF, graph or Sinkhorn.
- Update settings for the completed Plain run and first failed History-aware run: AdamW, lr=0.001, weight_decay=0, batch_size=256, seed=2026. Both start from the same complete Base weights, reset the optimizer at Update start and do not reinitialize codebooks. Section 6.4 separately records the retuned learning rate and clipping. The Llama content model is frozen; the RQ-VAE encoder is trainable.
- Full SID: 3 semantic digits plus a suffix of fixed capacity 256; token offsets=[1,257,513,769], PAD=0, EOS=1025, vocab_size=1026. The new version reserves old slots, retains suffixes for items whose prefix is unchanged, and appends other items in registry order.
- Base TIGER: randomly initialized repository T5, 4 encoder / 4 decoder layers, d_model=128, d_ff=1024, num_heads=6, d_kv=64, dropout=0.1; batch=128, lr=0.003, weight_decay=0.05, warmup=200, seed=2026.
- Base TIGER trains for up to 5000 steps. Selection uses full-denominator NDCG@10 on a fixed 512-query probe; step 4500 was selected. Final results use full Base validation, not the probe.
- Base compatibility ranking: 10924 Base train items at their old addresses; catalog-prefix beam search, beam=20, seen_item_filter=false. Base validation has 9192 queries, with 8689 servable old targets and 503 unavailable targets. Full-denominator metrics count those 503 as misses.
- History-only condition: freeze teacher parameters and re-encode history while keeping the complete old target SID, EOS, teacher-forced target prefix, availability mask and positions fixed. Previously unavailable history remains PAD even if the item has since been observed in Update.
- Sequence NLL sums the losses of 5 target tokens. ΔNLL=new−old; H=mean(max(ΔNLL,0)). H does not allow prediction gains to offset damage; mean ΔNLL does. The token CE difference equals sequence ΔNLL/5.
- Anchors: user-hash grouping with seed=2026; 4096 queries per cohort, with 2648 training users and 2613 holdout users, without user overlap. Holdout is excluded from historical-penalty training, not from Base TIGER training. It has been viewed in earlier exploration and is not a new blind test.
- Confidence intervals in the returned results use 500 user-cluster bootstrap replicates and query-weighted mean ΔNLL. They are not intervals for H or ranking differences. The new entry point in Section 6.4 separately computes an interval for the paired H difference; it must not be confused with the NLL-difference interval.

## 3. Content training results

### 3.1 Base and Plain Update milestones

| Experiment | Update steps | Evaluation catalog | Reconstruction MSE | Unique semantic codes | Largest current collision bucket | Codewords used at each level |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Base RQ-VAE, 1000 Base training steps | 0 | Base train, 10924 items | 0.002858462538 | 7952 | 45 | 100 / 147 / 202 |
| Base weights at Update start | 0 | Update train, 11707 items | 0.003109348266 | — | — | — |
| Plain Update pilot | 50 | Update train, 11707 items | 0.003102360316 | 8398 | 48 | 101 / 150 / 205 |
| Plain Update | 1000 | Update train, 11707 items | 0.002373193651 | 8928 | 35 | 120 / 181 / 230 |

The following before/after comparisons use the same catalog within each row. MSE values from different catalogs must not be treated as a paired difference.

| Item group | Items | MSE before Update | MSE after 1000 Plain steps | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Base train | 10924 | 0.002858462538 | 0.002261042332 | −20.90% |
| First seen in Base validation | 255 | 0.004093405791 | 0.003039789386 | −25.74% |
| First seen in Update train | 528 | 0.007824766873 | 0.004371601030 | −44.13% |
| Full Update train | 11707 | 0.003109348266 | 0.002373193651 | −23.68% |

Plain total loss changed from 0.006417348934 to 0.005955037059 (−7.20%); full-catalog quantization loss increased from 0.013232002630 to 0.014327373536. For items first seen in Update, quantization loss increased by 43.67% and total loss increased by 0.39%. Not all training objectives improved.

Plain resumed the 50-step pilot for another 950 steps, reaching 1000 total steps and processed_examples=254551. The 950-step segment took 25.394 seconds; the 50-step segment took 6.163 seconds. Their wall times sum to approximately 31.557 seconds and include per-segment loading/evaluation overhead; this is not a single measurement of pure training time.

For old Base train items, changed=10810/10924 and the three-level semantic code change fraction was 0.989564262175. Per-level change fractions were [0.486085682900, 0.850329549616, 0.955053094105]; prefix change fractions were [0.486085682900, 0.893903332113, 0.989564262175]. No codeword alignment was applied. This churn excludes suffixes and is not a prediction-damage rate.

### 3.2 Downstream compatibility ranking on full Base validation

The frozen TIGER, old targets and candidate catalog are identical; only history is replaced. Full denominator=9192.

| Condition | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Before Update: old history / old target | 1.294604% | 1.936466% | 0.008709150212 | 0.010793847025 |
| After Plain Update: new full history SID / old target | 1.033507% | 1.533943% | 0.006164498296 | 0.007795275995 |
| Relative change | −20.17% | −20.79% | −29.22% | −27.78% |

Conditioning on the 8689 servable queries, Recall@10 changed from 0.020485671539 to 0.016227413972, and NDCG@10 from 0.011418695115 to 0.008246538952. Both conditions had 503 unavailable targets and invalid_predictions=0.

These are neither post-Update TIGER fine-tuning rankings nor Future results. History-aware compatibility evaluation uses the same frozen-model definitions; fine-tuned recommender and Future results are recorded separately.

## 4. Anchor before/after loss and historical damage

### 4.1 New full-SID history with the old target fixed

| Cohort | Queries | Sequence NLL before Update | Sequence NLL after Plain Update | Mean ΔNLL | Positive-part damage H | User-cluster 95% interval for ΔNLL | Worsened fraction |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| anchor_train | 4096 | 8.908613241 | 9.149502934 | +0.240889693 | 0.348048894 | [0.208276862, 0.269093363] | 60.72% |
| anchor_holdout | 4096 | 9.068032124 | 9.303420701 | +0.235388577 | 0.331015811 | [0.207473320, 0.264947916] | 59.16% |
| Servable Base validation | 8689 | 10.221830663 | 10.257678380 | +0.035847717 | 0.193100594 | [0.018207850, 0.052405052] | 53.73% |

Both anchor cohorts had history_changed_fraction=0.99462890625; Base validation had 0.995281390264. The maximum absolute ΔNLL for unchanged histories was 0 in every cohort. Old validation NLL and reference ranking were reproduced, teacher parameter preservation checks passed, and new full SIDs were unique.

The run's historical_damage_detected=true requires both holdout mean ΔNLL and the lower interval bound to exceed 1e−6. It is not determined from H>0 alone.

### 4.2 Diagnostic: new semantic codes with old history suffixes

| Cohort | Mean ΔNLL | Positive-part damage H | User-cluster 95% interval for ΔNLL |
| --- | ---: | ---: | --- |
| anchor_train | +0.186982710 | 0.290458496 | [0.158887755, 0.210709133] |
| anchor_holdout | +0.189470680 | 0.283438526 | [0.165475138, 0.213284701] |
| Servable Base validation | +0.031896816 | 0.181010272 | [0.015339618, 0.047207628] |

Preserving old suffixes reduced damage but did not remove it. This diagnostic does not guarantee full-SID uniqueness, is not a serving mapping and does not identify a specific damage mechanism from the difference alone.

The full Plain E1 evaluation, including both loss conditions and old/new rankings, took 578.002 seconds. CUDA peak allocated=1124896768 bytes; reserved=2097152000 bytes.

## 5. Completed: Base/Plain Future evaluation

The server compact returned on 2026-10-19 reported execution_status=completed. All four conditions were completed: direct testing with frozen TIGER and testing after Update fine-tuning for each tokenizer. Entry point: `evaluate_incremental_future.py`; launcher: `scripts/run_llama_future.sh`. The numbers below come from the same server run, not cloud model recomputation.

### 5.1 Controls and temporal boundaries

- Two tokenizer arms: `base` keeps Base RQ-VAE weights frozen and assigns SIDs to newly observed Update items; `plain` uses the completed 1000-step Plain RQ-VAE. Neither RQ-VAE nor Llama is trained further in this evaluation.
- The Base control is extended to the same Update deployment catalog. Its results must not be labeled as the original 11179-item Base deployment snapshot. This controls candidate items and coverage when comparing tokenizer updates.
- Update train and Update validation use the 11707-item training catalog. Only Future uses the 11775-item deployment catalog, including the 68 items first seen in Update validation. Those items do not enter TIGER parameter training or validation ranking.
- Base extension retains existing full SIDs. Plain extension uses the versioned allocator, reserving old Base addresses before assigning suffixes to new items. Vacated old addresses are not reused, and the tested 11707-item mapping is not renumbered.
- History, target and candidate catalog all use the current full-SID mapping of their own arm. This is end-to-end Future ranking, distinct from the fixed-old-target history-only diagnostic in Section 4.
- Future uses the protocol's prequential history: only events strictly earlier than the target are available. Future-only items remain outside the catalog, with their history positions represented by four PAD tokens and attention=0. Future events do not update the model, mapping or catalog.
- Future contains 34189 queries: 27541 servable and 6648 with out-of-catalog targets, giving coverage of 80.555149%. Counts are identical across all four conditions. Full-denominator metrics count unavailable targets as misses; metrics conditional on servability are also reported.

### 5.2 TIGER fine-tuning and checkpoint selection

- Both arms start from the same selected Base TIGER checkpoint, Base step 4500. Each uses its own RQ mapping to encode both Update history and target. AdamW is reset rather than inheriting Base optimizer momentum.
- The preset budget is at most 2000 TIGER Update steps per arm; both completed 2000. Settings: batch_size=128, lr=0.003, weight_decay=0.05, warmup=200, constant_with_warmup, max_grad_norm=1, seed=2026. This follows the original E3 budget and was not adjusted using Future results.
- A fixed 512-query Update validation probe is evaluated at steps [0,100,250,500,1000,2000]. Selection maximizes full-denominator NDCG@10, retaining the earliest step on ties. Step 0 is eligible, allowing selection of no fine-tuning when fine-tuning does not improve the selection metric.
- Selected checkpoints are additionally evaluated on full Update validation. Pair checks inspect actual checkpoint optimizer parameter groups, sampler order/cursor/generator, processed-example counts and training budgets.
- Future examples are parsed and inference begins only after both arms have selected checkpoints and written `selection_locked.json`. Future is not used to select the learning rate, budget, tokenizer, penalty weight or checkpoint.
- Frozen conditions load Base TIGER directly; fine-tuned conditions load the TIGER selected on each arm's Update validation. All four use beam=20, Recall/NDCG@5/@10/@20 and seen_item_filter=false.

Actual pair checks passed: both arms started from identical Base TIGER weights, used the same optimizer recipe, Update training queries, step budget and sampler endpoint, and processed 255919 examples each. Warm-start and selected-checkpoint reload checks were true. Both selected Update step 500, rather than the last step 2000. The 2000-step recommender fine-tuning budget is separate from the 1000-step RQ-VAE Update budget.

### 5.3 Actual Future ranking

The primary full-denominator table is maintained at the [top of this report](#primary-downstream-results). It uses all 34189 queries, counting 6648 unavailable targets as misses. Recall is a percentage; NDCG is in raw units. All four conditions share the 11775-item deployment catalog, beam=20, seen_item_filter=false and invalid_predictions=0. Base rows refer to a frozen Base RQ-VAE extended to observed Update items, not the original Base deployment snapshot.

Conditioning on the same 27541 servable queries:

| Tokenizer | TIGER state | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Base RQ, extended catalog | Frozen Base TIGER | 0.591845% | 0.842380% | 1.158273% | 0.004188308572 | 0.004991939651 | 0.005795465121 |
| Plain RQ, 1000 Update steps | Frozen Base TIGER | 0.072619% | 0.123452% | 0.254167% | 0.000378498413 | 0.000546538244 | 0.000861703646 |
| Frozen Base RQ, extended catalog | Selected after Update fine-tuning | 0.867797% | 1.510475% | 2.251189% | 0.005496115920 | 0.007557646297 | 0.009408922212 |
| Plain RQ, 1000 Update steps | Selected after Update fine-tuning | 0.740714% | 1.216368% | 2.233034% | 0.004808322352 | 0.006338399058 | 0.008836870331 |

Full Update validation for the selected checkpoints contains 3280 queries: 3084 servable and 196 unavailable. Ranking columns below use the full denominator. Probe NDCG@10 uses the fixed 512-query selection set, with 483 servable queries, and is distinct from full-validation metrics.

| Tokenizer | Selected Update step | Probe NDCG@10 | Full validation Recall@10 | Full validation Recall@20 | Full validation NDCG@10 | Full validation NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Base RQ, extended catalog | 500 | 0.011866984365 | 2.896341% | 4.176829% | 0.012938729610 | 0.016154641217 |
| Plain RQ, 1000 Update steps | 500 | 0.011273458891 | 1.890244% | 4.268293% | 0.009451011768 | 0.015336322498 |

On the servable denominator of full Update validation, Base / Plain Recall@10 was 3.080415% / 2.010376%, and NDCG@10 was 0.013761035383 / 0.010051659728.

The server saved per-query metrics and NLL, old/new target groups, and low-frequency slices. Slices were omitted from the returned compact and have not been read in the cloud, so slice conclusions remain pending. Groups distinguish Base train, first seen in Base validation, first seen in Update train, first seen in Update validation, and Future-only items. Low frequency uses target counts in causally valid Update train examples, with threshold 5, not Future frequencies.

### 5.4 Conclusions, cost and evidence limits

1. With frozen TIGER, Plain Future Recall@10 was 85.34% lower than Base (0.678581% → 0.099447%; hits 232 → 34), and NDCG@10 was 89.05% lower (0.004021264440 → 0.000440264698). This is end-to-end incompatibility after both history and target re-encoding. The full decrease cannot be attributed to the history-only damage in Section 4.
2. After both arms were fine-tuned on Update and selected step 500, Plain Recall@10 remained 19.47% lower (1.216766% → 0.979847%; hits 416 → 335), and NDCG@10 was 16.13% lower (0.006088073259 → 0.005105906826). Fine-tuning narrowed the gap, but Plain did not exceed the matched fine-tuned control with frozen Base RQ.
3. Fine-tuned Recall@20 was only 0.81% lower, and NDCG@20 was 6.08% lower. The gap varies with k; the @10 effect must not be generalized to every cutoff. Coverage is identical, so ranking differences are not explained by different numbers of servable targets.
4. Each arm's Future ranking improved after its own fine-tuning. Plain fine-tuned outperforming Base frozen is not evidence of a codebook-update benefit. The primary tokenizer comparison must match TIGER state and budget.
5. This run uses only seed=2026. There are no user-cluster confidence intervals for Future ranking differences or multi-seed results; statistical significance and general superiority are not claimed. Both arms start from the same old TIGER, so the results concern warm-start compatibility under the preset fine-tuning budget. They do not establish that new SIDs are worse after sufficient retraining. History-aware benefits remain unverified. Observed Future results must not guide changes to λ, budget or checkpoint selection.

Diagnostic NLL and runtime are below. Future target addresses differ across mappings, so cross-mapping NLL is not a substitute for recommendation quality or the historical damage in Section 4. Fine-tuned inference weights are also frozen during evaluation: teacher_frozen_during_evaluation=true does not mean the preceding Update fine-tuning was skipped.

| Tokenizer | TIGER state | Future servable sequence NLL | Future decoding time, seconds |
| --- | --- | ---: | ---: |
| Frozen Base RQ, extended catalog | Frozen Base TIGER | 11.993350524562 | 881.079 |
| Plain RQ, 1000 Update steps | Frozen Base TIGER | 20.501776810613 | 930.938 |
| Frozen Base RQ, extended catalog | Selected after Update fine-tuning | 10.462548662853 | 884.338 |
| Plain RQ, 1000 Update steps | Selected after Update fine-tuning | 11.811874873376 | 892.202 |

TIGER fine-tuning and validation took 409.705 / 426.078 seconds for the two arms. Total Future pipeline runtime was 4456.991 seconds (74.28 minutes), including both fine-tuning/selection phases, validation, four Future evaluations and pipeline overhead. CUDA peak allocated=3720788992 bytes; reserved=4506779648 bytes. Total runtime is not the update time of one arm and excludes preceding Base/Plain RQ training and Llama encoding. There is no equal-total-compute comparison.

The server compact reported pair_checks=passed, future_used_for_selection=false, selection_locked_before_future=true and original_checkpoints_unchanged=true. Cloud checks covered compact counts, coverage, conversion between full/servable denominators, Plain−Base differences and relative changes. Server checkpoints and per-query files were not read to independently recompute metrics.

### 5.5 Engineering checks

These are code-delivery checks, separate from the actual server results above.

- Eight new cloud protocol checks passed: temporal/same-timestamp exclusion, sample deduplication, registry prefixes, availability masks, exclusion of Future-only items from catalogs, old suffix-slot reservation, full/servable denominators and slices.
- The existing 19 temporal round1 checks passed, as did Bash and new Python syntax checks.
- Six simulated launcher checks passed: fresh run, explicit resume, stop on check failure, preserve/resume after execution failure, dirty workspace and missing input. Simulations do not load real Torch models.
- Four real tiny TIGER integration checks were added: four-condition model training/decoding; exact interruption recovery with dropout and deferred Future access; resume bindings/result checks; and reuse of locked models and completed predictions after Future interruption. The upstream E1/RQ context is a synthetic stub; TIGER, Trainer and metrics use real implementations. PyTorch was unavailable in the cloud, so these four checks were not run there. The server launcher requires them to pass before Beauty execution.
- Fresh outputs refuse overwrite. Explicit resume requires the same source, data, configuration and budget, reuses completed training/evaluation, and preserves a separate log for each attempt. Original Base/Plain RQ-VAE and TIGER checkpoints are preserved and hashes rechecked.
- The returned server run passed the startup-check stage and completed all four Beauty conditions. checks.log has not been returned separately; real tiny TIGER checks skipped in the cloud are not labeled as cloud passes. Section 5.3's server compact supports the actual Future results. The failed History-aware run is recorded in Section 6.2.

## 6. Paired History-aware experiments

The first full-SID training run stopped before its budget; final anchor evaluation was not run. The retuned recipe is tracked separately in Section 6.4.

### 6.1 Original preset and launch procedure

This experiment uses the existing scalar-penalty entry point `train_historical_incremental_rqvae.py`, with objective `RQ loss + 0.001 × H`. Settings are fixed at λ=0.001, temperature=0.1, anchor minibatch=32, and explicit training history condition `new_full_sid`. These are preset values, not validated optimal weights; λ is not selected on the previously viewed holdout.

The launcher covers RQ updating, anchor damage and frozen TIGER ranking on Base validation. It does not automatically run the History-aware Future conditions with frozen/fine-tuned TIGER. Those evaluations require a later integration and remain not_run.

History-aware starts fresh from the same Base RQ-VAE for an independent 1000 Update steps. The penalty is not added by continuing from the completed Plain endpoint. The model structure, content batch_size/lr/weight_decay/seed, optimizer start and `Trainer.fit_steps` content sampler match Plain. RQ-VAE encoder, decoder and all three codebooks remain trainable. Historical STE gradients enter the encoder and three semantic codebooks; suffixes have no gradient. Llama features and TIGER teacher remain frozen, with old targets fixed. Only anchor_train enters the penalty; holdout and validation are excluded from training.

`scripts/run_llama_history_aware.sh` performs the following:

1. Check that Plain E1 is complete and includes ranking; the budget must not exceed the completed Plain budget.
2. Run the existing real-model synthetic checks in `test_historical_update_round12.py` on the server; stop on failure.
3. Train History-aware into a fresh run, preserving logs, checkpoints and source/config/data bindings. Exact historical-checkpoint resume is supported; Plain optimizer state is not reused.
4. After 1000 steps, inspect actual checkpoint content settings, optimizer parameter groups, initial content metrics, sampler endpoint and Base/teacher/Plain identities.
5. Run `evaluate_historical_update.py --ranking` using identical cohorts, old labels, teacher, catalog and full denominator. Report H, mean ΔNLL, paired History-aware−Plain NLL differences with user-cluster intervals, Recall/NDCG, reconstruction and churn.

Equal Update steps do not imply equal compute. Historical penalties add forward/backward and catalog re-encoding work, so runtime is reported separately. Sampler matching is supported by configuration, implementation and endpoint state; it is not an independently saved complete per-step batch trace.

Partial runs below 1000 steps can be saved and resumed, but final paired ranking is not run at unequal budgets. Full-SID export or pair-check failure stops subsequent evaluation and preserves logs; suffix capacity is not expanded and checks are not bypassed.

Cloud delivery checks passed Bash syntax, syntax of three embedded Python blocks, and eight simulated launch scenarios covering the full workflow, partial/resume, model-check failure, training failure, sampler mismatch, evaluation failure, budget/weight/Plain-ranking prechecks, and dirty-workspace/artifact preservation. Simulations use fake torch and real CLI parsers, not real RQ-VAE/TIGER or GPU execution. PyTorch was unavailable in the cloud; the server launch procedure runs real-model checks. The server failure below is separate from successful delivery checks.

### 6.2 Actual execution: suffix capacity exhausted

On 2026-10-19, Hao returned training logs and a traceback for run `llama_history_aware_20261019T155320Z_Yk6Nlc`: 11707 content items, 4096 training anchors, λ=0.001, temperature=0.1, anchor batch=32, condition=`new_full_sid`, and target budget 1000 Update steps. Training passed startup checks and old anchor NLL caching. The last periodic log returned was step 400, followed by `ValueError: New-version collision suffix capacity exhausted` in `HistoryPenalty.input_embeds` → `versioned_suffixes`. This was a training exception, not merely an SSH disconnect or missing terminal output.

The returned minibatch milestones are below. RQ, reconstruction and quantization losses use that step's content batch; H uses that step's 32 anchors. Capacity statistics are computed before the parameter update, not on the post-update catalog.

| Completed step | Minibatch reconstruction MSE | Minibatch quantization loss | Minibatch H | Weighted historical term λH | Largest current collision bucket | Maximum required suffix slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.003124332521 | 0.009211765602 | 0.000000000000 | 0.000000000000 | 55 | 55 |
| 100 | 0.003842768725 | 0.012881316245 | 0.119086772203 | 0.000119086772 | 52 | 60 |
| 225 | 0.004056828097 | 0.064984902740 | 0.670326828957 | 0.000670326829 | 108 | 108 |
| 350 | 0.003523970023 | 0.032825231552 | 0.401989817619 | 0.000401989818 | 131 | 131 |
| 400 | 0.003486433532 | 0.029009586200 | 0.280607461929 | 0.000280607462 | 102 | 102 |
| 418 | 0.005391825922 | 0.045405112207 | 0.398684948683 | 0.000398684949 | 228 | 228 |

At step 400, processed_examples=101848, anchor_sampler_epoch=4 and anchor_changed_history_fraction=1.0. Combined gradient norms were encoder=0.033068541437, decoder=0.003261050209 and codebooks=0.007837447338. These are total-objective norms, without separate RQ/history gradients; their values or the scalar λH share do not identify a gradient mechanism.

For each semantic prefix, the allocator reserves previously occupied Base addresses and appends old items that changed prefix and new items. Required capacity is reserved old-slot width plus current movers/newcomers, which may exceed the current bucket size. Each snapshot rebuilds from the fixed Base registry; slots are not permanently accumulated across optimizer steps. The original traceback alone established that at least one prefix needed more than 256 slots. It did not provide the failed state's bucket size, unique codes or per-level usage. Bounded replay later established that this overflow bucket had no reserved old slots; see Section 6.3.

On 2026-10-20, Hao returned a read-only inspection: run_commit=`35d135b4273adf5203e42ec9100f706563b4daa1`; last successful record step=418, epoch=10, processed_examples=106387, total_loss=0.017141789198, rq_loss=0.016743104249, mean sequence ΔNLL=+0.264655411243, H=0.398684948683, anchor_changed_history_fraction=1.0 and anchor_sampler_epoch=4. Total-objective gradient norms were encoder=0.037496332079, decoder=0.006189642940 and codebooks=0.001894956222.

`metrics.jsonl` writes and flushes every successful optimizer step; the terminal prints every 25 steps. Its final row and the traceback confirm that attempt 419 failed during allocation before its backward/optimizer.step. Step 418's capacity 228 describes the parameters before that update, after 417 completed updates. After update 418, allocation was infeasible. A logged feasible=true must not be interpreted as post-update feasibility.

The inspected `last.pth` was at step 400, epoch=9, processed_examples=101848 and cursor=8192. The final 18 unsaved updates cannot be read directly from it. compact, training_summary and evaluation_summary did not exist. This failure path exited without a rejected-state snapshot or a successful rollback record; step 400 is not the failure-state checkpoint.

The 1000-step budget, final full-SID export, final pair checks, full anchor_train/anchor_holdout evaluation, Base validation ranking and History-aware Future evaluation were not completed. A 32-query minibatch H must not be compared with Plain's full 4096-query holdout H, nor used as an equal-budget benefit estimate. Capacity failure is retained as a negative feasibility result. There is neither evidence that History-aware beat Plain nor a completed paired evaluation establishing that it performed worse.

### 6.3 Bounded failure diagnosis: overflow reproduced

Independent tools were added: `diagnose_historical_capacity.py` and `scripts/run_llama_capacity_diagnostic.sh FAILED_RUN_DIR`. In a fresh output directory, they restore the step-400 model, optimizer, content/anchor samplers and RNG, replay steps 401–418 under the original settings, then inspect allocation for attempt 419. They do not continue the original 1000-step run or write a new training checkpoint. Original engine, allocator, suffix capacity and source/data/environment bindings are retained, with no λ change or switch to the old-suffix condition.

All original logged fields are compared step by step: exact equality for integers/booleans, and relative_tolerance=1e-6, absolute_tolerance=1e-7 for floating-point values. Any mismatch stops replay. Agreement establishes only that saved observables match within tolerance; there are no per-step reference weights, so this is not bitwise trajectory reproduction. Replay has a hard limit of 50 attempts. This run expected 18 successful updates followed by the failed attempt 419.

The diagnostic records assignment statistics for step 400, the last matched pre-update state and the failed attempt: unique semantic codes, per-level codebook usage, largest prefix/full-code buckets, largest-bucket fraction, whether all items share one three-level code, and each overflowing prefix's current items, reserved Base-slot width, appended items and required capacity. Input/checkpoint hashes are checked before and after replay. outcome=`overflow_reproduced` requires matching records for 401–418 and the original allocator raising its capacity exception on attempt 419. Non-reproduction, earlier failure, log mismatch and input changes are reported separately and are not treated as confirmation of the original failure state.

Seven local protocol checks, Python/Bash syntax and diff checks passed. They cover replay limits, log continuity/numerical deviations, full/partial concentration, invalid codes, input-directory protection and path-text parsing. Two real RQ-VAE/TIGER small-model replay checks were also provided: restoring model/optimizer/two samplers/RNG from an intermediate state and matching logs; and stopping on the first log deviation while preserving input files. These two were not run in the cloud because Torch was unavailable. The server launcher requires them to pass before the actual diagnostic. The diagnostic uses neither holdout nor Future and is not a new optimization variant or final anchor evaluation.

On 2026-10-20, the returned diagnostic compact reported execution_status=completed and outcome=overflow_reproduced. Output: `results/incremental/Beauty/llama_capacity_diag_20261020T090834Z_Zic7HT`. After restoring disk step 400, all recorded observables for the 18 successful steps 401–418 matched within tolerance. On parameters after 418 completed optimizer updates, the original allocator raised the same exception on attempt 419. No optimizer update occurred on that attempt.

| Completed optimizer updates in parameters | Allocation attempt | Unique three-level semantic codes | Largest current full-code bucket | Maximum required suffix capacity | Codewords used, each level capacity 256 | Largest prefix buckets: L1 / L1–L2 / L1–L3 | Full-SID allocation feasible |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 400, disk checkpoint | 401 | 7366 | 118 | 118 | 101 / 176 / 228 | 1409 / 250 / 118 | true |
| 417, before last matched update | 418 | 5980 | 228 | 228 | 92 / 168 / 214 | 2050 / 566 / 228 | true |
| 418, before failed attempt | 419 | 5981 | 266 | 266 | 94 / 175 / 216 | 2054 / 613 / 266 | false |

The only returned overflow bucket was semantic code `[13,26,8]`: current_bucket_items=266, reserved_old_slots=0, reserved_slot_width=0, appended_items=266 and required_capacity=266. It exceeded fixed suffix_capacity=256 by 10 slots. This failure was caused directly by the current bucket exceeding capacity, not by reserved Base slots. Section 6.2's reservation rule is the general allocation rule, not the specific explanation for this bucket.

At failure, the largest complete semantic bucket contained 266/11707=2.272145% of the catalog; the codebooks still used 94, 175 and 216 codewords, and all_items_share_one_full_semantic_code=false. This is local overconcentration making full-SID allocation infeasible, not all-item three-level collapse. From step 400, the largest bucket rose from 118 to 266 while unique codes fell from 7366 to 5981. However, step 417→418 increased unique codes slightly from 5980 to 5981 while enlarging the largest bucket. Unique-code counts and per-level usage alone do not guarantee per-bucket feasibility. The evidence identifies the direct allocation failure, without isolating causal contributions from historical gradients, content gradients, learning rate or STE. There is also no paired same-step Plain concentration diagnostic.

The original run was not continued (original_run_continued=false), no new training checkpoint was written (checkpoint_written=false), and monitored input hashes were unchanged (observed_inputs_unchanged=true). holdout_evaluated=false and future_evaluated=false. Replayed step-418 parameters existed only in diagnostic memory, not as a new disk training checkpoint; the disk save remained step 400. Diagnostic runtime was 35.141653 seconds, not the original training runtime. Returned diagnostic_source_sha256=`a3a5098769fe37dc9390250755367286e7f73e1d81f85fb4c87055adf4ab6646` matched the delivered diagnostic file. Original run commit remained `35d135b4273adf5203e42ec9100f706563b4daa1`.

Cloud review covered the compact, source semantics and counts. Server weights, attempts.jsonl and per-step logs were not read for an independent replay. Agreement of logged observables is not bitwise weight agreement.

This stage completed failure diagnosis and report updating only. It changed no capacity, offsets, allocator, λ, optimizer or training procedure, and did not fix the failure. A subsequent fix should address post-update capacity feasibility and preservation of failed states. Rollback, smaller-step retry or other constraints must be labeled as a new optimization variant with costs recorded, not as completion of the original paired 1000-step experiment. No new full-anchor/holdout, Base validation or History-aware Future results were produced by this diagnostic.

### 6.4 Retuned paired experiment: preset and implementation

The requested response to unhealthy collisions is a retuned paired run. Direct evidence is the step 400→418 increase in the largest full-code bucket from 118 to 266, with unique codes falling from 7366 to 5981. Even the final slight increase in unique codes did not prevent overflow. Tail concentration made full-SID allocation infeasible under the fixed capacity; per-level usage does not rule out this problem.

The following mechanisms are hypotheses from source inspection, not isolated causal findings. Historical STE uses softmax(-distance/(fixed Base scale × temperature)); gradients propagate through residuals into the encoder and all three codebooks. Encoder changes affect the whole catalog. At temperature=0.1, the surrogate may be sensitive to distance changes, and a 32-query anchor minibatch may add variability. The original objective includes reconstruction, quantization and frozen-teacher positive-part damage, with no load-balancing or per-bucket capacity term. lr=1e-3 without clipping may be too aggressive for the combined objective. Plain completing 1000 steps at that learning rate does not establish stability after adding historical gradients. Existing logs contain only combined gradients and do not establish which component caused concentration.

At step 418, λH was approximately 2.33% of total loss; this does not bound its gradient influence. STE, residual propagation and Adam's adaptive parameter scaling prevent a scalar-loss share from diagnosing update size. Higher temperature, lower λ and lower learning rate are candidate stabilization settings; none guarantees proportionally smaller gradients or parameter displacement. Several parameters change together, so this is a recipe rerun, not an individual-parameter ablation.

| Setting | First failed History-aware | New Plain control | New History-aware |
| --- | ---: | ---: | ---: |
| Content learning rate | 0.001 | 0.0001 | 0.0001 |
| Historical penalty λ | 0.001 | 0 | 0.0001 |
| STE temperature | 0.1 | No historical gradients | 0.5 |
| Anchor minibatch | 32 | No historical training | 128 |
| Global gradient clipping max_norm | None | 0.1 | 0.1 |
| Update optimizer steps | Target 1000; completed 418 | Fixed 1000 | Fixed 1000 |
| Content batch / weight_decay / seed | 256 / 0 / 2026 | 256 / 0 / 2026 | 256 / 0 / 2026 |

Both new arms start fresh from the same Base RQ-VAE `23d4163c…` with reset AdamW. Neither continues from failed step 400/418 or old Plain step 1000. The 3-hidden/e64 architecture, 3×256 codebooks, RQ loss and trainability of all parameters are retained. Llama features, Base teacher, full SID / 256 suffix, old targets, masks/positions and original cohorts follow the same protocol. Only anchor_train enters historical training. Training length is fixed, with no holdout/validation/Future checkpoint selection; both use the last step 1000. The old lr=1e-3 Plain result remains on record; the new paired Plain is the primary control for this recipe.

Independent entry points `train_retuned_historical.py` and `scripts/run_llama_history_retuned.sh E1_DIR [EXISTING_RUN_TO_RESUME]` leave the original E1 hash-bound training/allocation sources unchanged. After every optimizer.step, both arms check original full-SID capacity on all 11707 items and record post-update unique codes, largest bucket, per-level usage and capacity audits. H and gradient metrics are explicitly pre-update. Every 25 steps, gradient probes record grouped RQ and unweighted-history norms, weighted history/RQ ratios and cosines. Every step records the combined pre-clipping norm and whether clipping occurred.

Both arms share a new protection: retain the last feasible model, optimizer, content sampler, anchor sampler and RNG. If post-update capacity is infeasible, save an isolated `rejected_candidate.pth`, explicitly unsuitable for export or ordinary resume, restore `last.pth` and stop. There is no automatic retuning/retry, suffix expansion or switch to an old-suffix objective. The protection does not prevent all unhealthy concentration below overflow; the full collision trajectory still needs review. If either arm fails to finish the budget, final paired evaluation is skipped and the shorter run is not reported as a successful main experiment. Normal interruption supports explicit resume under the same commit/config. Capacity/training errors preserve failure results and prohibit automatic retry in the same output.

After both arms complete, checks compare initialization, configuration differing only in λ, optimizer parameter groups and content-sampler endpoint. Saved checkpoint assignments must match after reload. The same frozen teacher then evaluates anchor_train, anchor_holdout and Base validation, reporting old NLL, new full-SID NLL, mean ΔNLL, H, reconstruction, churn, collisions and Base validation ranking. Old-suffix results remain a separate diagnostic. User-cluster bootstrap intervals are computed separately for History-aware−new Plain mean NLL and positive-part damage H; the NLL interval is not reused for H. Ranking retains old Base addresses, full denominator and the original beam/seen-filter protocol, and reproduces old Base ranking. Additional TIGER Update fine-tuning and History-aware Future evaluation remain not_run. Anchor results cannot establish Future benefits.

Fresh outputs use `results/incremental/Beauty/llama_history_retuned_<timestamp>_<suffix>/experiment`, containing `request.json`, both arms' checkpoints/summaries, per-step metrics, `pair_checks.json`, per-query evaluation files, `summary.json` and `compact.json`. Original Base, old Plain, teacher checkpoint and E1 summary hashes are checked before/after. The original failed run is not written. Subsequent attempts use new log directories; interrupted partial evaluation is recomputed in a fresh attempt, preserving prior files. The launcher permits the explicitly acknowledged untracked `verify_incremental_embeddings.py` to remain in place; other workspace changes stop execution.

The recipe adds historical computation, gradient probes, full-catalog checks and state snapshots. Equal steps do not mean equal compute. It records training-loop time for each arm, total runtime and overall peak CUDA memory, without claiming equal total compute. Four cloud protocol checks and Python/Bash syntax passed. Two real-model integration checks were provided but not run in the cloud because PyTorch was unavailable: full pairing, fixed labels, ranking and exact interruption recovery including optimizer/two samplers/RNG; and a synthetic audit failure testing post-update rollback, rejected-candidate isolation and unchanged original inputs. The latter tests protection logic, not a new observed collapse. The server launcher requires the original real-model checks and these two new checks before training.

Latest execution evidence: Hao reported checking out delivery `c61bc13f46374790ac6e95f4f225d11e114ecd2c` and submitting a background launch with PID=1104312 and log `results/incremental/Beauty/history_retuned_launch_ZfPUB4.log`. This established launch submission only. A later log tail now provides partial evaluation results, recorded in Section 6.5. Completion was unconfirmed at that point. The later extracted results confirm completion in Section 6.6; History-aware Future remains not_run. HANDOFF was not updated.


### 6.5 Retuned result excerpt: preliminary compatibility ranking

This is the earlier partial-evidence record for the same run, not a separate experiment. Its pending items are superseded by the completed extracted results in Section 6.6.

On 2026-10-20, Hao returned the end of `results/incremental/Beauty/llama_history_retuned_20261020T123132Z_Edpqdn/attempt_20261020T123132Z_RwmG3i/train.log`. The final ranking entry and its `historical_minus_plain_full_denominator` values match the History-aware entry position in the delivered driver's output. Only this tail is available; the complete compact, training summaries, pair_checks.json and per-query records have not been read. The full Base validation ranking table is maintained [above](#latest-base-validation-compatibility-ranking), separately from Future results.

Reported full-denominator differences, History-aware minus retuned Plain:

| Metric | Absolute difference, raw metric units | Relative change versus retuned Plain |
| --- | ---: | ---: |
| Recall@5 | 0.0 | 0.00% |
| Recall@10 | +0.000761531766753698 | +4.64% |
| NDCG@5 | +0.0001597065864385374 | +2.06% |
| NDCG@10 | +0.00038011447632696656 | +4.12% |

Recall@10 corresponds to 151 retuned Plain hits and 158 History-aware hits out of 9192 queries, a gain of 7 hits. Recall@5 is 110 hits in both arms. The reported History-aware conditional-on-servable values are Recall@5=0.012659684658763954, Recall@10=0.018183910691679135, NDCG@5=0.008386162698220873 and NDCG@10=0.010161953442855951. decoded_queries=8689. Its mean_sequence_nll_servable=10.246039268624482 and mean_token_ce_servable=2.0492078537248966; these are Base validation metrics, not full anchor_train/anchor_holdout damage.

The excerpt reports ranking_catalog=base_train, invalid_predictions=0, seen_item_filter=false, decoding=catalog_prefix_beam_search and num_beams=20. It also reports stage=paired_retuned_scalar_history, original_inputs_unchanged=true, runtime_seconds=2014.67658434622 (approximately 33.58 minutes), peak_cuda_allocated_bytes=2508070912 and peak_cuda_reserved_bytes=3424649216. These are server-reported fields; cloud checks have not independently verified input hashes or model execution.

This is a positive directional compatibility-ranking signal versus the new Plain control, not a completed assessment of the method. History-aware still trails the original Base reference by 11.24% in Recall@10 and 11.01% in NDCG@10. There is no ranking interval or repeated-seed evidence; seven extra hits do not establish statistical significance. The full compact is needed to confirm both 1000-step budgets, pair checks, anchor H/mean ΔNLL and their paired intervals, reconstruction, churn, and final collision/capacity statistics. The tail does not establish that unhealthy concentration has been resolved. recommendation_training_status=not_run and future_test_status=not_run; the completed Future table remains unchanged. No parameter change, new training or Future evaluation was initiated from this excerpt.


### 6.6 Completed retuned paired experiment: capacity feasible, historical damage worse

On 2026-10-20, Hao returned the requested key-field extraction from `results/incremental/Beauty/llama_history_retuned_20261020T123132Z_Edpqdn/experiment/compact.json`. This completes the evidence previously available only as the tail in Section 6.5; it is the same run, not an additional replication. The returned status is execution_status=completed and original_inputs_unchanged=true. This is server text evidence, not independent cloud execution or recomputation of weights and per-query records.

**Main result:** both arms completed their budgets without a reported capacity failure, but the historical penalty did not reduce hard historical damage. History-aware has higher H on both anchor_train and anchor_holdout, and both paired H-difference intervals exclude zero in the harmful direction. Its small compatibility-ranking gain must be reported alongside this negative primary-objective result.

#### Completion, matching and settings

- Plain and History-aware each completed 1000 Update optimizer steps and processed 254551 content examples.
- Pair checks passed: same_base=true, same_content_recipe=true, same_optimizer_and_clipping=true, same_content_sampler_endpoint=true, same_update_steps=1000, processed_examples=254551. same_compute=false.
- Both report teacher_unchanged=true and checkpoint_reload_assignments_equal=true. Original inputs are reported unchanged; those hashes have not been independently rechecked in the cloud.
- Returned settings: max_steps=1000, lr=0.0001, history_weight=0.0001, temperature=0.5, anchor_batch_size=128 and max_grad_norm=0.1. Plain has no historical penalty; both use the matched content recipe in Section 6.4 and the same Base start. Checkpoint selection remains the last fixed-budget step.
- The key-field extract does not include source/data/checkpoint/cohort hash bindings, full training summaries, per-step gradients or capacity traces, per-arm runtime, or a separately read pair_checks.json. Completion and matching are reported server results, with these audit limits retained. The launch checkout evidence was delivery `c61bc13f46374790ac6e95f4f225d11e114ecd2c`.

#### Reconstruction, quantization and assignments

All RQ metrics below use the same 11707-item Update training catalog. These RQ totals exclude the historical penalty.

| State | RQ total loss | Reconstruction MSE | Quantization loss | Unique semantic codes | Largest current bucket | Codewords used at each level | Old Base semantic churn |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Shared Base weights before Update | 0.006417348934 | 0.003109348266 | 0.013232002630 | — | — | — | — |
| Retuned Plain after 1000 steps | 0.006498323446 | 0.002695704708 | 0.015210474678 | 8794 | 30 | 105 / 152 / 210 | 79.384841% |
| Retuned History-aware after 1000 steps | 0.006507471001 | 0.002695613611 | 0.015247429571 | 8795 | 32 | 103 / 150 / 207 | 80.300256% |

Each codebook retains capacity 256. duplicate_assignment_fraction is 0.24882548902366108 for Plain and 0.2487400700435637 for History-aware. The final largest buckets, 30 and 32, are far below the failed recipe's bucket of 266. Completion under the delivered guard/export procedure is consistent with feasible accepted updates; the per-step capacity trace has not been independently read. This establishes successful completion of this recipe, not a guarantee against concentration under other settings or seeds.

Reconstruction improves from the shared start by 13.30% / 13.31%. History-aware−Plain MSE is approximately −9.109724166e-8 (-0.00338% relative), effectively unchanged at this scale. RQ total loss increases by 1.26% / 1.40% because quantization loss increases. It is incorrect to describe all RQ objectives as improved. History-aware churn is 0.915416 percentage points higher; it does not preserve more old semantic assignments.

#### Full-cohort hard historical damage

Each arm uses the same old targets and frozen teacher. Old-reference NLL and cohort sizes match across arms; anchor_train and anchor_holdout remain disjoint by user under the fixed protocol. Base validation loss uses only its 8689 servable queries, distinct from the 9192-query ranking denominator.

| Cohort | Arm | Queries | Users | Old sequence NLL | New sequence NLL | Mean ΔNLL | Positive-part damage H |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor_train | Retuned Plain | 4096 | 2648 | 8.908613241336 | 8.950773072895 | +0.042159831559 | 0.121296057536 |
| anchor_train | Retuned History-aware | 4096 | 2648 | 8.908613241336 | 8.967521098442 | +0.058907857107 | 0.138890273578 |
| anchor_holdout | Retuned Plain | 4096 | 2613 | 9.068032124196 | 9.128443209687 | +0.060411085491 | 0.133387686394 |
| anchor_holdout | Retuned History-aware | 4096 | 2613 | 9.068032124196 | 9.140940013051 | +0.072907888854 | 0.147488629329 |
| base_validation | Retuned Plain | 8689 | 3955 | 10.221830663040 | 10.245090198319 | +0.023259535279 | 0.109843461008 |
| base_validation | Retuned History-aware | 8689 | 3955 | 10.221830663040 | 10.246039283579 | +0.024208620539 | 0.116588338223 |

The following intervals are for each arm's mean ΔNLL against its own old-history reference. They are not H intervals and are not the paired-arm intervals in the next table.

| Cohort | Plain mean ΔNLL 95% interval | History-aware mean ΔNLL 95% interval |
| --- | --- | --- |
| anchor_train | [0.029572713791, 0.054135531712] | [0.046162745427, 0.071182653117] |
| anchor_holdout | [0.048020362216, 0.073443784949] | [0.058905421789, 0.087377704660] |
| base_validation | [0.012476796592, 0.032709633236] | [0.013540367209, 0.034720519544] |

Paired differences below are **History-aware minus retuned Plain**. Positive values mean more damage. Both interval types use 500 user-cluster bootstrap replicates with query-weighted resampled clusters.

| Cohort | Mean sequence NLL difference | NLL-difference 95% interval | Positive-part H difference | H-difference 95% interval | Relative H increase |
| --- | ---: | --- | ---: | --- | ---: |
| anchor_train | +0.016748025548 | [0.007296662874, 0.026062053982] | +0.017594216042 | [0.009639276321, 0.025667953749] | +14.51% |
| anchor_holdout | +0.012496803363 | [0.004108865857, 0.021746903177] | +0.014100942935 | [0.006135259296, 0.023099341622] | +10.57% |
| base_validation | +0.000949085260 | [-0.007932871644, 0.009171002961] | +0.006744877216 | [-0.000552602887, 0.013950998956] | +6.14% |

For anchor_train, both NLL and H difference intervals are above zero; the same holds for anchor_holdout. Historical damage is therefore worse in this paired run on the training cohort as well as the held-out cohort. This is not explained by holdout generalization alone. The previously viewed holdout and single preset seed limit broader inference.

For Base validation, H increases by approximately 6.14%, but its paired H interval spans zero, as does its paired NLL interval. There is no supported directional difference on these validation loss metrics at the reported interval level.

#### Ranking and interpretation

The confirmed rankings are maintained at the [top of the report](#latest-base-validation-compatibility-ranking). All conditions share 9192 queries, 8689 servable queries, 503 unavailable targets and invalid_predictions=0. History-aware−Plain gains remain Recall@10 +4.64% and NDCG@10 +4.12%, with seven extra top-10 hits (151 → 158); Recall@5 stays at 110 hits. Both remain below the original Base reference. There is no returned ranking confidence interval or multi-seed result.

H measures the average positive increase in teacher-forced sequence NLL on each query; ranking measures the position of an item in a decoded list. A small increase in top-10 hits can coexist with worse mean positive damage, so the ranking gain does not override the negative anchor result. The returned NLL from the paired Base validation loss path is 10.246039283578806 for History-aware, versus 10.246039268624482 in the earlier ranking-path tail; the approximately 1.5e-8 numerical difference does not change the conclusion.

The conservative Plain recipe itself has holdout H about 59.70% below the earlier lr=0.001 Plain result (0.331015811 → 0.133387686394), while its reconstruction MSE is about 13.59% higher (0.002373193651 → 0.002695704708). This is a comparison of different content recipes, including learning rate and clipping, not an isolated learning-rate ablation. It must not be credited to the historical penalty. Under the matched new recipe, the historical penalty adds damage rather than reducing it.

These results do not identify whether the negative effect comes from the STE gradient estimate, conflicts with content gradients, parameter routing or optimization dynamics. That would require inspecting the saved training evidence or a separately declared diagnostic. Do not select a new λ, training budget or checkpoint using this viewed holdout/Future. No new training or automatic parameter search was started from this result.

#### Cost, artifacts and remaining scope

Reported runtime_seconds=2014.67658434622 (approximately 33.58 minutes), peak_cuda_allocated_bytes=2508070912 and peak_cuda_reserved_bytes=3424649216. This is the driver runtime for both arms and evaluation, not a per-arm or equal-compute comparison; per-arm runtime was omitted from the extract.

Run root: `results/incremental/Beauty/llama_history_retuned_20261020T123132Z_Edpqdn`. The returned fields were extracted from `experiment/compact.json`; the delivered workflow also writes training/evaluation summaries, checkpoints, mappings, pair checks and attempt logs under this run. Those other files were not independently read here. The final metrics confirm the earlier derived Plain ranking values. No summary-file hash or new checkpoint hashes were supplied in this extraction.

The retuned RQ update and fixed-model compatibility evaluation are completed as reported. Update TIGER fine-tuning and Future evaluation for these retuned arms remain **not_run**. The primary Future table at the top is unchanged. The first failed run and its diagnostic remain on record; this completed run does not retroactively complete that failed experiment. HANDOFF, RESEARCH_SPEC and training sources are unchanged.

## 7. Artifact and identity index

Paths below are relative to the server project root. These server files have not been read directly in the cloud.

| Artifact | Path |
| --- | --- |
| Fixed temporal protocol | `cache/incremental/Beauty/beauty_20261007T104452Z` |
| Fixed Llama features | `cache/incremental/Beauty/llama7b_features_20261016T132127Z` |
| Base RQ-VAE | `ckpt/incremental/Beauty/llama_base_3hidden_e64_20261016T153649Z_R1DwE8_1000` |
| Base SID | `cache/incremental/Beauty/llama_base_sids_20261019T091637Z_QHuOlK` |
| Base TIGER | `ckpt/incremental/Beauty/llama_base_tiger_full_20261019T094547Z_vHLdGQ_5000`; selected=`best.pth`, step 4500 |
| Plain 50-step checkpoint | `ckpt/incremental/Beauty/llama_plain_update_20261019T105512Z_sVgAhp_50` |
| Plain 1000-step checkpoint | `ckpt/incremental/Beauty/llama_plain_update_20261019T110104Z_mdchRh_1000` |
| Plain 1000-step training run | `results/incremental/Beauty/llama_plain_update_20261019T110104Z_mdchRh` |
| Plain E1 evaluation | `results/incremental/Beauty/llama_plain_history_damage_ranking_20261019T111607Z_zCkKTi/evaluation` |
| Base/Plain Future run | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL` |
| Future summary / compact | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/summary.json`; `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/compact.txt` |
| Fine-tuned and selected TIGER models | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/{base,plain}/last.pth`, both step 2000; `best.pth` in the same directories, both step 500 |
| Future selection lock and pair checks | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/selection_locked.json`; `pair_checks.json` in the same directory |
| Four-condition Future metrics and per-query records | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/{base,plain}_{frozen,finetuned}.json`; corresponding `*_queries.jsonl` and `*.sha256.json` |
| Future startup checks and execution logs | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/attempt_20261019T142547Z_IpXMsA/{checks,evaluate}.log` |
| Future report fragment | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/experiment_report_fragment.md` |
| Failed History-aware run | `results/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc`; `train.log`, `paths.txt`, `commit.txt` |
| History-aware partial checkpoint / step logs | `ckpt/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc_1000/last.pth` at step 400; `metrics.jsonl` in the same directory ends at successful step 418 |
| History-aware capacity diagnostic | `results/incremental/Beauty/llama_capacity_diag_20261020T090834Z_Zic7HT`; diagnostic outputs `summary.json`, `compact.json`, `attempts.jsonl`, `replay.jsonl`. Only the compact and path were returned; the other files have not been independently read. |
| First failed History-aware run's final evaluation, not run | `results/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc/evaluation` is a preset output path, not evidence of valid generated results |

Retuned completed run: `results/incremental/Beauty/llama_history_retuned_20261020T123132Z_Edpqdn/experiment/compact.json` (key fields returned by the user; no file SHA256 supplied).

Returned SHA256 values:

- Feature summary: `7680a9258ac379e7d8f1dc24a62048847b177597fb071f9092f1e6b9b3e91810`.
- Base embeddings: `40efd83af24e11a6d72328da24dab9d8652d10b1634f358796ed3c08355add85`.
- Base RQ-VAE checkpoint: `23d4163c8cb458d7830cdca66900a2e15727d33456fc530b6b026c856fceab55`.
- Base SID summary: `2815fe097446687f9c23fd364450d283c899e36ebb59301b9f7dcc60a303977b`.
- Selected Base TIGER: `83308fa0bb71f398de22f6a3c5652dbb94da323a4bffcd9cc1cae584e03438fa`.
- Plain Update 1000 checkpoint: `1a37ad18ef1fa196416dd7bb373cfa86c3ee169e302dbe7b6e7d4a9024ad62cb`.
- Future Base fine-tuned selected TIGER, Update step 500: `668c9b475103b512779f0c4b659d1662fb23b07b4f1ca49207c8b53e455ca732`.
- Future Plain fine-tuned selected TIGER, Update step 500: `c255c61e4ecef797291c1febf9be869780b2fcb425965f11c7cece3abe500567`.
- Returned attachment `粘贴的文本 (1)(20261019-154219).txt`, containing the compact and path text: SHA256=`f5a8e9c13cf8b3fc3392d4d1b84314a71afa6201f4656b05d9cf68f16b9fc98d`. This is not the server summary.json hash. The original attachment filename is retained for identification.

Code-delivery index: server runs separately save `commit.txt`. Unless explicitly stated, that file has not been independently read, and a delivery SHA is not a newly verified execution SHA.

- Base 3-hidden/e64: `2531623ee02dc62e4de47a2f282a93ee958c65a8`; Base resume: `71984e9d9be8e1cfbaf7895cf176da8515fd732b`.
- Base TIGER launcher: `ad57e414e5ca52a3412275e07d7067509cfc4154`.
- Plain Update launcher: `039d09cc67f815bb4ee82f4fc25d3546982780f4`.
- Plain E1 launcher: `ce6214090ea635835fed4bee89dcb645a8570a06`.
- First ongoing experiment report and History-aware launcher: `271e7b9071106fdfd52666c2caa0199d8f6dab18`.
- Delivery for both Future evaluation modes: `35d135b4273adf5203e42ec9100f706563b4daa1`. The returned compact did not include server commit.txt, so this is not an independently verified Future execution SHA.
- User-reported execution SHA of the failed History-aware run: `35d135b4273adf5203e42ec9100f706563b4daa1`, distinct from later report/diagnostic deliveries.
- Retuned paired launcher: `c61bc13f46374790ac6e95f4f225d11e114ecd2c`; checkout and background submission were reported, followed by completed extracted results in Section 6.6. The extract does not include server commit.txt or source bindings for an independent execution-identity audit.
- Current report revision: use this file's Git commit and the delivery reply, avoiding a self-referencing SHA.

## 8. Ongoing update rules

After each returned server experiment result, update this file and commit to `vibe-dev`. Record settings, status, key metrics, conclusions, failures, artifact paths, checkpoint/source identities, runtime and evidence limits. Preserve earlier and negative results. Missing fields remain unknown or not_run; do not fill them with zero or mark expected outcomes completed.

Maintain the **Primary downstream results** table at the top. Add completed downstream conditions with their tokenizer, TIGER state, selected Update step and Recall/NDCG@5/@10/@20. Preserve metric units, full denominator, catalog/coverage and checkpoint-selection context. New protocols must be identified separately rather than mixed into the same comparison. Section 5.3 links to the top table instead of duplicating it.

Keep fixed-model compatibility evaluation, post-fine-tuning validation and Future test separate. The two Future workflows and four actual conditions in Section 5 have been returned and recorded. The first full-SID History-aware run stopped before its budget; final anchor, ranking and Future results remain missing. The retuned paired run has completed training, anchor, collision and Base validation results as reported in Section 6.6; History-aware H is worse than matched Plain on both anchor cohorts. Its TIGER fine-tuning and Future evaluation remain not_run. Future is for final evaluation only, not penalty-weight, training-length or checkpoint selection.
