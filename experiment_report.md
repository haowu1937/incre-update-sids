# SID Incremental Update: Experimental Settings and Results

Updated: 2026-11-02. Recall is a percentage; NDCG, H and MSE are raw values. `—` means no result. Provisional results have not completed scientific acceptance.

## 1. Downstream Future results

### 1.1 Content RQ-VAE + TIGER, seed 2026

Future: 34,189 queries, 27,541 servable, 6,648 unavailable targets counted as misses; 11,775 Update candidates; coverage 80.555149%. Beam=20; no seen-item filtering. Each fine-tuned TIGER ran 2,000 steps at lr=0.003; the selected checkpoint is shown below.

| Tokenizer | TIGER state | Selected Update step | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Base RQ, extended catalog | Frozen Base TIGER | 0 | 0.476762% | 0.678581% | 0.933049% | 0.003373898224 | 0.004021264440 | 0.004668545582 |
| Plain RQ, 1000 Update steps | Frozen Base TIGER | 0 | 0.058498% | 0.099447% | 0.204744% | 0.000304899962 | 0.000440264698 | 0.000694146659 |
| Frozen Base RQ, extended catalog | Selected after Update fine-tuning | 500 | 0.699055% | 1.216766% | 1.813449% | 0.004427404386 | 0.006088073259 | 0.007579371337 |
| Plain RQ, 1000 Update steps | Selected after Update fine-tuning | 500 | 0.596683% | 0.979847% | 1.798824% | 0.003873351250 | 0.005105906826 | 0.007118554090 |
| Retuned History-aware RQ, 1000 Update steps | Frozen Base TIGER | 0 | 0.257393% | 0.321741% | 0.389014% | 0.001442893261 | 0.001653433621 | 0.001817588659 |
| Retuned History-aware RQ, 1000 Update steps | Selected after Update fine-tuning | 2000 | 0.778028% | 1.389336% | 2.152739% | 0.004622809742 | 0.006579084913 | 0.008508993425 |

The Plain RQ uses update lr=1e-3; retuned Historical uses lr=1e-4 and λ=1e-4. These two Future rows are not a matched codebook-update comparison.

### 1.2 CF/LETTER-adapted + GPT, seed 2026

<!-- cf-letter-downstream-primary:v1 -->

SASRec CF input; 10,924 common old-item candidates. Future: 34,189 queries, 20,281 servable; coverage 59.320249%. Each GPT ran 2,000 Update steps at lr=1e-4 and selected step 250 on the fixed 512-query validation probe. Codebook update lr=1e-5; λ=0 for Plain and 0.000638660634 for Historical. β₁=0 is an ablation; β₁=0.9 is the default.

| Mapping | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- |
| FT-old: original C0(E0) SIDs | 0.675656% | 1.237240% | 1.994794% | 0.004020170015 | 0.005816194284 | 0.007737786721 |
| Plain, lr1e-5, beta1=0.9 | 0.742929% | 1.275264% | 2.038667% | 0.004445055967 | 0.006156421457 | 0.008079779016 |
| Historical, lr1e-5, beta1=0.9 | 0.734154% | 1.324988% | 2.032818% | 0.004370220192 | 0.006262911584 | 0.008043950622 |
| Plain, lr1e-5, beta1=0 | 0.722455% | 1.313288% | 2.062067% | 0.004313334323 | 0.006212234431 | 0.008092132197 |
| Historical, lr1e-5, beta1=0 | 0.716605% | 1.304513% | 2.041592% | 0.004287552409 | 0.006186972486 | 0.008051356587 |

Paired Historical−Plain differences; 500 user-cluster bootstrap replicates. Recall intervals are in percentage points.

| Setting | Metric | Difference | 95% CI |
| --- | --- | --- | --- |
| β₁=0.9 | recall@10 | 0.049724 pp | [0.002940, 0.099819] |
| β₁=0.9 | ndcg@10 | 0.000106490 | [-0.000116122, 0.000351958] |
| β₁=0 | recall@10 | -0.008775 pp | [-0.052693, 0.031244] |
| β₁=0 | ndcg@10 | -0.000025262 | [-0.000232085, 0.000198500] |

<!-- /cf-letter-downstream-primary:v1 -->

### 1.3 CF/LETTER-adapted, five seeds

<!-- cf-letter-multiseed-primary:v1 -->

Seeds 2027–2031; fixed Base models and refreshed CF embeddings. Each seed trains Plain/Historical codebooks for 1,000 steps each (lr=1e-5, β₁=0.9) and FT-old/Plain/Historical GPTs for 2,000 steps each (lr=1e-4). The seed-2026 pilot is excluded. Catalog and Future denominators match Section 1.2.

| Method | Recall@5 mean | Recall@10 mean | Recall@20 mean | NDCG@5 mean | NDCG@10 mean | NDCG@20 mean |
| --- | --- | --- | --- | --- | --- | --- |
| FT-old | 0.502501% | 0.922519% | 1.550206% | 0.003076684 | 0.004413289 | 0.006000526 |
| Plain | 0.593758% | 1.072275% | 1.782445% | 0.003588309 | 0.005123426 | 0.006920523 |
| Historical | 0.569189% | 1.026061% | 1.675978% | 0.003524038 | 0.004980780 | 0.006619144 |

Across-seed uncertainty: equal seed weights, sample SD and Student-t 95% CI, df=4; conditional on the fixed Base/CF inputs.

| Method | Recall@10 SD (pp) | Recall@10 95% CI (%) | NDCG@10 SD | NDCG@10 95% CI |
| --- | --- | --- | --- | --- |
| FT-old | 0.123343 | [0.769368, 1.075670] | 0.000751756 | [0.003479861, 0.005346718] |
| Plain | 0.254809 | [0.755888, 1.388662] | 0.001223473 | [0.003604284, 0.006642569] |
| Historical | 0.187473 | [0.793282, 1.258840] | 0.000993455 | [0.003747243, 0.006214317] |

| Historical−Plain metric | Mean difference | Seed 95% CI |
| --- | --- | --- |
| Recall@10 | -0.046214 pp | [-0.419857, 0.327429] |
| NDCG@10 | -0.000142646 | [-0.002127572, 0.001842279] |

| Seed | FT-old selected step | Plain selected step | Historical selected step | Plain Recall@10 | Historical Recall@10 | Plain NDCG@10 | Historical NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2027 | 0 | 500 | 500 | 0.827752% | 0.824827% | 0.003964980 | 0.004014707 |
| 2028 | 500 | 250 | 250 | 1.213841% | 1.219691% | 0.005900212 | 0.005932997 |
| 2029 | 250 | 100 | 250 | 1.313288% | 1.096844% | 0.006259318 | 0.005153042 |
| 2030 | 0 | 100 | 500 | 1.240165% | 0.827752% | 0.005857047 | 0.003885316 |
| 2031 | 250 | 500 | 250 | 0.766328% | 1.161192% | 0.003635574 | 0.005917838 |

All GPTs trained 2,000 steps; step 0 is an eligible selected checkpoint. Selection candidates: 0/100/250/500/1000/2000; earliest maximum fixed-probe NDCG@10. Future was previously observed; these are fixed-configuration repeats.

<!-- /cf-letter-multiseed-primary:v1 -->

### 1.4 Post-codebook Hungarian mapping: TIGER

<!-- tiger-hungarian-primary:v1 -->

Matched retuned content Plain/Historical codebooks, frozen after 1,000 updates each. Four position-wise Hungarian bijections are fitted only on shared Base-train item identities; the fourth position is a bijective suffix rename, with no suffix reallocation. Current history and target both use the same translated mapping. Planned new fitting: three TIGER arms × 2,000 steps, same validation selection and Future protocol as Section 1.1.

Result status: failed before production fitting; 0 new optimizer steps and 0 new Future passes. The reused Historical control retains its original accepted result.

| Matched retuned mapping | Planned new TIGER steps | Reported new steps | Existing TIGER steps | Selected step | Future Recall@10 | Future NDCG@10 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Plain, unmapped | 2000 | 0 | 0 | — | — | — | Failed before fitting |
| Plain + Hungarian | 2000 | 0 | 0 | — | — | — | Failed before fitting |
| Historical, unmapped | 0 | 0 | 2000 | 2000 | 1.389336% | 0.006579084913 | Original accepted control |
| Historical + Hungarian | 2000 | 0 | 0 | — | — | — | Failed before fitting |

<!-- /tiger-hungarian-primary:v1 -->

## 2. Shared experimental settings

| Setting | Content RQ-VAE + TIGER | CF/LETTER-adapted |
| --- | --- | --- |
| Dataset | Beauty 5-core; fixed global chronological Base / Update / Future split | Same split and query identities |
| Query counts | Base train 76,252; Base val 9,192; Update train 28,023; Update val 3,280; Future 34,189 | Same raw counts; GPT Update fitting uses 24,039 eligible I0-target queries |
| Input embeddings | Frozen Llama7B; title/categories/description; 1,024-token sum/1024 pooling; float32, 4,096 dimensions | SASRec, 64 dimensions; 2 layers, 2 heads, FFN 256, dropout 0.3 |
| Quantizer | Encoder 4096→512→256→128→64; mirrored decoder; 3×256 books; suffix capacity 256; quantization outer weight 0.25 | Identity encoder/decoder; 3×512 books; suffix capacity 512; commitment 0.25; quantization outer weight 1 |
| Update training catalog | 11,707 content rows; deployment catalog 11,775 | 11,691 CF rows; serving catalog I0=10,924 |
| Trainable quantizer parameters | Encoder, decoder and codebooks | Three codebooks only; CF rows fixed |
| Retriever | Random T5; encoder/decoder 4/4 layers; d_model 128, FFN 1024, 6 heads, d_kv 64, dropout 0.1 | Random native GPT2 decoder; 4 layers, d_model 768, FFN 1024, 6 heads, dropout 0.1, 8,000 user buckets |
| History / target | 50 raw events; four SID digits + EOS | 20 raw events; four SID digits, no EOS |
| Retriever Update optimizer | Reset AdamW; lr=0.003, batch 128, weight decay 0.05, clip 1; 200-step warmup then constant | Reset AdamW; lr=1e-4, batch 256, weight decay 0.01, clip 1; constant LR |
| Validation selection | Fixed 512 queries; full-denominator NDCG@10; earliest tie; full 3,280-query Update validation reported separately | Same rule; full validation overlaps the selection probe |
| Future decoding | Catalog-constrained beam 20; top-k 5/10/20; no seen-item filtering; unavailable targets count as misses | Same rule, with the I0 catalog |
| Availability | Raw history is truncated before masking; unavailable slots retain positions | Same rule |

Historical damage: **H = mean(max(NLL_new − NLL_old, 0))** under the frozen Base retriever, with old target SIDs, masks and positions fixed. NLL sums five tokens for TIGER and four for CF/GPT. The CF reference remains the original E0 likelihood after CF refresh. H-only optimizes H; RQ-only optimizes native RQ loss; Mixed optimizes RQ + λH. MSE is per item/per dimension in the corresponding embedding space.

Anchor train: 4,096 queries / 2,648 users. Content anchor holdout: 4,096 queries / 2,613 disjoint users, excluded from historical-penalty training. Holdout/Future were viewed in earlier exploration. Query CIs use 500 user-cluster bootstrap replicates; seed CIs use the five-seed rule in Section 1.3.

## 3. Base and quantizer results

### 3.1 Full Base validation

9,192 queries, 8,689 servable. Base training/selection budgets are listed separately from Update fine-tuning.

| Model | Actual Base steps | Selected step / epoch | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Content RQ + TIGER | 5,000 | 4,500 | 1.294604% | 1.936466% | — | 0.008709150 | 0.010793847 | — |
| SASRec CF source | 44,700 | 37,250 / 125 | 1.566580% | 2.991732% | 5.036989% | 0.008203680 | 0.012750642 | 0.017899974 |
| CF SID + GPT retriever | 8,940 | 1,192 / 4 | 1.958225% | 3.111401% | 4.862924% | 0.012056704 | 0.015726218 | 0.020088908 |

Base quantizer fitting: 1,000 steps in each chain. CF Base SASRec: 150 epochs; Base GPT: 30 epochs. CF Base hard MSE: 0.003605380→0.002218748; native RQ loss: 0.018433030→0.012625291; 10,554 semantic codes / 10,924 items, maximum bucket 71, codebook use 512/512/512.

### 3.2 Content quantizer updates

Seed 2026; AdamW, content batch 256, weight decay 0. Retuned pairs use lr=1e-4, clip=0.1; Historical λ=1e-4, temperature=0.5, anchor batch 128. Original Plain uses lr=1e-3 and no historical penalty.

| State / evaluation catalog | Update steps | RQ total loss | Reconstruction MSE | Unique semantic codes | Maximum bucket | Base semantic churn |
| --- | --- | --- | --- | --- | --- | --- |
| Base RQ / Base train 10,924 | 0 (Base fitted 1,000) | — | 0.002858463 | 7,952 | 45 | — |
| Shared Base weights / Update train 11,707 | 0 | 0.006417349 | 0.003109348 | — | — | — |
| Original Plain, lr=1e-3 / Update train | 1,000 | 0.005955037 | 0.002373194 | 8,928 | 35 | 98.956426% |
| Retuned Plain after 1000 steps | 1,000 | 0.006498323446 | 0.002695704708 | 8794 | 30 | 79.384841% |
| Retuned History-aware after 1000 steps | 1,000 | 0.006507471001 | 0.002695613611 | 8795 | 32 | 80.300256% |

Fixed-old-target historical damage:

| Cohort | Queries | Original Plain H | Retuned Plain H | Retuned Historical H | Retuned Historical−Plain H | Paired H 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| anchor_train | 4096 | 0.348048894 | 0.121296057536 | 0.138890273578 | +0.017594216042 | [0.009639276321, 0.025667953749] |
| anchor_holdout | 4096 | 0.331015811 | 0.133387686394 | 0.147488629329 | +0.014100942935 | [0.006135259296, 0.023099341622] |
| Servable Base validation | 8689 | 0.193100594 | 0.109843461008 | 0.116588338223 | +0.006744877216 | [-0.000552602887, 0.013950998956] |

Frozen Base TIGER compatibility ranking: full Base validation and original Base-train candidate addresses; these are not Future scores.

| Condition | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Original Base reference | 1.294604% | 1.936466% | 0.008709150212 | 0.010793847025 |
| Retuned Plain | 1.196693% | 1.642733% | 0.007767552735 | 0.009225761662 |
| Retuned History-aware | 1.196693% | 1.718886% | 0.007927259322 | 0.009605876138 |

### 3.3 CF refresh, without codebook training

SASRec refresh: 12,240 steps / 30 cumulative epochs; selected step 3,264 / epoch 8 by Update-validation probe NDCG@10. Training vocabulary: 11,691 rows (767 new). Frozen Base quantizer and GPT; 4,096 anchor_train queries.

| CF input to frozen quantizer | Anchor mean sequence NLL | Positive damage H | Old-item SID change | Old-item reconstruction MSE |
| --- | ---: | ---: | ---: | ---: |
| Original Base CF | 7.490371 | 0.000000 | 0.0000% | 0.002218748 |
| Refreshed CF, selected SASRec epoch 8 | 7.552976 | 0.102947 | 24.9359% | 0.002279873 |

SASRec full Update-validation Recall@10=1.128049%, NDCG@10=0.004745465; 2,234/3,280 servable targets. Initial full-E1 quantizer native RQ=0.012321118, hard MSE=0.002196955.

### 3.4 CF H-only / RQ-only / Mixed, 1,000 steps

Same C0/E1 initial state; H=0.102947295. AdamW lr=1e-4, β₁=0.9, weight decay 0, clip=0.1, content batch 4,096, anchor batch 128, temperature=0.5. Mixed λ=0.000638660634. Tail monitors: steps 780/810/840/870/900/930/960/990; step 1,000 is a separate endpoint.

| Arm | Endpoint full H | Endpoint signed delta NLL | Native RQ loss | Hard MSE (all E1 rows) | Last-eight full-monitor H mean | Tail H trend span | Joint stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H-only | 0.235105047 | +0.092519539 | 0.014198393 | 0.002680671 | 0.217281609 | 13.3096% | Failed |
| RQ-only | 0.285447898 | +0.194572997 | 0.011248636 | 0.001924145 | 0.276949294 | 7.1468% | Failed |
| Mixed | 0.265476771 | +0.173788772 | 0.011269451 | 0.001929527 | 0.256898976 | 7.3104% | Failed |

### 3.5 CF H-only optimizer ablation — provisional

Each new arm: 1,000 steps; same starting H=0.102947295, native RQ=0.012321118 and hard MSE=0.002196955. β₂=0.999; other settings match Section 3.4. The lr=1e-4, β₁=0.9 control is the H-only row above. Scientific acceptance remains blocked for this screening study.

| LR | beta1 | Endpoint full H | Tail H mean | Tail H reduction vs initial | Tail H trend span | Tail native RQ increase | Tail hard MSE increase | Eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3e-5 | 0.9 | 0.129549846 | 0.121356144 | -17.8818% | 13.9037% | +1.9138% | +2.9218% | No |
| 1e-5 | 0.9 | 0.087607179 | 0.085674517 | +16.7783% | 3.2706% | +0.3259% | +0.4766% | No |
| 1e-4 | 0 | 0.225744341 | 0.212134346 | -106.0611% | 13.6033% | +12.7950% | +18.6777% | No |
| 3e-5 | 0 | 0.133391841 | 0.120451668 | -17.0032% | 15.6570% | +1.9036% | +2.9006% | No |
| 1e-5 | 0 | 0.086996541 | 0.085361878 | +17.0820% | 2.7191% | +0.3241% | +0.4735% | No |

### 3.6 CF paired optimizer settings

Each RQ-only/Mixed arm: 1,000 steps from the same C0/E1 state; lr=1e-5, Mixed λ=0.000638660634. Same tail window as Section 3.4.

| β₁ | Objective | Final H | Tail H | Tail native RQ | Tail hard MSE | H trend span | Joint stability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | rq_only | 0.13964783516712487 | 0.13611830442823702 | 0.01198027491986527 | 0.0021114599949214607 | 4.6936% | Failed |
| 0.9 | mixed | 0.13614308775868267 | 0.13289818072371418 | 0.011981874362110272 | 0.002111845387844369 | 5.2627% | Failed |
| 0 | rq_only | 0.13997216755524278 | 0.13632531157782068 | 0.01197979469698884 | 0.002111282665282488 | 4.3502% | Failed |
| 0 | mixed | 0.13675530976615846 | 0.13278793675999623 | 0.011981385019510854 | 0.0021116150310263038 | 5.4820% | Failed |

Tail H reduction vs matched RQ-only: 2.3657% (β₁=0.9), 2.5948% (β₁=0). Required reduction ≥5%; RQ/MSE increases ≤2%; H trend span ≤2%, H range ≤10%, RQ drift ≤2%, RQ CV ≤3%.

### 3.7 Five-seed CF codebook results

Same codebook recipe as Section 3.6, β₁=0.9; 1,000 steps per arm, seeds 2027–2031. Initial H=0.102947295. All ten arms failed joint RQ/H tail stability.

| Seed | Plain final fullH | Historical final fullH | TailH reduction Historical vsPlain | Historical H trend | Plain tail nativeRQ | Historical tail nativeRQ | Plain tail hardMSE | Historical tail hardMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2027 | 0.139738543 | 0.136297506 | 2.372326% | 5.161285% | 0.011979977 | 0.011981493 | 0.002111375 | 0.002111675 |
| 2028 | 0.140076078 | 0.135933177 | 2.754242% | 4.941405% | 0.011979731 | 0.011981211 | 0.002111283 | 0.002111604 |
| 2029 | 0.139794499 | 0.136512513 | 2.451633% | 6.279039% | 0.011979435 | 0.011981114 | 0.002111190 | 0.002111560 |
| 2030 | 0.139636095 | 0.137113644 | 2.187648% | 5.666552% | 0.011979820 | 0.011981084 | 0.002111355 | 0.002111575 |
| 2031 | 0.140369191 | 0.135503447 | 3.024048% | 5.847443% | 0.011979860 | 0.011980756 | 0.002111327 | 0.002111410 |

## 4. GPT learning-rate ablation: Update validation only

<!-- cf-letter-gpt-lr-primary:v1 -->

Frozen accepted five-seed Plain/Historical maps and Base GPT. New lr=3e-5: ten arms × 1,000 steps. Baseline lr=1e-4: reuse the first 1,000 steps of the prior runs. Other optimizer settings unchanged. Select at 0/100/250/500/1000 on the same 512-query probe; full validation: 3,280 queries, 2,234 servable. No new Future evaluation. **Status: provisional; training complete, scientific acceptance pending.**

| GPT lr | Frozen mapping | Selected-checkpoint Recall@10 mean | NDCG@10 mean | NDCG seed95%CI |
| --- | --- | ---: | ---: | --- |
| 1e-4, reused prefix | Plain | 1.579268% | 0.007753045 | [0.005348849, 0.010157242] |
| 1e-4, reused prefix | Historical | 1.518293% | 0.007545810 | [0.005745780, 0.009345840] |
| 3e-5, new training | Plain | 1.591463% | 0.008403694 | [0.007103197, 0.009704192] |
| 3e-5, new training | Historical | 1.646341% | 0.008477703 | [0.007148398, 0.009807008] |

| Seed | Plain selected step,1e-4 | Historical selected step,1e-4 | Plain selected step,3e-5 | Historical selected step,3e-5 |
| --- | ---: | ---: | ---: | ---: |
| 2027 | 500 | 500 | 500 | 500 |
| 2028 | 250 | 250 | 1000 | 1000 |
| 2029 | 100 | 250 | 500 | 1000 |
| 2030 | 100 | 500 | 1000 | 500 |
| 2031 | 500 | 250 | 500 | 500 |

| Paired NDCG@10 contrast | Mean difference | Seed 95% CI |
| --- | --- | --- |
| lr1e4 historical minus plain | -0.000207235 | [-0.003461302, 0.003046831] |
| lr3e5 historical minus plain | 0.000074008 | [-0.001827790, 0.001975807] |
| lr3e5 minus lr1e4 plain | 0.000650649 | [-0.002929448, 0.004230746] |
| lr3e5 minus lr1e4 historical | 0.000931893 | [-0.001619152, 0.003482937] |

<!-- /cf-letter-gpt-lr-primary:v1 -->

## 5. Additional content-chain training results

### 5.1 Fixed-checkpoint five-step diagnostic

All branches start from the same final retuned Historical checkpoint: full anchor_train H=0.138890298, fixed-128 H=0.074603535, reconstruction MSE=0.002695613611. Fresh SGD has no momentum/decay; AdamW is reset. Mixed λ=1e-4; temperature=0.5, clip=0.1. Five updates per branch; no downstream fine-tuning.

| Optimizer | Learning rate | Objective | Full train H, step 5 | Fixed-128 H, step 5 | Catalog MSE, step 5 |
| --- | --- | --- | --- | --- | --- |
| SGD | 1e-6 | H-only | 0.138890298 | 0.074603535 | 0.002695613618 |
| SGD | 1e-6 | RQ-only | 0.138890298 | 0.074603535 | 0.002695613600 |
| SGD | 1e-6 | Mixed (lambda=1e-4) | 0.138890298 | 0.074603535 | 0.002695613600 |
| SGD | 1e-5 | H-only | 0.138890298 | 0.074603535 | 0.002695618788 |
| SGD | 1e-5 | RQ-only | 0.138890298 | 0.074603535 | 0.002695613610 |
| SGD | 1e-5 | Mixed (lambda=1e-4) | 0.138890298 | 0.074603535 | 0.002695613613 |
| SGD | 1e-4 | H-only | 0.138855056 | 0.074603535 | 0.002695686638 |
| SGD | 1e-4 | RQ-only | 0.138890298 | 0.074603535 | 0.002695613532 |
| SGD | 1e-4 | Mixed (lambda=1e-4) | 0.138890298 | 0.074603535 | 0.002695613571 |
| AdamW reset | 1e-4 | H-only | 0.149320051 | 0.094059799 | 0.002875301512 |
| AdamW reset | 1e-4 | RQ-only | 0.139117855 | 0.067200996 | 0.002734218417 |
| AdamW reset | 1e-4 | Mixed (lambda=1e-4) | 0.131612893 | 0.067681655 | 0.002734290681 |

### 5.2 Joint RQ/H training grid

Seed 2026; content batch 256, anchor batch 128, temperature=0.5, clip=0.1. LR schedule: warmup 50, cosine decay to 0.1× peak at step 2,500, then constant. Screen: 1,500 steps per cell; the lr=3e-5 Plain / λ=1e-4 pair reaches 3,000 steps.

| Phase | LR | lambda | Step | Full RQ loss | Anchor train H |
| --- | --- | --- | --- | --- | --- |
| screen | 3e-05 | 0.0 | 1500 | 0.00627220397 | 0.0966144718 |
| screen | 3e-05 | 0.0001 | 1500 | 0.00626442829 | 0.091643621 |
| screen | 3e-05 | 0.001 | 1500 | 0.00632254559 | 0.100183809 |
| screen | 3e-05 | 0.01 | 1500 | 0.00703948342 | 0.11022997 |
| screen | 0.0001 | 0.0 | 1500 | 0.00663655225 | 0.127460144 |
| screen | 0.0001 | 0.0001 | 1500 | 0.00663483573 | 0.122405993 |
| screen | 0.0001 | 0.001 | 1500 | 0.00669368518 | 0.117461895 |
| screen | 0.0001 | 0.01 | 1500 | 0.0071090093 | 0.176198837 |
| screen | 0.0003 | 0.0 | 1500 | 0.00690470124 | 0.145349686 |
| screen | 0.0003 | 0.0001 | 1500 | 0.00696031732 | 0.164356615 |
| screen | 0.0003 | 0.001 | 1500 | 0.00724488451 | 0.158640338 |
| screen | 0.0003 | 0.01 | 1500 | 0.00936493993 | 0.298922757 |
| extended | 3e-05 | 0.0 | 3000 | 0.00628179942 | 0.10248367 |
| extended | 3e-05 | 0.0001 | 3000 | 0.00628192233 | 0.0994473656 |

Eight-completed-epoch tail means for the 3,000-step pair:

| Arm | Tail RQ loss | Tail H | New-item MSE | Tail stability |
| --- | --- | --- | --- | --- |
| Plain | 0.006282328315 | 0.102028577319 | 0.005940554466 | Passed |
| Historical, λ=1e-4 | 0.006285275639 | 0.100669074935 | 0.005947839793 | Passed |

Historical tail H reduction vs Plain=1.332472%; required ≥5%. The final quality criterion was not met.

### 5.3 Historical-gradient measurements

41 probes from the 1,000-step retuned Historical run, λ=1e-4; component statistics below are training gradients.

| Parameter group | Mean RQ gradient norm | Mean unweighted H gradient norm | Mean weighted H/RQ norm ratio | Mean cosine (defined probes) | Negative cosine | Zero H gradient |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder | 0.005929175 | 0.930419261 | 1.6793% | -0.023598 (40) | 22/40, 55.0% | 1/41, 2.4390% |
| Decoder | 0.002094012 | 0 | 0% | Undefined (0) | Undefined | 41/41, 100% |
| Codebooks | 0.000244023 | 0.081744113 | 3.3919% | -0.001734 (40) | 21/40, 52.5% | 1/41, 2.4390% |

The earlier content Historical preset (lr=1e-3, λ=1e-3, temperature=0.1, anchor batch 32, no gradient clipping) completed 418/1,000 updates; required suffix capacity reached 266/256. No completed final H or downstream result is available for that preset.

Settings and raw result files: [experiment manifests](docs/experiments/) · [result artifacts](https://github.com/haowu1937/incre-update-sids/tree/lab-results/artifacts).

<!-- history-gradient-logs:0a94dcff5c46ed991481852e9799f1c65be18e24b4e789b82710b394f9cab1c9 -->
<!-- history-gradient-driver:e225ebd9547aa6eb89a23e4a4f909ca4e9e0608d -->
<!-- history-gradient-preflight-failure:e87e19e408001f2d04bf6932e64d591d57f8927de8031038f779d2b157f2415f -->
<!-- history-gradient-controlled:b71607a509ac8c259b0d58487f7ca9a289c041d9d1f5b9f3278d709e2d9d9f4c -->
<!-- joint-convergence-cloud:1d34004ae25f2216edd0586b5dcd9ef83501fcf646e10227db063163775f26d5 -->
<!-- cf-letter-design:v1 -->
<!-- sid-main-acceptance:966416120907d6c205b188b08b36bafe511efe8783c1291109e4a0832f0e056a -->
<!-- joint-convergence:1d34004ae25f2216edd0586b5dcd9ef83501fcf646e10227db063163775f26d5 -->
<!-- cf-letter-base-implementation:v1 -->
<!-- lab-task:cf-letter-base-001:d468e76eb33177a7729b5a25d734278a78e28e8fe94e723fb4e4785096b0de5c -->
<!-- cf-letter-base-cloud:ff051c739b8af53ffc305fee72ee2dc56fc512e0e0493e34005c7dd580de18bb -->
<!-- cf-letter-refresh-implementation:v1 -->
<!-- lab-task:cf-letter-refresh-diag-001:efaa01a74a4b0c9bfff67963947e17e1b849436e54562137ff72ec49d313dbc8 -->
<!-- cf-letter-refresh-cloud:ca3e8ca977b712aef504b7b8c4854fb5b4a7b40d9a2682a50950486a3639a58a -->
<!-- cf-letter-history-long-plan:v1 -->
<!-- cf-letter-history-long-preflight-failure:v1 -->
<!-- lab-task:cf-letter-history-long-repair-001:f09fca24f697ad728d31cc39e27bb953bd486b290a07bd00e42abbf8f4f34cc7 -->
<!-- cf-letter-history-long-cloud:84490dc7aa8074fa4d9b8d2183e94a19f8e4031ffcc2c5bb59f915ee61788c0f -->
<!-- cf-letter-optimizer-plan:v1 -->
<!-- cf-letter-optimizer-acceptance-blocker:v1 -->
<!-- lab-task:cf-letter-optimizer-audit-001:41f343af0e7885722fe96931ec48054806643bca563ab29a85ab0cc9ebd9b095 -->
<!-- cf-letter-optimizer-pair-plan:v1 -->
<!-- lab-task:cf-letter-optimizer-pair-001:83736d739f210db9879c27027350c96ebe4b820fcd355a7c6c348776d6fb4e41 -->
<!-- cf-letter-optimizer-pair-analysis:49ec6a3d761d46fc4340f5cddc070b5163ca0f2bb17597265b270de7226ae4cd -->
<!-- cf-letter-gpt-future-plan:v1 -->
<!-- cf-letter-gpt-future-preflight-failure:v1 -->
<!-- lab-task:cf-letter-gpt-future-repair-001:43229d9a847c55171f07bea93f55598a7793fe405d42aea1f77e59f907155585 -->
<!-- cf-letter-downstream-analysis:8629100e69cfbf478212b30f8aaded9df74cbf559c1a774d36e7ebd0b64b060d -->
<!-- cf-letter-multiseed-plan:v1 -->
<!-- cf-letter-multiseed-failure:4dac80ba8f86723c910cfe1f235453d03c1632c6e69a8202e0dbe0d3648e62ec -->
<!-- cf-letter-multiseed-resume-plan:v1 -->
<!-- lab-task:cf-letter-multiseed-resume-001:88724c922ed5c63c032d463e21265d657cb683d6a1a5d766b68032ce6e3a42c3 -->
<!-- cf-letter-multiseed-resume-analysis:91f2a15c9df6df64fa85c865ba0c4c234f4321f19efa97353984b94fb5456513 -->
<!-- cf-letter-gpt-lr-plan:v1 -->
<!-- cf-letter-gpt-lr-delivery-failure:542db0a1aa901aaee0593d0cef3be373093eb7050e62effd4b650b4ccbe5a74a -->
<!-- tiger-hungarian-plan:v1 -->
<!-- tiger-hungarian-analysis:1d2faf2545fe346c0aa90b1ecaf2c2e991bc202f51df31be62cd5ef62bfa098f -->
