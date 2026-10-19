# History-aware SID Update 实验报告

更新日期：2026-10-19。当前主线：Amazon Beauty / 冻结 Llama7B 内容特征 / 纯内容 RQ-VAE / TIGER。

本报告持续记录用户 GPU 服务器回传的实验结果。云端没有 Beauty 数据或模型 checkpoint；表内真实训练和评估数字来自用户 compact，不代表云端独立重算。代码交付、本地模拟检查与服务器实验状态分别记录。旧 sentence-T5 结果保留在存量报告中，不与本轮 Llama 结果混作严格 encoder 消融。

## 1. 最新状态与结论

| 阶段 | 状态 | 已有证据 |
| --- | --- | --- |
| Llama 内容特征 | completed，用户回传 | 4096 维、float32；共享物品特征不变 |
| Base RQ-VAE | completed，用户回传 | 累计 1000 步，3 个 hidden layers，量化维度 64 |
| Base SID 导出 | completed，用户回传 | 原 Base 分配不变；完整 SID 唯一 |
| Base TIGER | completed，用户回传 | 累计 5000 步，选中 step 4500，完整 Base validation 已评估 |
| Plain RQ-VAE Update | completed，用户回传 | 累计 1000 Update 步；父 checkpoint 保留、重载检查通过 |
| Plain anchor 与冻结 TIGER 排名 | completed，用户回传 | anchor_train、anchor_holdout、Base validation 的配对 NLL 和 Base 排名 |
| History-aware RQ-VAE + 同口径评估 | code_provided / not_run | 本轮交付训练、配对核对和评估启动脚本，尚无本轮服务器结果 |
| Update TIGER 微调与 Future ranking | not_run | 当前报告没有微调后或 Future 指标 |

当前结论：

1. Plain Update 在累计目录上的重构 MSE 下降 23.68%，但旧物品三层语义码变化率达到 98.96%。重构改善不能自动代表与旧推荐模型兼容。
2. 固定旧 target 和冻结旧 TIGER 后，Plain 的 anchor_holdout 平均序列 ΔNLL 为 +0.235389，用户聚类 bootstrap 95% 区间为 [0.207473, 0.264948]；正部损伤 H 为 0.331016。本轮观察到了历史输入重编码损伤。
3. Base validation 的 Recall@10 从 1.936466% 降至 1.533943%（相对下降 20.79%），NDCG@10 从 0.01079385 降至 0.00779528（相对下降 27.78%）。这是冻结 TIGER、旧 target 地址下的兼容性排名结果。
4. 保留旧 history suffix 后，anchor_holdout 平均 ΔNLL 仍为 +0.189471，H 为 0.283439，损伤仍存在。该条件是输入诊断，不是可部署的完整 SID 映射；不能将两个条件的差直接当作可加的 suffix 因果贡献。
5. 本轮 History-aware 的收益尚未验证。1000 步是当前配对预算，不是已证明收敛或最优的上限。

## 2. 固定设置与评价口径

- 数据：Amazon Reviews 2014 Beauty 5-core，固定已有全局时间切分和样本，不重新划分。
- Base train catalog：10924；Base 部署目录：11179；Update train catalog：11707。
- Update 训练物品组：Base train 10924、Base validation 首见 255、Update train 首见 528。
- 特征：本地冻结 Llama7B，title / categories / description；最后层 hidden states，长度 1024，padding 清零后 sum/1024；输出 float32 / 4096 维，无额外归一化、投影或微调。
- RQ-VAE：4096 → 512 → 256 → 128 → 64，镜像 decoder，3×256 semantic codebooks。外层 quantization loss weight=0.25，无 BN、dropout、CF、graph 或 Sinkhorn。
- Update：AdamW，lr=0.001、weight_decay=0、batch_size=256、seed=2026；从同一 Base 全权重开始，Update 起点重置优化器，不重新初始化码本。冻结的是 Llama 内容模型，不是 RQ-VAE encoder。
- 完整 SID：3 个 semantic digits + 固定容量 256 的 suffix；token offsets=[1,257,513,769]，PAD=0、EOS=1025、vocab_size=1026。新版本预留旧 slots、保留未移动物品的 suffix，其他物品按 registry 顺序追加。
- Base TIGER：随机初始化的 repository T5，4 encoder / 4 decoder layers，d_model=128、d_ff=1024、num_heads=6、d_kv=64、dropout=0.1；batch=128、lr=0.003、weight_decay=0.05、warmup=200、seed=2026。
- TIGER 最多训练 5000 步，按固定 512-query probe 的 full-denominator NDCG@10 选择 checkpoint；最终选中 step 4500。最终表格使用完整 Base validation，不用 probe 代替。
- Ranking：Base train 10924 个旧地址物品，catalog-prefix beam search，beam=20，seen_item_filter=false。Base validation 共 9192 queries，其中 8689 个旧 target 可服务，503 个不可服务；主表将 503 个计作 misses。
- History-only：teacher 参数冻结，history 重编码，target 完整旧 SID 与 EOS、teacher-forced target prefix、可用性 mask 和位置不变。原来不可用的 history 保持 PAD，不因 Update 已观察到物品而补入。
- NLL 为 5 个 target tokens 的序列损失之和。ΔNLL=new−old；H=mean(max(ΔNLL,0))。H 不允许预测增益抵消损伤；平均 ΔNLL 允许抵消。token CE 的差等于序列 ΔNLL/5。
- Anchor：seed=2026 的 user-hash 分组，各取 4096 queries；train=2648 users、holdout=2613 users，用户不重叠。holdout 仅对历史惩罚训练留出，并非对 Base TIGER 训练留出；此 cohort 已在旧探索中查看，不称为新的盲测。
- 95% 区间：500 次用户聚类 bootstrap，重采样后以 query 加权计算平均 ΔNLL；这里的区间不属于 H 或排名差。

## 3. 内容训练结果

### 3.1 Base 与 Plain 更新里程碑

| 实验 | Update 步数 | 评估目录 | 重构 MSE | Unique semantic codes | 最大当前碰撞桶 | 三层码本使用数 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Base RQ-VAE，Base 训练 1000 步 | 0 | Base train，10924 | 0.002858462538 | 7952 | 45 | 100 / 147 / 202 |
| Base 权重，Update 训练起点 | 0 | Update train，11707 | 0.003109348266 | — | — | — |
| Plain Update pilot | 50 | Update train，11707 | 0.003102360316 | 8398 | 48 | 101 / 150 / 205 |
| Plain Update | 1000 | Update train，11707 | 0.002373193651 | 8928 | 35 | 120 / 181 / 230 |

同目录的更新前后比较如下；不能直接把不同目录的两个 MSE 当作配对差。

| 物品组 | items | 更新前 MSE | Plain 1000 步后 MSE | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| Base train | 10924 | 0.002858462538 | 0.002261042332 | −20.90% |
| Base validation 首见 | 255 | 0.004093405791 | 0.003039789386 | −25.74% |
| Update train 首见 | 528 | 0.007824766873 | 0.004371601030 | −44.13% |
| 完整 Update train | 11707 | 0.003109348266 | 0.002373193651 | −23.68% |

Plain 总损失 0.006417348934 → 0.005955037059（−7.20%）；完整目录量化损失 0.013232002630 → 0.014327373536。Update 首见组量化损失上升 43.67%，该组总损失上升 0.39%，因此不能说全部训练目标都改善。

Plain 从 pilot 50 步续训 950 步，累计 1000 步，processed_examples=254551；950 步段运行 25.394 秒，50 步段为 6.163 秒。两段 wall time 相加约 31.557 秒，包含各段加载/评估等开销，不是统一计时的纯训练耗时。

旧 Base train 物品 changed=10810/10924，三层完整语义码变化率=0.989564262175。逐层变化率=[0.486085682900, 0.850329549616, 0.955053094105]；前缀变化率=[0.486085682900, 0.893903332113, 0.989564262175]。未进行 codeword alignment，此 churn 不含 suffix，也不等于预测损伤率。

### 3.2 完整 Base validation 下游兼容性排名

相同冻结 TIGER、相同旧 target 与候选目录，仅替换 history；full denominator=9192。

| 条件 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 更新前，旧 history / 旧 target | 1.294604% | 1.936466% | 0.008709150212 | 0.010793847025 |
| Plain 更新后，新完整 history SID / 旧 target | 1.033507% | 1.533943% | 0.006164498296 | 0.007795275995 |
| 相对变化 | −20.17% | −20.79% | −29.22% | −27.78% |

仅对 8689 个可服务 queries 计算时：Recall@10=0.020485671539 → 0.016227413972，NDCG@10=0.011418695115 → 0.008246538952。两种条件的 unavailable targets 均为 503，invalid_predictions 均为 0。

这不是 Update TIGER 微调后的部署排名，也不是 Future 结果。History-aware 将使用完全相同的冻结模型评价口径；推荐模型微调/Future 结果另列。

## 4. Anchor before/after loss 与历史损伤

### 4.1 新完整 SID history，旧 target 固定

| 数据组 | queries | 更新前序列 NLL | Plain 更新后序列 NLL | 平均 ΔNLL | 正部损伤 H | ΔNLL 用户聚类 95% 区间 | 变差比例 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| anchor_train | 4096 | 8.908613241 | 9.149502934 | +0.240889693 | 0.348048894 | [0.208276862, 0.269093363] | 60.72% |
| anchor_holdout | 4096 | 9.068032124 | 9.303420701 | +0.235388577 | 0.331015811 | [0.207473320, 0.264947916] | 59.16% |
| Base validation，可服务部分 | 8689 | 10.221830663 | 10.257678380 | +0.035847717 | 0.193100594 | [0.018207850, 0.052405052] | 53.73% |

Anchor 两组 history_changed_fraction 均为 0.99462890625；Base validation 为 0.995281390264。未变 history 的最大绝对 ΔNLL 均为 0。旧验证 NLL 与排名参考复现通过，teacher 参数保留检查通过，新完整 SID 唯一。

运行中的 historical_damage_detected=true 使用的是 holdout 平均 ΔNLL 及其区间下界均大于 1e−6；不是仅凭 H>0 判定。

### 4.2 新语义码 + 旧 history suffix 诊断

| 数据组 | 平均 ΔNLL | 正部损伤 H | ΔNLL 用户聚类 95% 区间 |
| --- | ---: | ---: | --- |
| anchor_train | +0.186982710 | 0.290458496 | [0.158887755, 0.210709133] |
| anchor_holdout | +0.189470680 | 0.283438526 | [0.165475138, 0.213284701] |
| Base validation，可服务部分 | +0.031896816 | 0.181010272 | [0.015339618, 0.047207628] |

保留旧 suffix 时损伤较低，但没有消失。它不保证完整 SID 唯一，不作为 serving 映射，也不能仅凭差值认定具体损伤机制。

本次完整 Plain E1（含两条件损失和 old/new 排名）运行 578.002 秒，CUDA peak allocated=1124896768 bytes、reserved=2097152000 bytes。

## 5. 下一项：配对 History-aware，code_provided / not_run

本轮明确使用已有标量历史惩罚入口 `train_historical_incremental_rqvae.py`，目标为 `RQ loss + 0.001 × H`。固定 λ=0.001、temperature=0.1、anchor minibatch=32，训练 history 条件显式指定 `new_full_sid`。这是本轮预设配置，尚未验证是最优权重；不根据已查看的 holdout 选择 λ。

从同一个 Base RQ-VAE fresh warm-start，独立训练累计 1000 Update 步，不在已训练的 Plain 终点上继续加惩罚。继承 Plain 的模型结构、内容 batch_size/lr/weight_decay/seed/优化器起点与 `Trainer.fit_steps` 内容采样实现。RQ-VAE encoder、decoder、三个码本保持可训练；历史 STE 梯度进入 encoder 与三个 semantic codebooks，suffix 无梯度。冻结 Llama 特征和 TIGER teacher，固定旧 target。仅 anchor_train 进入惩罚，holdout 与 validation 不参与训练。

调用 `scripts/run_llama_history_aware.sh` 时顺序为：

1. 检查 Plain E1 已完成且包含排名；预算不得超过已完成的 Plain 预算。
2. 在服务器运行已有真实模型合成检查 `test_historical_update_round12.py`，失败则停止。
3. 训练 History-aware，保存全新 run、日志、checkpoint、source/config/data binding。支持历史 checkpoint 精确 resume；不复用 Plain optimizer 状态。
4. 完成 1000 步后，核对实际 checkpoint 的内容配置、优化器参数组、初始内容指标、采样器终点及 Base/teacher/Plain 身份。
5. 调用 `evaluate_historical_update.py --ranking`，使用相同 cohorts、旧 labels、teacher、目录和 full denominator。报告 H、平均 ΔNLL、History-aware 相对 Plain 的配对 NLL 差与用户聚类区间、Recall/NDCG、重构和 churn。

等 Update 步数不等算力；历史惩罚有额外 forward/backward 和目录重编码成本，单独报告 runtime。配对核对的采样器证据是配置、实现与终点状态，不冒称独立保存了每步完整 batch trace。

小于 1000 步的 partial run 可保存和恢复，但不运行不同预算的最终配对排名。完整 SID 导出或配对核对失败时停止后续评估并保留日志，不扩充 suffix 容量或绕过校验。

本轮云端检查：Bash 语法、3 个嵌入 Python 块语法及 8 项模拟启动场景通过，覆盖完整流程、partial/resume、模型检查失败、训练失败、采样器不一致、评估失败、预算/权重/Plain 排名预检和 dirty workspace/产物保留。模拟使用 fake torch 与真实 CLI parser；不等于真实 RQ-VAE/TIGER 或 GPU 检查。云端缺少 PyTorch，真实模型合成检查由上述服务器启动流程执行；本轮尚未有 History-aware 的服务器结果。

## 6. 产物与身份索引

以下路径相对服务器项目根目录；云端未读取这些服务器文件。

| 产物 | 路径 |
| --- | --- |
| 固定 temporal protocol | `cache/incremental/Beauty/beauty_20261007T104452Z` |
| 固定 Llama 特征 | `cache/incremental/Beauty/llama7b_features_20261016T132127Z` |
| Base RQ-VAE | `ckpt/incremental/Beauty/llama_base_3hidden_e64_20261016T153649Z_R1DwE8_1000` |
| Base SID | `cache/incremental/Beauty/llama_base_sids_20261019T091637Z_QHuOlK` |
| Base TIGER | `ckpt/incremental/Beauty/llama_base_tiger_full_20261019T094547Z_vHLdGQ_5000`；selected=`best.pth`，step 4500 |
| Plain 50-step checkpoint | `ckpt/incremental/Beauty/llama_plain_update_20261019T105512Z_sVgAhp_50` |
| Plain 1000-step checkpoint | `ckpt/incremental/Beauty/llama_plain_update_20261019T110104Z_mdchRh_1000` |
| Plain 1000-step training run | `results/incremental/Beauty/llama_plain_update_20261019T110104Z_mdchRh` |
| Plain E1 evaluation | `results/incremental/Beauty/llama_plain_history_damage_ranking_20261019T111607Z_zCkKTi/evaluation` |

已回传 SHA256：

- Feature summary：`7680a9258ac379e7d8f1dc24a62048847b177597fb071f9092f1e6b9b3e91810`。
- Base embeddings：`40efd83af24e11a6d72328da24dab9d8652d10b1634f358796ed3c08355add85`。
- Base RQ-VAE checkpoint：`23d4163c8cb458d7830cdca66900a2e15727d33456fc530b6b026c856fceab55`。
- Base SID summary：`2815fe097446687f9c23fd364450d283c899e36ebb59301b9f7dcc60a303977b`。
- Selected Base TIGER：`83308fa0bb71f398de22f6a3c5652dbb94da323a4bffcd9cc1cae584e03438fa`。
- Plain Update 1000 checkpoint：`1a37ad18ef1fa196416dd7bb373cfa86c3ee169e302dbe7b6e7d4a9024ad62cb`。

代码交付索引（服务器 run 另存 `commit.txt`；未独立读取该文件，不将交付 SHA 冒称重新核验的实际执行 SHA）：

- Base 3-hidden/e64：`2531623ee02dc62e4de47a2f282a93ee958c65a8`；Base resume：`71984e9d9be8e1cfbaf7895cf176da8515fd732b`。
- Base TIGER 启动：`ad57e414e5ca52a3412275e07d7067509cfc4154`。
- Plain Update 启动：`039d09cc67f815bb4ee82f4fc25d3546982780f4`。
- Plain E1 启动：`ce6214090ea635835fed4bee89dcb645a8570a06`。
- 本轮 History-aware 启动和报告提交：以本文件所在 Git commit 及交付回复为准，避免自引用 SHA。

## 7. 持续更新规则

每次收到服务器实验结果后更新本文件并提交 `vibe-dev`：追加设置、状态、关键指标、结论、失败、产物路径、checkpoint/source 身份、耗时和证据范围。保留旧结果及负结果；缺失字段写未知或 not_run，不填零，不把预期结果标 completed。

固定模型的兼容性评估、推荐模型微调后的验证、Future test 分开记录。后续两组若进行 Update TIGER 微调，应从同一 Base TIGER、使用各自新映射同步编码 history/target，并使用相同训练预算与 Update validation 选择规则；Future 仅用于最终评估，不用于权重、训练长度或 checkpoint 选择。本轮没有执行该阶段。
