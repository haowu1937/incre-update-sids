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
| Base/Plain 冻结 TIGER 的 Future ranking | completed，用户回传 | 相同 Update 部署目录；Plain 的 Recall@10 / NDCG@10 相对低 85.34% / 89.05% |
| Base/Plain Update TIGER 微调后的 Future ranking | completed，用户回传 | 各训练 2000 步、均选 step 500；Plain 的 Recall@10 / NDCG@10 相对低 19.47% / 16.13% |
| History-aware RQ-VAE + 同口径评估 | failed / 预算未完成，用户回传 | full SID 训练至少完成 400 步，随后 suffix 容量耗尽；最终 anchor、排名与 Future 未运行 |

当前结论：

1. Plain Update 在累计目录上的重构 MSE 下降 23.68%，但旧物品三层语义码变化率达到 98.96%。重构改善不能自动代表与旧推荐模型兼容。
2. 固定旧 target 和冻结旧 TIGER 后，Plain 的 anchor_holdout 平均序列 ΔNLL 为 +0.235389，用户聚类 bootstrap 95% 区间为 [0.207473, 0.264948]；正部损伤 H 为 0.331016。本轮观察到了历史输入重编码损伤。
3. Base validation 的 Recall@10 从 1.936466% 降至 1.533943%（相对下降 20.79%），NDCG@10 从 0.01079385 降至 0.00779528（相对下降 27.78%）。这是冻结 TIGER、旧 target 地址下的兼容性排名结果。
4. 保留旧 history suffix 后，anchor_holdout 平均 ΔNLL 仍为 +0.189471，H 为 0.283439，损伤仍存在。该条件是输入诊断，不是可部署的完整 SID 映射；不能将两个条件的差直接当作可加的 suffix 因果贡献。
5. 本轮 History-aware 以 1000 步、λ=0.001 启动，至少完成 400 步后因新版本 suffix 容量耗尽中止，未形成完整 anchor/holdout 评估。该异常证明当时完整 SID 分配不可行，不足以证明码本完全坍缩；收益尚未验证。1000 步仍是配对预算，不是已证明收敛或最优的上限。
6. 四组 Future 结果已回传：相同覆盖率 80.555149% 下，Plain 冻结 TIGER 的 Recall@10 为 0.099447%，低于 Base 的 0.678581%；配对微调后为 0.979847%，仍低于 Base 的 1.216766%。微调缩小了差距，本轮尚未观察到 Plain 码本更新带来的 Future 排名收益。详见第 5 节，不与固定旧 target 的历史诊断混用。

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

## 5. 已完成：Base/Plain 的两种 Future evaluation

2026-10-19 收到服务器 compact：execution_status=completed。冻结 TIGER 直接测试和 Update 微调后测试四种条件均完成；入口为 `evaluate_incremental_future.py`，启动脚本为 `scripts/run_llama_future.sh`。以下数字来自同一次服务器 run，不是云端模型重算。

### 5.1 对照与时间边界

- 两个 tokenizer arm：`base` 保持 Base RQ-VAE 权重冻结，只为已观察的 Update 新物品追加 SID；`plain` 使用已完成的 1000 步 Plain RQ-VAE。RQ-VAE 和 Llama 均不再训练。
- Base control 会扩展到相同的 Update 部署目录，不能把这行结果称作原始 11179-item Base 部署快照。这样可在相同候选物品和覆盖率下比较 tokenizer 更新的影响。
- Update train 和 Update validation 使用 11707-item 训练目录。仅 Future 使用 11775-item 部署目录，包含 Update validation 首见的 68 项；它们不参与 TIGER 参数训练或 validation ranking。
- Base 扩展保留已有完整 SID；Plain 扩展复用版本化分配器，预留旧 Base 地址，再为新增物品分配 suffix。不会复用已经离开的旧地址，且不会重编号 11707 项已测映射。
- 此处 history 和 target 同时使用本组当前完整 SID，候选目录也使用当前映射。这是端到端 Future 排名，不是第 4 节固定旧 target 的 history-only 诊断。
- Future 保持协议中的 prequential 历史：可使用严格早于 target 的已发生事件，但 future-only 物品保持目录外，其历史位置为四个 PAD、attention=0。模型、映射、目录不会随 Future 事件更新。
- 实际 Future 有 34189 条 queries，可服务 27541 条、目录外 target 为 6648，覆盖率 80.555149%；四种条件完全相同。目录外 target 在完整分母中计作 misses，可服务分母指标另报。

### 5.2 TIGER 微调和 checkpoint 选择

- 两组均从已选中的同一个 Base TIGER（Base step 4500）开始；分别使用各自 RQ 映射同步编码 Update history 与 target。新建 AdamW，不继承 Base TIGER optimizer 动量。
- 预设每组最多 2000 个 TIGER Update 步，实际两组均完成 2000 步，batch_size=128、lr=0.003、weight_decay=0.05、warmup=200、constant_with_warmup、max_grad_norm=1、seed=2026。该预算沿用原 E3 计划，不根据 Future 调整。
- 在步数 [0,100,250,500,1000,2000] 上评估固定 512-query Update validation probe；按 full-denominator NDCG@10 选 checkpoint，平局保留最早一步。Step 0 也可被选中，表示微调没有在该选择指标上改善。
- 两组选中模型另行评估完整 Update validation。对实际 checkpoint 的 optimizer 参数组、sampler order/cursor/generator、处理样本数和训练预算进行配对核对。
- 只有两组均已选定并写入 `selection_locked.json`，才解析 Future 样本并运行 Future 推理。Future 不用于选择 lr、预算、tokenizer、权重或 checkpoint。
- 冻结条件直接加载 Base TIGER；微调条件加载各组 Update validation 选中的 TIGER。四种条件统一 beam=20，Recall/NDCG@5/@10/@20，seen_item_filter=false。

实际配对检查为 passed：两组从相同 Base TIGER 权重开始，采用相同 optimizer recipe、Update training queries、步数和采样器终点，各处理 255919 个 examples；warm-start 与 selected checkpoint reload 一致性均为 true。两组选中的 Update 步数均为 500，而不是最后的 2000 步。该 2000 步属于推荐模型微调预算，与 RQ-VAE 的 1000 步 Update 预算不同。

### 5.3 实际 Future ranking

主表使用完整分母 34189，将 6648 个不可服务 target 记作 misses。Recall 为百分比，NDCG 为原始数值；四组均使用相同 11775-item Update 部署目录、beam=20、seen_item_filter=false，invalid_predictions=0。Base 行表示冻结 Base RQ-VAE 并扩展到已观察 Update 目录，不是原始 Base 部署快照。

| Tokenizer | TIGER 状态 | 选中 Update 步数 | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RQ 冻结并扩展目录 | 冻结 Base TIGER | 0 | 0.476762% | 0.678581% | 0.933049% | 0.003373898224 | 0.004021264440 | 0.004668545582 |
| Plain RQ 更新 1000 步 | 冻结 Base TIGER | 0 | 0.058498% | 0.099447% | 0.204744% | 0.000304899962 | 0.000440264698 | 0.000694146659 |
| Base RQ 冻结并扩展目录 | Update 微调后选中 | 500 | 0.699055% | 1.216766% | 1.813449% | 0.004427404386 | 0.006088073259 | 0.007579371337 |
| Plain RQ 更新 1000 步 | Update 微调后选中 | 500 | 0.596683% | 0.979847% | 1.798824% | 0.003873351250 | 0.005105906826 | 0.007118554090 |

仅对相同的 27541 条可服务 queries 计算：

| Tokenizer | TIGER 状态 | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RQ 冻结并扩展目录 | 冻结 Base TIGER | 0.591845% | 0.842380% | 1.158273% | 0.004188308572 | 0.004991939651 | 0.005795465121 |
| Plain RQ 更新 1000 步 | 冻结 Base TIGER | 0.072619% | 0.123452% | 0.254167% | 0.000378498413 | 0.000546538244 | 0.000861703646 |
| Base RQ 冻结并扩展目录 | Update 微调后选中 | 0.867797% | 1.510475% | 2.251189% | 0.005496115920 | 0.007557646297 | 0.009408922212 |
| Plain RQ 更新 1000 步 | Update 微调后选中 | 0.740714% | 1.216368% | 2.233034% | 0.004808322352 | 0.006338399058 | 0.008836870331 |

选中 checkpoint 的完整 Update validation 共 3280 条 queries，其中 3084 条可服务、196 条不可服务。下表 ranking 指标使用完整分母；probe NDCG@10 来自固定 512 条、可服务 483 条的选择集，不与完整验证指标混用。

| Tokenizer | 选中 Update 步数 | Probe NDCG@10 | 完整验证 Recall@10 | 完整验证 Recall@20 | 完整验证 NDCG@10 | 完整验证 NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RQ 冻结并扩展目录 | 500 | 0.011866984365 | 2.896341% | 4.176829% | 0.012938729610 | 0.016154641217 |
| Plain RQ 更新 1000 步 | 500 | 0.011273458891 | 1.890244% | 4.268293% | 0.009451011768 | 0.015336322498 |

完整 Update validation 的可服务分母下，Base / Plain 的 Recall@10 为 3.080415% / 2.010376%，NDCG@10 为 0.013761035383 / 0.010051659728。

服务器保存逐 query 指标和 NLL、old/new target 组别及低频切片；本次 compact 排除了 slices，云端未读取这些明细，故切片结论仍待回传。组别按 Base train / Base validation 首见 / Update train 首见 / Update validation 首见 / Future-only 区分；低频使用 Update train 可形成因果样本的 target 次数、阈值 5，不使用 Future 频次。

### 5.4 关键结论、成本与证据范围

1. 冻结 TIGER 时，Plain 相对 Base 的 Future Recall@10 下降 85.34%（0.678581% → 0.099447%，命中 232 → 34），NDCG@10 下降 89.05%（0.004021264440 → 0.000440264698）。这是 history 和 target 同时重编码后的端到端不兼容表现，不能把全部下降归因于第 4 节的 history-only 损伤。
2. 两组均经 Update 微调且均选择 step 500 后，Plain 的 Recall@10 仍比 Base 低 19.47%（1.216766% → 0.979847%，命中 416 → 335），NDCG@10 低 16.13%（0.006088073259 → 0.005105906826）。微调明显缩小了差距，但本次配置下 Plain 没有超过冻结 Base RQ 的配对微调对照。
3. 微调后的 Recall@20 仅相对低 0.81%，NDCG@20 低 6.08%；差距随 k 变化，不能把 @10 的幅度推广到所有截断位置。四组覆盖率完全一致，观察到的排名差不是不同 target 可服务数量造成的。
4. 两组各自微调后的 Future 排名均高于各自冻结 TIGER；不能只用 Plain 微调后优于 Base 冻结 TIGER，就判断码本更新有收益。码本更新的主比较应匹配 TIGER 状态与预算。
5. 本轮只有固定 seed=2026 的结果，没有 Future 排名差的用户聚类置信区间或多 seed 结果，不声称统计显著或普遍优劣。两组从同一旧 TIGER 开始，结果针对 warm-start 兼容性与预设微调预算，不证明新 SID 在充分重训练后质量更差。History-aware 收益仍未验证；不根据已查看的 Future 修改 lambda、预算或 checkpoint 选择。

诊断 NLL 与耗时如下。Future 两组 target 地址随映射变化，跨映射 NLL 不作为推荐质量或第 4 节历史损伤的替代指标。微调模型的推理权重也保持冻结；teacher_frozen_during_evaluation=true 不代表未进行前面的 Update 微调。

| Tokenizer | TIGER 状态 | Future 可服务序列 NLL | Future 解码耗时（秒） |
| --- | --- | ---: | ---: |
| Base RQ 冻结并扩展目录 | 冻结 Base TIGER | 11.993350524562 | 881.079 |
| Plain RQ 更新 1000 步 | 冻结 Base TIGER | 20.501776810613 | 930.938 |
| Base RQ 冻结并扩展目录 | Update 微调后选中 | 10.462548662853 | 884.338 |
| Plain RQ 更新 1000 步 | Update 微调后选中 | 11.811874873376 | 892.202 |

两组 TIGER 微调及验证阶段分别耗时 409.705 秒 / 426.078 秒；Future 流水线总耗时 4456.991 秒（74.28 分钟），包括两组微调/选择、验证、四轮 Future 评估和相关流水线开销。CUDA peak allocated=3720788992 bytes，reserved=4506779648 bytes。总耗时不是单个 arm 的更新时间，也不包含先前 Base/Plain RQ 训练及 Llama 编码；本轮没有 equal-total-compute 对照。

服务器 compact 报告：pair_checks=passed、future_used_for_selection=false、selection_locked_before_future=true、original_checkpoints_unchanged=true。云端复核了 compact 内计数、覆盖率、完整/可服务分母换算、Plain−Base 差值和相对变化；没有读取服务器 checkpoint 或逐 query 文件独立重算指标。

### 5.5 工程检查

以下保留代码交付阶段的检查记录，区别于上面的真实服务器结果。

- 云端新增 8 项协议检查通过：时序/同时刻排除、样本去重、registry 前缀、可用性 mask、Future-only 不入目录、旧 suffix 槽位预留、完整/可服务分母和切片。
- 已有 temporal round1 的 19 项检查通过；Bash 和新增 Python 语法检查通过。
- 6 项模拟启动检查通过：新建、显式 resume、检查失败停止、运行失败后保留并续跑、dirty workspace、缺失输入。模拟不加载真实 Torch 模型。
- 新增 4 项 real tiny TIGER 合成测试：四组真实模型训练/解码、带 dropout 的中断精确恢复且 Future 延后、resume 绑定和结果校验、Future 中断后复用已锁定模型与已完成预测。上游 E1/RQ context 使用合成 stub，TIGER/Trainer/指标计算使用真实实现。云端缺少 PyTorch，尚未执行这 4 项；服务器启动脚本先运行它们，通过后才开始 Beauty 流程。
- 新输出拒绝覆盖；显式 resume 绑定相同源码、数据、配置和预算，可复用已完成的训练/评估，并为每次尝试保留独立日志。原 Base/Plain RQ-VAE、TIGER checkpoint 保留并复核 hash。
- 本次服务器流程已越过启动前合成检查并完成四组 Beauty 评估；尚未单独回传 checks.log，不把云端未运行的 real tiny TIGER 检查冒称为云端通过。真实 Future 结果以第 5.3 节的服务器 compact 为证据；History-aware 的中止另记于第 6.2 节。

## 6. 配对 History-aware（full SID 训练中止，最终 anchor 评估未运行）

### 6.1 预设实验及启动流程

本轮明确使用已有标量历史惩罚入口 `train_historical_incremental_rqvae.py`，目标为 `RQ loss + 0.001 × H`。固定 λ=0.001、temperature=0.1、anchor minibatch=32，训练 history 条件显式指定 `new_full_sid`。这是本轮预设配置，尚未验证是最优权重；不根据已查看的 holdout 选择 λ。

现有 History-aware 启动脚本覆盖 RQ 更新、anchor 损伤和冻结 TIGER 的 Base validation 排名，不自动执行 History-aware 的四组 Future 对照；其同口径冻结/微调 Future 评估须后续接入，当前为 not_run。

从同一个 Base RQ-VAE fresh warm-start，独立训练累计 1000 Update 步，不在已训练的 Plain 终点上继续加惩罚。继承 Plain 的模型结构、内容 batch_size/lr/weight_decay/seed/优化器起点与 `Trainer.fit_steps` 内容采样实现。RQ-VAE encoder、decoder、三个码本保持可训练；历史 STE 梯度进入 encoder 与三个 semantic codebooks，suffix 无梯度。冻结 Llama 特征和 TIGER teacher，固定旧 target。仅 anchor_train 进入惩罚，holdout 与 validation 不参与训练。

调用 `scripts/run_llama_history_aware.sh` 时顺序为：

1. 检查 Plain E1 已完成且包含排名；预算不得超过已完成的 Plain 预算。
2. 在服务器运行已有真实模型合成检查 `test_historical_update_round12.py`，失败则停止。
3. 训练 History-aware，保存全新 run、日志、checkpoint、source/config/data binding。支持历史 checkpoint 精确 resume；不复用 Plain optimizer 状态。
4. 完成 1000 步后，核对实际 checkpoint 的内容配置、优化器参数组、初始内容指标、采样器终点及 Base/teacher/Plain 身份。
5. 调用 `evaluate_historical_update.py --ranking`，使用相同 cohorts、旧 labels、teacher、目录和 full denominator。报告 H、平均 ΔNLL、History-aware 相对 Plain 的配对 NLL 差与用户聚类区间、Recall/NDCG、重构和 churn。

等 Update 步数不等算力；历史惩罚有额外 forward/backward 和目录重编码成本，单独报告 runtime。配对核对的采样器证据是配置、实现与终点状态，不冒称独立保存了每步完整 batch trace。

小于 1000 步的 partial run 可保存和恢复，但不运行不同预算的最终配对排名。完整 SID 导出或配对核对失败时停止后续评估并保留日志，不扩充 suffix 容量或绕过校验。

交付阶段的云端检查：Bash 语法、3 个嵌入 Python 块语法及 8 项模拟启动场景通过，覆盖完整流程、partial/resume、模型检查失败、训练失败、采样器不一致、评估失败、预算/权重/Plain 排名预检和 dirty workspace/产物保留。模拟使用 fake torch 与真实 CLI parser；不等于真实 RQ-VAE/TIGER 或 GPU 检查。云端缺少 PyTorch，真实模型合成检查由服务器启动流程执行。本轮服务器训练中止见下节，不将交付检查当作实验完成。

### 6.2 实际执行：suffix 容量耗尽

2026-10-19 用户回传训练日志与 traceback：run=`llama_history_aware_20261019T155320Z_Yk6Nlc`，11707 个内容物品、4096 条 training anchors、λ=0.001、temperature=0.1、anchor batch=32、condition=`new_full_sid`，目标 Update 步数 1000。训练已越过启动检查与旧 anchor NLL 缓存；最后回传的周期日志为 step 400，随后在 `HistoryPenalty.input_embeds` → `versioned_suffixes` 抛出 `ValueError: New-version collision suffix capacity exhausted`。这次明确有训练异常，不能仅归为 SSH 断开或终端输出丢失。

以下为用户已贴出的训练 minibatch 里程碑。RQ、重构和量化损失来自当步内容 batch；H 来自当步 32 条 anchor。分配容量统计在该步参数更新前计算，不是同一步更新后的完整目录评估。

| 已完成步数 | 当步重构 MSE | 当步量化损失 | 当步 H | 加权历史项 λH | 最大当前碰撞桶 | 最大所需 suffix 槽位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.003124332521 | 0.009211765602 | 0.000000000000 | 0.000000000000 | 55 | 55 |
| 100 | 0.003842768725 | 0.012881316245 | 0.119086772203 | 0.000119086772 | 52 | 60 |
| 225 | 0.004056828097 | 0.064984902740 | 0.670326828957 | 0.000670326829 | 108 | 108 |
| 350 | 0.003523970023 | 0.032825231552 | 0.401989817619 | 0.000401989818 | 131 | 131 |
| 400 | 0.003486433532 | 0.029009586200 | 0.280607461929 | 0.000280607462 | 102 | 102 |

Step 400 的 processed_examples=101848，anchor_sampler_epoch=4，anchor_changed_history_fraction=1.0；合成梯度范数为 encoder=0.033068541437、decoder=0.003261050209、codebooks=0.007837447338。这些是总目标的梯度范数，没有分别记录 RQ/历史梯度，不能据其数值或 λH 占比认定具体梯度机制。

分配规则是对每个 semantic prefix 保留 Base 已占用地址，再为当前改变 prefix 的旧物品和新物品追加 suffix。因此所需槽位为该 prefix 的旧保留槽宽加当前迁入/新增数，可能大于当前物品桶大小。每次使用固定的 Base registry 重建，不是跨 optimizer steps 永久累积已用槽位。异常只能确认失败时至少一个 prefix 需要超过固定 256 个槽位；失败快照的实际桶大小、unique codes、逐层码本利用率未保存到所贴日志，不能将其记作“完全坍缩”。

`metrics.jsonl` 每个成功 optimizer step 均写入并 flush，终端每 25 步打印；精确的最后成功步和失败尝试步仍待服务器回传该文件末行。代码每 50 步保存 `last.pth`，按现有周期日志推断最近定期保存点应为 step 400；尚未读取该 checkpoint 核实，不将推断当作恢复状态证据。异常发生在当次 backward/optimizer.step 之前；本次路径直接退出，没有生成最终训练 summary、失败快照或成功回滚记录。

本次 1000 步训练、最终完整 SID 导出、最终配对检查、完整 anchor_train/anchor_holdout、Base validation 排名及 History-aware Future 均未完成。不能将当步 32-query H 与 Plain 的完整 4096-query holdout H 比较，也不能用未完成预算的轨迹判断相同步数收益。训练可行性和容量失败需作为负结果保留；目前既没有 History-aware 胜于 Plain 的证据，也没有完整配对评估证明其更差。

本轮仅补充失败记录，不改变 source-bound 训练实现、suffix 容量或已有映射，也不自动重跑、改 λ 或换成 old-suffix 条件。接下来先只读核对 `metrics.jsonl`、`last.pth` 和运行元数据；容量或优化策略的修复应作为单独说明的后续变体。

## 7. 产物与身份索引

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
| Base/Plain Future run | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL` |
| Future summary / compact | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/summary.json`；`results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/compact.txt` |
| TIGER 微调与选中模型 | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/{base,plain}/last.pth`（均 step 2000）；同目录 `best.pth`（均 step 500） |
| Future 选择锁定与配对核对 | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/selection_locked.json`；同目录 `pair_checks.json` |
| 四组 Future 指标与逐 query 明细 | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/{base,plain}_{frozen,finetuned}.json`；对应 `*_queries.jsonl` 与 `*.sha256.json` |
| Future 启动检查和运行日志 | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/attempt_20261019T142547Z_IpXMsA/{checks,evaluate}.log` |
| Future 报告片段 | `results/incremental/Beauty/llama_future_20261019T142547Z_OQS0aL/evaluation/experiment_report_fragment.md` |
| History-aware 中止 run | `results/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc`；`train.log`、`paths.txt`、`commit.txt` |
| History-aware partial checkpoint / 逐步日志 | `ckpt/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc_1000/last.pth`、同目录 `metrics.jsonl`；实际保存步数待核实 |
| History-aware 最终评估（未运行） | `results/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc/evaluation` 为预设输出路径，不视作已生成有效结果 |

已回传 SHA256：

- Feature summary：`7680a9258ac379e7d8f1dc24a62048847b177597fb071f9092f1e6b9b3e91810`。
- Base embeddings：`40efd83af24e11a6d72328da24dab9d8652d10b1634f358796ed3c08355add85`。
- Base RQ-VAE checkpoint：`23d4163c8cb458d7830cdca66900a2e15727d33456fc530b6b026c856fceab55`。
- Base SID summary：`2815fe097446687f9c23fd364450d283c899e36ebb59301b9f7dcc60a303977b`。
- Selected Base TIGER：`83308fa0bb71f398de22f6a3c5652dbb94da323a4bffcd9cc1cae584e03438fa`。
- Plain Update 1000 checkpoint：`1a37ad18ef1fa196416dd7bb373cfa86c3ee169e302dbe7b6e7d4a9024ad62cb`。
- Future Base 微调选中 TIGER（Update step 500）：`668c9b475103b512779f0c4b659d1662fb23b07b4f1ca49207c8b53e455ca732`。
- Future Plain 微调选中 TIGER（Update step 500）：`c255c61e4ecef797291c1febf9be869780b2fcb425965f11c7cece3abe500567`。
- 本次回传附件 `粘贴的文本 (1)(20261019-154219).txt`（compact 与路径文本）SHA256：`f5a8e9c13cf8b3fc3392d4d1b84314a71afa6201f4656b05d9cf68f16b9fc98d`；不是服务器 summary.json 的 hash。

代码交付索引（服务器 run 另存 `commit.txt`；未独立读取该文件，不将交付 SHA 冒称重新核验的实际执行 SHA）：

- Base 3-hidden/e64：`2531623ee02dc62e4de47a2f282a93ee958c65a8`；Base resume：`71984e9d9be8e1cfbaf7895cf176da8515fd732b`。
- Base TIGER 启动：`ad57e414e5ca52a3412275e07d7067509cfc4154`。
- Plain Update 启动：`039d09cc67f815bb4ee82f4fc25d3546982780f4`。
- Plain E1 启动：`ce6214090ea635835fed4bee89dcb645a8570a06`。
- 首次持续实验报告与 History-aware 启动：`271e7b9071106fdfd52666c2caa0199d8f6dab18`。
- 两种 Future 评估交付版本：`35d135b4273adf5203e42ec9100f706563b4daa1`；本次 compact 未包含服务器 commit.txt，不把交付 SHA 当作独立核验的实际运行 SHA。
- 本轮 History-aware 启动和报告提交：以本文件所在 Git commit 及交付回复为准，避免自引用 SHA。

## 8. 持续更新规则

每次收到服务器实验结果后更新本文件并提交 `vibe-dev`：追加设置、状态、关键指标、结论、失败、产物路径、checkpoint/source 身份、耗时和证据范围。保留旧结果及负结果；缺失字段写未知或 not_run，不填零，不把预期结果标 completed。

固定模型的兼容性评估、推荐模型微调后的验证、Future test 分开记录。第 5 节的两种 Future 流程及四组实际结果已回传并记录；History-aware full SID 训练已中止，最终 anchor、排名和 Future 结果仍缺失。Future 仅用于最终评估，不用于权重、训练长度或 checkpoint 选择。
