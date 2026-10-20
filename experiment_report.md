# History-aware SID Update 实验报告

更新日期：2026-10-20。当前主线：Amazon Beauty / 冻结 Llama7B 内容特征 / 纯内容 RQ-VAE / TIGER。

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
| History-aware RQ-VAE + 同口径评估 | failed / 预算未完成，用户回传 | 完成 418 步，第 419 次更新分配时 suffix 容量耗尽；磁盘 checkpoint 为 step 400，最终 anchor、排名与 Future 未运行 |
| History-aware 容量失败受限重放 | completed / overflow_reproduced，用户回传 | 401—418 日志观测值匹配；第 419 次分配的桶 [13,26,8] 含 266 项，超过 256 容量；该桶无旧槽位预留 |
| 调参后配对 Plain / History-aware 主实验 | code delivered / server not_run | 两组均从同一 Base 重新开始；lr=1e-4、裁剪 0.1，历史组 λ=1e-4、temperature=0.5、anchor batch=128；各 1000 步，见第 6.4 节 |

当前结论：

1. Plain Update 在累计目录上的重构 MSE 下降 23.68%，但旧物品三层语义码变化率达到 98.96%。重构改善不能自动代表与旧推荐模型兼容。
2. 固定旧 target 和冻结旧 TIGER 后，Plain 的 anchor_holdout 平均序列 ΔNLL 为 +0.235389，用户聚类 bootstrap 95% 区间为 [0.207473, 0.264948]；正部损伤 H 为 0.331016。本轮观察到了历史输入重编码损伤。
3. Base validation 的 Recall@10 从 1.936466% 降至 1.533943%（相对下降 20.79%），NDCG@10 从 0.01079385 降至 0.00779528（相对下降 27.78%）。这是冻结 TIGER、旧 target 地址下的兼容性排名结果。
4. 保留旧 history suffix 后，anchor_holdout 平均 ΔNLL 仍为 +0.189471，H 为 0.283439，损伤仍存在。该条件是输入诊断，不是可部署的完整 SID 映射；不能将两个条件的差直接当作可加的 suffix 因果贡献。
5. 本轮 History-aware 以 1000 步、λ=0.001 启动，完成 418 步后在第 419 次更新的分配阶段中止；checkpoint 只保存到 step 400，完整 anchor/holdout 尚未评估。受限重放已复现：语义码 [13,26,8] 当前聚集 266 项，超过 suffix 容量 256，且该桶旧保留槽宽为 0。这次是当前碰撞桶本身超限，并非旧槽位预留挤占；仍有 5981 个不同三层语义码，不能称为完全坍缩。具体原因及证据边界见第 6.3 节；收益尚未验证，1000 步仍是配对预算，不是已证明收敛或最优的上限。
6. 四组 Future 结果已回传：相同覆盖率 80.555149% 下，Plain 冻结 TIGER 的 Recall@10 为 0.099447%，低于 Base 的 0.678581%；配对微调后为 0.979847%，仍低于 Base 的 1.216766%。微调缩小了差距，本轮尚未观察到 Plain 码本更新带来的 Future 排名收益。详见第 5 节，不与固定旧 target 的历史诊断混用。
7. 针对局部集中，下一轮降低更新幅度、平滑历史 STE 并增加 anchor batch，同时重跑相同内容配置的 Plain，避免将降低学习率本身带来的稳定性误归于历史惩罚。参数由训练失败证据确定，未通过 holdout/Future 调参。新配置尚无服务器结果，不预设其能改善碰撞或兼容性。

## 2. 固定设置与评价口径

- 数据：Amazon Reviews 2014 Beauty 5-core，固定已有全局时间切分和样本，不重新划分。
- Base train catalog：10924；Base 部署目录：11179；Update train catalog：11707。
- Update 训练物品组：Base train 10924、Base validation 首见 255、Update train 首见 528。
- 特征：本地冻结 Llama7B，title / categories / description；最后层 hidden states，长度 1024，padding 清零后 sum/1024；输出 float32 / 4096 维，无额外归一化、投影或微调。
- RQ-VAE：4096 → 512 → 256 → 128 → 64，镜像 decoder，3×256 semantic codebooks。外层 quantization loss weight=0.25，无 BN、dropout、CF、graph 或 Sinkhorn。
- 已完成 Plain 和首次中止 History-aware 的 Update：AdamW，lr=0.001、weight_decay=0、batch_size=256、seed=2026；从同一 Base 全权重开始，Update 起点重置优化器，不重新初始化码本。第 6.4 节新配对实验使用不同学习率和裁剪，单独记录。冻结的是 Llama 内容模型，不是 RQ-VAE encoder。
- 完整 SID：3 个 semantic digits + 固定容量 256 的 suffix；token offsets=[1,257,513,769]，PAD=0、EOS=1025、vocab_size=1026。新版本预留旧 slots、保留未移动物品的 suffix，其他物品按 registry 顺序追加。
- Base TIGER：随机初始化的 repository T5，4 encoder / 4 decoder layers，d_model=128、d_ff=1024、num_heads=6、d_kv=64、dropout=0.1；batch=128、lr=0.003、weight_decay=0.05、warmup=200、seed=2026。
- TIGER 最多训练 5000 步，按固定 512-query probe 的 full-denominator NDCG@10 选择 checkpoint；最终选中 step 4500。最终表格使用完整 Base validation，不用 probe 代替。
- Ranking：Base train 10924 个旧地址物品，catalog-prefix beam search，beam=20，seen_item_filter=false。Base validation 共 9192 queries，其中 8689 个旧 target 可服务，503 个不可服务；主表将 503 个计作 misses。
- History-only：teacher 参数冻结，history 重编码，target 完整旧 SID 与 EOS、teacher-forced target prefix、可用性 mask 和位置不变。原来不可用的 history 保持 PAD，不因 Update 已观察到物品而补入。
- NLL 为 5 个 target tokens 的序列损失之和。ΔNLL=new−old；H=mean(max(ΔNLL,0))。H 不允许预测增益抵消损伤；平均 ΔNLL 允许抵消。token CE 的差等于序列 ΔNLL/5。
- Anchor：seed=2026 的 user-hash 分组，各取 4096 queries；train=2648 users、holdout=2613 users，用户不重叠。holdout 仅对历史惩罚训练留出，并非对 Base TIGER 训练留出；此 cohort 已在旧探索中查看，不称为新的盲测。
- 已回传结果的 95% 区间：500 次用户聚类 bootstrap，重采样后以 query 加权计算平均 ΔNLL；这些区间不属于 H 或排名差。第 6.4 节新入口另外计算配对 H 差的专属区间，不能与平均 NLL 差区间混用。

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
| 418 | 0.005391825922 | 0.045405112207 | 0.398684948683 | 0.000398684949 | 228 | 228 |

Step 400 的 processed_examples=101848，anchor_sampler_epoch=4，anchor_changed_history_fraction=1.0；合成梯度范数为 encoder=0.033068541437、decoder=0.003261050209、codebooks=0.007837447338。这些是总目标的梯度范数，没有分别记录 RQ/历史梯度，不能据其数值或 λH 占比认定具体梯度机制。

分配规则是对每个 semantic prefix 保留 Base 已占用地址，再为当前改变 prefix 的旧物品和新物品追加 suffix。因此所需槽位为该 prefix 的旧保留槽宽加当前迁入/新增数，可能大于当前物品桶大小。每次使用固定的 Base registry 重建，不是跨 optimizer steps 永久累积已用槽位。原 traceback 仅能确认失败时至少一个 prefix 需要超过固定 256 个槽位；当时所贴日志未包含失败参数的桶大小、unique codes 或逐层码本利用率。后续受限重放补齐这些统计，确定本次溢出桶没有旧槽位预留，见第 6.3 节。

2026-10-20 用户回传只读检查结果：run_commit=`35d135b4273adf5203e42ec9100f706563b4daa1`；最后成功记录为 step 418、epoch=10、processed_examples=106387，total_loss=0.017141789198、rq_loss=0.016743104249、平均序列 ΔNLL=+0.264655411243、H=0.398684948683，anchor_changed_history_fraction=1.0、anchor_sampler_epoch=4。总目标梯度范数为 encoder=0.037496332079、decoder=0.006189642940、codebooks=0.001894956222。

`metrics.jsonl` 每个成功 optimizer step 均写入并 flush，终端每 25 步打印。该末行加此前 traceback 确认第 419 次更新在分配阶段失败，尚未执行其 backward/optimizer.step。Step 418 的容量 228 是该次参数更新前（参数已经完成 417 次更新）的统计；完成第 418 次更新后的实际分配不可行，不应把日志中的 feasible=true 当作更新后仍可行。

用户读取的 `last.pth` 为 step 400、epoch=9、processed_examples=101848、cursor=8192；未保存的最后 18 次更新不能直接从该 checkpoint 读取。compact、training_summary、evaluation_summary 均不存在。该路径直接退出，没有生成失败快照或成功回滚记录；不能把 step 400 checkpoint 当作失败时参数。

本次 1000 步训练、最终完整 SID 导出、最终配对检查、完整 anchor_train/anchor_holdout、Base validation 排名及 History-aware Future 均未完成。不能将当步 32-query H 与 Plain 的完整 4096-query holdout H 比较，也不能用未完成预算的轨迹判断相同步数收益。训练可行性和容量失败需作为负结果保留；目前既没有 History-aware 胜于 Plain 的证据，也没有完整配对评估证明其更差。

### 6.3 受限失败诊断（completed，用户回传 overflow_reproduced）

新增独立 `diagnose_historical_capacity.py` 与 `scripts/run_llama_capacity_diagnostic.sh FAILED_RUN_DIR`。仅在新输出目录中恢复 step 400 的模型、optimizer、内容/anchor sampler 和 RNG，按原配置重放 401—418，再检查第 419 次分配；不会续跑原 1000 步实验或写入新的训练 checkpoint。原 engine、原 allocator、suffix 容量和 source/data/environment 绑定不变，不改 λ、不切换 old-suffix 条件。

逐步比较原日志全部字段：整数和布尔值精确一致，浮点 relative_tolerance=1e-6、absolute_tolerance=1e-7；任一不一致即停止。相符仅证明已保存的观测值在该容差内相符，没有每步权重参考，不能称为逐 bit 轨迹重现。重放上限硬限制为 50 次尝试；本 run 预期为 18 次已完成更新加第 419 次失败尝试。

诊断保存 step 400、最后匹配的更新前状态及失败尝试的 assignment 统计：unique semantic codes、各层码本使用数、最大前缀桶/完整桶、最大桶占比、是否所有物品共享同一个三层语义码，以及各溢出 prefix 的当前桶数、Base 保留槽宽、迁入数量和所需容量。原输入文件与原 checkpoint 在前后复核 hash。只有在原记录 401—418 均匹配且第 419 次由原 allocator 抛出容量异常时，outcome 才为 `overflow_reproduced`；未重现、提前失败、日志不符或输入变化分别记录，不作原现场证据。

本地 7 项纯协议检查、Python/Bash 语法与 diff 检查通过，覆盖重放步数上限、日志连续性/数值偏差、完整与部分分配集中、非法码、输入目录保护和路径文本解析。另提供 2 项真实 RQ-VAE/TIGER 小模型重放检查：从中间模型/optimizer/两套 sampler/RNG 恢复并匹配原日志，以及首个日志偏差时停止并保持输入文件不变。云端没有 Torch，这两项未执行；服务器启动脚本要求它们通过后才运行实际诊断。诊断本身不使用 holdout/Future，不构成新的优化变体或最终 anchor 评价。

2026-10-20 用户回传实际诊断 compact：`execution_status=completed`、`outcome=overflow_reproduced`；输出为 `results/incremental/Beauty/llama_capacity_diag_20261020T090834Z_Zic7HT`。从磁盘 step 400 恢复，401—418 共 18 个成功步骤的已记录观测值均在上述容差内匹配；在已完成 418 次 optimizer 更新的参数上，第 419 次尝试由原 allocator 再次抛出同一容量异常。第 419 次没有执行 optimizer 更新。

| 参数已完成 optimizer 步数 | 诊断分配尝试 | 不同三层语义码数 | 最大当前完整语义桶 | 最大所需 suffix 容量 | 三层码本使用数 / 各层容量 256 | 各长度最大前缀桶（L1 / L1–L2 / L1–L3） | 完整 SID 可分配 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 400（磁盘 checkpoint） | 401 | 7366 | 118 | 118 | 101 / 176 / 228 | 1409 / 250 / 118 | true |
| 417（最后匹配尝试的更新前） | 418 | 5980 | 228 | 228 | 92 / 168 / 214 | 2050 / 566 / 228 | true |
| 418（失败尝试的更新前） | 419 | 5981 | 266 | 266 | 94 / 175 / 216 | 2054 / 613 / 266 | false |

唯一回传的溢出桶为三层语义码 `[13,26,8]`：current_bucket_items=266、reserved_old_slots=0、reserved_slot_width=0、appended_items=266、required_capacity=266，固定 suffix_capacity=256，超出 10 个槽位。因此，本次具体触发原因是当前桶本身已超过容量，不是 Base 旧槽位预留挤占。第 6.2 节的预留规则仍是分配器的一般规则，但不能用作这次溢出桶的具体解释。

失败参数下，最大完整语义桶占目录 266/11707=2.272145%，三层码本仍分别使用 94、175、216 个 codeword；`all_items_share_one_full_semantic_code=false`。应描述为局部过度集中导致完整 SID 不可分配，不能称为“所有物品的三层语义码完全坍缩”。相对 step 400，最大完整桶由 118 增至 266，不同完整语义码由 7366 降至 5981；但是 step 417→418 的 unique codes 从 5980 略增至 5981，同时最大桶继续扩大，说明单看 unique codes 或逐层利用率不足以保证每桶容量可行。现有证据定位了分配失败的直接原因，没有隔离历史梯度、内容梯度、学习率或 STE 对集中趋势的因果贡献，也没有同一步 Plain 的配对集中度诊断。

原 run 未被续跑（original_run_continued=false），没有写入新训练 checkpoint（checkpoint_written=false），受监测输入前后 hash 不变（observed_inputs_unchanged=true）；holdout_evaluated=false、future_evaluated=false。重放中的 418 步参数是诊断内存状态，不是新的磁盘训练 checkpoint；磁盘保存点仍为 400。诊断耗时 35.141653 秒，这是受限重放耗时，不是原训练总耗时。回传 diagnostic_source_sha256=`a3a5098769fe37dc9390250755367286e7f73e1d81f85fb4c87055adf4ab6646`，与交付诊断文件内容一致；原 run commit 仍为 `35d135b4273adf5203e42ec9100f706563b4daa1`。云端核对了 compact、源码语义和计数，没有读取服务器权重、attempts.jsonl 或逐步日志独立重放；日志观测值相符不等于逐 bit 权重相符。

本轮只完成失败诊断与报告更新，没有改变容量、offsets、allocator、λ、优化器或训练过程，没有完成修复。后续修复应显式处理参数更新后的容量可行性与失败状态保全；若引入回滚、减小步长重试或其他约束，应单独标为新优化变体并记录成本，不能将其冒称为原 1000 步配对实验完成。本诊断没有新增完整 anchor/holdout、Base validation 或 History-aware Future 结果。


### 6.4 调参后的配对主实验（代码已交付，服务器 not_run）

用户要求分析不健康碰撞并调整参数重跑。本轮直接证据是 step 400→418 最大完整语义桶由 118 增至 266，unique codes 从 7366 降至 5981；最后一步 unique codes 略增却仍溢出。固定容量下的尾部集中已破坏完整 SID 可分配性，逐层码本利用率正常不能排除此问题。

源码给出的待验证机制如下，尚未分别隔离因果贡献：历史 STE 的软分配为 softmax(-distance/(Base 固定尺度 × temperature))，经残差同时向 encoder 和三个码本反传；encoder 更新会影响整个目录。temperature=0.1 的代理梯度可能对距离变化敏感，32-query anchor minibatch 可能带来较大波动。原目标只有重构、量化和冻结 teacher 的正部损伤，没有均衡分配或每桶容量约束。原 lr=1e-3 配合无裁剪的混合目标可能过于激进；原 Plain 在此学习率完成 1000 步，并不证明加入历史梯度后仍稳定。日志只有混合梯度，尚不能证明是其中某一项导致集中。

step 418 的 λH 约占总 loss 的 2.33%，这不限制其梯度影响；STE、残差传播和 Adam 按参数自适应缩放使标量占比不足以诊断更新大小。提高温度、降低 λ 和学习率均为待检验的稳定化选择，不能据此保证梯度或参数位移严格按比例下降。多项参数同时改变，本轮是配方重跑，不是逐项消融。

| 设置 | 首次失败 History-aware | 新 Plain 对照 | 新 History-aware |
| --- | ---: | ---: | ---: |
| 内容学习率 | 0.001 | 0.0001 | 0.0001 |
| 历史惩罚 λ | 0.001 | 0 | 0.0001 |
| STE temperature | 0.1 | 不应用历史梯度 | 0.5 |
| anchor minibatch | 32 | 无历史训练 | 128 |
| 全局梯度裁剪 max_norm | 无 | 0.1 | 0.1 |
| Update optimizer steps | 目标 1000，实际完成 418 | 固定 1000 | 固定 1000 |
| 内容 batch / weight_decay / seed | 256 / 0 / 2026 | 256 / 0 / 2026 | 256 / 0 / 2026 |

两个新分支均 fresh warm-start 同一 Base RQ-VAE `23d4163c…` 并重置 AdamW。不会从失败 step 400/418 或旧 Plain step 1000 接着训练；保持 3-hidden/e64、3×256 码本、RQ 损失和全部参数可训练。Llama 特征、Base teacher、完整 SID/256 suffix、旧 target、mask/位置与原 cohorts 保持原协议。只有 anchor_train 进入历史训练；训练长度固定，不按 holdout/validation/Future 选择 checkpoint。最终均使用 last step 1000。旧 lr=1e-3 Plain 保留为已有结果，新配对 Plain 是新主比较对象。

新增独立 `train_retuned_historical.py` 和 `scripts/run_llama_history_retuned.sh E1_DIR [EXISTING_RUN_TO_RESUME]`，不修改 E1 hash 绑定的原训练/分配源码。两组都在每次 optimizer.step 后对全部 11707 项检查原完整 SID 分配容量，写出更新后 unique codes、最大桶、每层使用数和容量审计；历史 H 和梯度指标明确属于更新前，避免重复原日志的时间点混淆。每 25 步记录 RQ 与未加权历史梯度的分组范数、加权历史/RQ 比值及夹角，并记录每步合成梯度的裁剪前范数和是否裁剪。

两组采用同一项新增保护：保存最近可行模型、optimizer、内容 sampler、anchor sampler 和 RNG；若更新后的容量不可行，保存独立 `rejected_candidate.pth`（明确不可导出/不可作为普通 resume checkpoint），回滚 `last.pth` 并停止。不自动调参重试、不扩充 suffix、不改用 old-suffix 目标。此保护不能防止所有不健康但尚未溢出的集中；完整碰撞轨迹仍需审核。若任一组不能完成预算，不执行最终配对评估，也不将较短轨迹标成主实验成功。正常中断可在同一 commit/config 下显式 resume；容量/训练错误保留失败结果，禁止在同一输出里自动重试。

两组完成后核对同一初始化、配置（仅 λ 不同）、optimizer 参数组和内容 sampler 终点；验证保存 checkpoint 重载后 assignments 相同。随后自动评估同一 frozen teacher：anchor_train、anchor_holdout、Base validation 的旧 NLL、新 full-SID NLL、平均 ΔNLL、H、重构、churn、碰撞与 Base validation 排名。old-suffix 仅单列为诊断条件。新入口对 Historical−新 Plain 的平均 NLL 差及正部损伤 H 差分别做用户聚类 bootstrap 区间；H 差区间不借用 NLL 区间。排名保留旧 Base 地址目录、full denominator、原 beam/seen-filter 口径并复核旧 Base 排名。额外的 TIGER Update 微调和 History-aware Future 仍为 not_run，不能从 anchor 结果推断 Future 收益。

产物在全新 `results/incremental/Beauty/llama_history_retuned_<timestamp>_<suffix>/experiment`，含 `request.json`、两组 checkpoint/summary、每步 metrics、`pair_checks.json`、evaluation 逐 query 明细、`summary.json` 和 `compact.json`。原 Base、旧 Plain、teacher checkpoint 与 E1 summary 前后核对 hash；原失败 run 不写入。后续 attempt 使用新日志目录，部分 evaluation 中断后在新 attempt 重算，保留旧文件。启动脚本允许已获用户说明的未跟踪 `verify_incremental_embeddings.py` 留在现场；其他工作区修改仍停止。

该配方有额外历史计算、梯度探针、全目录检查和状态快照成本，等步数不等算力；记录两组训练 loop 耗时、总耗时和总峰值 CUDA 内存，不声称两组总算力相同。云端 4 项纯协议检查、Python/Bash 语法通过；2 项真实模型集成检查尚未执行（云端无 PyTorch）：一项覆盖完整配对、固定 labels、排名及含 optimizer/双 sampler/RNG 的中断精确恢复；一项用合成审计失败验证更新后回滚、候选隔离与原输入不变。后者是保护机制测试，不是另一次真实坍缩证据。服务器启动脚本要求原真实模型检查与这两项新集成检查通过后才训练。当前没有新服务器训练/评估数字，状态 not_run；没有更新 HANDOFF。

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
| History-aware partial checkpoint / 逐步日志 | `ckpt/incremental/Beauty/llama_history_aware_20261019T155320Z_Yk6Nlc_1000/last.pth` 为 step 400；同目录 `metrics.jsonl` 最后成功步为 418 |
| History-aware 容量失败诊断 | `results/incremental/Beauty/llama_capacity_diag_20261020T090834Z_Zic7HT`；诊断输出 `summary.json`、`compact.json`、`attempts.jsonl`、`replay.jsonl`（本轮回传 compact 与目录，其他文件未独立读取） |
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
- History-aware 中止 run 的用户回传执行版本：`35d135b4273adf5203e42ec9100f706563b4daa1`；与报告/诊断代码的后续交付 commit 区分。
- 本轮 History-aware 启动和报告提交：以本文件所在 Git commit 及交付回复为准，避免自引用 SHA。

## 8. 持续更新规则

每次收到服务器实验结果后更新本文件并提交 `vibe-dev`：追加设置、状态、关键指标、结论、失败、产物路径、checkpoint/source 身份、耗时和证据范围。保留旧结果及负结果；缺失字段写未知或 not_run，不填零，不把预期结果标 completed。

固定模型的兼容性评估、推荐模型微调后的验证、Future test 分开记录。第 5 节的两种 Future 流程及四组实际结果已回传并记录；History-aware full SID 训练已中止，最终 anchor、排名和 Future 结果仍缺失。Future 仅用于最终评估，不用于权重、训练长度或 checkpoint 选择。
