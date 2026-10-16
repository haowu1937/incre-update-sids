# History-aware SID Update：研究规格

## 0. 当前状态与接续边界

更新日期：2026-10-16。当前研究仍处于 Amazon Beauty 可行性验证阶段。

最近完成的是固定 Base/teacher/cohorts、冻结 RQ-VAE encoder、每组 100 步的三 Update seed 权重扫描。`rho=[0.1,0.3,1,3,10]` 没有选中稳定改善的配置；15 组中 14 组最终训练历史损伤高于对应 Plain。新增权重没有进行 holdout 或排名评估。此结果支持当前实现/配置下的负结论，不证明研究思路整体无效。

接续工作的首要事项是核对迁移 commit 与最近成功服务器运行的源码差异、运行清单和证据，不自动扩大训练。本轮交接只整理文档。完整状态、指标及路径见 [HANDOFF.md](HANDOFF.md)。

## 1. 规格来源与代码基准

- 当前仓库：<https://github.com/haowu1937/incre-update-sids>。
- 指定迁移基准：`main`，`e1a4cf42f40a67fd750e80ae7c3d0e9549eb0188`。本次已通过 GitHub 读取该 commit；读取时 main 指向它。
- 上游仓库：<https://github.com/hirc-airecs/graph-informed-sids>。
- 原 prompt 审阅的上游快照：`master`，`5c2ca487f961e91e75e7d8f3ca78349faf3dc82e`。它不是历史服务器实验自动记录的 Git SHA。
- 已找回并完整读取 `agent_prompt_web(2).md`：55,760 bytes，SHA256 `cecf4ece02a5990b4178e514101ae7c8866fb5173c5f7414c7887f5f094a856a`。原 prompt 不在迁移 commit 中。本文保留其目标、方法与实验计划的整理版，不声称逐字替代原文。
- 原 prompt 提到的 `history_aware_sid_research_report.md` 本次未读取，其内容未知/待核实。

若文档与源码不一致，应保留差异并说明原因。不能把后来的实现选择自动当成原始规格，或把 planned 当成 completed。

## 2. 原始研究目标

增量 SID 更新应同时适应新目录/新行为信号，并保持与已有生成式推荐模型的兼容性。部署假设是：完整下游重训练成本较高，因此 tokenizer 更新后，下游推荐模型从旧 checkpoint 微调。这个假设不代表所有推荐系统的部署方式。

核心研究对象是历史输入重编码的影响：物品 A 的 SID 改变，不仅改变 A 作为 target 的地址，也可能改变 A 出现在历史中时对 B、C、D 的预测。SID churn 是观测指标，不能替代下游兼容性，也不应作为要求各方法相同的训练参数。

原始三个研究问题：

1. 固定 target 地址时，history 重编码是否独立造成有意义的预测损伤？
2. 该损伤能否成为 tokenizer 增量训练的有效信号？
3. 在下游微调后、计入新数据适应能力与额外计算成本后，收益是否仍然存在？

不要把推荐梯度、replay、STE 或 joint training 本身作为创新性结论。负结果必须保留，不能通过改协议使方法获胜。

## 3. 方法与损伤口径

### 3.1 原始方法

保留旧 tokenizer `C_old`、旧映射和冻结旧推荐模型 `M_old`。对同一个历史样本，比较：

- reference：旧 history SID，旧 target SID；
- history-only：当前 tokenizer 重编码 history，target 仍使用完整旧 SID。

原 prompt 定义每样本有效 target token 的平均交叉熵 `ell`，并使用：

\[
H=\frac1{|A|}\sum_{q\in A}\max(\ell_{new}(q)-\ell_{old}(q)-\delta,0),
\qquad L=L_{shared}+\lambda_H H.
\]

先逐样本取正部，再平均；其他样本的预测增益不能抵消受损样本。`delta` 必须使用相同损失单位。原方法初始 `delta=0`。

冻结 `M_old` 参数并设为 eval，但新 history 分支需要保留对输入 embedding 的计算图。旧 reference 可缓存并 detach。target、padding、EOS、用户 token、suffix 的身份与布局必须受控。

整数 argmin 后直接查 embedding 不会自动提供 SID 分配梯度。原计划使用距离 softmax 和 hard-forward/soft-backward STE，把 assignment 权重乘以旧 GR 模型对应层的固定 token embedding。RQ codeword 向量和 GR token embedding 属于不同空间，不能直接替换。

训练结束必须导出真实 hard mapping。真正的下游微调应让 history 和 target 同时使用新映射；new-history/old-target 仅用于隔离诊断及历史约束。

### 3.2 当前实现与原规格的差异

当前实验证据及源码使用完整 target 序列的 **NLL 求和**，标签长度为 5：3 个 semantic token、1 个 suffix、EOS。报告中：

- `mean_delta_sequence_nll`：每样本新旧序列 NLL 差的平均，允许正负抵消；
- `positive_damage` / `H`：每样本序列 NLL 差取正部后平均；
- `mean_delta_token_ce = mean_delta_sequence_nll / 5`。

在当前固定长度且 `delta=0` 时，序列口径与平均 token 口径仅差固定 5 倍；固定 lambda 的尺度相应不同。以后引入非零容差、变长 target 或切换口径时必须显式换算。该实现差异不是已核实的用户逐条批准。

最新实现使用逐层梯度混合：

\[
g_l=g_{RQ,l}+\alpha_{l,t}g_{H,l},\quad
\alpha_{l,t}=\operatorname{stopgrad}\left(\rho\frac{\|g_{RQ,l}\|}{\|g_{H,l}\|+10^{-12}}\right).
\]

任一梯度范数为零时，该层历史项缩放为零。当前历史梯度仅进入三层 codebook；encoder 输出在历史分支 detach。冻结 encoder 的实验中，内容目标只训练 decoder 和 codebooks。最新扫描放大的是 `rho`，不能当作原始固定 scalar lambda 扫描。

逐层动态缩放不等于同一个全局固定权重标量目标；原始梯度范数比例也不保证 Adam 参数更新比例或真实 hard H 的下降。当前 STE 前向使用完整新 SID，suffix 保持 hard/detached；距离尺度取自 Base 并冻结，temperature=0.1。

## 4. 数据与时间协议

### 4.1 数据身份与处理

默认数据集为 Amazon Reviews 2014 Beauty 5-core；本次文件含 198,502 条 review。它们作为隐式交互使用，不能称作点击。默认保留不同时间的重复物品交互；同 user/item/time/rating 的 exact duplicate 去重；实际本次未发现 exact duplicate；无 rating threshold。

已回传报告声明 `2014_beauty_5core`，但 `dataset_identity_verified=false`、`source_url_declared=null`。文件哈希已知不等于官方来源已验证。5-core 过滤使用完整时期，时间切分不能消除其选择偏差。更强的时间因果结论原计划需要 raw Beauty、仅依赖已观察前缀的 eligibility；该实验未运行。

内容字段白名单为 title、brand、categories、description。排除 review 文本、全期评分、sales rank、also-bought/also-viewed 和全期流行度。静态 metadata 缺少历史发布日期，须披露离线近似。目录可用时间以 first observed interaction 推断。

### 4.2 划分与训练边界

原计划按全局事件数约 60% Base、20% Update、20% Future 切分，Base/Update 各自最后约 10% 为 validation。边界只能位于时间戳边缘；实际比例偏差须记录。当前采用无 validation 后参数 refit；训练目录和部署目录分开。

| split | UTC 日期范围 | 事件数 | 有历史 examples |
| --- | --- | ---: | ---: |
| base_train | 2002-06-12—2013-08-01 | 107108 | 76252 |
| base_val | 2013-08-02—2013-09-27 | 11962 | 9192 |
| update_train | 2013-09-28—2014-02-22 | 35689 | 28023 |
| update_val | 2014-02-23—2014-03-06 | 3930 | 3280 |
| future_test | 2014-03-07—2014-07-23 | 39813 | 34189 |

每个 history event 的时间必须严格小于 target；同时刻交互互不进入 history。初始 max_history=50，至少一个先前交互；空历史排除数单列。显式 causal example 只能 tokenize 一次，不能再展开成所有前缀。legacy leave-one-out 仅作原路径保留。

| catalog | items | 用途 |
| --- | ---: | --- |
| base_train | 10924 | Base 训练及 Base validation ranking |
| base | 11179 | Base 部署目录，追加 Base validation first-seen 255 项 |
| update_train | 11707 | Update tokenizer 训练目录，再追加 528 项 |
| update | 11775 | Update 部署目录，追加 Update validation first-seen 68 项 |

追加已观察身份不代表可用 validation 拟合参数。禁止把 future-only 项加入训练、mapping 或检索目录。Future 可采用 prequential history：使用此前已发生的 future 交互作 history，但不更新参数、目录或映射；原计划另有 fixed-history sensitivity，均未形成当前 future 结果。

### 4.3 Anchor 与可服务性

原 prompt 要求 Base training 内按用户划分 replay 与独立历史诊断集，初始各至多 5000 条，先尝试 uniform-by-user。实际当前 cohort 为固定 seed=2026 的 user-hash 二分，再对组内 example 以 sample-hash 排序截取各 4096 条。它保证 train/holdout 用户不重叠，但不是严格每用户等权采样；评估平均按 query 加权。

当前 anchor_train 4096 queries/2648 users，anchor_holdout 4096/2613 users。历史参考缓存与旧映射、旧 checkpoint、样本身份绑定。不得用 holdout 训练或选择权重。该 holdout 已在多轮探索中查看，不能称作从未使用的新测试集。

Base validation 共 9192 条，其中 8689 个 target 可服务，503 个不可服务。实际 NLL/H 对 8689 条计算；ranking 同时报告 conditional-on-servable 与 full-denominator，后者把 503 条记作 misses。当前 seen-item filter=false。

不可用 history 的每个物品保留原位置的 4 个 PAD、attention=0，EOS 保持有效；不静默压缩序列位置。Update validation 不可服务 target=196，Future=6648，须单列覆盖率。

## 5. 模型、身份和导出约束

- 使用 `gensid.models.rqvae.RQVAE` 的可训练 codebooks；不要换成 TIGER 内部 EMA RQVAE，并误称其获得 codebook 梯度。
- 首轮已明确选择纯内容 embedding + RQ-VAE + TIGER；用户允许 TIGER/HSTU，实际使用 TIGER。graph-informed SID、LETTER、GRIS 是后续范围。
- sentence-t5-base 原生 768 维；不加 128 维投影，不额外归一化，不微调内容 encoder。实际 encoder 输出约为单位范数，不代表本代码另加归一化。
- RQ-VAE：768→[512,256,128,64]→32，3×256 semantic codebooks，dropout=0、BN=false、无 graph/CF/Sinkhorn；Base 初始化一次，加载后不得重新 k-means。
- 需区分外层 `quant_loss_weight=0.25` 与每层内部 commitment `mu=0.25`。当前量化项为各层 codebook MSE + mu×commitment MSE 的平均，再乘外层权重。
- 当前 native quantized-vector STE 的重构梯度不直接进入 codebook。不能把 frozen-encoder 下的优化解释成所有参数都接收重构梯度。
- 物品 raw ASIN、GR ID、gensid global ID、local row、graph node 必须显式对应；GR 0 是 PAD，gensid global ID=GR ID−1；不能用过滤后的 local row 替代 global ID。
- 3 个 semantic digit 加 1 个固定容量 suffix；容量均 256。token offset=[1,257,513,769]，PAD=0、EOS=1025、vocab=1026；n_user_tokens=0，input 最大 201 tokens，label=5。
- 禁止导出后 Sinkhorn 修复改变 semantic assignments。suffix 使用确定性、版本化、zero-based persistent slots；保留旧已占用位置，未变 prefix 保留 suffix，mover/newcomer 按 registry 顺序追加。不得 wrap、默默扩容或随机消歧。
- 原型更新中 suffix 失效时回滚 model/optimizer/sampler/RNG，记录停止；最后可导出 checkpoint 不等于该配置完成预算。
- TIGER 使用 repository T5，随机初始化，无 pretrained model/tokenizer 下载。当前 full SID prefix-constrained beam=20，报告 item-level @5/@10，invalid_predictions=0；不是 future @10/@20 主实验。
- graph 原计划为 observed adjacent co-occurrence、一次 APPNP、alpha=0.5，默认仅 update_train 窗口；旧节点孤立时 content fallback，cumulative graph 作敏感性。graph input 已准备，graph/APPNP 主实验未知/未获执行证据。

## 6. 原始实验计划与当前覆盖

| 阶段 | 原始要求 | 当前覆盖/缺口 |
| --- | --- | --- |
| E0 | 合成因果/身份/梯度/映射/恢复检查，再跑 Beauty old tokenizer+GR | 多轮合成检查与 Base 内容链路已回传；不能把 graph 完整 E0 自动标完成 |
| E1 | 隔离 history-only、target-only、joint、reference；共同旧物品；graph feature×parameter 2×2；Update LR 初试 1e-5/1e-4/1e-3 | 已有内容 Plain 1e-3/1000 步的 history-only 证据及 suffix 诊断；完整四组合、LR sweep、graph 2×2 未获执行证据 |
| E2 | Frozen deployment、refreshed graph frozen RQ、Naive、representation anchor、assignment KL anchor、History-aware | 已做 Plain/History-aware、修复及训练诊断；完整六方法对照未完成 |
| E3 | 新映射 history+target，从同一旧 GR 微调；固定步数及总时间比较 | 未运行 Update GR 微调、future ranking 或 equal-total-compute 比较 |
| E4 | 相对单侧损伤 vs replay NLL；replay 大小/近期采样；约束对照；重复更新 | 仅有若干机制诊断；系统性 E4 未完成 |
| E5 | 经论文/作者代码验证的 SID Alignment、DACT 等 | 未运行；相关工作讨论不等于复现 |

E0 原始必检项还包括：同映射 H≈0、整数/hard embedding logits 一致、只改 history 能影响固定 target、teacher 不变、活跃 penalty 的有限非零梯度、noncontiguous ID、`skip_n_last=0`、代码簿不重初始化、固定 offsets、metadata 不传入 T5，以及旧路径兼容。后续冻结 encoder 是明确实验变体，不能用它声称原始 encoder+codebooks 梯度检查全部完成。

E1 原计划的主切片为 target 完整旧地址不变的历史诊断，并补报全部 eligible old targets。当前主要回传的是固定 old labels 的全体旧 target 结果；该地址不变主切片的完整结果未知/待核实。target-only/joint 的 token loss 不可作加性因果分解。

原 E2 建议固定 lambda=[0,0.1,1,10]、delta=0，在 Update validation 调参；之后才试 delta=[0.01,0.05]、temperature=[0.1,0.5,1]。原计划 selected configs 用 seeds=[42,43,44]。当前固定 Base/cohort seed=2026、Update seeds=[2026,2027,2028]，不是全流水线三 seed，也不是上述原始 seed 方案已完成。

原 E3 上限初设 2000 GR steps，预定评估 [0,100,250,500,1000,2000]；主指标为可服务 future item Recall/NDCG@10/@20，并报覆盖率、old/new/long-tail 切片。部署 checkpoint 只在 Update validation 选择；future 曲线不能反向选择 checkpoint。需比较相同步数与相同总更新时间，并计入 tokenizer、history probe、重编码、trie、GR、调参成本。

原 E4 的 replay sizes=[1000,5000,10000]，视可用样本；多次更新比例为 Base50%、Update1 15%、Test1 5%、Update2 10%、Test2 5%、Update3 10%、Test3 5%，按时间戳边缘切分。每种方法延续自身部署状态；后来的观察只能用于之后的 deployment。

E5 的 per-layer alignment 必须同时处理 hard mapping、soft history 路径、token embeddings/targets、resolver；仅在导出做 permutation 的结果须独立标注。representation/assignment anchor 只是通用对照，不能叫 DACT 复现。外部论文的不同数据切分指标不能直接与本实验比较。

## 7. 后续明确确认的调整

用户在本会话明确确认：

1. 先跑纯内容路线，图与其他 SID 版本后续再加入；实际用 TIGER。
2. 跳过建议的热门商品 baseline，进入下一步；该 baseline 未运行。
3. 提供手动安装/运行命令；ZIP 始终放项目根目录，不是 Downloads。
4. 保留 2026，新增 2027/2028 做配对 Update seed 实验。
5. 权重逐步放大、多个 seed 并行；实际交付了有上限的 rho 扫描。
6. 当前迁移为 GitHub 分支/PR 交付，用户在 Linux GPU 服务器拉取与运行。
7. 本轮只做交接文档，不启动下一项实验。

当前扫描采用训练集预定筛选：连续两档 rho 在全部三个 seed 上 H 降低至少 1%，训练配对 user-bootstrap CI 上界<0，MSE≤对应 Plain×1.01，完成 100 步且 SID 有效；只评估选中权重的 holdout。最大 rho=10。该规则是本轮已交付并实际运行的实现规则，不是原 prompt 的 Update-validation 调参方案，也不是多重检验校正后的泛化证明。

## 8. 对照、证据与未决问题

对照必须配对相同 Base/teacher/特征/样本、catalog、内容 batch 顺序、step budget、optimizer、初始化、特殊 token 和 decoding budget。冻结 encoder 的 History-aware 必须比较冻结 encoder 的 Plain，不能直接归因于与早期可训练 encoder Plain 的差异。

每组保存原始指标、逐 query 差值、source/config/data/checkpoint/mapping/cohort hashes、实际 step/例数、失败、成本。跨 seed 报 paired difference 的均值与样本 SD（ddof=1），不能把相同固定数据上的多次结果当成更多独立用户。bootstrap 按 user cluster 重采样，保留 query-weighted 估计量。等 tokenizer 步数不代表等算力。

尚未确定/未验证：

- 官方数据来源验证、raw prefix-eligible 数据实验；
- 当前 STE、Adam 轨迹及离散 SID/suffix 变化各自造成负结果的份额；
- 原始 scalar history 目标是否能在其他优化设计下有效；
- target 地址不变切片、四种 E1 mapping 组合及 graph feature/parameter 分离；
- 泛化到不同 Base/teacher/cohort seed、其他数据集及 SID 版本；
- 下游微调后的收益、新数据适应程度、future ranking 和等总计算成本；
- 与 DIGER、SID Alignment、DACT 的创新性边界及忠实复现。

以上问题不能用当前负结果、合成梯度检查或单个 seed 的改善自动作答。
