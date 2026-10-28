# 项目协作规则

## Optimizer 验收阻塞：只读诊断（2026-10-28 当前继续）

`cf-letter-optimizer-001`训练wrapper已completed、报告5000步，五组无eligible且配对未跑；科学verifier在CPU full-monitor canonical codes逐项比较时失败，状态needs_diagnosis。保留原request/result/needs_attention receipt、全部checkpoint/索引/H记录，不重训、不改2% H阈值、不把浮点差异当作已确认原因。原7000 scope未完成科学验收，剩余2000不挪用。

当前只允许完成既有实验检查/故障诊断的零optimizer只读任务 `cf-letter-optimizer-audit-001` / `cf-letter-optimizer-audit-v1`，manifest `docs/experiments/cf_letter_optimizer_audit.json`。绑定原compact/index/code SHA，检查所有原文件hash前后不变和原durable5000计数；对170个monitor测CPU与保存codes、完整目录与old-item chunk差异，对涉差异点重载GPU并记录保存prefix条件下float64距离。仅生成独立新诊断输出，不改原产物、原索引、算法或验收标准。原验收必须保持blocked，此任务验收仅确认诊断证据采集，不冒充原实验通过。无optimizer、训练、推荐预测/holdout/Future或新参数实验。

沿用generic v1，0-step reservation；不升级Runner/helper/cron。当前本地PAUSED由原verifier故障触发，Git不能越过；先准备具体任务后仅对精确故障暂停做有证据归档，不清其他暂停或终止进程。GPU外部占用等待。提交英文报告须标明provisional/acceptance blocked，诊断回传后再判断数值或保存逻辑原因，不自动修改通过标准/重跑原实验。


## H-only 学习率／动量消融（2026-10-28 最新明确授权）

用户在审阅5组H-only＋条件性RQ-only/Mixed配对方案后明确“下发这个任务”。授权 `cf-letter-optimizer-v1` / `cf-letter-optimizer-001`，manifest=`docs/experiments/cf_letter_optimizer.json`，说明同名.md。新增预算最多7000 steps（5x1000筛选＋合格时2x1000配对），最多2并行；复用已验收旧H-only lr1e-4/beta1=.9，不重跑旧对照、不回收旧预算。矩阵lr=[1e-4,3e-5,1e-5] x beta1=[.9,0]除旧对照外5组；beta2=.999、eps1e-8、wd0、clip.1、temperature.5，固定原C0/E1/teacher、采样、STE/残差/原RQ梯度路径。

筛选只用最后8个scheduled完整anchor_train/RQ监测点：原joint稳定标准＋尾部平均H较初始下降>=5%，native RQ及hard MSE较初始各<=+2%，1000步完成、全程suffix可行。最低尾部H、再最低RQ、再声明顺序平局。无合格则停止；有合格才用其lr/beta1独立重启配对RQ-only/Mixed各1000，lambda固定旧值0.0006386606341359484，不扫描/重校准。原始E0 reference/target/mask/I0不变，四token无EOS。原每步/固定128每10步/全4096每10完整内容epoch监测和失败证据保留。

沿用已安装generic v1、原子SHA/contract/scope/reservation，不升级Runner/helper/cron/policy，不切主工作区。绑定Base、refresh、已验收旧对照的只读依赖，所有新输出独立；硬链接只在新job已复制的私有输入间。每个arm启动前持久预算领取，最多2个受管GPU子进程，外部GPU等待，4h/暂停/状态不明后不启动新arm，运行中的完成，不终止。8个bounded产物内无损分组所有曲线。服务器Base+long+optimizer真实CPU preflight通过才训练，独立科学verifier验收后英文报告/receipt去重；手动和小时入口共享状态。无holdout/Future/排名、推荐模型/SASRec训练、额外seed、自动重跑或延长。完整矩阵/条件配对验收分析后停止；以下旧scope为历史。


## 同预算零步软件修复（2026-10-28）

首发 `cf-letter-history-long-001` 在真实CPU preflight停止：encoded user hash与raw user ID共用prepared.npz的anchor_users键。task-runtime的command_started=false，真实Beauty steps=0；原失败与receipt保留。新job `cf-letter-history-long-repair-001` / scope `cf-letter-history-long-repair-v1`仅修复序列化键命名空间，不改实验/λ/STE，实际step家族总上限仍3000。任务先校验旧失败runtime完整SHA及无真实输出，才允许替代执行。新增纯NumPy round-trip regression，真实tiny verifier仍为必须通过项。不要重复旧job、删除失败或升级Runner。服务器因main_service异常自动写本地PAUSED，Git下发不能绕过；仅可核对精确失败后归档对应暂停，不触碰其他暂停/进程/记录。已准备最小恢复命令；不能将排队称为已恢复运行。


## CF 码本三组长程诊断（2026-10-28 最新明确授权）

用户审阅 H-only / RQ-only / Mixed 各1000步方案后明确“开始下发实验任务”。授权 `cf-letter-history-long-repair-v1` / `cf-letter-history-long-repair-001`，总上限3000码本optimizer steps；配置见 `docs/experiments/cf_letter_history_long.json`，说明见同名.md。覆盖旧refresh完成停止条件及未执行60步proposal；不复用旧预算、不执行旧proposal。

三组同一Base C0 + 已验收固定E1（11691x64）独立reset，仅更新3x512x64码本；冻结selected Base GPT(1192步/epoch4)、CF和SASRec。AdamW reset、lr1e-4、weight_decay=0、clip0.1、temperature0.5；相同content batch4096及rotating anchor_train batch128顺序。四个正H训练batch校准一次lambda0=0.1*median(gQ/gH)，任一零/非有限则保留并停止，不扫描λ/STE。原E0旧reference/target/mask/positions/I0不变；初始全anchor H须复现0.10294729462475516。每步记录真实batchH及梯度/裁剪，固定128条每10步，全4096条与全目录RQ/MSE每10完整内容epoch（30步）；0/1000端点评估，末8个scheduled monitor按预声明阈值分析。每步完整suffix容量检查，保留rejected/last_valid，不扩容。只读Base_train/Update_train；无validation排名、holdout/Future、推荐模型训练或SASRec更新。串行、4h后不启动下一arm，已运行继续。

沿用已安装generic v1 Runner，完整SHA/contract/scope/reservation原子下发；不改host/helper/policy/cron，不要求手动升级，不动主工作区和旧产物。Base+refresh为manifest逐文件hash绑定的只读依赖，在新独立输出中复制和复核。服务器真实CPU preflight通过后才启动Beauty；独立verifier验收原始H/容量/模型与预算，英文报告和receipt去重，完成本轮后停止。下面旧scope记录为历史。


## CF 刷新诊断（2026-10-27 最新“继续”，覆盖已完成 Base 的停止条件）

用户确认继续上一条建议，仅执行新 Base anchor_train 参考、SASRec 累积刷新与冻结量化器的 feature-only 诊断。配置见 `docs/experiments/cf_letter_refresh_diagnostic.json`，scope=`cf-letter-refresh-diag-v1`，job=`cf-letter-refresh-diag-001`。新增上限 12,240 steps / 30 cumulative epochs，Base train + Update train、seed2026、lr1e-4、串行；固定 Update-validation 512-query probe，旧 I0 catalog 的 full-denominator NDCG@10 选模，earliest tie。量化器/GPT 不训练，不运行 Plain/Historical、anchor_holdout 或 Future，不自动调参/重跑。完成报告后停止。

明确授权只读复用已验收 `cf-letter-base-001` 的 SHA 绑定产物：任务 manifest 列明 predecessor ID、selected checkpoints 与 table 的完整 hashes；任务驱动/科学 verifier 执行依赖复制前后与源输入结束时的 transitive checks，新副本仅在新 job 输出目录。host 的直接原始数据 binding 仍是既有 main-deployment.json；不扩大 host policy、不改 installed helper、read roots 或 cron，不要求用户升级 Runner。旧任务/主工作区/产物保留。用既有 E1 anchor_train 的同一批查询身份，重新计算新 Base GPT 的四 token 旧 NLL，验证 same-map H=0；仅重编码历史，固定旧 target/mask/positions/I0。按旧槽位保留规则检查完整 suffix，溢出报告负结果，不静默扩容。服务器必须先通过两个真实 CPU preflight suites；云端 skip 不算通过。

## Base 链路验收（2026-10-27 最新授权，覆盖方案未执行状态）

用户明确“先验收Base链路”。当前仅授权 `cf-letter-base-v1`：SASRec 最多 150 epochs / 44,700 steps，冻结 CF 表的直接残差量化器 1,000 steps，GPT/LETTER-style Base 生成模型最多 30 epochs / 8,940 steps；总上限 54,640，seed=2026，串行，12h 后不启动下一阶段。配置见 `docs/experiments/cf_letter_base_acceptance.json`。只读既有 Base train/Base validation；以固定 512-query Base validation probe 的 full-denominator NDCG@10 选 checkpoint，最早平局优先。完整 Base validation 仅在选定后报告。无 Update、Historical、holdout 或 Future 执行，不调参、不自动重跑，不回收旧 scope 的剩余预算。

通过已安装的通用 v1 task contract 下发 `cf-letter-base-001`，无需升级 Runner，不改 installed helper/gate/cron。数据依赖绑定既有 main-deployment.json 及其中的 Base 文件 hashes；所有新输出在独立 job workspace。服务器 preflight 必须运行真实 Torch/transformers CPU 模型测试；云端跳过项不能视为通过。数据、合法 SID 解码、checkpoint 重载与只读科学 verifier 验收后更新英文报告并停止。保留旧 Future 表和负结果。整个迁移方案的其余部分仍为 design_only。

## 新范式方案阶段（2026-10-23 最新请求）

用户要求先构建 SASRec CF embedding + LETTER-style 新范式的实验方案。方案见 `docs/experiments/cf_letter_migration_plan.md` 与配套 JSON，当前为 design_only，未授权执行。沿用已安装通用 Runner，不能再为实验安装 gate；不得因看到方案而下发任务、复用上一轮剩余 9,000 步或改动已完成的 control scope。既有研究结果与 Future 主表保留。先区分 Mitigating 的纯 CF / decoder-only 变体与完整原版 LETTER，再按新预算获准后实施。

## Runner 与任务边界（2026-10-23 最新授权）

用户要求修复反复手动升级 Runner 的边界。先读 `docs/lab_buffer/task_runtime_v1.md`。当前联合收敛任务已经在 1bc0b43 成功接入并运行，保持其代码、任务、预算、输出和进程。新的通用 v1 入口只做一次等待空闲后的核心迁移；部署以 `workers/server2-task-runtime.json` 为准，不能把交付当作已部署。此后实验入口、参数、检查和科学 verifier 随任务完整 SHA/不可变 Git contract 下发，禁止为新实验或 repair job ID 再新增 installed gate/bridge 或要求用户升级。Git CAS、共享 lease、预算 reservation、输入校验、GPU 等待、失败证据和既定停止条件保留。本次基础设施修复不新增实验或计算预算。

## 当前联合收敛实验（2026-10-23，覆盖已完成诊断的停止条件）

用户已确认按已提交方案继续开发并执行，新增 optimizer step 总上限 30,400（含 400 步吞吐标定、12x1500 筛选、最多四组 1500 步延长、条件性 2x3000 第二 seed 复验）。先读 `docs/experiments/joint_convergence_plan.md`、`docs/lab_buffer/joint_convergence_plan.json` 和 CONTROLLER 最新 amendment。完整 RQ loss 与 hard full-SID H 为训练侧目标；不运行 holdout、Base/Update validation 排名、Future 或 TIGER 训练。固定 STE/temperature，不扩展矩阵、预算或自动重跑。并行上限两组，标定吞吐不改善则串行；保留外部 GPU 占用等待、失败/中断/原输入证据。用独立新入口保留旧 source-bound 文件。已有 Runner 必须先通过 pinned 一次性接入及真实 RQ/TIGER CPU 检查；不把云端无 Torch 的协议测试记作真实模型通过。阶段间保留优化器、LR schedule、两个 sampler 和 RNG。每个 epoch 全目录 RQ、完整 anchor_train H；不把变化 minibatch 的日志当作收敛曲线。完成矩阵/条件复验、自动英文报告及回执后停止。

## 当前诊断任务（2026-10-22，覆盖已完成计划的停止条件）

用户已明确授权：先读取现有retuned Historical梯度日志，再从同一H>0最终checkpoint执行训练侧最小受控H-only/RQ-only/Mixed诊断。先读`docs/lab_buffer/CONTROLLER.md`最新scope amendment和`history_gradient_plan.json`；全部新微更新预算上限60步，不改λ/STE，不做holdout/Base validation/Future评分，不新增TIGER训练。既有Future结果与负anchor结论保留。通过原Runner的有记录兼容升级接入，不切换主工作区、不覆盖输入/旧结果；英文报告及时提交。

## 当前主实验Git接入（2026-10-20，覆盖早期阶段限制）

用户授权在既定计划和预算内接入自动迭代。每次手动“继续”或定时检查均先读`docs/lab_buffer/CONTROLLER.md`、`docs/lab_buffer/main_protocol.json`及`lab-requests:control/main.json`，共用原子领取和回执。两个History-aware Future条件已收到完成结果；剩余任务仅为真实服务器产物/模型重载验收和报告，不新增训练预算或研究实验。保留旧负结果，不用holdout/Future调参，不更新HANDOFF。暂停自动迭代仅停止新任务下发，终止正在运行的实验需用户明确指令。实际部署和恢复Web监控必须等待服务器验收证据，不能将云端检查冒充部署完成。

## 当前优先任务（2026-10-19，用户最新指令）

先补齐 Llama Base/Plain 的 Future evaluation，History-aware 暂缓。用户确认冻结 TIGER 直接测试和 Update 微调后测试两种都评估并记录。当前状态以根目录 `experiment_report.md` 为准；下文早期 Llama pilot 的 not_run 描述为历史状态。Base RQ-VAE 保持冻结并为已观察 Update 物品追加 SID，Plain 使用已完成的 1000 步 checkpoint；两组采用相同 Update 部署目录，history/target 使用各组自己的完整 SID。TIGER 从同一 Base checkpoint 配对微调，仅用 Update validation 选择；两组选定 checkpoint 后才能运行 Future 推理。真实 Future 指标须等待服务器回传，不将本地协议检查或 code_provided 当作完成结果。

## 先读与当前任务

先阅读 [docs/HANDOFF.md](docs/HANDOFF.md) 的最新状态，再读 [docs/RESEARCH_SPEC.md](docs/RESEARCH_SPEC.md)。迁移基准为 `main@e1a4cf42f40a67fd750e80ae7c3d0e9549eb0188`；开始工作时核实实际HEAD、工作区差异及相关目录规则。

迁移来源与CUDA启动核对已结束。用户已确认Llama4096维完整特征及全目录完整性检查通过，路径为`cache/incremental/Beauty/llama7b_features_20261016T132127Z/`；跳过推理smoke和重新编码。当前最小任务是独立Llama Base的50步可行性试训及Base-only尺度/均值MSE报告，尚未运行；这不是把旧T5主实验自动替换为Llama，也不是启动新history训练。原训练代码、特征、T5结果与source绑定保持不变。用户新增验证脚本提交`6e8de385e10fa3ee506b04acd8a2423a4a4f8e60`已接入`vibe-dev`。旧rho扫描负结果保留，不重复扫描或迁移核对。原始权重与历史GPU逐bit执行没有独立复核；后续状态和证据边界见HANDOFF。

## 分工与Git交付

- Codex在云端读取源码、修改代码、做可行的本地检查，通过GitHub工作分支或PR交付。默认不直接修改main，不自动合并。
- 持续使用`vibe-dev`，首次从已核实的基准创建；每轮检查diff、必要验证并提交，报告工具确认的完整SHA和链接。服务器remote为`research`。写入工具不可用时交付绑定基准SHA的patch及完整应用/提交命令。
- 用户Hao在自己的Linux GPU服务器拉取指定分支/commit，执行实验，回传compact、summary、必要日志和traceback。
- 云端不等于用户服务器。没有实际权限和执行证据时，不得声称访问过服务器、GPU、数据、checkpoint，或完成真实Beauty训练。
- GPU、数据和大型checkpoint继续留在服务器；代码交付改用Git，不把旧ZIP安装目录当作活跃源码。
- 改动前读取实际函数/CLI和调用关系。缺少必要信息时只索取相关文件、少量日志或只读命令结果；不猜接口、路径、指标和版本。

## 开发检查

- 首先`git status --short`、`git rev-parse HEAD`，修改后`git diff --check`。
- 运行与改动相关的已有合成测试，必要时新增能验证协议边界/真实故障的测试；检查入口见HANDOFF。
- 优先覆盖hard IDs与embedding输入一致性、旧target不变、teacher冻结、梯度路由、目录/时间因果、suffix容量、warm-start不重初始化、精确resume、seed配对及失败回滚。
- CPU/模拟通过只记为相应范围的实现证据，不能自动升级成GPU或研究结论。不得因缺少GPU而伪造通过状态。
- requirements.txt与实际frec-gpu环境不一致，未经核实不要升级服务器依赖。
- 保留上游与legacy路径。不要另建独立推荐框架，也不要把gensid RQVAE与TIGER内部EMA RQ混为一谈。

## 实验约束

- 遵守RESEARCH_SPEC中的时间、catalog、ID、旧labels、full SID和统计口径；任何更改须明示并单独命名。
- train/holdout用户分离，不用holdout或Future选训练权重、方向、checkpoint；已多次查看的holdout不能称作全新盲测。
- 配对实验使用相同Base/teacher/样本/特征/目录/内容batch顺序、初始化、优化器和预算；明确只变哪些seed。
- 所有预声明seed和失败都必须报告；不能挑选正结果。等步数不等算力，额外history计算需记录。
- 固定容量及token offsets；suffix溢出应明确失败/回滚，不随机消歧、wrap或静默扩容。
- 新输出不得覆盖旧run。保留source/config/data/checkpoint/mapping/cohort hashes、实际步骤、RNG/optimizer/sampler等恢复状态。
- source绑定变化可能使旧checkpoint拒绝恢复。先保留旧版与记录，再设计显式迁移；不通过关闭校验掩盖差异。
- 把事实、推测、待验证、not_run分开；负结果保留。不得为得到正结果改变评估协议。

## 每轮交付

每轮给出：目标与前提、分支/commit或PR、文件变更及原因、已运行检查和适用范围、用户服务器的精确命令、成功/失败标准、少量需回传的证据、尚未运行事项及下一动作。

命令从`/home/hao/frec/graph-informed-sids-web`执行，使用已验证的`frec-gpu`；保持`set -o pipefail`，耗时实验写文件日志并提供compact摘要。历史手工文件均放项目根目录，不用Downloads路径。

用户最新约定（2026-10-19）：每次收到实验执行结果后，更新根目录 `experiment_report.md` 并提交到 `vibe-dev`，记录实验设置、最新指标、关键结论、产物路径及证据边界；保留此前结果，区分冻结模型的兼容性排名与推荐模型微调/Future 指标。按用户要求不更新 HANDOFF 文件。研究规格发生已确认变化时再同步 RESEARCH_SPEC。未获得用户回传前保持 code_provided/not_run，不把预期当结果。只有用户请求新实验时才准备相应运行步骤；“只分析”或“只文档”期间不启动训练。
