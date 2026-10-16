# 项目交接：History-aware SID Incremental Update

## 0. 2026-10-16 Llama完整特征已完成；独立Base试训准备

**用户回传的已完成产物**：`cache/incremental/Beauty/llama7b_features_20261016T132127Z/`。本地冻结Llama7B从title、首条categories路径、description编码，最后一层隐藏状态，1024 tokens截断/补齐，padding清零后sum/1024，模型FP16、pooling float32，输出float32/4096维，无L2归一化、降维或微调。用户明确报告完整性通过：BaseTrain10924×4096、UpdateTrain11707×4096、Update部署11775×4096；无缺失、非有限或全零向量，共享物品跨阶段向量完全一致。商品仅统一编码一次。当前用户提供的事实足以跳过smoke与重编码；编码时的实际Git SHA、完整summary/hash、总耗时与峰值显存未在本条回传，不补填。

**代码同步**：进一步发现feature分支提交`6e8de385e10fa3ee506b04acd8a2423a4a4f8e60`（父6d69252），仅新增`verify_incremental_embeddings.py`。已读完并合入持续分支；保留与`339bf6fa990430d5962699af17138fa7a0f64b2e`两条历史。该脚本检查protocol/row-map/hash/finite/nonzero/shared-prefix，用户已经报告过这些检查通过，本轮不要求再运行。原Llama编码、temporal/content、训练实现和RESEARCH_SPEC均不修改。

**下一项最小任务（code_provided/not_run）**：新建独立Llama Base RQ-VAE，只在`base_train`10924项上fresh训练50步。实际入口`train_incremental_rqvae.py`→`gensid.incremental_content.run`→repository `RQVAE`/`Trainer.fit_steps`。`in_dim=features.shape[1]`，因此自动适配4096维，无需修改源码或加入投影。沿用原recipe：layers512/256/128/64，e_dim32，3×256，外层quant_loss_weight0.25、内部commitment沿用0.25，Base kmeans一次（4096样本、50iters），AdamW lr0.001/wd0、batch256、seed2026、无BN/dropout/CF/graph/Sinkhorn，原始向量不变。新的输入/输出层维度和特征会改变参数初始化及输入尺度，不能称为与旧T5共享Base或严格encoder-only消融。输出全新目录、每25步保存、每10步日志；不从旧T5 checkpoint加载，不训练TIGER/History-aware。

该50步只是新增表示的可行性pilot，不替换现有T5主实验，不定义完整Llama对照方案。当前没有Llama RQ-VAE运行结果，更没有兼容性/排序收益结论。RESEARCH_SPEC不更新为已完成新路线；后续若用户执行并决定扩大，记录该确认后再安排累计1000/5000、SID和新teacher。

**新增`report_content_rqvae.py`**：只读取BaseTrain特征，chunked float64计算L2 norm分位数、zero predictor MSE、Base均值向量predictor MSE；不读取Update/Future向量、不归一化或拟合推荐模型。可读取与同一feature-summary绑定的Base RQ-VAE摘要，报告initial/final MSE、相对initial及mean predictor的改善、码本使用/碰撞和checkpoint reload一致性。MSE由原训练入口的全BaseTrain评估提供，本工具不加载checkpoint独立重算。输入尺度不同，不能直接以T5与Llama原始MSE大小判断表示优劣；均值基线也不是推荐质量指标。

本地NumPy解析例验证通过（已知zero MSE=11、mean MSE=1、分块、零方差、绑定报告比较、拒绝覆盖及错绑定）；CLI help和diff通过。云端无PyTorch，不能运行RQ-VAE模型检查或真实Beauty训练。服务器执行命令及准确交付SHA见本轮回复；命令记录commit、参数、seed、feature/run路径、train.log和compact；无冲突强制清理。

**需回传**：新report的compact（含norm分布、MSE/ratio、codebook统计及reload标志），异常则train.log最后100行。若50步有限、重构较初始化下降且无明显坍缩，可讨论从该checkpoint延长累计预算；50步未超过mean baseline本身不证明失败。出现nonfinite或坍缩时先检查原始尺度/损失/初始化，不自动归一化、改学习率、扩大容量或进入history训练。现有全量embedding与旧结果均保留。

## 0.1 迁移核对结束；接入Llama特征提交（历史记录）

**新的用户执行证据**：`results/incremental/Beauty/cuda_startup_check_20261016T131826Z_PIdjP4`，按交付`vibe-dev@8ef11cfc60cedb99c58e8e5b81866a352013d261`执行。6项测试在服务器通过（0.120s）；归档元数据链`metadata_consistent`，fix记录SHA256=`08cd9f9138afe5e8053123e11659d5cc21d34dd72777690829821300d3ef3613`；plan只有controller source hash变化，engine未变。15组比较中14组History H更高。真实`cuda:0` probe passed，进入engine前CUDA已初始化，seed2026，单张量值1.0；torch2.6.0+cu118、CUDA runtime11.8、Tesla V100-SXM2-32GB。未训练、未加载checkpoint。用户重复贴出的两份相同JSON是同一run的打印，不记作两次独立验证。

**关闭事项**：迁移来源/启动修复的一致性核对在所声明的文件/元数据与启动路径范围内完成；不再重复旧扫描或启动probe。原始权重/历史GPU逐bit执行和指标重算仍不在验证范围内。这是工程结论，没有新增推荐质量或History-aware收益结论。

**用户最新代码**：找到并实际读取feature分支`feature/llama7b-temporal-embeddings`的提交`6d692529e38a6c17c2f2aaf558fb86828dec7f89`（父提交2fb3426），新增`prepare_incremental_llama.py`、`docs/LLAMA_EMBEDDINGS.md`、`tests/incremental/test_llama_content.py`。本轮将其合入持续分支`vibe-dev`，保留feature与8ef11cf两条提交历史，不写入main。原encoder文件及其source-bound依赖`genrec/temporal.py`、`genrec/temporal_content.py`与feature提交逐字相同；不改变可能存在的编码断点绑定。

代码复用用户已有Beauty protocol/content/raw reviews及本地`/home/shared/huggingface/llama-7b`，不重切数据，不下载权重。按现有registry保持BaseTrain10924、UpdateTrain11707、Update部署11775项和原ID顺序；共享编码后按前缀导出，排除future-only物品。输出4096维float32；模型默认FP16、SDPA、长度1024，padding置零后除以固定长度1024，无额外归一化/投影。已实际读取作者notebook `<https://github.com/iskbaga/semantic-id-alignment/blob/main/notebooks/AmazonBeautyDownload.ipynb>`，blob确认为`20f09c2d081b8ddc7b7825d43ff3bf626e879521`，核实title/categories/description文本格式及固定长度pooling。FP16/SDPA及本地checkpoint身份/时间切分差异仍按LLAMA文档披露，不宣称逐bit复现作者结果。

**新增最小验证**：独立`check_incremental_llama.py`，默认只推理2项、seed2026，沿用上述输入校验、local backend和pooling；输出到新的results目录，记录当前commit、source/model/input hashes、样本维度/范数/显存/耗时。仅生成`llama_content_smoke_v1`样本结果，不能作为`E0_fixed_content_embeddings`训练输入。拒绝覆盖或使用`--resume`；不会写入原全量编码的output或`.incomplete`。首次检查仍需hash全部模型分片并加载完整7B模型，不能把这段等待说成两条文本的推理耗时。全量源入口不变。

本地检查：两份`test_llama*.py`共15项，14项合成检查通过；1项tiny真实后端测试因云端缺少torch/transformers而跳过。新增4项验证样本输出不冒充完整特征、拒绝覆盖、保留已有partial目录及错误日志保留；CLI help、diff检查通过。Llama7B/V100推理、全量编码是否完成尚无本会话用户证据，不能标completed；若用户已完成，先接收其原compact，不重复加载模型或编码。

**研究边界**：Llama实现还改变了文本字段/格式、长度和pooling，不能称作仅替换encoder的严格消融。旧T5 checkpoint和结果仍是独立路线；不能将768维Base RQ-VAE直接接4096维特征，也不能复用其SID/TIGER/anchor绑定。是否用Llama重新开展完整Base/Update对照尚未作为研究决策确认，RESEARCH_SPEC不变。先前提出的训练日志只读分析仍停留在方案，未实现/未交付。

**本轮下一步**：若还未有真实编码结果，按交付命令执行2项smoke（FP16、SDPA、1024、batch1，独立输出）；回传测试末尾和compact/traceback。若已有全量结果，则直接回传原compact及其输出目录。只有输入身份、维度、有限非零向量、显存和实际速度得到验证后再安排全量编码/新Base训练；新Base训练与公平对照协议须另行说明，不能据新embedding推断会解决历史损伤负结果。

## 0.2 服务器核对回传与CUDA修复接入（历史记录）

**用户实际回传**：2026-10-16，`source_audit_20261016T105113Z_v4eZXM`。8项合成测试在服务器通过（1.299s）；收集器`collected`、`training_started=False`。运行前branch=`main`、HEAD=`2fb3426615ec0033dd6b65734593075f7e8380e0`、工作区clean。相对迁移基准e1a4cf4，41个静态/绑定源码中2项不同：`fix_history_weight_cuda_startup.py`与`run_history_weight_sweep.py`。来源声明对原工作区mismatch=0，15组比较全部找到、14组History H更高，历史sweep completed、selected_rho=None、CUDA修复记录1份、issues=0。随后服务器已切换`vibe-dev@cbe6761b7441f3eadef6e92c7cf19e7808adace3`。

原始证据仍在服务器：`results/incremental/Beauty/source_audit_20261016T105113Z_v4eZXM/report/report.json`及同名run的`.tar.gz`。本轮用户只粘贴compact，未上传archive；不能声称云端已直接审阅原始fix记录、运行元数据或独立重算权重/指标。compact没有暴露来源冲突，足以继续核对公开代码并接入已经存在的修复。

**新增源码证据**：已fetch并逐项检查main提交`2fb3426615ec0033dd6b65734593075f7e8380e0`（message=`add cuda fix`）。相对e1a4cf4共4个文件：controller仅增加CUDA子进程bootstrap，另外是修复安装脚本及修复前controller/plan备份。engine及内容/历史算法文件没有变化。controller的新SHA256=`1cc112fcab6e2c582ac219c71305e3844f3b69d888e15a1342cb0a17a9c4e7fa`；engine仍为`096b8b500446ea2057aa6d75e2f59b80ca55c85842c73c4f6260d77c9d80879a`。备份路径为`cuda_startup_fix_backups/20261015T165206566101Z/`，备份plan仍记录旧controller SHA256，符合修复脚本的备份流程。

**本轮实现**：将上述main提交合入持续分支`vibe-dev`，保留两条历史的提交祖先；不写入main。controller与main修复版逐字相同；不重新实现bootstrap、不改engine、不重新运行安装脚本或改写旧plan。此前`cbe6761`源自e1a4cf4，用户切换到它后controller会回到未修复版本；本轮交付恢复修复版，避免后续使用时版本倒退。commit SHA以本轮交付及Git记录为准。

新增`scripts/check_history_cuda_startup.py`，从已保存的审计副本检查关键文件checksum、单个fix记录、修复前备份plan/修复后plan的哈希链、仅controller hash变化、归档源码及15个终点的一致性。验证通过才经真实controller.command启动一个临时probe（仅在当前进程替换ENGINE指针，finally恢复，磁盘engine不变）。probe在入口确认CUDA已初始化，固定seed=2026，reset显存统计并分配1个float张量；不加载模型/数据/checkpoint、不训练、不调用sweep.run。保存commit、源码hash、参数、child.log、config/summary/compact；新输出目录拒绝覆盖。若记录链不一致则在启动CUDA前失败。

新增`tests/incremental/test_history_weight_cuda_startup.py`共6项。检查修复版controller/原engine的固定hash、独立子进程的CUDA初始化顺序及CPU直接启动、seed/CLI传递、失败日志与ENGINE指针恢复、fix链及归档篡改拒绝。使用fake torch完成云端检查，6项本地通过；这不是V100验证。原有8项来源测试的服务器通过来自用户回传。新增真实CUDA probe为code_provided/not_run，需用户执行。RESEARCH_SPEC未修改。

**本轮服务器命令接口**（在准确交付SHA、frec-gpu下；输出用全新`$CHECK_OUTPUT`）：

```bash
python -B tests/incremental/test_history_weight_cuda_startup.py
python -u -B scripts/check_history_cuda_startup.py \
  --audit-report results/incremental/Beauty/source_audit_20261016T105113Z_v4eZXM/report/report.json \
  --output "$CHECK_OUTPUT" --device cuda:0 --seed 2026
```

完整安全fetch/switch命令以本轮交付回复为准。需回传6项测试末尾及新compact（异常时含child.log）。成功标准为metadata_consistent、startup passed、initialized_at_engine_entry=true、tensor_value=1、training_started=false。这只确认元数据/文件一致性和启动路径，不验证历史GPU逐bit重现。通过后结束本项迁移核对并进入负结果的机制分析/下一项最小实验设计；不自动重跑rho、选holdout或运行E3。若失败，按具体记录/hash或CUDA traceback定位，不重复训练。

## 0.3 2026-10-16 首轮接管开发（历史记录）

用户已授权采用持续工作分支`vibe-dev`开发并提交；本节覆盖旧交接轮的“只整理文档”限制，不更改研究协议。已实际读取三份附件，并clone开发仓库。GitHub与本地均核实`main=e1a4cf42f40a67fd750e80ae7c3d0e9549eb0188`；核实时无其他远端分支，现已从该SHA创建`vibe-dev`。本轮提交的准确SHA以Git提交记录及交付回复为准，不在文件中写入自引用SHA。

**本轮目标**：完成服务器来源核对所需的最小工具，使下一轮能比较迁移基准与最新成功扫描的实际文件。**当前状态为代码已实现、独立本地检查通过；服务器证据收集尚未运行，来源一致性尚未确认。** 不重跑训练，不修改旧engine、controller、manifest或checkpoint，不实施CUDA修复，不更改RESEARCH_SPEC。附件规格原样纳入`docs/RESEARCH_SPEC.md`。

本轮从活跃源码确认调用链：`run_history_weight_sweep.command`→根目录`history_weight_update.py`的train/evaluate→`history_aware_frozen_encoder`→`history_aware_update_v2`→`gensid.historical_incremental_content`，最终复用`gensid.models.rqvae`与repository TIGER/T5。`history_update_multiseed`及`audit_history_pair`参与绑定/独立评估。两个关键文件SHA256与下文第3节相符，controller没有CUDA bootstrap，engine仍在模型`.to(device)`前reset peak memory stats。未改bundle/handoff/complete_files副本。

新增交付：

- `scripts/audit_history_sources.py`：标准库只读收集器。保存原始plan（含发现的plan备份）、selection、sweep/子任务/旧seed/E1摘要、request/config、CUDA修复JSON、checkpoint attempts存在性及相关日志最后64 KiB。记录原工作区分支/HEAD、当前Python/包版本（仅metadata，不import训练包）、来源声明与基准/工作区哈希比较、静态本地import闭包、原始源码快照/差异。核对selection的plan hash、旧seed摘要hash及终点指标是否与子摘要一致。仅报告发现的证据与缺项，`collected`不等于来源已验证。
- `scripts/collect_history_source_audit.sh`：从已fetch的准确交付SHA提取收集器，激活`frec-gpu`，先保存服务器原始源码再尝试切换版本。工作区dirty或本地`vibe-dev`无法fast-forward时保留现场、输出blocked及证据包；clean时切换到准确交付SHA并执行新增的轻量检查。新目录为`results/incremental/Beauty/source_audit_<UTC>_<随机后缀>`，同时生成同名`.tar.gz`。只读取历史run，不打开权重/数据/逐query损失文件，不调用GPU，不删除缓存。
- `tests/incremental/test_history_source_audit.py`：8项独立合成检查，覆盖原始字节和hash/diff、嵌套来源声明、缺项、错误plan/终点、重复输出拒绝、路径边界，以及clean/dirty/diverged三种同步流程。同步测试使用临时Git仓库和模拟Conda，不能视作用户服务器环境验证。
- 纳入附件`AGENTS.md`，仅补充本轮已授权的Git开发方式和当前任务。

本地检查：上述8项在Python 3.12.14通过；`bash -n scripts/collect_history_source_audit.sh`、`git diff --check`通过。在真实clone上运行收集器可列出36个静态源码文件，当前与基准一致；缺失服务器产物被明确报告，没有当作completed实验。当前云端没有PyTorch，未重跑旧模型合成检查或任何Beauty/GPU实验。新工具支持服务器Python 3.10。

**服务器执行协议**：用户按本轮交付命令进入`/home/hao/frec/graph-informed-sids-web`，检查分支/HEAD/status，`git fetch research main vibe-dev`，用`git show <交付SHA>:scripts/collect_history_source_audit.sh`提取并执行该SHA脚本。脚本将原现场及所有收集器参数、交付SHA保存到独立run。即使本地有未提交的CUDA修复也先收集，随后停止切换；不得通过stash/reset/关闭校验掩盖差异。原始源码快照在`report/sources/workspace`，基准在`report/sources/baseline`，差异在`report/diffs`，收集清单在`report/files_sha256.json`。包中有`run_command.sh`和`delivery_commit.txt`，可据准确SHA重建此次只读命令。

**需回传**：终端打印的compact及`workspace_sync`状态，以及`RETURN_ARCHIVE=`指向的单个`.tar.gz`（含小型摘要/源码/差异，不含模型和数据）。若报错，回传traceback及新run路径，不重启训练。

**下一步判断**：先读原始plan、三seed preflight、12组新增训练摘要、修复记录和源码diff。若差异仅为有记录的启动bootstrap且其余绑定及终点一致，记录这一有限的一致性结论，并单独决定如何固化修复。若engine/绑定/plan不一致，定位具体字段和时间链；缺项只索取相关产物。不迁移旧checkpoint、不改写旧plan、不扩大rho扫描。最新负结论仍仅来自下文已回传U结果；本轮没有新增实验结论。

## 0.4 迁移交接时的状态（历史记录）

更新日期：2026-10-16。仓库 <https://github.com/haowu1937/incre-update-sids>，本次迁移基准 `main@e1a4cf42f40a67fd750e80ae7c3d0e9549eb0188`。本次已通过 GitHub 读取 commit、目录树及关键源码；读取时 main 指向该 commit。

**最近成功运行**：服务器权重扫描 `weight_sweep_20261015T161912Z` 已 completed。固定 Base/teacher/cohorts，冻结 encoder，gate disabled；Update seeds 2026/2027/2028，各 100 步，rho=0.1/0.3/1/3/10。没有选中配置，`selected_rho=None`；15 组中 14 组最终训练历史损伤高于对应 Plain。新增权重未进入 holdout/ranking，Future 未运行。

**最近回传的执行失败**：同一扫描首次启动，在 `history_weight_update.py:108` 的 `torch.cuda.reset_peak_memory_stats(device)` 报 `RuntimeError: Invalid device argument`，训练尚未开始。会话交付过 CUDA 启动修复脚本，随后用户回传扫描完成结果；修复脚本的应用输出、服务器最终源码哈希和完整 provenance 未回传。

**迁移基准确有差异**：本次读取的 commit 中，`run_history_weight_sweep.py` 仍直接启动 engine，没有先初始化 CUDA 的 bootstrap；engine 仍在模型上 GPU 前重置显存峰值统计。修复文件 `fix_history_weight_cuda_startup.py` 不在该 commit。不能宣称该 commit 已包含成功运行所需修复，也不能把后来的服务器结果等同于此 commit 的逐字复现。

**第一优先任务，仅一项**：完成“迁移基准—最近成功运行”的源码及证据一致性核对。读取服务器 sweep 的 plan、源码哈希、CUDA 修复记录和结果摘要，核对该 commit，形成可审阅的差异与 provenance 记录。先保留现有负结果；不自动再跑训练、扩大权重或执行 E3。

研究规格见 [RESEARCH_SPEC.md](RESEARCH_SPEC.md)，协作规则见 [../AGENTS.md](../AGENTS.md)。本轮只新增交接文档；未向 GitHub 写入、创建分支/PR，也未运行新的模型实验。

## 1. 证据分级与来源

| 标识 | 来源 | 能支持什么 |
| --- | --- | --- |
| P | 完整读取的 `agent_prompt_web(2).md` | 原始目标、方法、阶段计划；不证明执行 |
| C | 指定 commit 的源码、目录树、README、manifest、路径文件 | 该版本接口与实现；不证明服务器当前文件相同 |
| U | 本会话用户粘贴的 summary、指标、日志和 traceback | 对应实际运行及报告值；未远程重算原数据 |
| U-latest | 用户附件 `粘贴的文本 (1)(20261016-082341).txt` | 最新完整 compact、rho=0.3/seed2026 日志 |
| S | 先前助手独立 CPU/模拟检查 | 局部实现证据；不是用户服务器/GPU/Beauty 实验 |
| I | 基于以上结果的解释 | 推测，不能写成根因已证实 |

旧 `README.md` 实际是 Round 1 的说明，`implementation_plan_round1.md` 与 `experiment_report_round1.md` 是早期状态。它们不能覆盖最新 U 证据。旧 bundle 的 validation/local_checks 也不能自动升级为服务器 user_verified。

历史真实实验大多发生于导入 Git 之前，部分 summary 的 `source_sha=not_available`。迁移 commit 不是这些实验已经核实的训练 Git SHA；应依靠各 run 的 source-file hashes 恢复版本。

## 2. 当前源码结构与运行入口（C 已核对）

| 活跃路径 | 作用 |
| --- | --- |
| `genrec/temporal.py`, `temporal_artifacts.py` | 安全解析、全局时间窗、causal examples、registry、manifest |
| `prepare_incremental_data.py` | 从已有 review gzip 生成 temporal artifacts |
| `genrec/temporal_content.py`, `prepare_incremental_content.py` | metadata 白名单与固定文本 embedding；子命令 metadata/encode |
| `temporal_graph_inputs.py`, `temporal_graphs.py` 及根目录 prepare 脚本 | 图输入/构图的增量适配；图输入成功不等于图训练成功 |
| `gensid/models/{rqvae,rq,vq,layers}.py`, `gensid/trainer.py` | 复用仓库 RQ-VAE 与训练器 |
| `gensid/incremental_content.py`, `train_incremental_rqvae.py` | Base 内容 RQ-VAE；恢复参数名 `--resume`，max-steps 为累计总步数 |
| `genrec/temporal_sids.py`, `prepare_incremental_sids.py` | 冻结 Base SID、suffix、causal TIGER tensors |
| `genrec/frozen_sid_tokenizer.py`, `temporal_tiger.py`, `train_incremental_tiger.py` | Base TIGER 固定步数训练及 constrained ranking；恢复 `--resume-from` |
| `gensid/plain_incremental_content.py`, `train_plain_incremental_rqvae.py` | 原始 Plain Update，encoder/decoder/books 均训练，恢复 `--resume-from` |
| `genrec/historical_damage.py`, `measure_historical_damage.py` | E1 cohort、固定旧 target 的 history 重编码损伤、ranking |
| `gensid/historical_incremental_content.py`, `train_historical_incremental_rqvae.py`, `evaluate_historical_update.py` | 较早 scalar-lambda History-aware 更新与评估 |
| `diagnose_historical_collapse.py`, `check_history_gradient_direction.py`, `diagnose_history_update_step.py` | 坍缩、方向及单步候选诊断；不是完整主实验 |
| `repair_historical_codebooks.py` | 从 Plain Update 出发的 codebook repair/evaluate |
| `history_aware_update_v2.py` | 全 anchors、full SID、逐层归一化历史梯度；内容 encoder 可训练 |
| `history_aware_frozen_encoder.py` | frozen encoder 的 Plain/History-aware 配对，train/compare |
| `history_aware_hard_gate.py` | 每步 local content/joint 候选 hard-H gate，train/compare |
| `audit_history_pair.py` | 独立 raw IDs→hard SID→logits NLL 复算；preflight/compare |
| `history_update_multiseed.py` | 显式 Update seed、batch trace、train/evaluate/aggregate |
| `history_weight_update.py` | 独立 rho≤10 engine；train/evaluate，支持 `--train-only` |
| `run_history_weight_sweep.py` | 多进程 seed 并行、权重逐档、训练筛选、最终 selected evaluation |
| `tests/incremental/` | 各轮合成协议、模型接口、恢复、审计和调度检查 |

以上入口委托现有 gensid/genrec 模型；不是另起推荐框架。原 prompt 的 `main_incremental.py`、`gensid/history_aware_loss.py` 等是建议接口，不能当作已实现的命令。

### 历史目录与补丁应用方式

- `round1_handoff`—`round8_handoff`：历史 ZIP 展开目录，通常有 README、manifest、完整文件副本、`.patch` 和当轮计划/报告。
- `round9_bundle`—`round12_bundle`、`round12_fix`：后期 installer、补丁、查找路径与 compact helper。
- `codebook_repair_bundle`、`collapse_diag`、`history_direction`、`history_step_bundle`、`history_v2_bundle`、`frozen_encoder_bundle`、`hard_gate_bundle`、`multiseed_bundle`、`weight_sweep_bundle`：对应历史交付包的安装、运行、报告及源文件副本。
- `complete_files/`：完整替换文件的历史副本。根目录同名目录及各 bundle 内副本都不是默认 import 的活跃源码；执行入口通常是项目根目录文件及 `genrec/`、`gensid/`。
- `*_paths.sh`：某次运行的路径记录；部分已提交，但其指向的 cache/ckpt/results 通常未提交。

早期方式是 ZIP 放项目根目录，`unzip ... -d <handoff>`，然后 `git apply --check`，通过后 `git apply`；后期是 `python <bundle>/apply_*.py --repo .`，校验 manifest、备份后安装。不能强制忽略 checksum、不看版本直接复制 complete_files，或重放所有 ZIP。

迁移 commit 已导入很多补丁成果；新 Codex 应以活跃源码为起点，通过 Git 分支/PR 交付。历史包用于 provenance/接口追踪，不继续充当默认发布方式。旧 repair README 的 Downloads 路径已被用户明确纠正。

## 3. 服务器、数据、checkpoint 与版本

服务器：`hao@aistar-server2`；项目根目录 `/home/hao/frec/graph-informed-sids-web`；conda `frec-gpu`。服务器 OS 据用户记录为 Ubuntu 22.04，GLIBC 2.35；本轮未重新取得系统 audit。会话实际 GPU 输出为 Tesla V100-SXM2-32GB，driver 580.178.04；nvidia-smi 显示 CUDA ceiling 13.0，不代表 PyTorch runtime 是 13.0。

实际训练报告：Python 3.10.18、torch 2.6.0+cu118、NumPy 2.2.6、sklearn 1.7.1、transformers 4.56.1；embedding 报告 sentence-transformers 5.1.0。其他运行包的完整 lockfile、当前 RAM/空闲显存、并行总耗时未知/待核实。

**依赖文件差异**：迁移 commit 的 requirements.txt 写 torch==2.11.0、CUDA 13 索引等，与真实 V100 运行环境不同。不要直接 pip install -r requirements.txt 来覆盖服务器现有环境，也不能把该文件当作已验证的实验 lockfile。

以下相对路径均以服务器项目根目录为基准：

| 内容 | 已知路径/状态 |
| --- | --- |
| review 原始文件 | `cache/AmazonReviews2014/Beauty/raw/reviews_Beauty_5.json.gz` |
| temporal protocol | `cache/incremental/Beauty/beauty_20261007T104452Z` |
| content metadata artifact | `cache/incremental/Beauty/content_20261008T123254Z`；原始 metadata 文件路径未知/待核实 |
| 后续训练使用 features | `cache/incremental/Beauty/features_20261008T135919Z` |
| 较早成功 features 日志 | `features_20261008T125632Z`；不能混同为后续 checkpoint 的绑定 features run |
| Base RQ-VAE 1000 | `ckpt/incremental/Beauty/base_rqvae_20261008T152400Z` |
| Base RQ-VAE 5000 | `ckpt/incremental/Beauty/base_rqvae_5000_20261009T082337Z` |
| 原 Plain Update 1000 | `ckpt/incremental/Beauty/plain_update_rqvae_1000_20261009T124124Z` |
| Base SID/TIGER 目录 | 完整路径未知/待核实；从 E1 summary 的 `sid_dir`、`tiger_dir` 读取，不能猜测时间戳 |
| E1 | `results/incremental/Beauty/historical_damage_20261009T142825Z` |
| Joint v2 | `ckpt/incremental/Beauty/historical_joint_v2_20261014T124630Z_1000`；评估 `results/incremental/Beauty/historical_joint_v2_20261014T124630Z_evaluation` |
| frozen encoder seed2026 | `ckpt/incremental/Beauty/frozen_encoder_20261015T101457Z_plain_100` 与 `_history_100` |
| hard gate | `hard_gated_20261015T115808Z_paths.sh` 记录；完整 checkpoint 路径待读取核实 |
| paired seeds | `results/incremental/Beauty/paired_seeds_20261015T135505Z`；新 checkpoints 在 `ckpt/incremental/Beauty/paired_seeds_20261015T135505Z/seed2027_plain`、`seed2027_history`、`seed2028_plain`、`seed2028_history` |
| 最新 sweep | `results/incremental/Beauty/weight_sweep_20261015T161912Z` |
| sweep 子任务 | `checkpoints/rho_<tag>/seed<seed>/attempt_<NNN>`，相对于 sweep root；具体有效 attempt 查目录与 summary，不猜测 |
| 小 lambda、repair 及部分方向诊断 | 本会话有结果，完整 run ID/路径未知/待核实 |

关键校验值：

| 对象 | SHA256 |
| --- | --- |
| review gzip | `e5925ec99023f1dc9c7d3dff7ef34cfeacd40d1b981b5525e1e4ba9c8abc18fe` |
| protocol | `97ecf43d164f74e9b13f2c814d3edd9cace9130e8f9a9d2581421b7b2b6e2e66` |
| Base embeddings.npy | `305dc6a7a73feb2ffd39077f2609413fb38987709ac34c41ba4745e59a91f09d` |
| Base RQ-VAE 5000 last.pth | `b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d` |
| Base TIGER best.pth | `4538d5d8bcab84e6f6919cdd66a041c257d1fa62f8519bd787918f00a68d8451` |
| commit 中 sweep controller | `1afee70be91aca93e27cc607708bb1de9a0c0ba14db6d98b377d456b612a1b47` |
| commit 中 weight engine | `096b8b500446ea2057aa6d75e2f59b80ca55c85842c73c4f6260d77c9d80879a` |

## 4. 已完成实验与实际结果（U）

### 4.1 数据、内容、Base 模型

数据准备 completed；198502 事件全部保留，exact duplicate=0。数据身份仍未正式验证。图输入准备 completed：Base 107108 事件/10924 nodes；Update 35689 事件/11707 nodes，2440 个节点在 Update 窗口没有交互。graph construction、APPNP、graph feature extraction 没有获得执行证据。

固定内容 embedding completed：sentence-t5-base revision `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`，768 维 float32；BaseTrain 10924 rows，UpdateTrain 11707，Update deployment 11775。无 graph、projection、finetuning，旧特征不变。较早回传的 encode runtime=43.03 秒，batch progress≈32 秒；之前长时间 GPU idle 的启动等待根因未获证实，不能写成 GPU 编码耗时半小时。

Base RQ-VAE：seed2026，batch256，AdamW lr=0.001、wd=0，kmeans init_samples4096、iters50；768→[512,256,128,64]→32；3×256 books，无 BN/dropout/SK/CF。

| Base 预算 | 全目录 reconstruction MSE | unique codes | 最大桶 | books used |
| --- | ---: | ---: | ---: | --- |
| 1000 累计 steps | 0.0001423425414480207 | 6287 | 37 | 10/127/62 |
| 5000 累计 steps | 0.00010876538503473197 | 10090 | 26 | 38/224/192 |

5000 从 1000 checkpoint 继续；不是额外 5000 步。平均 embedding baseline MSE=0.00023159830016084015；用户回传的 38.539% 改善对应 1000 步 MSE，不能错误标到 5000 步。5000 runtime=107.49 秒（该次续跑报告）。

冻结 Base SID 导出 completed：BaseTrain 10924 full SIDs 唯一，最大 semantic 桶26，suffix最高25；Base deployment11179。RQ frozen、无 refit。Base train tensors=76252×201/labels×5；Base val labeled8689，另503不可服务。

Base TIGER 已训练到累计5000，best probe step3000，4,590,976参数，随机 T5；full Base validation completed：

| full denominator | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Base old model | 0.0165361184 | 0.0253481288 | 0.0105087632 | 0.0133026481 |

NLL_servable=10.9256017483，invalid=0，catalog=base_train，beam20，seen filter=false。选择用512-query probe 的 full-denominator NDCG@10；best probe NDCG@10=0.0178129149，不能混作全9192条结果。该次续跑 runtime=975.26秒，peak allocated=3695589376 bytes。模型架构/优化器完整 resolved config 应从保存的 config 核实；源码默认不能自动认定全部是用户实际 CLI。

### 4.2 原 Plain Update 与 E1

Plain：seed2026，从 Base5000 原权重 warm-start、reset AdamW、lr0.001、batch256，Update 1000 steps。encoder/decoder/books 全训练，catalog11707，含10924 BaseTrain、255 BaseVal-first、528 UpdateTrain-first。

- recon MSE=0.00010460418691563549；unique10903、max桶20；Base语义 churn=10906/10924=0.9983522519。
- runtime28.06秒，peak allocated95578624 bytes；这不是 History-aware 的等计算成本。
- E1：anchor_holdout new_full_sid 的 mean Delta_NLL=+0.0523633182，user-cluster CI95=[0.0392136356,0.0659748654]；positive damage=0.1481531531。
- Base val mean Delta_NLL=+0.0001097354，CI跨零；不能声称验证集平均 NLL 明确受损。
- `new_full_sid` 表示 history 使用新完整地址，target 仍固定旧 labels；不代表 target 被改成新 SID。

### 4.3 较早 History-aware、repair 与机制诊断

| 实验 | 配置/预算 | 实际结果与限制 |
| --- | --- | --- |
| scalar lambda=0.001 | 从 Base 更新，1000 steps | 完全坍缩：unique1、max桶11707、books1/1/1，recon≈0.000232，full SID不可导出；不能记作成功方法 |
| scalar lambda=1.74475612832e-08 | 1000 steps | unique10880、max15，recon0.0001048194326；holdout full-H0.160086，高于Plain0.148153；未获得可靠收益 |
| STE/tangent 方向检查 | 512 queries，多个 scope/步长 | 两种 surrogate 梯度相同，连续方向与 hard-H 方向可不一致；局部反向也可能改善，不足以定位唯一根因 |
| repair，梯度512 queries | Plain1000 出发，freeze encoder/decoder | 5提案、0接受，最终等于Plain |
| repair，梯度4096 queries | 同起点，codebooks-only，hard训练H筛选 | 22提案、6接受，训练 old-suffix H0.144634149→0.142943433；recon0.000104609888 |
| Joint Update v2 | Base出发，1000 steps、rho0.1，内容encoder可训练，H仅books，全4096anchors/步 | 训练full-H0.242539345，holdout0.232033459，均劣于原Plain；recon0.000104663086；约3小时 |
| frozen encoder pilot | Base出发，100 steps、rho0.1，seed2026，两组都freeze encoder | Plain训练H0.087290558，History0.0974458815；差+0.0101553234，CI=[0.0035512692,0.0166640293] |
| local hard-H gate | 同frozen设置，100 steps | 历史候选47次接受、53次fallback；最终H0.100405086，比独立Plain高0.013114528，CI=[0.0064845952,0.0188940976] |

repair holdout：保留旧suffix时H差−0.0014563108，CI=[−0.0027801589,−0.0004104441]；完整新SID时差−0.0006886942，CI跨零。其 Base validation full-H略高且CI跨零。该结果只支持部分条件的改善，不是主实验已经成功。

Joint v2 holdout full-H相对Plain差+0.0838803062，CI=[0.0656373210,0.1020187133]；RQ质量接近并未带来兼容性。median step9.10787秒，history query处理4096000，peak raw梯度比例≈0.1。原始梯度比例受控不等于参数步长受控。

单步诊断：v2 step1000 状态上，content Adam候选 hard-H差+0.01159035，joint+0.01069666，codebook-only joint−0.00550743；训练steps committed=0。说明局部历史修正可能帮助当前状态，但不能保证整条联合轨迹胜过独立Plain。local gate 的 content候选也只属于自己的当前轨迹，不能称其为独立Plain。

历史 ranking 的 full denominator 统一为9192：

| 方法（旧 target/catalog 的诊断） | Recall@10 | NDCG@10 |
| --- | ---: | ---: |
| old | 0.0253481288 | 0.0133026481 |
| 原 Plain1000 新history | 0.0239338555 | 0.0126165707 |
| 小lambda History-aware | 0.0233899 | 0.0122823 |
| codebook repair | 0.0241514360 | 0.0126783399 |
| Joint v2 | 0.0237162750 | 0.0123613843 |

这些不是 Update GR 微调后的部署 ranking，也不是 Future 指标。不同 encoder/预算的实验不能直接当成严格消融。

### 4.4 三个 Update seed 的配对结果

run=`paired_seeds_20261015T135505Z`，100steps、rho0.1、encoder frozen、gate disabled。2026重用已有两组，2027/2028各新训练Plain/History。相同seed内content顺序一致，跨seed不同；Base/teacher/cohort始终seed2026。

| seed | cohort | Plain H | History H | History−Plain |
| --- | --- | ---: | ---: | ---: |
| 2026 | train | 0.0872905617 | 0.0974458734 | +0.0101553117 |
| 2027 | train | 0.0915346648 | 0.0905229643 | −0.00101170048 |
| 2028 | train | 0.0935252069 | 0.0996246640 | +0.00609945712 |
| 2026 | holdout | 0.0961020831 | 0.0994556448 | +0.00335356171 |
| 2027 | holdout | 0.0999859306 | 0.0954230545 | −0.00456287607 |
| 2028 | holdout | 0.1016917530 | 0.1002828170 | −0.00140893640 |
| 2026 | Base val | 0.0916032363 | 0.0926572834 | +0.00105404708 |
| 2027 | Base val | 0.0897233789 | 0.0924885529 | +0.00276517400 |
| 2028 | Base val | 0.0977823836 | 0.0903605272 | −0.00742185641 |

跨seed paired H差均值±样本SD：train +0.00508102278±0.005653；holdout −0.000872750253±0.003985；Base val −0.00120087845±0.005455。平均 RQ MSE Plain0.000106929703，History0.000107032747。

seed2028 Base val H差CI=[−0.0124374634,−0.0024458622]，仅支持该run的用户条件改善，不能升级为跨seed稳定结论。三次Update采样共享一个Base/teacher/cohort，不是全流水线多seed。此轮ranking未评估，compute不相等。

### 4.5 最新权重扫描：完整终点记录

U-latest 回传的15组均100steps；rho0.1复用前轮，其余12组新训练，从同一Base开始。

| rho | seed | Plain H | History H | 差值 | reconstruction MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.1 | 2026 | 0.0872905617 | 0.0974458734 | +0.0101553117 | 0.000107348371 |
| 0.1 | 2027 | 0.0915346648 | 0.0905229643 | −0.00101170048 | 0.000106917266 |
| 0.1 | 2028 | 0.0935252069 | 0.0996246640 | +0.00609945712 | 0.000106832604 |
| 0.3 | 2026 | 0.0872905617 | 0.0908828719 | +0.0035923102 | 0.000107277879 |
| 0.3 | 2027 | 0.0915346648 | 0.0938739842 | +0.0023393194 | 0.000106866187 |
| 0.3 | 2028 | 0.0935252069 | 0.1036881650 | +0.0101629585 | 0.000107002265 |
| 1 | 2026 | 0.0872905617 | 0.1031256360 | +0.0158350744 | 0.000107938092 |
| 1 | 2027 | 0.0915346648 | 0.1149294390 | +0.0233947740 | 0.000107517530 |
| 1 | 2028 | 0.0935252069 | 0.1032095350 | +0.00968432785 | 0.000107608324 |
| 3 | 2026 | 0.0872905617 | 0.1169974910 | +0.0297069297 | 0.000108806347 |
| 3 | 2027 | 0.0915346648 | 0.1066859030 | +0.0151512384 | 0.000108826514 |
| 3 | 2028 | 0.0935252069 | 0.1118848150 | +0.0183596082 | 0.000108680262 |
| 10 | 2026 | 0.0872905617 | 0.1239286890 | +0.0366381273 | 0.000111506213 |
| 10 | 2027 | 0.0915346648 | 0.1425341060 | +0.0509994414 | 0.000110858186 |
| 10 | 2028 | 0.0935252069 | 0.1321333780 | +0.0386081713 | 0.000111030720 |

三个seed平均H：Plain0.0907834778；rho0.1/0.3/1/3/10分别0.0958645006/0.0961483404/0.1070882033/0.1118560697/0.1328653910，较Plain增加5.60%/5.91%/17.96%/23.21%/46.35%。这些均值是从已回传终点计算，非额外实验。

所有training_pass=false；`stop_reason=rho_limit_reached_without_stable_training_improvement`。不是仅因阈值严格：新增12组全部H更差。较大rho的MSE也恶化；没有观察到“牺牲重构换兼容性”的收益。

rho0.3/seed2026详细日志：全4096anchor/步、总history queries409600；实际历史梯度比例≈0.3；终点unique10824、max桶28、完整SID有效，recon0.00010727787875868421。第100步H从0.0910464546降到0.0908828729，但终点仍差于Plain。局部下降不保证终点优势。其他权重的完整码本占用、逐步日志、sweep总runtime/peak memory未知/待核实。

## 5. 已验证事实、解释与阻塞

已验证事实：内容Base链路、原Plain更新、固定旧target的E1损伤、repair条件性改善、v2/frozen/gate负结果、三seed不稳定、最新增权重负结果均有U证据。完整E0、完整原计划E1/E2、E3—E5不可因局部成功自动标completed。

解释 I：当前结果削弱“只需加大历史权重即可改善”的解释；与STE代理方向、离散边界/suffix切换及Adam轨迹共同作用的可能性相容。不能据此确定损失符号错误、STE必然无效、纯粹过拟合或研究命题被否定。

当前阻塞：

1. 指定commit与最新成功运行的CUDA启动实现/源码provenance尚未对齐。
2. cache/ckpt/results未随Git提交，云端不能读取服务器原始产物；部分路径/环境/运行成本未知。
3. 方法尚未稳定降低训练H，尚不能主张下游微调或Future收益。
4. 尚无完整公平baseline集合、等总成本、多Base seed及graph主实验。

## 6. 复现与接续命令：只作为文档，本轮不执行

### 6.1 第一优先任务的只读证据

用户在服务器项目根目录执行，下次提供必要片段，不上传数据集或checkpoint：

```bash
git rev-parse HEAD
git status --short
sha256sum run_history_weight_sweep.py history_weight_update.py
cat results/incremental/Beauty/weight_sweep_20261015T161912Z/compact.txt
cat results/incremental/Beauty/weight_sweep_20261015T161912Z/predeclared_plan.json
```

还需核对该run的`cuda_startup_fix_*.json`（是否存在未知/待核实）、`selection.json`、三个preflight及子任务summary。不能为了通过校验而删除manifest或改写旧实验事实。源码迁移应保留原plan备份和修复记录。

### 6.2 最新成功流程的历史启动/续跑接口

当时交付的续跑命令如下；服务器使用后回传completed，但完整shell执行记录未回传。指定commit未含启动修复，核对前不能直接当成无条件可运行的复现命令。

```bash
conda activate frec-gpu
cd /home/hao/frec/graph-informed-sids-web
source weight_sweep_20261015T161912Z_paths.sh
set -o pipefail
python -u -B run_history_weight_sweep.py \
  --previous-root "$WS_PREVIOUS_ROOT" \
  --e1-dir "$WS_E1_DIR" \
  --output "$WS_OUTPUT" \
  --workers 3 --devices 0 \
  2>&1 | tee "${WS_RUN_ID}_cuda_fix_resume.log"
```

路径文件在commit中：previous_root为`results/incremental/Beauty/paired_seeds_20261015T135505Z`，e1_dir为`results/incremental/Beauty/historical_damage_20261009T142825Z`。并行是在同一个GPU0上运行三个独立进程，不保证三倍加速。

**已完成后不再运行该命令来“继续找正结果”**。调度器可复用匹配的已完成结果、从中断checkpoint写入新attempt；它会拒绝不匹配plan及不完整evaluation目录。源文件变化影响source绑定，不能把warm-start与精确resume混淆。

### 6.3 较早阶段的可重建接口示例

以下参数由源码核对，输出必须是新目录；不是所有历史原始CLI已找回，也不是本轮要求执行：

```bash
python -B prepare_incremental_data.py \
  --reviews cache/AmazonReviews2014/Beauty/raw/reviews_Beauty_5.json.gz \
  --output "$NEW_PROTOCOL_DIR" --max-history 50

python -B train_incremental_rqvae.py \
  --features-dir cache/incremental/Beauty/features_20261008T135919Z \
  --output "$NEW_BASE_DIR" --device cuda --max-steps 5000 \
  --batch-size 256 --lr 0.001 --seed 2026

python -B train_plain_incremental_rqvae.py \
  --protocol-dir cache/incremental/Beauty/beauty_20261007T104452Z \
  --features-dir cache/incremental/Beauty/features_20261008T135919Z \
  --base-rqvae-dir ckpt/incremental/Beauty/base_rqvae_5000_20261009T082337Z \
  --output "$NEW_PLAIN_DIR" --device cuda --max-steps 1000 \
  --batch-size 256 --lr 0.001 --seed 2026
```

`NEW_*` 是操作者必须指定的全新目录，不是历史产物路径。Base实际曾从50→1000→5000累计恢复，因此上面fresh 5000模板不能声称逐字重现其原始命令。Base SID/TIGER命令先读取E1绑定，不猜目录；原始完整命令、所有显式/默认参数仍需保存config核实。

### 6.4 开发检查入口

先对照测试文件内容和依赖；下面仅列已有接口，本轮未重新运行这些模型测试：

```bash
git diff --check
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B tests/incremental/test_frozen_encoder_update.py
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B tests/incremental/test_multiseed_history_update.py
USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B tests/incremental/test_history_weight_sweep.py
```

先前助手报告过sweep 8项CPU合成测试、CUDA启动修复5项模拟测试通过；均为S级证据，不是用户V100回归。用户明确回传过Round3的5项synthetic检查通过。不能写“全部测试已在迁移commit/服务器通过”。

## 7. 未完成事项与交付约束

Update GR fine-tuning、Future test、主实验E3、完整E4/E5、LETTER/GRIS/graph对比、官方dataset identity验证、等总算力比较：未运行或未获证据。不要填零指标。

Git忽略cache、ckpt、results、二进制checkpoint、NumPy特征、ZIP和日志；部分根目录compact/path文件已跟踪。完整结果长期归档位置未知/待核实。后续每轮应提交精简、可审阅的事实摘要和产物哈希/路径，避免默认提交大数据、权重或私密环境文件。

本次交接没有授权新研究设计或额外训练。接手者先完成第0节的一项一致性核对，再基于明确需求安排后续实现。
