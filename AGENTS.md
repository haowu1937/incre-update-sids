# 项目协作规则

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
