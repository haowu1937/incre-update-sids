# Round 9：Base 纯内容 SID → 仓库 TIGER

前置条件：Round 1–8 文件已安装；Base frozen SID preparation 已完成。
本轮直接使用仓库 `genrec.models.TIGER.model.TIGER`、`Trainer`、`Evaluator`，
不重新训练内容编码器或 RQ-VAE，不加载图，不训练 Update，也不评估 Future test。

## 安装与验证

在 `graph-informed-sids-web` 仓库根目录执行。把下载的 ZIP 放到此目录：

```bash
unzip round9_base_tiger_bundle.zip -d round9_bundle
python -B round9_bundle/apply_round9.py --repo .
python -B tests/incremental/test_temporal_tiger_round9.py
```

安装器先检查全部文件的原始版本和包内 SHA，再写入并备份两份原有文件。
重复执行不会覆盖已安装的同一版本。若本地原文件有额外修改，安装器在写入前退出。
测试只使用合成数据和真实小型 RQ-VAE/TIGER；也检查实际默认 TIGER 架构、
1026 词表、201 输入位置和目录约束解码。不读取服务器的真实 Beauty 数据。

## 50 步 GPU 检查

```bash
set -o pipefail
SID_DIR="$(python -B round9_bundle/find_base_sids.py)"
printf 'SID_DIR=%s\n' "$SID_DIR"
TIGER_SMOKE_ID="base_tiger_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
TIGER_SMOKE_DIR="ckpt/incremental/Beauty/$TIGER_SMOKE_ID"

python -u -B train_incremental_tiger.py \
  --sid-dir "$SID_DIR" \
  --expected-rqvae-sha256 b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d \
  --output "$TIGER_SMOKE_DIR" \
  --device cuda \
  --max-steps 50 \
  --smoke \
  2>&1 | tee "${TIGER_SMOKE_ID}.log"
```

查找脚本按已确认的协议 SHA 与 5000 步 Base RQ-VAE SHA 筛选，打印最新的已完成 SID
目录。训练入口随后验证全部 SID 文件的 SHA、可服务目录、目标标签、历史 mask 与审计行。
如需指定另一份相同来源的导出，可直接设置 `SID_DIR`。

50 步检查包含固定 512 个原始验证样本的目录约束排名，其中目录外目标也计入分母。
日志显示 `TIGER ready` 后，每 25 步输出 CE、学习率和耗时，验证阶段显示批次进度。
成功后检查：

```bash
cat "$TIGER_SMOKE_DIR/summary.json"
```

应出现：`optimizer_steps: 50`、`invalid_predictions: 0`、
`selected_checkpoint_reload_equal: true`。Smoke 不评估完整 Base 验证集。
GPU 适配尚需这一步在你的 V100 上确认；本地检查使用 CPU。

## 继续到累计 5000 步并完整验证

在同一终端执行，保留上面的 `SID_DIR` 与 `TIGER_SMOKE_DIR`：

```bash
TIGER_RUN_ID="base_tiger_5000_$(date -u +%Y%m%dT%H%M%SZ)"
TIGER_DIR="ckpt/incremental/Beauty/$TIGER_RUN_ID"

python -u -B train_incremental_tiger.py \
  --sid-dir "$SID_DIR" \
  --expected-rqvae-sha256 b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d \
  --output "$TIGER_DIR" \
  --device cuda \
  --max-steps 5000 \
  --resume-from "$TIGER_SMOKE_DIR/last.pth" \
  2>&1 | tee "${TIGER_RUN_ID}.log"

cat "$TIGER_DIR/summary.json"
cat "$TIGER_DIR/base_validation.json"
```

`--max-steps 5000` 是累计总步数，续训新增 4950 步。
如换终端，重新指定之前的 `SID_DIR` 和 `TIGER_SMOKE_DIR` 路径。
任意中断后可指定已写出的 `last.pth`，使用新的输出目录续训至更大的累计步数。
`last.pth` 默认每 250 步以及本次最后一步写入；写入通过临时文件再原子替换。
输出目录必须不存在。改变模型、数据、软件版本、设备、batch size 或训练/评估配置时，
续训会拒绝；不要在 smoke 与正式训练之间修改这些值。

## 本轮明确的训练与评估选择

- 模型结构沿用仓库 TIGER：4 层 encoder、4 层 decoder，d_model 128、d_ff 1024、
  6 个 attention heads、d_kv 64、dropout 0.1，随机初始化 T5Config，无 HF 下载。
- 词表由冻结 SID 决定：三层语义码加一位固定碰撞后缀，padding 0、EOS 1025、
  vocab 1026、n_user_tokens 0。完整 SID 唯一。Base validation 仅允许 10924 个
  Base training 商品；Base deployment 目录的额外 255 个商品不进入候选。
- AdamW lr 0.003、weight_decay 0.05 沿用仓库 TIGER 配置；batch 128、FP32、
  梯度裁剪 1.0。学习率采用 200 步 warmup 后恒定，便于精确延长训练预算。
  这项调度和 batch 与原 epoch trainer 默认值不同，完整保存于 config.json。
- 精确保存 optimizer、scheduler、随机状态与 CPU 随机采样顺序/位置。
  评估前后恢复训练随机状态，不改变后续 dropout 序列。无 early stopping。
- 输入 batch 仅裁掉尾部 padding；无法服务的历史事件仍占原四个位置，attention 为零。
  序列中的 EOS 始终有效。目标 SID 的四位加 EOS 共五位参与交叉熵。
- 每 500 步及本次最后一步评估固定 512 个验证样本。样本根据 seed 与 sample_id 的
  SHA 顺序从全部 9192 个样本抽取，保留目录外样本。probe.json 保存名单。
- 用 probe 的完整分母 NDCG@10 选择 best，分数相同时保留较早 checkpoint。
  Smoke 最后一步也成为候选；续训保存它的 best 状态。完成后从导出的 best.pth
  重新加载模型，对全部 9192 个 Base 验证样本计算一次指标。
- Beam 20、top-k 5/10，按 Base training 完整 SID 前缀约束，EOS 在四位 SID 后结束。
  报告 beam search 的排名指标；它是 beam 近似，并非穷举全目录精确 top-k。
  不过滤历史出现过的商品。对每个最终候选检查目录合法性、唯一性和有限得分。
- 完整分母为 9192，其中 503 个不可服务目标计为未命中；条件分母为 8689。
  CE/NLL 只在可服务目标上定义，同时提供 token CE 和五位 sequence NLL。
- 本轮只有 Base 推荐训练和 Base validation；E0_full_status 与 E1_status 保持 not_run。
  正式运行完成时 E0_base_end_to_end_status 为 completed，并不声明已收敛。

## 输出文件

`last.pth` 保存当前训练状态，供精确续训；`best.pth` 保存选中的 Base 推荐模型，供后续
冻结旧模型及 E1 实验使用。`train_metrics.jsonl` 逐步保存 CE、梯度范数和学习率。
`base_validation.json` 保存正式运行的完整排名和 NLL。
`summary.json` 保存数据/代码/软件绑定、步数、最佳步数、目录/分母、运行时间和
本进程的 CUDA peak allocated/reserved 内存。完整验证的耗时可能明显超过 probe。

本地证据见 `validation.json` 和 `test_round9.log`。不能据合成测试宣称真实 Beauty 推荐效果。
