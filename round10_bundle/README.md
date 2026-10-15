# Round 10：普通增量内容 RQ-VAE

已按要求跳过热门商品基线。本包直接进入普通增量码本更新，作为 historical-aware
training 的对照。当前没有加入历史预测惩罚，也没有计算历史预测损伤。
所有真实数据训练命令由你在自己的服务器上手动执行；本地验证仅在独立 CPU 沙箱中运行。

## 安装

把 ZIP 上传到仓库根目录，在你已经配置的 `frec-gpu` 环境中执行：

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip round10_plain_update_rqvae_bundle.zip -d round10_bundle
python -B round10_bundle/apply_round10.py --repo .
python -B tests/incremental/test_plain_update_rqvae_round10.py
```

只新增三个 Python 文件，不修改 Round 1–9 的文件。前置依赖为 Round 7 的真实 RQ-VAE
与 step trainer，以及 Round 8 的协议工具和合成测试 fixture。无需下载模型或额外安装依赖。
测试应显示 `Ran 8 tests`、`OK`，使用真实 RQ-VAE 的小型合成输入。

## 普通 Update 训练：1000 个增量步

```bash
set -o pipefail

PROTOCOL_DIR="cache/incremental/Beauty/beauty_20261007T104452Z"
FEATURES_DIR="cache/incremental/Beauty/features_20261008T135919Z"
BASE_RQVAE_DIR="ckpt/incremental/Beauty/base_rqvae_5000_20261009T082337Z"
UPDATE_RUN_ID="plain_update_rqvae_1000_$(date -u +%Y%m%dT%H%M%SZ)"
UPDATE_RQVAE_DIR="ckpt/incremental/Beauty/$UPDATE_RUN_ID"

python -u -B train_plain_incremental_rqvae.py \
  --protocol-dir "$PROTOCOL_DIR" \
  --features-dir "$FEATURES_DIR" \
  --base-rqvae-dir "$BASE_RQVAE_DIR" \
  --expected-base-sha256 b1d696a8bd086b8c76f01afafd4380bfba14c05413abe17923459b73ac0b7b8d \
  --output "$UPDATE_RQVAE_DIR" \
  --device cuda \
  --max-steps 1000 \
  --batch-size 256 \
  --lr 0.001 \
  --seed 2026 \
  2>&1 | tee "${UPDATE_RUN_ID}.log"

cat "$UPDATE_RQVAE_DIR/summary.json"
cat "$UPDATE_RQVAE_DIR/base_semantic_churn.json"
```

1000 步是 Update 阶段的新增预算，不包含 Base 的 5000 步。
本次默认先直接运行 1000 步，不增加额外的真实数据 smoke 或热门商品验证环节。
初始化阶段检查文件 SHA、Base checkpoint 数据/代码/软件绑定、时间协议及全部特征行的
ID 对应关系。出现 `Plain Update ready` 后每 50 步打印损失，每 100 步保存断点。

## 训练规则

- 从已确认的 Base RQ-VAE 5000 步 checkpoint 加载 encoder、decoder 和三层 codebook
  的全部权重；保留已初始化标记，不调用 k-means，不重编号/对齐 codeword。
- Update 开始时重新建立 AdamW。默认 lr 0.001、weight decay 0、batch 256、seed 2026。
  架构和重构/量化损失参数继承 Base 配置。所有模型参数均参与普通重构训练。
- 使用 `update_train` 目录的全部 11707 个固定内容向量，均匀随机按 epoch 洗牌。
  这是目录级内容训练，包含旧商品内容，不是只训练新增商品或按交互次数加权。
- 其中 10924 个商品在 Base training 目录，255 个首次出现于 Base validation，
  528 个首次出现于 Update training。后两组共 783 个是 Base training 之后新增的商品。
- 不读取 `features/update/` 的 11775 商品部署视图；其中仅在 Update validation
  出现的 68 个额外商品不参与训练。也不读取 Future test 内容或样本。
- 固定内容向量保持不变，无额外 normalization、PCA、图、CF loss、Sinkhorn
  或 content encoder finetuning。本阶段没有历史样本预测损失，也没有历史损伤惩罚。
- 完成时选择固定 Update 步数预算末尾的 `last.pth`。不根据历史损伤或验证指标选择码本。
  输出目录必须是新目录；原始 Base checkpoint 和 Base SID 文件不被覆盖。

## 输出与解释

- `last.pth`：普通更新后的 RQ-VAE，checkpoint format 为 `plain_update_rqvae_v1`。
- `semantic_assignments.jsonl`：新码本对 Update training 全目录的三位原始语义码。
- `old_semantic_assignments.jsonl`：原 Base 码本对同一目录的三位原始语义码。
- `base_assignment_changes.jsonl`：10924 个 Base training 商品的逐行新旧码和变化标记。
- `base_semantic_churn.json`：旧商品整体、逐层和前缀变化比例，不包含碰撞后缀。
- `metrics.jsonl`：本次运行的每步损失；续训日志从该次开始步数记录。
- `summary.json`：来源绑定、增量步数、各组重构损失、碰撞统计、重载一致性、时间和
  CUDA peak memory。初始 Base 语义码必须与原导出完全一致。

为避免导出 batching 的数值差异，Base 商品沿用原 Base 导出记录的 evaluation batch size；
新增商品按两个稳定 registry 区间导出。此约定保存于 config 和 binding，便于后续复现。

语义码变化比例本身不是历史预测损伤。当前不生成新版本的碰撞后缀或 TIGER 训练张量，
也不训练推荐模型。`historical_prediction_damage_status` 与 `E1_status` 保持 `not_run`。
下一步才建立版本化完整 SID，以冻结的 Base TIGER `best.pth` 比较旧历史和新历史输入，
主实验保持目标的旧 SID 标签固定。

## 中断续训

若需要继续某份已写出的 Update `last.pth`，保持数据/代码/软件/设备/训练配置不变，
设定新的输出目录，在上面的命令中增加：

```bash
--resume-from "ckpt/incremental/Beauty/<previous_update_run>/last.pth"
```

此时 `--max-steps` 表示累计 Update 步数。例如 1000 步断点继续到 2000，会新增 1000 步。
`--resume-from` 不接受 Base RQ-VAE 或 TIGER checkpoint。普通更新开始的父模型始终仍是
上面已确认的 Base checkpoint，不能在续训时改变。

请返回正式运行的 `summary.json` 和 `base_semantic_churn.json`。
