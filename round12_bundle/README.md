# Round 12: 历史兼容性 RQ-VAE Update

本包由你在自己的 Linux 服务器安装和运行。这里的 CPU 合成测试不代表已经运行 Beauty 或 V100 实验。

## 当前 E1 结果

普通 Update 后，anchor_holdout 的完整 SID 输入使平均旧目标序列 NLL 增加 0.0523633，用户簇 bootstrap 95% 区间为 [0.0392136, 0.0659749]。这个留出组只相对于未来兼容性惩罚训练留出；Base TIGER 已经见过这些 Base 训练样本。

Base 验证集平均 ΔNLL 为 0.0001097，区间 [-0.0125450, 0.0124373]，不能认定平均 NLL 上升。完整分母 Recall@10 从 0.0253481 降到 0.0239339（233→220 次命中），NDCG@10 从 0.0133026 降到 0.0126166；相对降幅约 5.58% 和 5.16%。排名降幅尚未做配对置信区间。不能声称所有指标或所有样本都恶化。

保留旧 suffix 的输入诊断仍有正损伤，不支持“全部损伤来自 suffix”解释；这不是可以部署的唯一完整 SID 编码。

## 训练目标与范围

从 C_old 的全部权重重新开始，沿用普通 Update 的 AdamW 重置、学习率、batch size、seed、内容采样与 1000 个 Update optimizer steps。不是继续训练普通 Update 的 C_new，也不重新初始化码本。

`loss = repository_RQ_loss + lambda * mean(relu(NLL(M_old, new_history, old_target) - old_NLL))`

第一轮固定 lambda=0.001、temperature=0.1、anchor batch size=32。RQ loss 为仓库原始重构+量化损失。NLL 是三位语义码、一位 suffix 和 EOS 的五个旧目标 token 损失之和。lambda 用序列 NLL 尺度定义，不是平均 token CE 尺度。

仅 E1 保存的 anchor_train 参与惩罚；anchor_holdout 和 Base validation 不参与梯度、调参或 checkpoint 选择。内容训练仍均匀采样 UpdateTrain 的 11707 个 eligible 商品（包括旧商品），不加入 68 个 UpdateVal-only 商品或 Future-only 商品。固定内容 encoder 不训练。TIGER 权重、旧目标 SID、旧候选目录保持固定。未运行新推荐模型或 Future test。

## 梯度近似

每个 optimizer step 都按原来的分组和 batch 重新量化整个 eligible 目录，用 Round11 同一个版本分配规则保留旧 suffix 槽位并分配新完整 SID；硬前向包括这些真实 suffix。与旧历史相同的输入，其 ΔNLL 强制为零，避免预缓存参考 NLL 的 batch 浮点差异造成假惩罚。

整数 argmin 本身没有可用的反向导数。代码使用 straight-through token embedding：`hard_embedding + soft_embedding - soft_embedding.detach()`，实现中先计算括号内零项，保证前向逐位等于真实硬 embedding。概率来自三层残差量化距离的 softmax。各层距离尺度在 C_old 的前至多 4096 个 Base 商品上一次性校准并固定。残差软路径让 encoder 和三层 codebook 都能接收梯度；TIGER 参数 requires_grad=False，但该前向不能放在 no_grad 中。

**suffix 分配仍是离散过程，没有通过分配器的梯度。** 它的实际 token 会进入前向 NLL，但兼容性梯度只对语义码使用近似。训练中不做码字对齐。报告不能把这个梯度近似称为完整离散 SID 目标的精确导数。最终评估重新读取硬 assignment 和真实完整 SID，并同时保留旧 suffix 诊断。

训练使用原仓库 `Trainer.fit_steps` 和 optimizer，通过 opt-in loss hook 加兼容性项。本包只增加四个新文件，不修改任何绑定到现有 checkpoint 的源文件。lambda=0 的合成测试要求权重、sampler 和指标逐位复现普通 Update；非零 lambda 的断点续训要求与不中断训练逐位一致。

## 手动安装

把 ZIP 放到仓库根目录：

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip round12_historical_update_bundle.zip -d round12_bundle
python -B round12_bundle/apply_round12.py --repo .
```

保持原有环境和依赖版本，无需重新安装依赖或下载模型。可选的 CPU 合成检查：

```bash
python -B tests/incremental/test_historical_update_round12.py
```

该合成测试不会读写真实 Beauty 目录或运行 GPU。

## 先运行 50 步，再续到累计 1000 步

```bash
set -o pipefail
E1_DIR="$(python -B round12_bundle/find_e1.py)"
printf 'E1_DIR=%s\n' "$E1_DIR"
HIST_RUN_ID="historical_rqvae_$(date -u +%Y%m%dT%H%M%SZ)"
HIST_SMOKE_DIR="ckpt/incremental/Beauty/${HIST_RUN_ID}_smoke"
HIST_RQVAE_DIR="ckpt/incremental/Beauty/${HIST_RUN_ID}_1000"

python -u -B train_historical_incremental_rqvae.py \
  --e1-dir "$E1_DIR" --output "$HIST_SMOKE_DIR" \
  --device cuda --max-steps 50 \
  --compatibility-weight 0.001 --temperature 0.1 --anchor-batch-size 32 \
  2>&1 | tee "${HIST_RUN_ID}_smoke.log"

python -u -B train_historical_incremental_rqvae.py \
  --e1-dir "$E1_DIR" --output "$HIST_RQVAE_DIR" \
  --device cuda --max-steps 1000 \
  --compatibility-weight 0.001 --temperature 0.1 --anchor-batch-size 32 \
  --resume-from "$HIST_SMOKE_DIR/last.pth" \
  2>&1 | tee "${HIST_RUN_ID}_1000.log"
```

`max-steps` 是累计 Update 步数。正式第二段只新增 950 步，不是再新增 1000 步。两段的目标和超参数必须相同；绑定不一致会拒绝续训。训练日志每 25 步打印 RQ loss、原始/加权历史惩罚、anchor ΔNLL、梯度范数；每 50 步保存 last.pth。完整目录再编码和 frozen TIGER 反向传播会比普通 RQ-VAE Update 慢，实际耗时由你这次 GPU 日志确定。

若 `find_e1.py` 没找到目录，手动令 `E1_DIR=results/incremental/Beauty/你的实际E1目录`。入口会验证其协议、已选择 teacher、普通 Update、保存的 anchor 与源文件绑定；不按名称跳过这些检查。

## 相同步数的匹配评估

```bash
HIST_EVAL_DIR="results/incremental/Beauty/${HIST_RUN_ID}_evaluation"
python -u -B evaluate_historical_update.py \
  --e1-dir "$E1_DIR" --historical-rqvae-dir "$HIST_RQVAE_DIR" \
  --output "$HIST_EVAL_DIR" --device cuda --batch-size 128 --ranking \
  2>&1 | tee "${HIST_RUN_ID}_evaluation.log"

python -B round12_bundle/show_results.py "$HIST_EVAL_DIR"
```

入口拒绝把 50 步 smoke 与 1000 步 plain 作为匹配对照。正式评估沿用 E1 的同一用户/样本、旧标签、旧历史可用性、旧候选目录以及 beam 设置。验证 NLL 只使用 8689 个可标注样本，排名仍以 9192 为完整分母，其余 503 个目标计为 miss。

优先查看：

1. anchor_holdout 的 `mean_compatible_minus_plain_sequence_nll` 与用户簇配对 CI：负值说明比 plain 更好；CI 上界小于零才支持该方向的明确证据。
2. 与 C_old 比较的 ΔNLL、正部均值和恶化比例，避免均值中改善/损伤抵消。
3. Base validation NLL 和 Recall/NDCG，排名差异仍只有点估计。
4. RQ 重构/量化指标、语义 churn、codebook 使用与 collision，观察兼容性与内容学习之间的取舍。

降低 churn 本身不是成功判据；通过冻结编码而失去 Update 内容学习也不算完成研究目标。这里仅验证旧 TIGER 的历史输入兼容性，尚未评估新推荐模型性能。
