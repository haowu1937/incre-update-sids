# 三个 Update seed 的配对实验

保留已完成的 2026，新增 2027 和 2028；每个新 seed 各训练 Plain 与无门控 History-aware **100 步**，共四个新训练任务。Base RQ-VAE、旧 TIGER、内容特征、catalog、anchor_train／holdout、超参数均固定。本轮 encoder 仍冻结，历史梯度仅更新 codebook，rho=0.1。

## 安装和运行（ZIP 放项目根目录）

```bash
conda activate frec-gpu
cd ~/frec/graph-informed-sids-web
unzip history_update_multiseed.zip -d multiseed_bundle
python -B multiseed_bundle/apply_multiseed.py --repo .
source frozen_encoder_20261015T101457Z_paths.sh
bash multiseed_bundle/run_multiseed.sh "$FX_E1_DIR" "$FX_PLAIN_DIR" "$FX_HISTORY_DIR"
```

继续使用已有 frec-gpu，无新依赖，不下载模型、不改旧训练文件或已有 checkpoint。安装包包含之前的独立 auditor；若该文件已安装且完全相同，安装器会跳过它，若有不同本地修改则在写入前报错。

脚本会：

1. 保存预先声明的三 seed 计划及所有输出路径。
2. 对已有 2026 两组做独立预检。失败即停止，后续不训练。
3. **只评估、不重新训练 2026**，获得相同 anchors 上的独立 train／holdout／Base validation 可服务部分 NLL/H。
4. 按顺序训练 2027 Plain、2027 History-aware、2028 Plain、2028 History-aware，每组从完全相同的原始 Base 权重和重置 AdamW 开始。每完成一对，就做独立评估。
5. 汇总全部 2026、2027、2028，输出每个 seed 的差值与跨 seed 均值、样本标准差（ddof=1）。不按结果删掉 seed，不选择正结果。

预计约 40–60 分钟，实际以服务器运行时间为准。脚本没有 2026 的训练调用。即使先前相同 seed 的 rerun 已完成，也无需再训练 2026；可将其已完成两组路径作为第二、第三个参数。

## seed 到底改变什么

新入口显式使用 `--update-seed 2027`，写入 config 和 checkpoint。旧 Base checkpoint 的训练 seed、E1 cohort seed 仍保持 2026。新模型随机初始化的临时权重会被 Base state_dict 完全覆盖，已有 codebook 不重新 K-means 初始化；内容 MLP 无 dropout/BN，旧 TIGER eval 且冻结。因此，此实验考察的变化主要是 **Update 内容采样顺序**，不是重新训练 Base/TIGER，也不重新采样 anchors。

同一 seed 的两组使用相同的 private CPU sampler。每个真实已提交 batch 的 item IDs 被纳入可续训的链式 SHA256，终点评估会重建预期 batch 序列并严格校验。2026 的旧 sampler 没有链式字段，会根据其实际 epoch/order/cursor/generator state 与同一确定性 sampler 对照；新增两 seed 则还检查逐 batch 链。

新训练循环按之前已审查实现保留，只增加 seed 配置、来源绑定与 batch digest，无运行时替换函数。CPU 测试验证新入口 seed=2026 时两组的模型和 AdamW 与旧版完全相同。

## 固定比较设置

- Base、teacher、特征和全部 cohorts 来源绑定相同；不同 seed 的 cohort SHA256 必须相同。
- 每组固定 100 步终点，warm start 使用 C_old，不用旧 Plain Update 或 repair/gate 终点。
- Plain 历史梯度为零；History-aware 用完整 4096 训练 anchors，每步 batch=32 累积历史梯度，逐 codebook rho=0.1；temperature=0.1。
- 旧目标完整 SID 加 EOS 保持不变，仅重编码可用历史；完整 SID 容量仍严格检查，非法候选回滚并停止，不扩容。
- 独立评估从原始历史构造 token IDs，手动右移旧 decoder 标签，使用 log-softmax/gather 计算 NLL，不复用 STE 的 inputs_embeds 前向评估。
- 训练/holdout 用户不重叠，holdout 不用于训练或挑选 checkpoint；它在之前探索中已被查看，不应视为全新盲测。
- 排名与 future test 不运行；Base validation 的 NLL 只对有旧标签的可服务目标定义。
- 同一步数不等算力；History-aware 的历史前向/反向额外计算量会如实保留。

## 输出与解读

项目根目录保存 `paired_seeds_<UTC>_paths.sh`、各阶段日志及 `paired_seeds_<UTC>_compact.txt`。只需分享最后一个约 30 行摘要：

```bash
cat "$(ls -t paired_seeds_*_compact.txt | head -1)"
```

负的 `history-minus-plain` 表示 History-aware 的正部损伤更小。先看每个 seed 的方向，再看三个配对差值的均值和样本标准差。用户聚类 bootstrap 区间保存在每个 seed 的 JSON 报告中，不能把这些 query 区间当成跨 seed 的稳定性证据，也不将所有 seeds 的重复 query 合并成更多独立样本。

这是**固定一个 Base/teacher/cohort 条件下的三个 Update sampling seeds**，不能称为整条训练 pipeline 的多 seed 结果。仅三个 seed 的标准差也不是充分的跨 seed 显著性结论。

## 中断后恢复

每 10 步保存一次 checkpoint。若中断，不要再次运行整个 shell 脚本丢弃已完成任务。加载根目录的路径文件后，用新目录恢复未完成组，保持同 seed／mode／配置：

```bash
source paired_seeds_实际时间_paths.sh
python -u -B history_update_multiseed.py train \
  --mode history-aware --update-seed 2027 --e1-dir "$MS_E1_DIR" \
  --output "${MS_HISTORY_2027}_resumed" --resume-from "$MS_HISTORY_2027/last.pth" \
  --device cuda --max-steps 100 --history-gradient-ratio 0.1 \
  --anchor-batch-size 32 --temperature 0.1 --log-every 10 --save-every 10 \
  --history-progress-every 0
MS_HISTORY_2027="${MS_HISTORY_2027}_resumed"
```

达到 100 步后不再继续训练。完成剩余组，再分别评估和汇总。示例：

```bash
python -u -B history_update_multiseed.py evaluate \
  --update-seed 2027 --expected-steps 100 --e1-dir "$MS_E1_DIR" \
  --plain-dir "$MS_PLAIN_2027" --history-dir "$MS_HISTORY_2027" \
  --output "$MS_EVAL_2027" --device cuda --batch-size 32 --progress

# 其余 seed 评估相同，只替换 seed 与对应路径。已完成的评估不用重复。
python -u -B history_update_multiseed.py aggregate \
  --evaluation-dirs "$MS_EVAL_2026" "$MS_EVAL_2027" "$MS_EVAL_2028" \
  --output "$MS_AGGREGATE_DIR"
python -B multiseed_bundle/show_multiseed.py "$MS_AGGREGATE_DIR" \
  | tee "${MS_RUN_ID}_compact.txt"
```

恢复会重新缓存一次固定旧 NLL，因此累计实际缓存计算计数可增加，但模型、AdamW、RNG、sampler 和 batch digest 必须精确复现不中断的运行。所有训练/评估都拒绝覆盖现有输出目录，避免混合不同版本。

## 验证范围

CPU 合成数据与仓库实际 RQ-VAE/TIGER：旧版 seed=2026 的双组精确数值等价；新增 seed 的 Base/teacher/anchors 固定、同 seed 两组批次一致、不同 seed 批次不同；两组精确断点续训和 digest；独立三 seed 评估、配对均值和样本标准差；拒绝错误 seed/预算、错误 digest、重复或缺失 seed、跨 seed cohort 不一致。详情见 validation.json。真实 Beauty/GPU 实验由用户运行。
