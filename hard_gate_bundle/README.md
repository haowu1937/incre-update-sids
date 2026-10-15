# Hard-H candidate gate: frozen encoder Update pilot

这轮从原始 Base C_old 重新开始训练 100 个已提交步骤，并复用已完成的冻结 encoder Plain 100 步参照。可同时展示上一轮未加 gate 的 History-aware 结果。不从失败的 History-aware 终点继续。

## 在项目根目录安装运行

把 ZIP 放在 `~/frec/graph-informed-sids-web/`，使用现有 frec-gpu 环境：

```bash
conda activate frec-gpu
cd ~/frec/graph-informed-sids-web
unzip history_aware_hard_gate.zip -d hard_gate_bundle
python -B hard_gate_bundle/apply_gate.py --repo .
source frozen_encoder_20261015T101457Z_paths.sh
bash hard_gate_bundle/run_gate.sh "$FX_E1_DIR" "$FX_PLAIN_DIR" "$FX_HISTORY_DIR"
```

`FX_*` 来自你刚完成的冻结 encoder 两组实验。安装器校验此前源码，新增本轮训练文件和合成测试文件，不修改旧版实现。无新依赖，无模型下载。脚本只训练新组，已有 Plain 和未加 gate 的 History-aware 不重跑。若不需要展示未加 gate 的结果，可省略第三个参数。

## 固定协议

- 冻结内容 RQ-VAE encoder，并检查其权重与 Base 完全相同；decoder 和三个 codebook 接收内容损失梯度。旧 TIGER、内容特征编码器与旧目标标签保持固定。
- 仍使用原来的内容 batch size、AdamW、seed、训练目录、采样顺序与末尾小 batch。Historical 梯度只更新 codebook，逐层归一化比例 rho=0.1、temperature=0.1，与刚失败的冻结 encoder History-aware 相同。
- 每步从同一当前模型、AdamW moments、RNG 和同一内容 batch，生成两个候选。内容梯度只计算一次；历史梯度使用全部 4096 anchor_train，并分批 32 累积。
- **内容候选**：只用内容梯度执行一次 AdamW proposal。
- **历史候选**：用相同内容梯度加归一化历史梯度执行一次 AdamW proposal。若两组梯度完全相同，则复用同一候选，避免重复前向。历史梯度不能覆盖 decoder 的内容梯度。
- 两个候选都以完整新 SID 历史重编码，测量真实 `H = mean(max(NLL_new-NLL_old,0))`；旧目标完整 SID 加 EOS 不变。每个候选单独检查完整目录 MSE 与严格 suffix 分配容量。
- 历史候选必须满足 `H_joint < H_content - 1e-6`、`MSE_joint <= 1.01*MSE_content` 且 SID 分配可行，才被采用；否则采用内容候选。平局采用内容候选。
- 如果内容候选本身不合法，则回滚本轮并停止在上一个合法 checkpoint。不会在缺少合法内容参照时继续，也不切换旧 suffix、不扩容。
- 选中后一起提交该候选的**模型、AdamW 状态、RNG 状态**，采样器只推进一次。两个 proposal 不是连续训练两步。checkpoint 每 10 个已提交步骤保存一次。
- 不要求 `H_selected < H_previous`。初始 H=0 不阻断内容更新。这里约束的是**同一当前起点**的内容候选，非独立 Plain 轨迹。
- 固定终点 100 步。gate 只用训练 anchors；不评估 holdout、Base validation 预测/排名或 future test。已有 E1 报告仅用于输入来源绑定校验。

## 能保证什么

每个成功提交步骤满足 `H_selected <= H_content_candidate`。这个局部条件**不保证最终 H 优于独立 Plain**：两条模型与 AdamW 轨迹会在接受历史候选后分开。重构 cap 也是相对当步内容候选的局部 cap，不能当作终点相对独立 Plain 的 cap。

最终仍需对比独立冻结 encoder Plain 的训练 H、MSE、碰撞和配对差值区间。区间只描述训练 cohort，不能证明 holdout 改善。即使训练 H 下降，也需要后续单独的 holdout 验证。

## 运行时间和输出

比前一轮每步额外增加候选 hard-H 测量与全目录 MSE 评估，不能视为等算力对照。按前轮速度粗略估计，本轮 100 步可能需约 25–40 分钟；以日志实际 step 时间为准。控制台输出第 1 步及每 10 步的简洁进度；详细候选记录保存到 `metrics.jsonl`。

项目根目录会保存：

- `hard_gated_<UTC>_paths.sh`：本轮路径。
- `hard_gated_<UTC>_train.log` 和 `_comparison.log`。
- `hard_gated_<UTC>_compact.txt`：约 30 行摘要，只需分享这个文件。

checkpoint 在 `ckpt/incremental/Beauty/`，结果对照在 `results/incremental/Beauty/`。compact 里的 `content_candidate_H` 是 gate 自己当前轨迹上的内容候选，不是独立 Plain 在相同步骤的分数。

手动重新打印摘要：

```bash
source hard_gated_实际时间_paths.sh
python -B hard_gate_bundle/show_gate.py "$HG_COMPARE_DIR"
```

## 中断后续训

使用新输出目录，从本轮保存的 `last.pth` 继续到**累计 100 步**；不能从旧版未加 gate 的 checkpoint 恢复。超参数、Plain 来源和源码绑定必须相同。若 checkpoint 已达 100 步，无需继续训练。示例：

```bash
source hard_gated_实际时间_paths.sh
python -u -B history_aware_hard_gate.py train \
  --e1-dir "$HG_E1_DIR" --plain-dir "$HG_PLAIN_DIR" \
  --output "${HG_RQVAE_DIR}_resumed" --device cuda --max-steps 100 \
  --history-gradient-ratio 0.1 --anchor-batch-size 32 --temperature 0.1 \
  --min-h-improvement 0.000001 --rq-mse-ratio-cap 1.01 \
  --log-every 10 --save-every 10 --resume-from "$HG_RQVAE_DIR/last.pth"

HG_RQVAE_DIR="${HG_RQVAE_DIR}_resumed"
python -u -B history_aware_hard_gate.py compare \
  --rqvae-dir "$HG_RQVAE_DIR" --output "$HG_COMPARE_DIR" \
  --ungated-dir "$HG_UNGATED_DIR"
python -B hard_gate_bundle/show_gate.py "$HG_COMPARE_DIR" \
  | tee "${HG_RUN_ID}_compact.txt"
```

如未提供未加 gate 的参照，省略 `--ungated-dir`。恢复输出中的 `metrics.jsonl` 只含本次新增步骤，之前日志保留在原目录，累计验收计数保存在 checkpoint 中。

## 验证范围

CPU 合成数据、仓库实际 RQ-VAE 与 TIGER：验收门槛与质量/容量拒绝、两个候选恢复相同 AdamW moments、所选模型/优化器/RNG 一起提交、rho=0 精确复现 Plain、真实候选 hard-H 局部条件、精确断点续训、内容候选无效时完整回滚、配对训练报告和错误输入。安装器也验证重复安装和冲突时不写入其他文件。详情见 validation.json 与 validation.log。

合成测试不证明本方法在 Beauty 上有效；真实 GPU 实验由用户执行。
