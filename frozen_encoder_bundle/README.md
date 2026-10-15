# Frozen encoder: matched Plain / History-aware pilot

在项目根目录安装并运行。真实数据和 GPU 运行由用户完成。

```bash
unzip history_aware_frozen_encoder.zip -d frozen_encoder_bundle
python -B frozen_encoder_bundle/apply_frozen.py --repo .
bash frozen_encoder_bundle/run_pair.sh \
  results/incremental/Beauty/historical_damage_20261009T142825Z
```

运行脚本顺序完成：冻结 encoder 的 Plain 100 步、冻结 encoder 的 History-aware 100 步、训练 anchors 对照汇总。保持现有 frec-gpu 环境，无需下载模型或安装新依赖。仅新建两个源码文件，安装前检查依赖源码版本；不会更改已绑定 checkpoint 的 v2 实现。

## 固定协议

- 两组都从 E1 绑定的 **原始 Base C_old** 出发。不是从失败的 v2 1000 步、Plain Update 或 repair checkpoint 继续。
- 冻结的是内容 RQ-VAE 的 encoder MLP。两组 decoder 和三个 codebook 仍接收内容损失梯度。检查 encoder 权重/缓冲与 Base 完全相同，encoder 没有梯度；AdamW 对无梯度的 encoder 不更新或创建状态。
- Plain: rho=0。History-aware: rho=0.1，逐层 `g = g_RQ + stopgrad(rho*||g_RQ||/(||g_H||+1e-12))*g_H`。任一范数为零时额外梯度为零。使用相同的 AdamW 设置、seed、batch size、完整/末尾小 batch 与采样顺序。
- 每个 History-aware 步使用全部 4096 anchor_train，分批 32 累积均值。H 使用完整新 SID 历史和冻结旧 TIGER，旧目标标签固定。三语义位使用 v2 相同 STE，suffix 不求导。Plain 不计算 H 梯度。
- 固定终点 100 步，不按 holdout 或训练损伤挑选终点。每 10 步额外测量一次更新后的全训练 anchors hard H，便于两组比较；记录中的 `history_before_optimizer_step` 是更新前值，勿与 after 混用。
- 每步严格检查完整 SID 分配容量；若溢出，则回滚模型、AdamW、sampler、RNG 并停止，保留最后可行 checkpoint，不扩容、不切换旧 suffix。
- 最终报告包含 full-catalog reconstruction MSE、碰撞、base churn、逐查询 training H 与按用户聚类的配对差值区间。该区间仅描述训练 cohort，不能证明泛化改善。
- 不计算 holdout、Base validation 的预测或排名，不运行 future test。旧 E1 输入绑定校验会读取已有报告，但不把 holdout 用作训练、选择或本轮预测评估。
- 等步数不等计算量。History-aware 会多处理 409,600 个历史梯度 queries；按前轮约 9 秒/步估计，100 步 History-aware 约需 15 分钟，另加初始化与探测时间，实际以本机日志为准。

## 输出

项目根目录自动保存 `frozen_encoder_<UTC>_paths.sh`、两组日志和 `frozen_encoder_<UTC>_compact.txt`。只需要分享 compact 文件（通常约 25 行）。模型在 `ckpt/incremental/Beauty/`；对照在 `results/incremental/Beauty/`。

负的 `history_aware_minus_plain_positive_damage` 表示 History-aware 在这组训练 anchors 上损伤更小。还需同时看两组绝对 H、重构误差、碰撞及轨迹；本轮无论结果如何，都不能直接下 holdout 改善的结论。原始未冻结 Plain 的 1000 步结果仅保留为背景引用，不能替代本轮匹配 Plain 对照。

## 中断后恢复

每 10 步原子保存 `last.pth`。若进程中断，使用新输出目录和同一组的 `--resume-from OLD_DIR/last.pth`。不要用另一组或 v2 checkpoint。继续目标仍为累计 100 步；有效步数已达到 100 时无需重跑训练。`--mode`、rho、anchor batch size、device 和 temperature 必须匹配。示例：

```bash
source frozen_encoder_实际时间_paths.sh
python -u -B history_aware_frozen_encoder.py train \
  --mode history-aware --e1-dir "$FX_E1_DIR" \
  --output "${FX_HISTORY_DIR}_resumed" \
  --device cuda --max-steps 100 --history-gradient-ratio 0.1 \
  --anchor-batch-size 32 --log-every 10 --save-every 10 \
  --history-progress-every 0 --resume-from "$FX_HISTORY_DIR/last.pth"
```

成功后将 `FX_HISTORY_DIR` 指向新目录，再手动比较：

```bash
FX_HISTORY_DIR="${FX_HISTORY_DIR}_resumed"
python -u -B history_aware_frozen_encoder.py compare \
  --plain-dir "$FX_PLAIN_DIR" --history-dir "$FX_HISTORY_DIR" \
  --output "$FX_COMPARE_DIR"
python -B frozen_encoder_bundle/show_frozen.py "$FX_COMPARE_DIR" \
  | tee "${FX_RUN_ID}_compact.txt"
```

恢复运行的 `metrics.jsonl` 只含本次新增步数，完整历史日志仍在之前目录；compact 轨迹展示当前两份日志共有的 after-probe 步，endpoint 对照不受影响。恢复 Plain 同理，使用 `--mode plain`。

## CPU 验证

附带 `tests/incremental/test_frozen_encoder_update.py` 使用仓库真实 RQ-VAE、真实 TIGER 和合成数据，验证冻结干预、独立内容训练参照、两组精确断点续训、梯度路由、full-SID forward、尾部 anchor 小 batch、训练对照、容量回滚、输入不可变与错误输入。通过记录见 validation.json。这里的合成测试不代替用户服务器上的 Beauty 实验。
