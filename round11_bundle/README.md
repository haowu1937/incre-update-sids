# Round 11：E1 历史输入重编码损伤测量

真实实验由你在自己的服务器上运行。本包的本地证据来自独立 CPU 沙箱和合成数据，
使用现有仓库真实 RQ-VAE 与 TIGER/T5，不替代你的 Beauty 结果。

## 主问题和比较条件

冻结已经选出的 Base TIGER 第 3000 步 `best.pth`。保持同一历史样本、同一历史位置、
同一可用性 mask、同一旧目标标签和 teacher-forcing 的旧目标前缀，只替换历史商品编码。
每个样本的目标包含三位语义码、一位旧碰撞后缀和 EOS，共五位参与序列 NLL。

主损伤定义：

`Delta NLL = NLL(M_old, new_history, old_target) - NLL(M_old, old_history, old_target)`

正值表示重编码后旧目标的预测损失上升。它不混入目标标签变更，不训练任何模型。

三种条件：

1. `old`：原 Base 完整 SID 历史输入。
2. `new_full_sid`：普通 Update 码本的新三位语义码，加新版本分配的碰撞后缀，主条件。
3. `new_semantics_old_suffix`：只替换三位语义码，保留每个旧商品的旧后缀，诊断条件。

诊断条件可能出现完整编码碰撞，它只用于拆分输入损伤来源，不能当作部署用 SID。
主条件的 Update training 全目录完整 SID 必须唯一，固定词表容量不增长。

新版本后缀分配先保留 Base deployment 所有旧完整标识的占用，再在新语义前缀下追加。
语义前缀不变的旧商品保留原后缀。旧前缀的槽位即使商品离开也不回收，避免旧标识被
另一个商品复用。超出固定后缀容量时失败，不扩词表。该策略是本实验明确的版本策略，
不是声称所有 SID 更新方法均如此。原 Base allocation 文件和 checkpoint 不会改写。

## 样本组

- `anchor_train`：从 Base training 按用户划分的一半用户中，确定性抽取最多 4096 个
  历史样本。保存 raw 历史及旧标签对应的样本 ID，供下一轮历史惩罚训练使用。
- `anchor_holdout`：另一半用户中抽取最多 4096 个样本，作为主损伤测量组。
  两组用户、样本都不重叠。后续历史惩罚训练不得使用这一组。
- `base_validation`：全部 8689 个目标属于旧训练目录的验证样本，用作额外损伤测量。
  503 个旧目录外目标没有旧模型目标标签，NLL 不定义，不用零替代。

这里的 anchor holdout 是相对于未来的历史惩罚训练留出。Base TIGER 先前已经在
Base training 数据上训练，不能把它称为旧推荐模型从未见过的测试数据。
Base validation 也参与过旧模型选步；Future test 在本轮始终未使用。

历史中原本属于 Base training 目录之外的商品仍是四个零 token 和零 attention。
不会因为 Update 目录现在包含它就激活那些位置。目标标签永远保持原样。

## 安装

把 ZIP 上传到仓库根目录，手动执行：

```bash
cd /home/hao/frec/graph-informed-sids-web
source activate frec-gpu
unzip round11_historical_damage_bundle.zip -d round11_bundle
python -B round11_bundle/apply_round11.py --repo .
```

仅新增三个 Python 文件，不修改 Round 1–10 的实现。安装器检查包内 SHA，重复安装
同一版本不再写入，遇到不同的同名文件在写入前退出。
如需在服务器重跑合成正确性检查：

```bash
python -B tests/incremental/test_historical_damage_round11.py
```

本地已通过 10 项检查：真正的 identity 重编码损失/KL 为零、旧标签/mask 保留、
用户不交叉、版本后缀唯一且保留旧槽、溢出失败、冻结模型不变、完整指标分母、
旧模型基线复现、损坏输入拒绝及输出不覆盖。

## 运行 E1

```bash
set -o pipefail
TIGER_DIR="$(python -B round11_bundle/find_base_tiger.py)"
printf 'TIGER_DIR=%s\n' "$TIGER_DIR"

UPDATE_RQVAE_DIR="ckpt/incremental/Beauty/plain_update_rqvae_1000_20261009T124124Z"
E1_RUN_ID="historical_damage_$(date -u +%Y%m%dT%H%M%SZ)"
E1_DIR="results/incremental/Beauty/$E1_RUN_ID"

python -u -B measure_historical_damage.py \
  --protocol-dir cache/incremental/Beauty/beauty_20261007T104452Z \
  --tiger-dir "$TIGER_DIR" \
  --expected-tiger-sha256 4538d5d8bcab84e6f6919cdd66a041c257d1fa62f8519bd787918f00a68d8451 \
  --update-rqvae-dir "$UPDATE_RQVAE_DIR" \
  --output "$E1_DIR" \
  --device cuda \
  --batch-size 128 \
  --anchor-size 4096 \
  --bootstrap-reps 500 \
  --ranking \
  2>&1 | tee "${E1_RUN_ID}.log"

cat "$E1_DIR/summary.json"
cat "$E1_DIR/base_validation_ranking.json"
```

查找脚本按你报告的 selected TIGER SHA 筛选已完成的 Base 训练目录。
测量入口从 TIGER summary 中读取冻结 SID 路径；如你移动过该目录，可显式增加
`--sid-dir <new_path>`，内容与 binding SHA 必须一致。
先验证旧、新 checkpoint、源代码、软件、特征和协议绑定；重新构建 raw 历史的旧张量，
与冻结文件逐项比较，然后才创建新的输出目录。无需下载模型。

此命令还测量完整 Base validation 的旧、新历史输入排名。候选始终为旧 Base training
目录的 10924 个旧完整 SID，不扩展新目标词表；所有 9192 个原始查询保留排名分母，
503 个没有旧目录目标的查询计为未命中。它衡量旧模型输入兼容性，不是新推荐模型部署效果。
旧条件的 Recall/NDCG 应复现你前一轮结果，否则入口会失败。旧条件平均序列 NLL
应复现约 10.9256017483（允许浮点累计误差）。排名沿用旧配置的 beam 20、batch 16。
去掉 `--ranking` 可只做损失测量；这不是热门商品基线。

## 如何阅读结果

首先查看 `groups.anchor_holdout.new_full_sid`：

- `mean_delta_sequence_nll`：成对平均序列损失变化。
- `worsened_fraction`：损失上升超过 1e-6 的样本比例。
- `mean_positive_part_delta`：逐样本 `max(Delta NLL, 0)` 的平均值。
- `cluster_bootstrap_mean_delta_ci95`：按用户重采样的均值差 95% bootstrap 区间。
- `positive_damage_detected`：均值和区间下界均大于 1e-6 时为 true。

这是预设的主检测规则。损伤不显著时如实报告，不要求每个样本都受损，不根据结果
挑选新码本或样本。bootstrap 按用户聚类，保留同一用户序列的相关性；它不替代多随机种子。

再对照 `new_semantics_old_suffix`，判断去掉后缀变化后是否仍有损伤。输入不变的样本
通过 `max_abs_delta_unchanged_history` 检查。报告按五个输出位置分解损失差。
附加 KL 只是在同一旧目标前缀下的五个 teacher-forced token 分布 KL 之和，
不是全目录下一商品分布的精确 KL。

`historical_damage_detected` 来自主 holdout 的上述规则。
`E1_status: completed` 只表示损伤测量完成，不论结果是否显著。
`historical_aware_training_status` 仍为 `not_run`；没有 optimizer 更新、推荐模型微调或 Future test。

## 输出文件

`anchor_train.jsonl`、`anchor_holdout.jsonl` 保存后续可复用且不交叉的原始样本名单。
三组 `*_paired_losses.jsonl` 保存逐样本旧/新 NLL、差值、KL、旧目标标签、历史变化标记。
`new_allocation_registry.json`、`new_item2tokens.json` 保存本次新版本的唯一完整 SID。
`base_validation_ranking.json` 保存额外排名指标。
`summary.json` 保存来源 SHA、配置、统计、冻结状态、运行时间和显存。

运行完成后请贴回 `groups.anchor_holdout`、`groups.base_validation`，以及
`base_validation_ranking.json`。这些结果用于决定下一轮历史惩罚训练的实现与尺度。
