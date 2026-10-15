# Round 1: temporal and identity correctness

## 范围和前提
使用已存在的 frec-gpu 环境。用户授权跳过环境 audit；不安装依赖。
补丁依据公开 commit 5c2ca487f961e91e75e7d8f3ca78349faf3dc82e。
本轮只交付 E0 的时间/身份/解析子集。训练接入、图处理、TIGER、
梯度检查、Beauty 端到端与 E1 都没有执行。

## 文件清单
| Path | Change | Purpose |
| --- | --- | --- |
| genrec/temporal.py | new | Safe parsing, timestamp boundaries, causal examples, append-only registry snapshots |
| genrec/datasets/AmazonReviews2014/dataset.py | patch | Replace eval-based gzip parsing with the safe reader |
| tests/incremental/test_temporal_round1.py | new | 19 standard-library CPU synthetic correctness checks and evidence output |

complete_files/ 包含全部三个文件的完整内容，供审阅；已有的 Amazon
dataset.py 推荐使用窄补丁，避免覆盖本地其他修改。
manifest.json 记录代码文件与公开基准文件校验和。

## 协议约定
- 输入使用原仓库 _load_reviews 的 user/item/time/rating 四元组；
  reviews_from_sequences 可读取现有 all_seqs 格式中的真实时间。
- 默认去除同 user/item/time/rating 的重复观测。不同时间的重复物品保留。
- 全局约 60/80% 切分，base/update 后 10% 为验证，边界只能落在时间戳边缘。
  为避免空阶段，base/update 各至少两个时间戳组，future 至少一个；
  边界移动导致的比例偏差被记录。
- 无验证后 refit。验证目录参考相应训练前缀；最终部署 registry 可追加
  验证期间已观察身份，但参数/图不能用验证数据拟合。
- 同时间戳不进入相互的历史；已完成时间戳组按 user/item/rating 确定顺序。
  item_seq 是原始 ASIN 的 history + one target。默认历史上限 50。
- GR 实物 ID 从 1 开始，0 为 padding；对应 gensid global ID = GR ID - 1。
  registry 快照不包含 future-only 物品或用户。
- 未部署目录的目标和历史物品保留并显式打 availability 标记。
  此轮不决定/执行模型输入过滤，也不计算推荐指标。

注意：不要把此轮 causal rows 直接送入当前训练 tokenizer。
原 tokenize_function(train) 仍会展开前缀；下一轮交付 explicit-example
集成后才能使用。原 leave-one-out 配置未被改为 temporal。

## 应用顺序
下载 zip 到服务器仓库根目录并执行以下命令。目录名可改成新名字，
不要覆盖已有同名交付目录。

```bash
source activate frec-gpu
unzip -n round1_temporal_bundle.zip -d round1_handoff
git apply --check round1_handoff/round1.patch
```

仅在 check 成功后执行：

```bash
git apply round1_handoff/round1.patch
git diff --check
python -c "from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014; print('AMAZON_IMPORT_OK')"
```

如果 git apply --check 失败，不强制应用/覆盖文件。返回完整错误和现有
genrec/datasets/AmazonReviews2014/dataset.py；如果提示新文件已存在，
也返回对应文件的当前内容。

## CPU smoke
每次使用新的 run 目录，测试会拒绝覆盖已存在的结果。

```bash
set -o pipefail
RUN_ID="round1_$(date -u +%Y%m%dT%H%M%SZ)"
python -B tests/incremental/test_temporal_round1.py \
  --output "results/incremental/round1/$RUN_ID" \
  2>&1 | tee "round1_${RUN_ID}.log"
```

## 检查条件与文件
- git diff --check 无 whitespace error，实际 Amazon 模块能够导入。
- unittest 跑完 19 个检查，终端报告 OK；命令退出码 0。
- checks.json: execution_status=passed, tests_run=19, tests 每项 passed。
- synthetic_protocol.json: 保存合成行、目录快照、边界及计数；所有 history
  timestamp 严格小于 target；所有窗口不跨相同时间戳；快照不含 future-only ID。
- 结果 schema 含 source_sha、三个源文件校验和、配置及配置 hash。
  inputs 必须是 synthetic_only；Beauty/E0_full/E1 保持 not_run。
  未测量成本为 not_measured，不填写虚构指标。

返回完整日志、checks.json、synthetic_protocol.json。
失败时返回执行命令和完整 traceback；本轮不需要数据集或 checkpoint。

## 记录与下一轮
implementation_plan_round1.md 是最新计划，手动合并到已有
implementation_plan.md，避免覆盖已有计划与本地修改。
experiment_report_round1.md 是待追加进报告的状态区块；保持旧结果，
同一块只追加一次。收到服务器证据前状态为 code_provided。
下一轮：接入现有 AbstractDataset/TIGER 的 explicit-example 模式及
阶段 cache/manifest；仍不宣称 E0 完成。

## 沙箱验证的边界
助手仅在独立 Linux 沙箱 Python 3.12.14 上使用合成 fixture 运行本轮
19 个标准库检查；均通过。测试里的 Amazon parser 通过实际源文件 AST
提取执行，未验证你的服务器环境或完整 Amazon 模块依赖。
补丁另在公开原始文件组成的临时 Git 工作区应用检查。
这些不是 Beauty 结果，也不是服务器 user_verified 状态。
