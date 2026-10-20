# Git 实验消息队列：第一阶段部署

本轮只接通通信闭环。服务器默认 `allowed_kinds=["smoke"]`，Web 只自动下发第二个通信测试；不会提交主实验任务、修改主实验代码、切换主工作区、加载模型或占用 GPU。真实服务器安装与运行状态，以 `lab-results` 的实际回传为准。CPU 合成检查不等于服务器部署完成。

## 存储与角色

同一仓库 `haowu1937/incre-update-sids` 使用三个分支：

| 分支 | 写入方 | 路径 |
| --- | --- | --- |
| `vibe-dev` | 开发会话 | 源码；任务绑定完整 40 位 SHA |
| `lab-requests` | 一个 Web 控制器 | `jobs/JOB_ID.json`，不可修改的任务；`receipts/JOB_ID.json`，处理回执 |
| `lab-results` | 一个服务器 Runner | `workers/server2.json`，`jobs/JOB_ID.json`，`logs/JOB_ID.tail.txt`，`artifacts/JOB_ID/...` |

通信分支禁止合入代码分支，不从通信分支运行训练。所有本地状态、Git 仓库和运行目录在 `/home/hao/frec/git-lab-buffer`；Runner 不访问原 `/home/hao/frec/graph-informed-sids-web` 工作区。首次部署的 shell 命令只从原工作区读取 Git remote URL。

Runner 使用 Python 标准库，无 pip/conda 依赖安装，无 sudo。每分钟一个 cron tick；同一 root 的进程锁确保只有一个执行者。长任务由该 tick 持续管理，SSH 断开后由 cron 启动的进程继续工作。重启后若存在未确认结束的进程状态，会暂停整个队列，不擅自重跑或杀死任何旧进程。

## 安装

使用交付回复中的**完整 commit SHA**建立独立 bootstrap clone，运行：

```bash
bash BOOTSTRAP/scripts/lab_buffer/install.sh \
  /home/hao/frec/git-lab-buffer \
  /home/hao/miniconda3/envs/frec-gpu/bin/python
```

安装步骤：检查 Python、Git、crontab；拒绝覆盖已存在的 root；拉取三个分支；实际提交 `workers/server2.json` 以检查无人值守读写凭据；复制固定版本 Runner；备份原 crontab；仅追加带 `# git-lab-buffer-server2` 的每分钟条目；立即执行一次 tick。任何 Git 操作设置 `GIT_TERMINAL_PROMPT=0`，无法无人值守认证时停止，不等待输入密码。仅有 fetch 权限不足，服务器还需向 `lab-results` push 的权限。

安装脚本不变更系统 cron 服务配置。Ubuntu 通常已有 cron；若 crontab 命令/服务不可用，先报告该错误，不修改主任务或自行安装系统包。安装中途失败时保留目录和诊断；不要删除后盲目重跑。成功后 bootstrap clone 不再参与执行；Runner 不自动升级自身版本。

查看本地状态：

```bash
tail -n 30 /home/hao/frec/git-lab-buffer/runner.log
cat /home/hao/frec/git-lab-buffer/state/mailbox-smoke-001.json
```

安装脚本的即时 tick 会将结果直接打印；定时 tick 写入 runner.log。GitHub 回传由 Web 定时任务读取，之后不再需要手动复制实验日志。

## 任务与回传

首个测试：`mailbox-smoke-001`。收到真实 completed 及正确 artifact 后，Web 自动创建 `mailbox-smoke-002`，使用相同代码 SHA。第二个测试通过后写回执并通知用户，不自动启动第三个任务。

任务示例（把 CODE_SHA 替换为实际完整 SHA；只在消息分支提交）：

```json
{
  "schema_version": 1,
  "job_id": "mailbox-smoke-001",
  "worker_id": "server2",
  "kind": "smoke",
  "code_sha": "CODE_SHA",
  "timeout_seconds": 60,
  "created_at": "2026-10-20T10:00:00+00:00"
}
```

smoke 不接受自定义 argv 或 artifacts。它仅运行该 SHA 的 `scripts/lab_buffer/smoke.py`，隐藏 CUDA、不 import torch，只读取 Python/package metadata，生成 `outputs/lab-buffer-smoke.json`。此检查不证明 GPU、数据、模型或真实训练环境可用。

每项任务结果记录 request SHA256、指定/实际代码 SHA、Runner 版本、状态、PID、命令、退出码、耗时、日志路径及显式产物。最多上传日志末 12 KiB 和最多 8 个文本产物（每个最多 64 KiB）；不推送数据、权重、完整日志或访问凭据。

## 去重与故障语义

- 任务唯一 ID + 本地 durable claim + 远端 claim + 非阻塞进程锁。启动前落盘并上传领取记录。任务文件不可改；改动会暂停队列。
- 已完成/失败/blocked 任务不自动重试；需要新 job_id。远端有记录时，即使本地该 job 状态文件丢失，也不会重新执行。
- 上传失败先保留本地状态，下次 tick 只重传结果。所有 pending 状态先成功交付，才读取新任务。
- 崩溃发生在领取或运行阶段时，标记 interrupted，保存 `PAUSED.json` 并停止新任务。先检查记录中的 PID、日志、产物，再决定是否恢复。Git 不提供跨进程事务，因此不声称 exactly-once。
- 子进程超时由独立 watchdog 管理，Git 上传阻塞不会延长计算期限。只终止该 Runner 创建的进程组，不按名字 pkill，不操作主任务 PID。
- 工作区按 SHA 独立创建；不 pull 正在执行的代码，不 reset/clean 原工作区。结果发布采用 Git plumbing，基于最新结果分支创建提交，普通快进 push；冲突重取再构建，不强推。
- 该部署是单服务器、单 Runner。不要把同一 worker_id 分给另一台服务器或复制 root。需要多服务器时应增加跨主机 claim 协议。

## 暂停与停止

暂停后续任务（保留所有记录，已启动的子进程继续由原 Runner 管理）：

```bash
/home/hao/miniconda3/envs/frec-gpu/bin/python -B - <<'PY'
import json
from pathlib import Path
p = Path('/home/hao/frec/git-lab-buffer/PAUSED.json')
p.write_text(json.dumps({'reason': 'user paused new jobs'}) + '\n')
PY
```

不要通过删除 state/workspaces、恢复旧 crontab 快照或清理 Git 来恢复任务。停止调度时只删除自己带 `# git-lab-buffer-server2` 标记的 cron 行，保留其他条目。Web 监控可以独立暂停。恢复 interrupted 任务需要先核实原 PID 已退出，再由用户决定新任务 ID；不清空旧失败证据。

## 后续接入真实实验

当前禁止自动开启实验。`kind=experiment` 在 smoke-only 配置下返回 blocked，不创建运行进程。将来由用户明确开启后，才允许 `allowed_kinds=["smoke","experiment"]`。实验必须逐项提供 argv、预算、预期产物、数据/checkpoint 的**准确路径**；不把主目录整体软链接进去，不猜测原始绑定路径，也不复用会写主任务目录的命令。

opt-in 实验前还检查可见 GPU compute PID，发现已有进程则延后。这是一次性空闲检查，不能防止外部任务随后启动，不能代替共享 GPU 的统一调度。当前 smoke-only 硬门禁是本轮不占 GPU 的保证。

Web 端用 GitHub 工具读取协议与回执，不依赖前一轮本地目录。当前工具每小时轮询。待两次通信检查完成，再确定主实验的可自动变更范围、总预算与停止条件；本轮不更新研究指标、不改 HANDOFF/RESEARCH_SPEC、不接管在跑的主实验。

本地检查入口：`python -B -m unittest discover -s tests/lab_buffer -v`。测试用临时本地 Git 仓库和假的 crontab，不执行真实 cron、不联网、不访问 GPU。
