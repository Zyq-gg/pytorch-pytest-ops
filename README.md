# PyTorch Pytest Operations Skill

这是一个面向 Codex 的运维型 skill，用于管理本地 PyTorch 测试流程：根据需求生成可直接运行的命令、检查测试目录状态、判断是否正常结束，以及解释失败报告和 timeout/crash 恢复结果。

## 背景

PyTorch 测试并不只有一种入口。本项目所在环境同时使用：

- 直接 `python3 -m pytest` 的普通文件队列
- PyTorch 官方 `test/run_test.py` 队列
- distributed-tests 专项队列
- 普通 pytest 子集和历史失败文件补跑
- case 级稳定失败重测

这些入口的清单、checkpoint、日志和结束标志不同。长时间任务还涉及 `nohup`、多 GPU 调度、中断续跑、stepcurrent `--rs/--scs`、文件级 timeout/crash 补跑，以及最终 case 级 CSV 报告。这个 skill 把这些约定整理成一套稳定的决策和检查流程，避免只根据 `ps` 或单个日志误判运行状态。

## 能做什么

- 生成普通 pytest 全量、dry-run、子集和续跑命令
- 生成官方 `run_test.py` normal/distributed 队列命令
- 根据已有 `failure_report.csv` 补跑文件级 timeout/crash
- 生成稳定失败 case 重测命令
- 保持自定义环境变量在首次运行和续跑之间一致
- 检查计划清单、checkpoint、summary 和最终报告是否对齐
- 区分文件级 unresolved 与已经定位到 nodeid 的 case 级 Crash/Timeout
- 查询某个 case 的失败记录和原始日志
- 给出下一步应运行的精确命令

## 与测试脚本的关系

仓库同时包含 Codex skill 和当前推荐的 PyTorch runner。clone 完成后不再依赖原机器上的 `/workspace/torch_test`，默认环境为：

```text
PyTorch source: /workspace/pytorch
Test runners:   /workspace/pytorch-pytest-ops/runners
Environment:    /home/tmp/python_and_sh/env.sh
Workflow doc:   /workspace/pytorch-pytest-ops/docs/PYTORCH_PYTEST_WORKFLOW.md
```

仓库中的 `runners/` 是可执行实现，skill 会先检查这些脚本源码和 `--help`，再生成命令。`docs/PYTORCH_PYTEST_WORKFLOW.md` 是与该版本 runner 配套的详细操作手册。

运行测试仍需要目标机器上已有：

- 一份 PyTorch 源码树，例如 `/workspace/pytorch`
- 能够导入该 PyTorch 构建的 Python/ROCm 测试环境
- 对应的环境初始化脚本，例如 `/home/tmp/python_and_sh/env.sh`
- Linux、Python 3、pytest，以及测试需要的 GPU/ROCm 运行环境

PyTorch 源码路径、环境脚本、GPU 和输出目录都可以在实际命令中修改，不要求与示例机器完全一致。

## 安装

克隆到 `/workspace`：

```bash
git clone https://github.com/Zyq-gg/pytorch-pytest-ops.git \
  /workspace/pytorch-pytest-ops
```

链接到 Codex skill 目录：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s /workspace/pytorch-pytest-ops \
  "${CODEX_HOME:-$HOME/.codex}/skills/pytorch-pytest-ops"
```

如果目标链接已经存在，先确认它是否已经指向当前仓库，不要直接覆盖未知目录。

验证 runner：

```bash
python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py --help
python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py --help
python3 /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py --help
```

## 使用

在请求中显式引用 skill：

```text
使用 $pytorch-pytest-ops 给我一个普通 PyTorch 全量测试的 nohup 命令。
```

```text
使用 $pytorch-pytest-ops 检查这个目录是否跑完：
/home/tmp/torch2.13/log-final/pytest_full_nmz
```

```text
使用 $pytorch-pytest-ops 给我 distributed-tests 的中断续跑命令。
```

```text
使用 $pytorch-pytest-ops 检查 failure_report.csv 是否还有文件级 unresolved。
```

也可以直接运行只读状态检查器：

```bash
python3 /workspace/pytorch-pytest-ops/scripts/inspect_test_run.py \
  /home/tmp/torch2.13/log-final/pytest_full_nmz
```

输出 JSON：

```bash
python3 /workspace/pytorch-pytest-ops/scripts/inspect_test_run.py \
  /home/tmp/torch2.13/log-final/pytest_full_nmz \
  --json
```

状态检查器会汇总：

- 测试入口类型
- 清单和 checkpoint 数量
- PASS/FAIL/SKIP/TIMEOUT 状态
- 尚未进入 checkpoint 的项目
- `summary.json` 是否存在
- failure 和 unresolved 报告行数
- 自动文件补跑是否完成
- 本机命令行中包含该 work-dir 的进程

进程检查只覆盖执行脚本的当前机器。如果日志目录位于共享存储、进程实际运行在另一节点，应在对应节点检查进程，并以 checkpoint、summary 和最终报告共同判断结果。

## Clone 后直接跑测试

只生成普通 pytest 清单，不执行测试：

```bash
source /home/tmp/python_and_sh/env.sh

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --work-dir /home/tmp/torch2.13/pytest_dry_run \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only
```

正式后台运行普通 pytest 全量：

```bash
source /home/tmp/python_and_sh/env.sh

WORK=/home/tmp/torch2.13/pytest_full
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  --fresh \
  > "$WORK/runner.out" 2>&1 &

echo $! > "$WORK/runner.pid"
```

distributed-tests、子集、历史失败补跑和稳定失败重测命令见 `docs/PYTORCH_PYTEST_WORKFLOW.md`。

## 验收原则

普通 pytest 任务不能只凭“进程消失”认定完成。至少应确认：

1. 计划清单非空。
2. 所有实际测试文件或官方模块都有终态 checkpoint。
3. 根目录存在 `summary.json`。
4. `latest/failure_report.csv` 已生成。
5. `unresolved_process_failure_count` 为 `0`。

具体 `file.py::Class::case` 的 `error_type=Crash/Timeout` 表示异常已经定位到明确 case，不属于文件级遗漏，不应为了让错误类型消失而过滤真实失败。

## 目录结构

```text
pytorch-pytest-ops/
  README.md
  SKILL.md
  agents/openai.yaml
  docs/
    PYTORCH_PYTEST_WORKFLOW.md
  references/
    commands.md
    runner-selection.md
    status-and-reports.md
  scripts/
    inspect_test_run.py
  runners/
    run_pytorch_tests_prefix.py
    run_pytorch_subset.py
    rerun_stable_failures.py
    run_official_run_test_queue.py
    run_test-2.13-official-queue.sh
```

- `SKILL.md`：Codex 的核心操作规则
- `runners/`：clone 后可直接执行的测试 runner
- `docs/PYTORCH_PYTEST_WORKFLOW.md`：完整测试流程文档
- `references/commands.md`：常用命令模板
- `references/runner-selection.md`：测试入口选择
- `references/status-and-reports.md`：状态和报告语义
- `scripts/inspect_test_run.py`：只读目录检查器

## 安全边界

- 状态查询默认只读，不自动启动、停止或清理进程。
- 新任务可以使用 `--fresh`；续跑同一目录时不能继续使用 `--fresh`。
- 自定义环境变量必须在 dry-run、正式运行、自动补跑和续跑时保持一致。
- 普通 direct-pytest、官方 custom handlers 和 distributed-tests 是不同覆盖类别，不能互相替代。
