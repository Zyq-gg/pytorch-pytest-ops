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
- 检查官方队列的 `module_status.csv`、`coverage_report.json` 和 `incomplete_modules.txt`
- 官方队列续跑时复用历史时间戳目录中已有具体 case 的 FAIL，只重跑无报告 FAIL、TIMEOUT 和缺 checkpoint 模块
- 区分文件级 unresolved 与已经定位到 nodeid 的 case 级 Crash/Timeout
- 查询某个 case 的失败记录和原始日志
- 召回优先合并官方稳定失败清单与 pytest `FAILED nodeid`，避免异常结束前已出现的失败 case 被遗漏
- 给出下一步应运行的精确命令

## 与测试脚本的关系

仓库同时包含 Codex skill、日志解析器、只读状态检查器和当前推荐的 PyTorch runner。clone 完成后不再依赖原机器上的 `/workspace/torch_test`、`check_optest_results_v2.py`、`extract_optest_newfailed.py` 或其他仓库外解析脚本。

仓库可以 clone 到任意目录。文档中的下面路径只是当前容器示例：

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

普通 pytest runner 已兼容 PyTorch 2.9 的平面 stepcurrent cache（`stepcurrent/<key>`）和当前 2.13 的目录 cache（`stepcurrent/<key>/lastrun`）。

timeout/crash 恢复还支持三个独立边界：`--recovery-case-timeout` 控制单个精确 case，`--recovery-attempts` 控制同一异常 case 的重试次数，`--recovery-max-total-time` 控制一个文件全部恢复阶段的累计时间。累计预算耗尽会保留文件级 unresolved，不会生成虚假的恢复完成结论。

## 安装 Skill

克隆到任意目录：

```bash
git clone https://github.com/Zyq-gg/pytorch-pytest-ops.git "$HOME/src/pytorch-pytest-ops"
cd "$HOME/src/pytorch-pytest-ops"
```

安装到 Codex skill 目录并执行自检：

```bash
bash scripts/install_skill.sh
```

安装脚本根据自身位置创建链接，不要求仓库位于 `/workspace`，也不会覆盖未知的已有 skill。当前会话没有立即发现新 skill 时，打开新的 Codex 会话。

只验证仓库，不安装 skill：

```bash
python3 scripts/self_check.py
python3 scripts/self_check.py --pytorch-root /path/to/pytorch
```

自检覆盖 `SKILL.md`、references、全部 runner、日志解析入口、状态检查器、Python 语法、各命令 `--help` 和 shell 语法。仓库代码本身只使用 Python 标准库，不需要额外 `pip install`。

目标机器仍必须自行提供 PyTorch 源码、已构建的 `torch`、pytest、GPU/ROCm 驱动及其环境变量。这些是被测试环境，不是 skill 可移植依赖。环境已激活时无需 `env.sh`；否则通过 `ENV_SH=/path/to/env.sh` 传给包装器。

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

也可以从任意 clone 目录直接运行只读状态检查器：

```bash
OPS_ROOT="$HOME/src/pytorch-pytest-ops"
python3 "$OPS_ROOT/scripts/inspect_test_run.py" \
  /home/tmp/torch2.13/log-final/pytest_full_nmz \
  --pytorch-root /workspace/pytorch
```

输出 JSON：

```bash
python3 "$OPS_ROOT/scripts/inspect_test_run.py" \
  /home/tmp/torch2.13/log-final/pytest_full_nmz \
  --json
```

状态检查器会汇总：

- `Artifact verdict`：`COMPLETE`、`FINALIZED_INCOMPLETE` 或 `NOT_FINALIZED`
- 测试入口类型
- 清单和 checkpoint 数量，并区分 direct-pytest 的官方 virtual 目标与真实漏跑文件
- PASS/FAIL/SKIP/TIMEOUT 状态
- 尚未进入 checkpoint 的项目
- `summary.json` 是否存在
- failure 和 unresolved 报告行数
- 自动文件补跑是否完成
- 官方模块补跑是否完成，以及 `coverage_complete/planned/terminal/timeout/missing` 和各 timeout 的 idle/hard 类型
- 明确的 `Completion issues` 和旧版 runner/report 元数据提示
- 本机命令行中包含该 work-dir 的进程

进程检查只覆盖执行脚本的当前机器。如果日志目录位于共享存储、进程实际运行在另一节点，应在对应节点检查进程，并以 checkpoint、summary 和最终报告共同判断结果。

## Clone 后直接跑测试

先设置目标环境路径：

```bash
OPS_ROOT="$HOME/src/pytorch-pytest-ops"
PYTORCH_ROOT=/path/to/pytorch
ENV_SH=/path/to/environment.sh  # 环境已激活时留空
test -z "${ENV_SH:-}" || source "$ENV_SH"
```

只生成普通 pytest 清单，不执行测试：

```bash
python3 "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --work-dir /home/tmp/torch2.13/pytest_dry_run \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only
```

正式后台运行普通 pytest 全量：

```bash
WORK=/home/tmp/torch2.13/pytest_full
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  --fresh \
  > "$WORK/runner.out" 2>&1 &

echo $! > "$WORK/runner.pid"
```

distributed-tests、子集、历史失败补跑和稳定失败重测命令见 `docs/PYTORCH_PYTEST_WORKFLOW.md`。新建无人值守 distributed 全量使用完整官方队列的 `run-distributed`；`run_pytorch_subset.py run-test-resume` 只作为轻量探索和旧目录兼容入口。

官方 `run_test.py` 队列还会生成：

- `module_status.csv`：每个计划模块的最终 `status/elapsed/returncode/time/attempts/timeout_kind/source_log`；`TIMEOUT/MISSING` 都表示覆盖未闭合
- `coverage_report.json`：`planned/terminal/pass/fail/timeout/missing/unresolved_process_failures/timeout_details` 和最终 `coverage_complete`
- `incomplete_modules.txt`：仍为 TIMEOUT 或缺 checkpoint 的模块清单

官方队列即使命令退出并生成 `summary.json`，只要 `coverage_complete` 不是 `true`，就仍属于“运行已收尾但覆盖不完整”。

官方队列的 checkpoint 会记录产生当前模块状态的 `source_log`。中断续跑时，已有具体 case 报告的 FAIL 直接复用旧日志；只有没有可靠 case 的 FAIL、TIMEOUT 和未运行模块进入队列。旧版 checkpoint 没有 `source_log` 时，runner 会扫描历史 worker 日志中的模块终态标记兼容恢复，最终报告再按模块用新结果替换旧结果。

其中 `terminal` 只统计真正返回的 `PASS + FAIL`，不包含已经写入 checkpoint 的 `TIMEOUT`。因此 `completed_records == planned` 不能替代 coverage 验收；详细字段和示例见工作流文档第 6.4 节。

官方队列默认使用“连续 2 小时无子进程输出”与“单模块 72 小时硬上限”双 watchdog，避免数万 case 的活跃模块被旧的 12 小时绝对上限误杀。主 `failure_report.csv` 只保留具体 case nodeid，并按“不遗漏优先”合并官方稳定失败和普通 pytest FAILED，因此可能包含后来通过的 flaky 候选；模块级 TIMEOUT/MISSING/ProcessFailure 由 coverage、module status、incomplete 和 unresolved 文件单独记录。

## 验收原则

普通 pytest 任务不能只凭“进程消失”认定完成。至少应确认：

1. 计划清单非空。
2. 所有实际测试文件或官方模块都有终态 checkpoint。
3. 根目录存在 `summary.json`。
4. `latest/failure_report.csv` 已生成。
5. `unresolved_process_failure_count` 为 `0`。

官方队列还必须要求 `coverage_report.json.coverage_complete` 为 `true`。`completed_records` 可以包含 TIMEOUT，不能单独作为完成证据。

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
    install_skill.sh
    inspect_test_run.py
    self_check.py
  runners/
    run_pytorch_tests_prefix.py
    run_pytorch_subset.py
    rerun_stable_failures.py
    run_official_run_test_queue.py
    run_test-2.13-official-queue.sh
  tests/
    test_stepcurrent_cache_compat.py
    test_recovery_budget.py
    test_official_queue_completion.py
```

- `SKILL.md`：Codex 的核心操作规则
- `runners/`：clone 后可直接执行的测试 runner
- `docs/PYTORCH_PYTEST_WORKFLOW.md`：完整测试流程文档
- `references/commands.md`：常用命令模板
- `references/runner-selection.md`：测试入口选择
- `references/status-and-reports.md`：状态和报告语义
- `scripts/inspect_test_run.py`：只读目录检查器
- `scripts/install_skill.sh`：从任意 clone 路径安装 Codex skill
- `scripts/self_check.py`：不依赖第三方包的仓库完整性和命令入口自检
- `tests/test_stepcurrent_cache_compat.py`：PyTorch 2.9/2.13 stepcurrent 布局回归测试
- `tests/test_recovery_budget.py`：恢复超时预算与精确 case 重试回归测试
- `tests/test_official_queue_completion.py`：官方模块补跑、unresolved 和 coverage 回归测试

## 安全边界

- 状态查询默认只读，不自动启动、停止或清理进程。
- 新任务可以使用 `--fresh`；续跑同一目录时不能继续使用 `--fresh`。
- 自定义环境变量必须在 dry-run、正式运行、自动补跑和续跑时保持一致。
- `TORCHINDUCTOR_CPP_MARCH` 会改变 Inductor C++ 编译目标；不同值必须使用不同 work-dir，已有 `*_znver1*` 任务续跑时必须保持该值。
- 普通 direct-pytest、官方 custom handlers 和 distributed-tests 是不同覆盖类别，不能互相替代。
