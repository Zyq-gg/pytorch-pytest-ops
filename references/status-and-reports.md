# 状态检查与报告说明

## 只读检查优先

先使用仓库内的只读检查器：

```bash
python3 "$OPS_ROOT/scripts/inspect_test_run.py" "$WORK" --pytorch-root "$PYTORCH_ROOT"
```

再查看当前 work-dir 最近更新的 runner 输出和本机进程：

```bash
RUNNER_LOG=$(ls -1t "$WORK"/runner*.out 2>/dev/null | head -1)
test -z "${RUNNER_LOG:-}" || tail -100 "$RUNNER_LOG"

ps -ef | grep -E 'run_pytorch|run_test.py|python3 -m pytest' | grep -v grep
```

不要固定只看 `runner.out`：续跑通常写 `runner_resume.out`，只补跑不完整模块通常写 `runner_incomplete_rerun.out`。

进程查询只反映当前机器。如果 work-dir 位于共享存储，而任务实际运行在另一节点，应以制品检查进度，并到实际执行节点确认进程。

## 检查器制品判定的准确含义

检查器输出制品判定和明确的 completion issues：

- `COMPLETE`：适用的计划、checkpoint、补跑 summary、failure/unresolved 报告和 coverage 制品全部闭合；
- `FINALIZED_INCOMPLETE`：根 summary/report 已生成，但仍存在 unresolved、timeout、真实 missing 或补跑 summary 缺失；
- `NOT_FINALIZED`：必要的最终制品尚未生成。

这里的 `Artifact verdict` 只描述磁盘制品。当前实现会另外列出 `Local work-dir procs`，但活动进程不会自动把 `COMPLETE` 改成 incomplete。因此最终宣布任务结束前，仍必须单独确认实际执行机器上没有相关进程。

对于已经产生 `.test_progress.json` 的普通 direct pytest 目录，检查器会把五个已知官方 virtual/custom-handler 目标与真实 missing 文件分开。纯 dry-run 目录还没有 progress 时，当前检查器可能把模式显示为 `unknown`，并且 direct `test_files.txt` 可能不会被识别；此时直接检查清单文件即可。

`Legacy/version hints` 用于提示旧恢复输出、旧 summary 元数据，或 failure report 在根 summary 之后被重新生成。

## 普通 direct pytest 输出

- `test_files.txt`：官方 dry-run 解析出的原始计划，尚未过滤没有真实文件的 virtual 目标；
- `.test_progress.json`：文件级 checkpoint，主要读取 `tests`、`stats` 和 `updated`；
- `<timestamp>/gpu_*.log`：各 GPU worker 日志；
- `latest`：指向当前/最近 timestamp 的符号链接；
- `latest.txt`：无法创建 `latest` 符号链接时的回退记录；
- `latest/process_file_rerun/`：process-level 文件的自动大 timeout 补跑；
- `latest/failure_report.csv`：最终失败观察，direct 模式默认可以包含具体 case 和仍未定位的 process-level 行；
- `latest/unresolved_process_failures.csv`：没有可靠 case nodeid 的 process-level 失败；
- `summary.json`：根汇总和 failure report 路径；
- `latest/external_rerun_merge.json`：独立历史补跑发布后的持久合并元数据。

具体 `file.py::Class::case` 即使 `error_type` 是 Crash 或 Timeout，也已经定位到 case，不属于 process-level unresolved。不要为了让错误类型计数归零而删除真实 case 失败。

## 完整官方队列输出

- `run_test_tests.txt`：完整官方队列的模块计划；
- `run_test_modules.txt`：轻量 `run-test-resume` 使用的模块计划；
- `.run_test_progress.json`：模块 checkpoint，记录最新状态、elapsed、timeout 类型、`source_log` 和各次 attempt history；
- `<timestamp>/run_test_gpu_*.log`：Normal 合并 worker 日志；
- `<timestamp>/run_test_gpu_all.log`：Distributed 单 worker 日志；
- `<timestamp>/process_module_rerun*/`：权威的完整模块补跑日志和 summary；
- `latest/failure_report.csv`：只包含具体 case nodeid 的召回优先失败候选；
- `latest/unresolved_process_failures.csv`：没有可靠具体 case 的非零模块结果；
- `module_status.csv`：每个计划模块的最终状态、elapsed、return code、attempts、timeout 类型/上限、时间和权威 `source_log`；
- `incomplete_modules.txt`：仍为 MISSING 或 TIMEOUT 的模块；
- `coverage_report.json`：官方队列覆盖闭合的权威结果；
- `summary.json`：报告、补跑、progress 和 coverage 的根索引。

官方主 `failure_report.csv` 合并稳定失败汇总与普通 pytest `FAILED nodeid`，所以可能包含后来通过的 flaky 候选。需要精确的持续失败集合时，应继续执行稳定 case 重测。

## 官方状态与 coverage 语义

模块状态含义：

- `PASS`：官方模块返回码为 0；
- `FAIL`：官方模块正常返回非零；它是模块覆盖终态，但仍需要具体 case 报告或 unresolved 记录；
- `TIMEOUT`：被外层 hard/idle watchdog 杀死，不是 coverage 终态；
- `MISSING`：计划中存在，但没有 checkpoint。

`coverage_report.json.terminal` 只统计 `PASS + FAIL`，不包含 TIMEOUT。只有同时满足以下条件时，`coverage_complete` 才为 `true`：

- 没有 MISSING；
- 没有 TIMEOUT；
- 没有 unresolved process-level 失败。

因此 `.run_test_progress.json` 中的 completed record 数量等于计划数，也不能单独证明覆盖完整。

`timeout_details` 会记录：

- `timeout_kind`：`hard` 或 `idle`；
- 本次 hard/idle 上限；
- elapsed；
- attempts。

## 官方 resume 和历史日志

官方完整队列会复用 checkpoint 中的历史结果：

- PASS 直接复用；
- 有权威具体 case 报告的 FAIL 复用；
- 无具体报告的 FAIL、TIMEOUT 和 MISSING 重新执行；
- 本轮执行过的模块用新结果整体替换旧报告行；
- 未执行模块继续使用 checkpoint 保存的历史 `source_log`。

旧 checkpoint 没有 `source_log` 时，runner 会从历史 `run_test_gpu_*.log` 的终态标记寻找最近来源。

checkpoint 只在整个官方模块返回后更新。单个大模块运行期间，progress 数小时不变化并不一定是卡住；应同时查看当前 worker 日志的 mtime、文件大小和最近的 case/session 输出。

当前版本有一个重要限制：普通 `resume-normal`/`resume-distributed` 在计算出 `Need run: 0` 时仍会创建空线程池并报：

```text
ValueError: max_workers must be greater than 0
```

因此解析器升级后需要跨 timestamp 重建报告，或只想补跑不完整模块时，当前版本应使用：

```text
rerun-incomplete-normal
rerun-incomplete-distributed
```

这两个包装器动作使用 `--process-rerun-only`，不会进入空的首次执行队列；它会读取现有计划、checkpoint 和 `latest`，补跑已有 checkpoint 的 TIMEOUT、unresolved process failure 或无可靠 case 报告的 FAIL，然后重建 failure/unresolved、module status、coverage 和根 summary。它不会补跑完全缺少 checkpoint 的 MISSING 模块；存在 MISSING 时应使用普通 resume，此时 `Need run` 非零，不会触发空队列问题。

通用 `--analyze-only latest` 只扫描当前 `latest`，不能替代上述官方跨 timestamp 重建，因为可靠 FAIL 可能来自更早的 checkpoint `source_log`。

## 完成判定

普通 direct pytest 至少要求：

1. 计划清单非空；
2. 每个真实计划文件都有终态 checkpoint；
3. 根 `summary.json` 存在；
4. `latest/failure_report.csv` 存在；
5. `latest/unresolved_process_failures.csv` 存在且无数据行；
6. 如果触发 `process_file_rerun/`，其中有有效 `summary.json`；
7. 实际执行机器上没有相关进程。

完整官方队列还必须要求：

1. 每个计划模块有 PASS/FAIL 覆盖终态；
2. `module_status.csv` 和 `coverage_report.json` 存在；
3. `incomplete_modules.txt` 为空；
4. `coverage_report.json.coverage_complete` 为 `true`；
5. 被触发的每个 `process_module_rerun*/` 都有 `summary.json`。

没有进程但没有 summary，表示任务中断或未收尾，不代表成功。已有 summary 但仍有 missing、timeout 或 unresolved，则表示“命令已经留下最终诊断，但覆盖没有闭合”。

当前 direct runner 和 `pytest-list` 还有一个收尾边界：如果 resume 启动时所有文件已经有终态 checkpoint，它们会在生成新 summary/report 前直接返回。因此若上一次任务恰好在写完最后一个 checkpoint 后、最终收尾前中断，普通 resume 当前不能自动补齐缺失的根 summary。遇到这种目录时不能声明完成，应保留为 `NOT_FINALIZED`，等待修复 runner 或采用经人工验证的专门收尾方案。

## 计划数与 checkpoint 数不一致

比较计划时，应读取 progress JSON 的 `tests` 字段，不要使用 JSON 顶层 key 数量。

普通 direct pytest 的官方 dry-run 可能包含没有同名真实 `.py` 文件的 virtual/custom-handler 目标。当前已知目标是：

```text
doctests
test_autoload_disable
test_autoload_enable
test_cpp_extensions_aot_ninja
test_cpp_extensions_aot_no_ninja
```

实际名称应以当前 PyTorch 源树为准。分类前逐个检查：

```bash
test -f "$PYTORCH_ROOT/test/<item>"
```

只有真实计划文件缺少 checkpoint 才属于 direct 文件覆盖遗漏。

## 查询单个 case

先按完整 nodeid 查询最终 CSV，再搜索原始日志：

```bash
python3 - <<'PY'
import csv

p = "/path/to/latest/failure_report.csv"
needle = "test_file.py::ClassName::test_case"

with open(p, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if needle in row.get("nodeid", ""):
            print(row)
PY

grep -R -F "test_file.py::ClassName::test_case" \
  /path/to/latest --include='*.log'
```

`failure_report.csv` 是失败观察或候选集合，不是所有 case 的完整结果数据库。官方召回优先模式可能保留后来通过的 case；CSV 中缺少某个 nodeid 只表示解析器没有提取到该失败，不能据此证明它 PASS。最终 PASS/SKIP 应从原始日志确认，或使用稳定 case 重测。
