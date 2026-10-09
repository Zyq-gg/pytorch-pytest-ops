# Runner 选择指南

## 默认全量测试基线

默认使用 `$OPS_ROOT/runners/run_test-2.13-official-queue.sh` 及其 Python 队列 runner 作为全量入口。官方 Normal 与 Distributed 必须分开运行，使用不同 work-dir，并分别做 coverage 验收。

该入口保留官方 custom handler、模块参数和 `run_test.py` 行为，同时增加：

- 模块级 checkpoint；
- 动态 GPU 调度；
- hard/idle 双 watchdog；
- process-level 完整模块补跑；
- case 级失败报告；
- `module_status.csv`；
- `coverage_report.json`；
- `incomplete_modules.txt`。

官方 case 报告采用召回优先策略：合并官方稳定失败汇总和普通 pytest `FAILED nodeid`。因此初始报告可能保留后来通过的 flaky case；需要精确的持续失败集合时，再使用稳定 case 重测。

Normal 和 Distributed 仍不等于所有 PyTorch 类别。默认 Normal 参数排除 distributed 和 JIT executor，Distributed 单独覆盖 distributed 类别；被排除的 JIT executor 类别仍需另行规划。

## 直接 pytest 文件队列：增量验证与可选全量

`$OPS_ROOT/runners/run_pytorch_tests_prefix.py` 主要用于：

- 按前缀选择模块；
- 代码修改后的增量验证；
- 测试入口对比；
- timeout/crash 复现；
- 需要统一 pytest 文件日志的场景。

它也保留可选的普通文件全量能力：先从官方 `run_test.py --dry-run` 获取清单，再对每个真实文件执行 `python3 -m pytest`。文件按 GPU 动态分发，以文件为 checkpoint 粒度，并通过 stepcurrent `--rs/--scs`、collect/chunk fallback 和自动 process-level 文件补跑处理 timeout/crash。

该入口默认向 dry-run 添加：

```text
--exclude-jit-executor
--exclude-distributed-tests
```

官方 virtual/custom-handler 目标可能没有同名 `.py` 文件，direct runner 会把它们列入原始计划，但不会进入真实文件执行 checkpoint。因此 direct pytest 不能替代官方完整队列，也不能单独声明完整 PyTorch 类别覆盖。

当前 direct runner 默认启用失败日志分析和 process-level Timeout/Crash 文件自动补跑；可分别使用 `--no-analyze` 和 `--no-process-rerun` 关闭。关闭分析后不会产生完整 failure/unresolved 报告。

## 普通 pytest 子集

从已有测试清单筛选时，使用：

```text
$OPS_ROOT/runners/run_pytorch_subset.py pytest-list
```

它支持 include/exclude prefix 和正则过滤，并复用 direct runner 的 GPU 队列、文件 checkpoint、stepcurrent 与 crash 恢复逻辑。

从已有 failure CSV 中补跑 process-level 文件时，使用：

```text
$OPS_ROOT/runners/run_pytorch_subset.py pytest-failure-files
```

该模式从 CSV 中选出完整测试文件，通常配合更大的文件 timeout。若补跑已经完整且 unresolved 为 0，可增加 `--publish-to-work-dir`，把所选文件的新结果合并回原全量目录。

## 稳定 case 重测

输入已经包含具体 nodeid 时，使用：

```text
$OPS_ROOT/runners/rerun_stable_failures.py
```

`--attempts N` 表示稳定失败判定所需的连续失败次数。runner 精确重跑 case；某次 PASS 后立即停止，只有 N 次均未 PASS 才写入 `stable_failures.csv`。

默认跳过只有文件名、没有 `::case` 的 process-level 行。这类行应先用完整文件或官方模块补跑定位到具体 case。稳定 case runner 不需要文件级 `--rs/--scs` 继续机制。

默认使用 direct pytest 执行；需要官方 custom handler 语义时可使用 `--runner official`，并通过可重复的 `--official-run-test-arg` 传递额外官方参数。

## 轻量官方 run_test.py 入口

以下入口只用于轻量串行模块执行、distributed 探索和旧 work-dir 兼容：

```text
$OPS_ROOT/runners/run_pytorch_subset.py run-test-resume
```

它有模块 checkpoint 和可选 case 报告，但没有完整官方队列的 idle watchdog、coverage 文件或第二轮自动完整模块补跑。新的无人值守 Normal/Distributed 全量应使用完整官方队列。

`run-test` 子命令则只包装一次完整 `run_test.py` 调用，没有模块 checkpoint；中断后只能重跑整个命令。

## 如何描述覆盖范围

覆盖结论必须按类别陈述，不能把不同入口混为一谈：

- 默认全量基线：官方 Normal + 官方 Distributed；
- direct pytest：真实普通文件的增量、子集或可选文件全量；
- 官方 custom handler：必须由官方入口执行；
- distributed：必须由官方 Distributed 入口执行；
- JIT executor 等被显式排除的类别：另行规划。

报告中应明确写出实际运行的类别、筛选条件和 work-dir，不能仅用“全量通过”概括。

## 当前适用环境

这些 runner 会强制设置 `PYTORCH_TEST_WITH_ROCM=1` 等 ROCm 测试默认值，因此当前实现是 ROCm 定向的 PyTorch 测试运维工具。MIOpen 配置从启动环境继承。CUDA 或 CPU 环境不能只替换 GPU ID 就直接复用；应先审查 runner 的默认环境和设备可见性设置。
