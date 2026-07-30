# Runner 可靠性问题核验与修复说明（2026-07）

本文记录 2026-07-28 对当前 `main` 分支 runner 的实际核验、复现、根因和修复。结论来自当前脚本的最小复现和回归测试，不是只根据文档推测。

## 上游同步结果

开始修改前已执行 `git pull --ff-only origin main`，仓库从 `d1af828` 快进到 `b5dcc01`（`Avoid overwriting official queue runner logs`）。

该上游提交已经修复官方 shell 包装器覆盖日志的问题：

- 每次启动生成带时间戳和 shell pid 的独立 runner 输出；
- 同时写本次启动对应的 pid 文件；
- 为 run、resume 和 incomplete-rerun 分别维护 `*_latest.out` 符号链接；
- 状态查看选择最新 runner 输出，不再固定读取单个 `runner.out`。

因此这项问题不再重复修改 Python runner，只同步修正文档中“包装器不写 pid”和固定 runner 文件名的旧描述。

## 已确认并修复的问题

### 1. 官方队列在 `Need run: 0` 时崩溃

最小复现直接调用 `execute_queue(tests=[], gpu_ids=[0], ...)`，原实现把 worker 数计算为 0，再构造 `ThreadPoolExecutor(max_workers=0)`，实际抛出：

```text
ValueError: max_workers must be greater than 0
```

影响：

- 所有模块已有可复用终态时，普通 `resume-normal`/`resume-distributed` 不能正常收尾；
- 解析器升级后不能通过普通 resume 重建跨 timestamp 报告；
- 文档被迫把 `rerun-incomplete-*` 当作规避空队列异常的临时路径。

修复：

- `execute_queue()` 对空测试列表直接返回空结果；
- 官方主流程仅在 `remaining` 非空时创建执行队列；
- `Need run: 0` 时继续执行报告、module status、coverage 和根 summary 生成；
- 最终累计统计只看当前计划中的模块，避免同一目录残留的计划外旧 checkpoint 污染数量。

现在普通 resume 可以安全完成“零执行、只收尾”。`rerun-incomplete-*` 恢复为它本来的语义：只重跑已有 checkpoint 中的 TIMEOUT、unresolved 或无可靠 case 报告的 FAIL，不补完全缺 checkpoint 的 MISSING。

### 2. direct pytest 和 subset 在没有剩余测试时提前退出

原 direct runner 和 `pytest-list` 路径在过滤后没有 remaining 时直接 `return`，不会保证本轮生成：

- 根 `summary.json`；
- `latest/failure_report.*`；
- `latest/unresolved_process_failures.*`。

这会让一个 checkpoint 已经完整的续跑目录仍被检查器判定为 `NOT_FINALIZED`，也让新版解析器无法在零重跑的情况下刷新报告。

修复后的仅收尾路径会：

1. 解析已有 `latest` 或 `latest.txt`；
2. 用现有日志重新生成 failure/unresolved 报告；
3. 写入带 `finalize_only: true` 的根 summary；
4. 在 summary 中记录当前计划、计划内 checkpoint 状态和缺失项。

若 checkpoint 中存在 FAIL/TIMEOUT，但目录完全没有可分析的历史日志，runner 会明确拒绝生成“零失败”报告；只有全部为 PASS/SKIP 时才允许创建新的空日志目录完成收尾。

### 3. direct dry-run 接受非零返回码的半截清单

原 `run_dry_run()` 无论 `run_test.py --dry-run` 返回码是否为 0，都会把 stdout/stderr 交给解析器。实际构造“先输出一个合法 Serial tests 块、随后返回 1”的结果时，解析器会得到看似有效但不完整的 `test_partial.py`。

影响是最危险的一类静默漏跑：后续 checkpoint 和 summary 都可能围绕半截计划闭合。

修复后，dry-run 返回非零会立刻抛出异常，并附带最后 4000 字符输出用于诊断；不会写入测试计划或启动 worker。

### 4. 状态检查器混淆制品完成和进程仍在运行

原检查器有三个可复现问题：

- 没有 progress 的纯 direct dry-run 目录会显示 `unknown`；
- 状态统计会把计划外的陈旧 checkpoint 混入；
- 先计算 `Artifact verdict`，随后虽然能列出本机进程，却不会让总体结论变为运行中。

修复后：

- 根据 `test_files.txt`、`selected_test_files.txt`、`run_test_tests.txt` 等计划文件推断入口；
- 状态和非终态数量只统计当前计划内项目；
- 同时输出 `Artifact verdict` 和 `Run verdict`；
- 本机存在同 work-dir runner 时，`Run verdict` 为 `RUNNING`，即使制品暂时看起来已闭合；
- JSON 增加 `run_complete`，仅在制品完整且本机无匹配进程时为真。

进程判断仍只能覆盖执行检查器的机器。共享 work-dir 对应远端节点任务时，必须在实际节点再次确认。

### 5. checkpoint 没有可验证的运行语义

原实现要求人工保持源码、环境和参数一致，但 checkpoint 本身没有记录可比较的配置。误用同一个 work-dir 时，GPU 绑定、筛选条件、timeout、官方参数或环境变化都可能和历史结果混合。

新增的 `<work-dir>/.run_manifest.json` 覆盖：

- runner 类型；
- PyTorch 源码绝对路径、Git HEAD、已跟踪工作区状态与 diff 哈希；
- 计划数量和有序计划哈希；
- GPU、筛选条件、timeout、恢复策略、分析/发布等入口相关语义参数；
- 影响构建/测试的关键环境变量名和整体哈希。

为避免泄露路径中以外的环境内容，清单只保存入选变量名和整体 SHA-256，不保存环境变量原值。源码状态忽略 untracked 文件，避免 pytest 运行时新生成缓存导致无关续跑误拒绝；已跟踪修改仍通过状态和完整 diff 哈希区分。

复用 work-dir 时如果 `source`、`plan`、`environment` 或 `options` 变化，runner 会在覆盖计划文件和启动 worker之前拒绝运行。正确处理方式是恢复原语义配置，或换新 work-dir；不应删除清单绕过检查。

兼容旧目录时存在不可消除的信息边界：旧 checkpoint 没有历史配置。新版会先把新生成计划与旧目录已经保存的计划逐项比较；计划不同则拒绝接管。计划相同也只能把“当前源码、环境和参数”写为清单，并标记：

```json
"legacy_checkpoint_adopted_without_verification": true
```

这表示除已保存计划外的历史配置未经验证，不等于已证明全部一致。只读检查器会持续显示该提示。

### 6. 静态类型检查长期有 24 个错误

实际执行 `mypy runners scripts` 时，修改前可稳定得到 24 个错误，主要包括：

- PTY 执行函数声明只接受 `int`，调用方实际会传 `None` 表示无限 timeout；
- summary 被推断为过宽的 `object`，后续嵌套统计无法安全索引；
- GPU ID 的 `list[int]` 与包含 `None` 的“不绑定 GPU”语义不一致；
- coverage 中可空 checkpoint 没有显式缩窄；
- 检查器的 `Counter` 缺少元素类型。

这些错误已经通过准确补充类型和显式分支消除，没有通过全局忽略隐藏。当前 `mypy runners scripts` 对 7 个源码文件检查通过。

### 7. distributed custom handler 已打印 nodeid 仍落为 process failure

实际 distributed 日志存在一种不经过普通 pytest `FAILED` 或官方稳定失败汇总的结尾：

```text
Test exited with non-zero exitcode 124. Command to reproduce: ... --pytest-single-test file.py::Class::case
AssertionError: 1 unit test(s) failed:
        file.py::Class::case
```

原解析器的 nodeid 正则能匹配该文本，但抽取入口只接受稳定失败、普通 `FAILED` 和外层 timeout 当前项，因此 nodeid 没有进入报告。随后 `GPU all` 不被 `START`/文件名正则识别，VMFault 又会被最后一个 Python traceback 文件（常为 `common_distributed.py`）错误归属；checkpoint 模块过滤掉该错误行，最终合成 `<process-failure>`。

同一日志还暴露出队列输出读取问题：对 text-mode `proc.stdout` 设置非阻塞后，`TextIOWrapper.read()` 在底层暂时无数据时可能返回不完整内部状态，触发 `TypeError: can't concat NoneType to bytes`，使 runner 追加二次异常。

修复后：

- 从 `--pytest-single-test` 复现命令和 `AssertionError: N unit test(s) failed` 列表提取具体 nodeid；
- 子进程返回码 `124` 归类为 case 级 `Timeout`；
- 识别 `GPU all`、rank 前缀 `FAILED/ERROR` 和外层 `TIMEOUT`；
- 只把 `Running 1 item` 当作当前 case，忽略整模块 item 计划；
- crash 优先归属当前 case/模块，不使用 site-packages traceback 文件；
- 共享 worker 日志按规范化模块名过滤 process 行，避免一个模块的具体 case 误删另一个模块的 crash，也避免 `.py` 后缀差异留下重复 unresolved；
- subprocess pipe 改为 bytes 非阻塞读取和增量 UTF-8 解码。

### 8. Normal 模块已有 traceback 仍落为 process failure

一次真实 Normal 全量队列的 9 个 unresolved 中，旧解析器对全部目标都返回 0 条日志诊断，随后只根据 checkpoint 合成 `<process-failure>`。日志实际已经包含：

- `NameError`、`ImportError`、`RuntimeError`、`AssertionError` 等 import/collection 异常；
- DTK Stable ABI 头文件冲突和链接器失败；
- `TypeError: can't concat NoneType to bytes` runner exception；
- `test_ci_sanity_check_fail` 这一已知故意失败目标。

修复后，Normal 与 Distributed 共用模块阶段解析：按每个 `START` 到 `FAIL` 区段提取最终 Python 异常、根编译/链接错误或 runner exception，写为 `case_name=<module-stage>`。它表示“模块失败原因已经定位，但不存在 pytest case”，不会伪造 nodeid，也不再计入 unresolved；同模块已有具体 case 时仍以 case 为准。只有日志既没有 case、也没有明确模块阶段诊断时才生成 `<process-failure>`。

真实 9 模块历史日志验证结果：

| 分类 | 模块 |
| --- | --- |
| `ExpectedFailure` | `test_ci_sanity_check_fail` |
| `BuildError` | `test_cpp_extensions_aot_ninja`、`test_cpp_extensions_aot_no_ninja`、`cpp_extensions/test_libtorch_agnostic` |
| Python traceback | `test_overrides`、`functorch/test_vmap`、`inductor/test_ck_backend`、`inductor/test_reshape_pass` |
| `RunnerError` | `test_sparse_semi_structured` |

这 9 个模块从 0 条已解析诊断、9 条 unresolved 变为 9 条模块阶段诊断、0 条 unresolved。分类只说明解析完整性：环境、DTK 工具链、PyTorch 代码和故意失败测试本身仍需分别处理。

## 关键行为对照

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 官方 `Need run: 0` | 空线程池异常 | 跳过队列并重建全部最终制品 |
| direct/subset 无 remaining | 提前返回 | 仅收尾，生成报告和 summary |
| nonzero dry-run 有半截输出 | 可能接受半截计划 | 立即失败，不写计划 |
| 制品 COMPLETE 但本机 runner 活跃 | 总体仍像完成 | `Artifact=COMPLETE`，`Run=RUNNING` |
| 纯 dry-run 无 progress | 入口可能 unknown | 按计划文件识别入口 |
| 计划外旧 checkpoint | 混入状态统计 | 只统计当前计划 |
| 同目录改变源码/环境/参数 | 依赖人工发现 | manifest 校验失败并拒绝续跑 |
| 官方包装器再次启动 | 可能覆盖旧输出 | 独立时间戳日志、pid 和 latest symlink |
| structured distributed timeout | 已有 nodeid 仍变成 `<process-failure>` | 提取具体 case 并归类为 `Timeout` |
| `GPU all` / rank 前缀 / `ERROR` | 上下文或 case 可能丢失 | 统一识别并保留 |
| 非阻塞 text pipe | 可能追加 `NoneType` TypeError | bytes 读取并增量解码 |
| Normal import/collection/build 已有错误 | 仍合成 `<process-failure>` | 写入 `<module-stage>`，不伪造 case |
| `mypy runners scripts` | 24 个错误 | 7 个源码文件零错误 |

## 回归覆盖

自动化测试覆盖以下关键边界：

- 空官方执行队列为 no-op；
- 官方零 remaining 仍生成报告、coverage 和根 summary；
- 计划外陈旧 checkpoint 不污染官方最终统计；
- nonzero direct dry-run 的半截计划被拒绝；
- direct 仅收尾成功生成报告，失败 checkpoint 无日志时拒绝伪造结论；
- manifest 相同语义可续跑、参数变化被拒绝、特定 process-only timeout 字段可忽略；
- legacy checkpoint 被标记为未经历史验证；
- legacy checkpoint 的旧计划与新计划不同时拒绝接管；
- manifest 不落盘环境原值；
- manifest 拒绝新计划后，旧计划文件保持不变；
- 纯 dry-run 模式推断；
- 活动本机进程改变 `Run verdict`，但不篡改 `Artifact verdict`。
- official structured single-test timeout 能定位具体 case；
- 共享日志不同模块的 case 与 process crash 都被保留，同模块占位行会被具体 case 替代；
- `GPU all`、rank 前缀 `ERROR` 和只含外层 timeout 的当前 item；
- queue worker 的非阻塞输出不会再产生 `NoneType` text decoder 异常。
- Normal 模块级 traceback、构建错误和 runner exception 进入主报告且不计 unresolved；
- 同模块已有具体 case 时抑制 `<module-stage>` 兜底；
- 已知 `test_ci_sanity_check_fail` 被标记为 `ExpectedFailure`。

验证命令：

```bash
python3 -m compileall -q runners scripts tests
python3 -m unittest discover -s tests -v
python3 scripts/self_check.py --pytorch-root /workspace/pytorch
```

## 仍需遵守的边界

- `Run verdict` 的进程信息只来自本机，不是跨节点调度器；
- legacy checkpoint 的历史配置无法事后恢复，manifest 只能明确暴露这一不确定性；
- `.run_manifest.json` 用于防止语义混用，不替代计划/checkpoint/report/coverage 的完成验收；
- concrete nodeid 的 Crash/Timeout 是已经定位的 case 失败，不应当作 process-level unresolved 删除；
- 官方完整覆盖仍必须以 `coverage_report.json.coverage_complete=true` 为准，不能只看进程退出或 checkpoint 条数。
