# PyTorch 测试脚本使用说明

本文档按仓库 `runners/` 最新脚本整理，命令示例基于当前容器环境。仓库可以 clone 到任意目录；其他环境先设置 `OPS_ROOT=/实际/clone/路径`，并把示例中的 `/workspace/pytorch-pytest-ops` 替换为 `$OPS_ROOT`。日志解析、报告生成和状态检查所需脚本均已包含在本仓库，不依赖旧的 `/workspace/torch_test` 或 `/home/tmp/python_and_sh` 解析脚本。

本文提供两套测试入口，二者都必须把普通测试和 distributed-tests 分开运行：

1. 直接 pytest 入口：第 2 节用 `run_pytorch_tests_prefix.py` 跑真实 `.py` 文件，第 5 节用 `run_pytorch_subset.py run-test-resume` 补 distributed-tests。
2. 官方入口：第 6 节用 `run_official_run_test_queue.py` 跑 normal 模块，第 7 节用同一队列跑 distributed 模块；shell 包装器是 `run_test-2.13-official-queue.sh`。

直接 pytest 便于得到密集、统一的 pytest 日志；官方入口保留 custom handler 和官方 `--rs/--scs` 行为。不能只执行普通 pytest 命令就宣称覆盖了所有 PyTorch 测试类别。本文当前不把被明确排除的 JIT executor 测试算入普通全量；如需覆盖它，应按目标 executor 配置单独规划，不能简单删除排除参数后与普通 pytest 队列混跑。

本文中的“不遗漏”不是假设命令启动后必然成功，而是要求可验证地满足：计划清单非空；每个计划文件/模块都在 checkpoint 中有终态；runner 已写最终 summary/report；没有遗留无法定位到 case 的 process-level timeout/crash。若最后一项不满足，文档会明确要求继续补跑或复现，而不会把它算作完整 case 级结果。

本容器路径：

- PyTorch 源码：`/workspace/pytorch`
- 测试脚本：`/workspace/pytorch-pytest-ops/runners`
- 环境脚本：`/home/tmp/python_and_sh/env.sh`
- 普通 pytest 全量示例目录：`/home/tmp/torch2.13/pytest_full_nmz`
- distributed-tests 示例目录：`/home/tmp/torch2.13/run_test_distributed_resume_nmz`

## 0. 总体选择

当前推荐把测试分成四条主线：

| 场景 | 推荐入口 | 说明 |
| --- | --- | --- |
| 普通 pytest 全量 | `run_pytorch_tests_prefix.py` | 官方 dry-run 生成文件清单，直接 `python3 -m pytest` 跑文件，多 GPU 队列，timeout/crash 恢复与自动报告 |
| 普通 pytest 子集/补跑 | `run_pytorch_subset.py pytest-list` / `pytest-failure-files` | 从已有清单或失败报告筛文件重跑 |
| 轻量 official distributed | `run_pytorch_subset.py run-test-resume` | 按官方 dry-run 模块清单串行执行，带 checkpoint；见第 5 节 |
| 官方 run_test.py 完整队列 | `run_test-2.13-official-queue.sh` | normal/distributed、子集、失败补跑、稳定失败、覆盖对账；见第 6、7 节 |

稳定失败重测使用：

```text
/workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py
```

辅助和历史脚本的当前定位：

| 文件 | 当前用途 |
| --- | --- |
| `run_pytorch_tests_prefix.py` | 普通 pytest 主入口，当前推荐 |
| `run_pytorch_subset.py` | pytest 子集、历史 process-level 文件补跑、官方 run_test.py 单次或可恢复运行 |
| `rerun_stable_failures.py` | 对 case 级失败做 N 次稳定性确认 |
| `run_official_run_test_queue.py` / `run_test-2.13-official-queue.sh` | 官方 run_test.py 完整队列及其 shell 包装器 |
| `run_pytorch_tests_gpuid.py` | 较早的普通 pytest runner，不含当前完整恢复链路 |
| `analyze_pytest_*.py` / `pytorch_pytest_*.py` | 历史解析或 tmux 方案，新流程不以它们为主入口 |

新任务应先选定“直接 pytest”或“官方队列”入口，并在同一个 work-dir 内保持不变。第 5 节轻量 distributed 和第 7 节完整 distributed 的 checkpoint/报告语义不同，不能复用同一目录。

## 1. 环境验证

每个新 shell 先加载环境：

```bash
source /home/tmp/python_and_sh/env.sh
cd /workspace/pytorch
```

检查 torch、ROCm/HIP、pytest：

```bash
python3 - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("hip:", torch.version.hip)
print("ROCM_PATH:", os.environ.get("ROCM_PATH"))
print("PYTORCH_TEST_WITH_ROCM:", os.environ.get("PYTORCH_TEST_WITH_ROCM"))
PY

python3 -m pytest --version
```

本容器里如果不 `source /home/tmp/python_and_sh/env.sh`，`import torch` 可能因为动态库路径缺失失败。

如果没有 `rg`，本文中的进程查询可以用：

```bash
ps -ef | grep -E 'run_pytorch|run_test.py|python3 -m pytest' | grep -v grep
```

### 1.1 为一次测试添加自定义环境变量

所有 runner 都从启动它的 shell 继承环境。以 `TORCHINDUCTOR_CPP_MARCH=znver1` 为例，推荐在 `source env.sh` 后显式 `export`：

```bash
source /home/tmp/python_and_sh/env.sh
export TORCHINDUCTOR_CPP_MARCH=znver1
```

随后启动的 dry-run、正式运行、自动 timeout/crash 补跑、子集、失败文件补跑和稳定失败重测都会继承该值。也可以只对一个 nohup 命令设置：

```bash
mkdir -p /home/tmp/torch2.13/pytest_example
nohup env PYTHONUNBUFFERED=1 TORCHINDUCTOR_CPP_MARCH=znver1 \
  python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch --work-dir /home/tmp/torch2.13/pytest_example \
  --dry-run-only \
  > /home/tmp/torch2.13/pytest_example/runner.out 2>&1 &
```

官方 shell 队列可放在命令前：

```bash
TORCHINDUCTOR_CPP_MARCH=znver1 \
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-normal
```

环境变量传递规则：

- 直接 pytest 全量、3.1/3.2/3.3：放进 `nohup env ...`，或提前 `export`。
- 第 4 节稳定失败重测：同样放进 `nohup env ...`，或提前 `export`。
- 第 6、7 节官方 shell 队列：提前 `export`，或像上例放在 `bash` 命令前；shell 会继续传给 Python runner 和 `run_test.py`。
- dry-run 如果会受该变量影响，也必须使用与正式运行相同的值。
- 中断续跑必须继续使用完全相同的自定义环境变量，否则同一 checkpoint 会混入不同编译/运行配置的结果。
- 比较不同 `TORCHINDUCTOR_CPP_MARCH` 时应使用不同 work-dir，不要复用 checkpoint。

`TORCHINDUCTOR_CPP_MARCH` 会改变 TorchInductor C++ kernel 和 host-side C++ wrapper 的编译参数。x86_64 默认通常是 `-march=native`，设置 `znver1` 后变成 `-march=znver1`；它可能改变可用 SIMD、编译结果、性能、timeout 概率和部分数值路径。它不是所有节点都应设置的通用优化项：只有测试目标明确要求 `znver1` 时才设置，并在目录名或测试说明中标明该配置。当前 NMZ 容器是 Intel CPU，不应把 `znver1` 当作默认基线；已有 `*_znver1*` 目录的首次运行、续跑和重测则必须继续保持该变量，不能中途切回默认值。

确认当前 shell：

```bash
printenv TORCHINDUCTOR_CPP_MARCH
```

确认运行中进程（将 PID 换成实际 pytest/run_test PID）：

```bash
PID_TO_CHECK=12345
tr '\0' '\n' < "/proc/$PID_TO_CHECK/environ" | grep '^TORCHINDUCTOR_CPP_MARCH='
```

不要为了单次任务修改脚本里的 `DEFAULT_TEST_ENV`；那会变成所有任务的隐式默认值，不利于结果追踪。

## 2. 普通 Pytest 全量测试：run_pytorch_tests_prefix.py

### 2.1 这个脚本做什么

入口：

```text
/workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py
```

流程：

1. 进入 `/workspace/pytorch/test`
2. 调用官方：

```text
python3 run_test.py --dry-run --exclude-jit-executor --exclude-distributed-tests
```

3. 解析 `Serial tests` / `Parallel tests`，生成：

```text
<work-dir>/test_files.txt
```

4. 多 GPU 动态队列运行每个文件：

```text
python3 -m pytest --tb=long --color=no --sc=<stepcurrent_key> --print-items <test_file>
```

5. 每个 worker 通过以下环境变量绑定一个 ROCm GPU：

```text
HIP_VISIBLE_DEVICES=<gpu_id>
```

当前脚本不主动重写 `CUDA_VISIBLE_DEVICES`。在 ROCm 环境中，PyTorch API 仍使用 `cuda` 设备名，但可见设备由 `HIP_VISIBLE_DEVICES` 控制。若 `env.sh` 预先设置了 `CUDA_VISIBLE_DEVICES`，该值会被子进程继承。

6. 写文件级进度：

```text
<work-dir>/.test_progress.json
```

7. 生成失败报告：

```text
<work-dir>/<timestamp>/failure_report.csv
<work-dir>/<timestamp>/failure_report.json
<work-dir>/<timestamp>/failure_report.md
```

### 2.2 dry-run 只生成清单

```bash
source /home/tmp/python_and_sh/env.sh
cd /workspace/pytorch

mkdir -p /home/tmp/torch2.13/pytest_full_nmz

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_full_nmz \
  --dry-run-only
```

输出：

```text
/home/tmp/torch2.13/pytest_full_nmz/test_files.txt
```

验证清单不是空文件，并查看数量、头尾内容：

```bash
wc -l /home/tmp/torch2.13/pytest_full_nmz/test_files.txt
sed -n '1,10p' /home/tmp/torch2.13/pytest_full_nmz/test_files.txt
tail -10 /home/tmp/torch2.13/pytest_full_nmz/test_files.txt
```

`wc -l` 是本次普通 pytest 的文件总数。正式运行会重新执行一次相同 dry-run 并覆盖该清单；源码、环境和透传给 `run_test.py` 的参数不变时，数量应一致。脚本只接受官方输出中 `Serial tests (N):` 和 `Parallel tests (N):` 块内的条目，并忽略 `Name: excluded` 块；解析不到任何文件时会报错退出，不会以空清单继续。

官方 dry-run 还可能返回没有对应 `.py` 文件的 custom handler，例如 `doctests`、`test_autoload_enable/disable`、`test_cpp_extensions_aot_ninja/no_ninja`。普通 pytest runner 会追加 `.py` 后在 `filter_existing()` 阶段排除它们，因此某次实际结果可能出现“清单 640、checkpoint 635”。这 5 项不是尚未调度的普通文件，而是当前入口无法直接 pytest 的官方特殊目标；要覆盖它们必须执行第 6 节官方队列。官方队列保留原始模块名并调用 `run_test.py --include`，不会因文件不存在而漏掉 custom handler。

注意：默认 dry-run 带：

```text
--exclude-jit-executor --exclude-distributed-tests
```

所以这里不包含官方意义上的 `distributed/...` tests。普通文件名里带 `distributed` 的测试仍可能出现，例如：

```text
dynamo/test_fake_distributed.py
inductor/test_distributed_patterns.py
```

### 2.3 正式全量后台运行

下面命令可以直接在本容器运行。普通全量明确排除 distributed-tests，随后必须执行第 5 节的 distributed 命令补齐该类别。

```bash
source /home/tmp/python_and_sh/env.sh
cd /workspace/pytorch

mkdir -p /home/tmp/torch2.13/pytest_full_nmz

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_full_nmz \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  --fresh \
  > /home/tmp/torch2.13/pytest_full_nmz/runner.out 2>&1 &
```

参数说明：

- `--timeout 1800`：单个测试文件主流程最多运行 1800 秒
- `--recovery-case-timeout 600`：stepcurrent `--rs` 和 fallback 精确到单 case 后，每次最多运行 600 秒；设为 0 时复用文件 timeout
- `--recovery-attempts 3`：同一个 crash/timeout case 最多精确重测 3 次，任一次正常结束就停止重试；普通断言失败也算得到了明确结果
- `--recovery-max-total-time 7200`：一个文件进入 timeout/crash 恢复后，`--rs`、`--scs`、collect 和 chunk fallback 合计最多运行 7200 秒；设为 0 表示不限制
- `--process-rerun-error-types Timeout,Crash`：第一次全量结束后，自动补跑报告里仍然只有文件级结果的 timeout/crash 文件
- `--process-rerun-timeout 14400`：文件级 timeout/crash 自动完整文件补跑时最多 14400 秒；脚本未显式传入时默认是 `max(7200, --timeout * 4)`
- `--process-rerun`：显式启用自动文件补跑；当前脚本默认已启用，命令中保留它是为了让运行意图清楚
- `--fresh`：删除旧 `.test_progress.json`，从头开始
- 不加 `--fresh`：读取旧进度续跑
- `PYTHONUNBUFFERED=1 python3 -u`：让 `runner.out` 实时刷新

### 2.4 输出目录

一次完整运行目录大致如下：

```text
/home/tmp/torch2.13/pytest_full_nmz/
  runner.out
  test_files.txt
  .test_progress.json
  summary.json
  latest -> <timestamp>
  <timestamp>/
    gpu_0.log
    gpu_1.log
    ...
    process_file_rerun/             # 只有选中 process-level 文件时才出现
      test_files.txt
      summary.json
      gpu_0.log
      ...
    failure_report.csv
    failure_report.json
    failure_report.md
    unresolved_process_failures.csv
    unresolved_process_failures.json
    unresolved_process_failures.md
    unresolved_process_failure_files.txt
```

文件含义：

- `runner.out`：主调度器日志
- `test_files.txt`：官方 dry-run 得到的测试文件清单
- `.test_progress.json`：文件级 checkpoint
- `summary.json`：本次启动的 worker/case 统计、累计 checkpoint 状态和最终报告路径；续跑时 `total.passed/failed/skipped` 只统计本次实际执行，累计文件状态看 `.test_progress.json` 或 runner 尾部的 `Cumulative ...`
- `latest`：指向最近一次 timestamp 目录
- `gpu_*.log`：每张 GPU 的原始 pytest 日志
- `process_file_rerun/`：文件级 timeout/crash 自动补跑日志；没有 process-level 文件时不创建
- `failure_report.csv/json/md`：最终失败报告，以这个为准
- `unresolved_process_failures.csv/json/md`：最终仍只有文件级 `<timeout>/<crash>`、没有可靠 case nodeid 的独立报告
- `unresolved_process_failure_files.txt`：上述未定位文件的去重清单，可直接用于后续专项补跑

### 2.5 timeout/crash 现在怎么处理

当前全量脚本对第一次全量发现的 timeout/crash 有三层处理。

第一层：stepcurrent 定位当前 case。

主 pytest 命令带：

```text
--sc=<stepcurrent_key> --print-items
```

PyTorch `test/conftest.py` 会把当前运行的 item 写入 pytest cache。不同版本的物理布局不同：

```text
PyTorch 2.9:        /workspace/pytorch/.pytest_cache/v/cache/stepcurrent/<stepcurrent_key>
本地 PyTorch 2.13: /workspace/pytorch/.pytest_cache/v/cache/stepcurrent/<stepcurrent_key>/lastrun
```

runner 会先探测新版目录布局，再兼容读取 2.9 的平面文件布局。进程 timeout/crash 后，脚本优先读取这个断点，尽量得到具体 case。cache 不存在、尚未写入或文件/目录类型不匹配时只会判定 stepcurrent 不可用并进入 fallback，不允许因此让 GPU worker 异常退出。

第二层：借鉴官方 `run_test.py` 的 `--rs` / `--scs` 继续机制。

原进程死掉后不会原地继续。脚本会启动新 pytest 进程：

```text
--rs=<stepcurrent_key>
```

单独重跑 stepcurrent 记录的 case。如果这个 case 通过，再启动：

```text
--scs=<stepcurrent_key>
```

跳过已跑到的位置，继续后面的 case。如果同一个 case 连续 crash/timeout 达到 `--recovery-attempts`，则写成具体 case 级失败，再跳过它继续后面 case。`--rs` 使用 `--recovery-case-timeout`；`--scs` 仍使用文件级 `--timeout`，因为它运行的是从断点到文件结尾的一段 case。

旧版还有固定 `max_iterations=200`：它限制整个文件中 `--rs + --scs` 的总轮数，不是同一个 case 重跑 200 次。达到上限会退回全文件 collect/chunk，在数万 case 文件上可能非常慢。新版取消这个固定 200 轮切换，改为 `--recovery-max-total-time` 累计时间预算；同一个 case 的精确重测次数仍由 `--recovery-attempts` 单独限制。

累计预算耗尽时，脚本写入 `STEPCURRENT RECOVERY BUDGET EXHAUSTED` 或 `RECOVERY ABORTED`，保留文件级 process Timeout，并停止该文件恢复，不会写假的 `RECOVERY DONE`。首轮报告随后会把它交给 `process_file_rerun/`；如果大 timeout 补跑后仍耗尽预算，最终 unresolved 报告会明确保留该文件，不能算作 case 覆盖完整。

这里与官方 `run_test.py` 保持一个容易忽略的返回码约定：`pytest --scs` 返回 `5` 表示当前续跑分片中已经没有测试，不是测试失败。runner 会把它归一为成功并结束恢复；旧版把 `5` 当失败，会反复执行同一个空分片，直到命中恢复迭代上限后转入 collect/chunk fallback。日志中连续出现 `Running 0 items in this shard` 通常就是该旧问题。

第三层：文件级兜底补跑。

如果最终仍然只能得到文件级：

```text
case_name = <timeout>
case_name = <crash>
```

脚本会在分析失败报告后自动把这些文件拿出来，用更大的 timeout 重跑：

```text
process_file_rerun/
```

补跑完成后会重新生成最终 `failure_report.csv/json/md`。如果补跑通过，旧 `<timeout>/<crash>` 行会被过滤；如果补跑得到具体 case，最终报告保留具体 case。

报告器还会识别同一文件后续出现的终态标记：

```text
STEPCURRENT RECOVERY DONE
RECOVERY DONE
PASS
```

看到这些标记，说明原 timeout/crash 后的剩余区间或 fallback 已经执行完成。最终报告会删除该日志前面遗留的文件级 `<timeout>/<crash>` 占位行，同时保留恢复阶段抽取出的具体失败 case。没有完成恢复标记的进程级异常不会被删除。

这里的“自动补跑”不是简单地把原来的文件级行删掉：只有该文件确实进入 `process_file_rerun/test_files.txt` 后，初次运行产生的旧 process-level 行才会被替换；补跑日志本身若仍然 timeout/crash，最终报告仍会保留新的 `<timeout>/<crash>` 行，提醒该文件没有得到完整 case 级结论。

第四层：显式输出仍未定位的文件。

每次生成报告（包括自动补跑后的最终报告）都会额外生成：

```text
unresolved_process_failures.csv
unresolved_process_failures.json
unresolved_process_failures.md
unresolved_process_failure_files.txt
```

其中只包含没有 `::case` nodeid 或 `case_name` 仍为 `<timeout>/<crash>` 的 process-level 行。最终 `summary.json.failure_reports.unresolved_process_failure_count` 同步记录数量。数量为 0 才表示失败报告里不存在未定位到 case 的进程级异常；大于 0 时必须查看该独立报告，不能把对应文件算作已有完整 case 级结论。

当前脚本还对恢复底层做了多项与官方实现对齐的修正：stepcurrent 从仓库根目录 `/workspace/pytorch/.pytest_cache/...` 读取；timeout 时终止整个 pytest 进程组并保留终止前输出；`--scs` 返回码 `5` 按完成处理；报告器依据恢复完成标记过滤旧 process-level 占位行。collect-only fallback 同时识别 `test/foo.py::case` 和 `foo.py::case` 两种 nodeid。它们能显著减少误报的未定位项，但如果进程在收集/导入阶段、首个 pytest item 运行前就崩溃，stepcurrent 本来就没有 case 可记录，此时仍只能保留文件级异常。

默认自动补跑错误类型：

```text
Timeout,Crash
```

可配置：

```bash
--process-rerun-error-types Timeout,Crash
--process-rerun-error-types Timeout
--process-rerun-error-types Crash
--no-process-rerun
```

### 2.6 查看状态

主日志：

```bash
tail -f /home/tmp/torch2.13/pytest_full_nmz/runner.out
```

GPU 日志：

```bash
tail -f /home/tmp/torch2.13/pytest_full_nmz/latest/gpu_*.log
```

补跑日志：

```bash
tail -f /home/tmp/torch2.13/pytest_full_nmz/latest/process_file_rerun/gpu_*.log
```

进程：

```bash
ps -ef | grep -E 'run_pytorch_tests_prefix|python3 -m pytest' | grep -v grep
```

进度：

```bash
python3 - <<'PY'
import json
p="/home/tmp/torch2.13/pytest_full_nmz/.test_progress.json"
d=json.load(open(p))
print(d.get("stats", {}))
PY
```

输出中的 `total` 是已经写入 checkpoint 的文件数，不是 `test_files.txt` 的计划总数；`passed`、`failed`、`timeout`、`skipped` 是这些已记录文件的状态，`remaining` 只表示 checkpoint 内存在未知状态。因此运行中判断整体剩余量，应使用下面的清单对账命令，而不能只看这里的 `remaining`。

### 2.7 中断后续跑

续跑不要加 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh
cd /workspace/pytorch

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_full_nmz \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  > /home/tmp/torch2.13/pytest_full_nmz/runner_resume.out 2>&1 &
```

续跑规则：

- PASS：跳过
- SKIP：跳过
- FAIL：默认重跑
- 加 `--skip-fail`：跳过旧 FAIL

### 2.8 确认是否正常结束

先看进程：

```bash
ps -ef | grep -E 'run_pytorch_tests_prefix|python3 -m pytest' | grep -v grep
```

没有相关进程，再看 summary：

```bash
cat /home/tmp/torch2.13/pytest_full_nmz/summary.json
tail -120 /home/tmp/torch2.13/pytest_full_nmz/runner.out
```

检查最终报告是否存在：

```bash
ls -lh /home/tmp/torch2.13/pytest_full_nmz/latest/failure_report.*
ls -lh /home/tmp/torch2.13/pytest_full_nmz/latest/unresolved_process_failures.*
cat /home/tmp/torch2.13/pytest_full_nmz/latest/unresolved_process_failure_files.txt
```

最后做清单与 checkpoint 对账，这是确认普通 pytest 文件没有漏调度的关键检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path

work = Path("/home/tmp/torch2.13/pytest_full_nmz")
planned = [x.strip() for x in (work / "test_files.txt").read_text().splitlines() if x.strip()]
data = json.loads((work / ".test_progress.json").read_text())
done = data.get("tests", data)
virtual = [x for x in planned if not (Path("/workspace/pytorch/test") / x).is_file()]
real = [x for x in planned if x not in virtual]
missing = [x for x in real if x not in done]
unknown = [x for x in real if x in done and done[x].get("status") not in {"PASS", "FAIL", "SKIP", "TIMEOUT"}]
print("planned files:", len(planned))
print("real pytest files:", len(real))
print("virtual/custom-handler targets:", len(virtual))
print("checkpoint real files:", sum(x in done for x in real))
print("missing real files:", len(missing))
print("unknown status:", len(unknown))
for x in virtual[:20]:
    print("VIRTUAL", x)
for x in missing[:20]:
    print("MISSING", x)
PY
```

正常完整结束应满足：没有相关进程、`summary.json` 和最终失败报告存在、`missing real files: 0`、`unknown status: 0`。当前源码常见的 5 个 `VIRTUAL` 是 `doctests`、autoload 和 cpp extension AOT custom handler，不是普通 pytest 文件遗漏；它们由第 6 节官方 normal 队列覆盖。`FAIL` 代表测试确实执行后失败，不代表漏跑。最终报告仍有 `<timeout>`/`<crash>` 时，代表自动大超时补跑后仍无法得到完整 case 级结果，需要继续查看 `latest/process_file_rerun/`，不能把这次运行视为已有完整 case 结论。

### 2.9 查询某个 case

查失败报告：

```bash
python3 - <<'PY'
import csv
p="/home/tmp/torch2.13/pytest_full_nmz/latest/failure_report.csv"
needle="test_aot_compile"
for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
    text=" ".join(r.values())
    if needle in text:
        print(r["nodeid"], r["error_type"], r["error_message"])
PY
```

查原始日志：

```bash
grep -R "test_aot_compile" /home/tmp/torch2.13/pytest_full_nmz/latest/*.log
```

注意：`failure_report.csv` 只记录失败，不是所有 case 的结果数据库。查询通过 case 时应查原始 `gpu_*.log`；pytest 默认进度输出可能只显示 nodeid/状态，参数化 case 以完整 nodeid 为准。

### 2.10 历史结果迁移：stepcurrent cache 修复

旧版 runner 错误读取 `/workspace/pytorch/test/.pytest_cache/...`，而 pytest 和官方 `run_test.py` 实际写入：

```text
/workspace/pytorch/.pytest_cache/v/cache/stepcurrent/<key>/lastrun
```

新版已经修正。历史全量不需要全部重跑：先在 process rerun 已有 `summary.json` 后执行 `--analyze-only` 生成 `unresolved_process_failures.csv`，再按 3.3 只重跑其中的文件。重新分析只重建报告，不能追溯执行当时错过的 `--rs/--scs`。

2026-07-13 增加 PyTorch 2.9/2.13 cache 布局兼容。旧 runner 固定读取 `<key>/lastrun`，在 2.9 中 `<key>` 本身是文件，因此会抛 `NotADirectoryError` 并让 worker 提前退出。新版同时读取 `<key>/lastrun` 和 `<key>`，并把候选路径读取失败降级为 stepcurrent unavailable。已经因该异常退出的任务不会因代码升级自动补回：等待或停止旧 parent runner 后，使用同一个 work-dir、相同 GPU/timeout/环境参数续跑，并去掉 `--fresh`；checkpoint 中缺失的真实文件会重新进入队列。正在运行的 Python 进程已经加载旧模块，必须启动新的续跑进程才会使用修复。

2026-07-12 又修正了两个后续问题：空 `--scs` 分片返回码 `5` 未按官方语义处理，以及恢复完成后旧 process-level 行仍进入 unresolved 报告。已经完成的测试无需因此再执行；对已有日志运行 `--analyze-only` 即可按新规则重建报告。分析目标是当前 `latest` 时，脚本还会同步更新根目录 `summary.json.failure_reports`；分析旧 timestamp 时不会覆盖当前汇总。只有日志没有恢复完成标记、确实仍列在新 `unresolved_process_failure_files.txt` 中的文件，才需要按 3.3 补跑。

当前 `log-final` 最初的历史全量日志重新分析后曾得到 BW 15 个、NMZ 9 个 unresolved 文件；这是旧执行过程的结果，不是脚本固定值。NMZ 随后在 `unresolved_after_stepcurrent_fix` 中补跑这些文件，并在 2026-07-12 用修正后的报告器重新分析：共保留 103 条具体失败 case，process-level unresolved 为 0。每次都应以目标目录当前的 `unresolved_process_failures.csv` 和 `summary.json` 为准，不要沿用文档中的历史数字。

新启动的 3.1/3.2/3.3 任务都会加载修正后的相同 `run_gpu_tests`；第 4 节输入是精确 case，本身不需要 `--rs/--scs`。

以 `TORCHINDUCTOR_CPP_MARCH=znver1` 补跑历史 unresolved 文件时，两个节点分别使用独立 work-dir，示例命令见本节对应环境：

```text
BW:  /home/tmp/torch2.13/log-final/pytest_full_bw/unresolved_after_stepcurrent_fix_znver1
NMZ: /home/tmp/torch2.13/log-final/pytest_full_nmz/unresolved_after_stepcurrent_fix_znver1
```

正式命令采用 14400 秒文件 timeout；中断续跑必须保留该环境变量、输入 CSV、timeout 和 work-dir，并去掉 `--fresh`。

## 3. 普通 Pytest 子集和补跑

本节三个入口都以“选中的测试文件”为调度粒度。判断“不遗漏”统一使用三层标准：选中清单非空、清单中的每个文件都在 checkpoint 中有终态、最终报告不存在尚未解决的 process-level `<timeout>/<crash>`。前两项保证文件被调度，第三项才表示得到了完整 case 级结论。

恢复核心的复用关系：3.1 直接使用 `run_pytorch_tests_prefix.py`；3.2 的 `pytest-list` 从该模块导入同一个 `run_gpu_tests`；3.3 最终委托给 3.2。因此这三种模式在新启动的进程中都使用相同的 stepcurrent、`--rs/--scs`、collect/chunk fallback 和进程组 timeout 清理，也都支持 `--recovery-case-timeout`、`--recovery-attempts`、`--recovery-max-total-time`。已经启动或已经结束的旧进程不会因源码更新而被追溯修改。

### 3.1 使用 --include-prefix

`run_pytorch_tests_prefix.py` 可以在当前源码的官方 dry-run 结果上按逗号分隔的路径前缀过滤。它仍然排除官方 distributed-tests，适合 `inductor/`、`dynamo/` 等普通 pytest 子集，并且保留主脚本的 timeout/crash 自动文件补跑。

先确认选择结果：

```bash
source /home/tmp/python_and_sh/env.sh
mkdir -p /home/tmp/torch2.13/pytest_prefix_ind_dyn

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_prefix_ind_dyn \
  --include-prefix inductor/,dynamo/ \
  --dry-run-only

wc -l /home/tmp/torch2.13/pytest_prefix_ind_dyn/test_files.txt
```

正式后台运行：

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/pytest_prefix_ind_dyn

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_prefix_ind_dyn \
  --include-prefix inductor/,dynamo/ \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  --fresh \
  > /home/tmp/torch2.13/pytest_prefix_ind_dyn/runner.out 2>&1 &
```

注意：由于 dry-run 固定带 `--exclude-distributed-tests`，不要用这个方式跑官方 distributed tests。

输出目录与第 2.4 节相同，根目录换为 `/home/tmp/torch2.13/pytest_prefix_ind_dyn`，核心文件是 `test_files.txt`、`.test_progress.json`、`summary.json`、`latest/gpu_*.log`、`latest/process_file_rerun/` 和 `latest/failure_report.*`。

查看状态：

```bash
tail -f /home/tmp/torch2.13/pytest_prefix_ind_dyn/runner.out
ps -ef | grep -E 'run_pytorch_tests_prefix|python3 -m pytest' | grep -v grep
```

中断后使用完全相同的筛选和 timeout 参数，去掉 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir /home/tmp/torch2.13/pytest_prefix_ind_dyn \
  --include-prefix inductor/,dynamo/ \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --process-rerun \
  --process-rerun-error-types Timeout,Crash \
  --process-rerun-timeout 14400 \
  > /home/tmp/torch2.13/pytest_prefix_ind_dyn/runner_resume.out 2>&1 &
```

确认正常结束时，把第 2.8 节对账脚本中的 work 路径改为该目录。要求 `missing files: 0`、`unknown status: 0`，最终报告存在；若还有 `<timeout>/<crash>`，则文件已调度但 case 级覆盖仍不完整。

### 3.2 从已有 test_files.txt 筛选

入口：

```text
/workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-list
```

先 dry-run 生成本次固定选择清单：

```bash
source /home/tmp/python_and_sh/env.sh
mkdir -p /home/tmp/torch2.13/pytest_subset_ind_dyn

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-list \
  /workspace/pytorch \
  --test-list /home/tmp/torch2.13/pytest_full_nmz/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir /home/tmp/torch2.13/pytest_subset_ind_dyn \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only

wc -l /home/tmp/torch2.13/pytest_subset_ind_dyn/selected_test_files.txt
```

正式后台运行：

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/pytest_subset_ind_dyn

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-list \
  /workspace/pytorch \
  --test-list /home/tmp/torch2.13/pytest_full_nmz/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir /home/tmp/torch2.13/pytest_subset_ind_dyn \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --fresh \
  > /home/tmp/torch2.13/pytest_subset_ind_dyn/runner.out 2>&1 &
```

也可以用正则：

```bash
python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-list \
  /workspace/pytorch \
  --test-list /home/tmp/torch2.13/pytest_full_nmz/test_files.txt \
  --include-regex '^(inductor|dynamo)/.*compile.*\.py$' \
  --work-dir /home/tmp/torch2.13/pytest_subset_compile \
  --gpu-ids 0,1 \
  --dry-run-only
```

`pytest-list` 的筛选顺序是：读取并去重清单，应用 include prefix/regex，再应用 exclude prefix/regex，最后只运行源码树中真实存在的文件。它复用普通 runner 的 GPU 动态队列、stepcurrent crash 恢复、checkpoint 和失败报告，但不会重新调用官方 dry-run，也没有主全量脚本最后一层自动 process-file rerun。子集报告若仍有 `<timeout>/<crash>`，可再交给下一节的 `pytest-failure-files`。

输出目录：

```text
/home/tmp/torch2.13/pytest_subset_ind_dyn/
  runner.out
  selected_test_files.txt
  .test_progress.json
  summary.json
  latest -> <timestamp>
  <timestamp>/
    gpu_*.log
    failure_report.csv
    failure_report.json
    failure_report.md
    unresolved_process_failures.csv
    unresolved_process_failures.json
    unresolved_process_failures.md
    unresolved_process_failure_files.txt
```

这里的 `summary.json` 与全量 runner 一样，worker/case 数只描述本次启动实际执行的文件；跨续跑累计状态以 `.test_progress.json` 为准。`pytest-list` 没有全量 runner 的自动 `process_file_rerun/`，所以 unresolved 不为 0 时必须进入 3.3。

查看状态：

```bash
tail -f /home/tmp/torch2.13/pytest_subset_ind_dyn/runner.out
tail -f /home/tmp/torch2.13/pytest_subset_ind_dyn/latest/gpu_*.log
ps -ef | grep -E 'run_pytorch_subset.py pytest-list|python3 -m pytest' | grep -v grep
```

中断续跑，筛选条件必须与首次运行一致，去掉 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-list \
  /workspace/pytorch \
  --test-list /home/tmp/torch2.13/pytest_full_nmz/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir /home/tmp/torch2.13/pytest_subset_ind_dyn \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  > /home/tmp/torch2.13/pytest_subset_ind_dyn/runner_resume.out 2>&1 &
```

确认选中文件没有遗漏：

```bash
python3 - <<'PY'
import json
from pathlib import Path
work=Path('/home/tmp/torch2.13/pytest_subset_ind_dyn')
planned=[x for x in work.joinpath('selected_test_files.txt').read_text().splitlines() if x]
done=json.loads(work.joinpath('.test_progress.json').read_text()).get('tests', {})
missing=[x for x in planned if x not in done]
print('selected files:', len(planned))
print('checkpoint files:', sum(x in done for x in planned))
print('missing files:', len(missing))
for x in missing[:20]: print('MISSING', x)
PY
```

没有相关进程、`summary.json` 与 `latest/failure_report.*` 存在且 `missing files: 0`，表示所有选中文件已得到终态。由于本子命令没有自动 process-file rerun，还要按第 8.1 节检查 process-level 行；存在时继续用 3.3 补跑。

### 3.3 从 failure_report.csv 补跑文件级 timeout/crash

入口：

```text
/workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-failure-files
```

适合补救历史报告里的：

```text
case_name = <timeout>
case_name = <crash>
```

先看实际会选哪些历史 process-level 文件：

```bash
source /home/tmp/python_and_sh/env.sh
mkdir -p /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-failure-files \
  /workspace/pytorch \
  --failure-csv /home/tmp/torch2.13/pytest_full_bw/20260629_181949/failure_report.csv \
  --work-dir /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual \
  --error-type Timeout,Crash \
  --dry-run-only

cat /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual/failure_process_test_files.txt
```

正式补跑所有选中的 timeout/crash 文件，使用更大的文件 timeout：

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-failure-files \
  /workspace/pytorch \
  --failure-csv /home/tmp/torch2.13/pytest_full_bw/20260629_181949/failure_report.csv \
  --work-dir /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual \
  --publish-to-work-dir /home/tmp/torch2.13/pytest_full_bw \
  --error-type Timeout,Crash \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 14400 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 14400 \
  --fresh \
  > /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual/runner.out 2>&1 &
```

输出结构与 3.2 相同，选择清单文件名是 `failure_process_test_files.txt`。日志和报告位于该 work-dir 的 `latest/`。

查看状态：

```bash
tail -f /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual/runner.out
tail -f /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual/latest/gpu_*.log
ps -ef | grep -E 'run_pytorch_subset.py pytest-failure-files|python3 -m pytest' | grep -v grep
```

中断续跑时参数不变、去掉 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-failure-files \
  /workspace/pytorch \
  --failure-csv /home/tmp/torch2.13/pytest_full_bw/20260629_181949/failure_report.csv \
  --work-dir /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual \
  --publish-to-work-dir /home/tmp/torch2.13/pytest_full_bw \
  --error-type Timeout,Crash \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 14400 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 14400 \
  > /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual/runner_resume.out 2>&1 &
```

这个子命令只选择没有 case nodeid（没有 `::`）或 `case_name` 为 `<timeout>/<crash>` 的 process-level 行，再按 `--error-type` 过滤；普通 case 失败不会被误当成整文件补跑。实际输入清单写到 `<work-dir>/failure_process_test_files.txt`。中断后使用相同参数和 work-dir、去掉 `--fresh` 即可从 `<work-dir>/.test_progress.json` 续跑。

`--publish-to-work-dir` 指向原全量 work-dir。补跑结束后，脚本自动用这批文件的新结果替换主 `latest/failure_report.csv` 中对应文件的旧记录，并同步：

```text
latest/failure_report.csv/json/md
latest/unresolved_process_failures.csv/json/md
latest/unresolved_process_failure_files.txt
summary.json.failure_reports
latest/external_rerun_merge.json
```

`external_rerun_merge.json` 也是持久化合并依据。以后对主 `latest` 再执行 `--analyze-only` 时，报告器会重新应用这批补跑结果，不会因为只扫描到早期主日志而恢复旧的文件级异常。

发布不是无条件执行。只有以下条件全部满足才会修改主报告：选中文件在补跑 checkpoint 中全部为 PASS/FAIL/SKIP；补跑根目录已有 `summary.json`；补跑报告的 `unresolved_process_failure_count` 为 0。中断中的任务、缺少终态的文件或仍有文件级异常时会拒绝发布，因此不会提前隐藏主报告中的旧 `<timeout>/<crash>`。

这里验收的是“文件级 unresolved 为 0”。具体 `file.py::Class::case` 仍可能以 `error_type=Crash/Timeout` 出现在最终 CSV，它表示异常已经定位到明确 case，不属于漏跑或未定位；不要为了让错误类型消失而过滤这些真实失败。

不传 `--publish-to-work-dir` 时只生成独立补跑报告，不会修改主全量报告。第 2 节从头执行的全量 runner 使用自身的 `process_file_rerun/`，结束时本来就会自动更新主报告；该参数主要用于历史结果的独立补跑。

确认结束时用 3.2 的对账脚本，把清单文件名改为 `failure_process_test_files.txt`、work 路径改为本目录。要求 `missing files: 0`。然后检查新 `latest/failure_report.csv`：仍有 `<timeout>/<crash>` 说明即使 14400 秒补跑也未得到完整 case 结论；不能仅因 checkpoint 有 FAIL 就认为文件内部已完整运行。

如果补跑此前已经完成，只差发布，不要重新执行失败文件。增加 `--skip-fail` 并带发布参数即可复用现有 checkpoint、summary 和报告：

```bash
source /home/tmp/python_and_sh/env.sh

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py pytest-failure-files \
  /workspace/pytorch \
  --failure-csv /home/tmp/torch2.13/pytest_full_bw/latest/unresolved_process_failures.csv \
  --work-dir /home/tmp/torch2.13/pytest_full_bw/process_file_rerun_manual \
  --publish-to-work-dir /home/tmp/torch2.13/pytest_full_bw \
  --error-type Timeout,Crash \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 14400 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 14400 \
  --skip-fail
```

如果输入已经是新版生成的 `unresolved_process_failures.csv`，也可以直接作为 `--failure-csv`。这适合在修复恢复逻辑后只重跑历史结果中仍未定位的文件，而不必重跑整个全量清单。

## 4. 稳定失败重测

### 4.1 功能、输入和判定规则

入口：

```text
/workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py
```

它读取 `failure_report.csv`、`stable_failures.csv` 或 `rerun_all_results.csv`，规范化并按 `nodeid` 去重。默认跳过没有 `::` 的文件级 process row；这些行应先按 3.3 补成 case 级结果。每个 case 最多运行 `--attempts N` 次，一旦有一次 PASS 就提前停止；只有连续 N 次都失败，才写入 `stable_failures.csv`。timeout/crash 都按一次失败处理。

稳定失败重测不使用 `--rs/--scs`：它的输入本来就是一个精确 `file.py::Class::case`，每次尝试只运行该 case，不存在“从文件中当前 case 后继续”的需求。某次单 case 进程 crash/timeout 会被记录为该 nodeid 的一次失败，再按 `--attempts` 重试。文件级 `<crash>/<timeout>` 默认不会进入本脚本，应先通过 3.3 定位为 case。

可用任意输入列筛选，但必须指定一种匹配方式：

- `--filter-equals`：完全相等。
- `--filter-contains`：包含文本。
- `--filter-regex`：正则匹配。

### 4.2 dry-run 确认重测清单

先解析输入并记录 `Unique rerun rows` 和 `Missing targets`：

```bash
source /home/tmp/python_and_sh/env.sh

python3 /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/pytest_full_nmz/latest/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-list
```

`Unique rerun rows` 是计划 case 数；`Missing targets` 必须为 0，否则对应 nodeid 在当前 `/workspace/pytorch/test` 下已不存在或格式不能执行。dry-run 最多打印前 50 个 nodeid，但计数包含全部。

### 4.3 正式后台运行

正式跑 3 次确认稳定失败：

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/pytest_full_nmz/latest/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x \
  --fresh \
  > /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/runner.out 2>&1 &
```

`--fresh` 会删除同一 output-dir 的旧 `rerun_checkpoint.csv`，只应在确认从头开始时使用。正式命令必须和 4.2 使用相同输入和筛选参数，才能让计划数可对账。

### 4.4 输出目录和字段

输出：

```text
stable_rerun_3x/
  runner.out
  rerun_checkpoint.csv
  rerun_all_results.csv
  stable_failures.csv
  summary.json
  rerun_gpu_<id>.log
  rerun_worker_0.log        # 未传 --gpu-ids、继承全部可见 GPU 时
```

文件和字段含义：

- `rerun_checkpoint.csv`：每完成一个 nodeid 立即追加一行，中断恢复以它为准。
- `rerun_all_results.csv`：该 output-dir 中全部已完成重测结果，包括通过、波动失败和稳定失败。
- `stable_failures.csv`：只有连续失败达到 N 次的 case。
- `rerun_gpu_<id>.log`：传了 `--gpu-ids` 时，每个 GPU worker 的逐 attempt 原始日志。
- `rerun_worker_0.log`：未传 `--gpu-ids` 时的单 worker 日志，常用于 distributed case 串行重测。
- `attempts_run`：实际尝试次数；某次通过后会提前停止，所以可能小于 `--attempts`。
- `statuses`、`returncodes`、`error_types`：按尝试顺序记录每次状态、退出码和错误类型。
- `stable_failed=yes`：所有要求的尝试均失败；timeout/crash 也属于失败。

### 4.5 按列筛选

按任意列筛选后重跑，例如只跑 C++ 编译错误：

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/pytest_full_bw/stable_rerun_3x_march_x86_64/stable_failures.csv \
  --attempts 1 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --filter-column error_messages \
  --filter-equals 'CppCompileError: C++ compile error' \
  --output-dir /home/tmp/torch2.13/pytest_full_bw/stable_rerun_3x_march_x86_64_cpp_compile_after_fix \
  --fresh \
  > /home/tmp/torch2.13/pytest_full_bw/stable_rerun_3x_march_x86_64_cpp_compile_after_fix/runner.out 2>&1 &
```

筛选任务也应先把同样参数改成 `--dry-run-list` 验证数量，再执行正式命令。`--attempts 1` 表示只跑一次，不是在原有三次基础上追加一次。

### 4.6 查看状态和中断续跑

查看状态：

```bash
tail -f /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/runner.out
ps -ef | grep -E 'rerun_stable_failures|python3 -m pytest' | grep -v grep
```

查看已完成数量和当前稳定失败数：

```bash
python3 - <<'PY'
import csv
from pathlib import Path
p=Path('/home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/rerun_checkpoint.csv')
rows=list(csv.DictReader(p.open(newline='', encoding='utf-8'))) if p.exists() else []
print('completed cases:', len(rows))
print('stable failed so far:', sum(r.get('stable_failed') == 'yes' for r in rows))
PY
```

`completed cases` 是已经完成全部判定流程的唯一 nodeid 数，不是尝试次数；`stable failed so far` 是其中连续失败达到 N 次的数量。

中断后使用相同输入 CSV、筛选条件、`--attempts`、timeout、GPU 和 output-dir，去掉 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/pytest_full_nmz/latest/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x \
  > /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/runner_resume.out 2>&1 &
```

脚本按 nodeid 跳过 checkpoint 中已完成的行。不要在同一 output-dir 中改变 attempts 或筛选条件，否则旧结果会与新任务语义混合。

### 4.7 确认正常结束和不遗漏

```bash
ps -ef | grep -E 'rerun_stable_failures|python3 -m pytest' | grep -v grep
cat /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/summary.json
wc -l /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/rerun_all_results.csv
wc -l /home/tmp/torch2.13/pytest_full_nmz/stable_rerun_3x/stable_failures.csv
```

正常结束应满足：没有相关进程；`summary.json`、`rerun_all_results.csv`、`stable_failures.csv` 均存在；`summary.json` 的 `missing_targets` 为 0；`total_rerun` 等于 4.2 的 `Unique rerun rows`。CSV 的 `wc -l` 包含一行表头，所以 `rerun_all_results.csv` 行数应为 `total_rerun + 1`。`not_stable_or_passed` 包括通过和未达到连续 N 次失败的波动 case，不代表遗漏。

## 5. 官方 Distributed-Tests：轻量 run-test-resume 入口

### 5.1 为什么不用普通 pytest 全量脚本跑 distributed

官方 distributed tests 依赖 `run_test.py` 的特殊逻辑：

- backend/world size 环境
- `--include` 模块选择
- distributed custom handlers
- stepcurrent / case 级重试
- `--continue-through-error`

所以 distributed tests 必须通过官方入口跑。本节是单 worker、逐模块日志的轻量入口；新建无人值守任务如果还需要自动模块补跑、`module_status.csv` 和 `coverage_report.json`，优先使用第 7 节完整官方队列。

```text
/workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py run-test-resume
```

### 5.2 dry-run 生成模块清单

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/run_test_distributed_resume_nmz

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py run-test-resume \
  /workspace/pytorch \
  --work-dir /home/tmp/torch2.13/run_test_distributed_resume_nmz \
  --dry-run-only \
  --quiet-dry-run \
  -- \
  --distributed-tests
```

输出：

```text
/home/tmp/torch2.13/run_test_distributed_resume_nmz/run_test_modules.txt
/home/tmp/torch2.13/run_test_distributed_resume_nmz/run_test_dry_run.log
```

验证清单非空并检查头尾：

```bash
wc -l /home/tmp/torch2.13/run_test_distributed_resume_nmz/run_test_modules.txt
sed -n '1,10p' /home/tmp/torch2.13/run_test_distributed_resume_nmz/run_test_modules.txt
tail -10 /home/tmp/torch2.13/run_test_distributed_resume_nmz/run_test_modules.txt
```

脚本从 official dry-run 的 `Serial tests` / `Parallel tests` 块解析模块并忽略 excluded 块；解析为空时正式模式会退出，不会静默跑零个模块。

### 5.3 正式后台运行 distributed-tests

推荐命令：

```bash
source /home/tmp/python_and_sh/env.sh

mkdir -p /home/tmp/torch2.13/run_test_distributed_resume_nmz

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py run-test-resume \
  /workspace/pytorch \
  --work-dir /home/tmp/torch2.13/run_test_distributed_resume_nmz \
  --timeout 0 \
  --quiet-dry-run \
  --fresh \
  -- \
  --distributed-tests \
  --continue-through-error \
  --verbose \
  > /home/tmp/torch2.13/run_test_distributed_resume_nmz/runner.out 2>&1 &
```

关键点：

- `--timeout 0`：不设置外层 wrapper timeout，尽量让官方 `run_test.py` 自己完成 case 级 timeout、stepcurrent、重试和继续
- 如果担心模块永久卡死，可以设置很大的外层 timeout，例如 `--timeout 21600`
- `--fresh`：从头开始；续跑时不要加
- `--quiet-dry-run`：dry-run 输出写入文件，不刷满 `runner.out`
- `--distributed-tests --continue-through-error --verbose`：传给官方 `run_test.py`

### 5.4 run-test-resume 具体怎么跑模块

每次运行都会先 dry-run，并重写：

```text
run_test_modules.txt
```

随后逐模块执行：

```text
python3 run_test.py --include <module> --distributed-tests --continue-through-error --verbose
```

每个模块一个日志：

```text
<work-dir>/<timestamp>/0001_xxx.log
<work-dir>/<timestamp>/0002_xxx.log
...
```

每个模块结束后写：

```text
<work-dir>/.run_test_progress.json
```

这里的 checkpoint 粒度是官方模块名，不是 case。模块内部由官方 `run_test.py` 负责 pytest 参数、distributed custom handler、失败 case 重试和 stepcurrent 继续；外层脚本负责模块之间的 checkpoint。`run_test_modules.txt` 每次由当前源码的官方 dry-run 重新生成，因此源码中的模块增删会进入新的计划清单。

恢复期间不要切换源码版本、环境或 `--` 后的测试选择参数。计划发生变化时应使用新 work-dir，或确认后用 `--fresh` 从头执行。

### 5.5 distributed 输出目录

```text
/home/tmp/torch2.13/run_test_distributed_resume_nmz/
  runner.out
  run_test_dry_run.log
  run_test_modules.txt
  .run_test_progress.json
  summary.json
  latest -> <timestamp>
  <timestamp>/
    0001_xxx.log
    ...
    run_test_failures.csv
    failure_report.csv
    failure_report.json
    failure_report.md
    unresolved_process_failures.csv
    unresolved_process_failures.json
    unresolved_process_failures.md
    unresolved_process_failure_files.txt
```

文件含义：

- `run_test_modules.txt`：官方 dry-run 模块清单
- `.run_test_progress.json`：模块级 checkpoint
- `run_test_failures.csv`：失败/超时模块列表
- `failure_report.csv`：从官方 run_test.py 日志中抽取的 case 级失败报告
- `unresolved_process_failures.*`：仍无法映射到具体 case 的模块级 timeout/crash/process 行
- `summary.json`：总体结果、当前 `latest` 日志目录和报告路径

本入口不会生成第 6/7 节的 `module_status.csv`、`coverage_report.json`、`incomplete_modules.txt`，也不会自动执行第二轮完整模块补跑。`run_test_failures.csv` 是模块失败清单，`failure_report.csv` 是 case 解析结果，两者用途不同；模块失败但 case 报告没有对应行时，必须查看 unresolved 和原始模块日志。

### 5.6 distributed 失败 case 抽取逻辑

当前解析器会对 official `run_test.py` 日志做三类抽取：

1. 优先抽取：

```text
FAILED CONSISTENTLY: test/distributed/xxx.py::Class::case
The following tests failed consistently: [...]
```

2. 如果没有 `FAILED CONSISTENTLY`，解析普通 pytest：

```text
short test summary info
FAILED ... test/distributed/xxx.py::Class::case
```

3. 如果外层 wrapper timeout，尝试从最后的当前 item 提取：

```text
Running 1 items in this shard: test/distributed/xxx.py::Class::case
```

只要能找到 case 级 `nodeid`，最终 `failure_report.csv` 不再保留：

```text
common_distributed.py <crash>
```

这种旧的 process-level 兜底行。

### 5.7 查看状态和确认结束

查看主日志：

```bash
tail -f /home/tmp/torch2.13/run_test_distributed_resume_nmz/runner.out
```

查看某个模块日志：

```bash
tail -f /home/tmp/torch2.13/run_test_distributed_resume_nmz/latest/0214_distributed_test_c10d_nccl.log
```

查看进程：

```bash
ps -ef | grep -E 'run_pytorch_subset|run_test.py|python3 -m pytest' | grep -v grep
```

查看 checkpoint：

```bash
python3 - <<'PY'
import json
p="/home/tmp/torch2.13/run_test_distributed_resume_nmz/.run_test_progress.json"
d=json.load(open(p))
print(d.get("stats", {}))
PY
```

将模块清单与 checkpoint 对账：

```bash
python3 - <<'PY'
import json
from pathlib import Path

work = Path("/home/tmp/torch2.13/run_test_distributed_resume_nmz")
planned = [x.strip() for x in (work / "run_test_modules.txt").read_text().splitlines() if x.strip()]
data = json.loads((work / ".run_test_progress.json").read_text())
done = data.get("tests", data)
missing = [x for x in planned if x not in done]
print("planned modules:", len(planned))
print("checkpoint modules:", sum(x in done for x in planned))
print("missing modules:", len(missing))
for x in missing[:20]:
    print("MISSING", x)
PY
```

确认结束：

```bash
cat /home/tmp/torch2.13/run_test_distributed_resume_nmz/summary.json
ls -lh /home/tmp/torch2.13/run_test_distributed_resume_nmz/latest/failure_report.*
```

正常结束应满足：没有相关进程，`summary.json` 的 `remaining` 为 `0`，`total = passed + failed + skipped`，并且上面对账输出 `missing modules: 0`。还要检查 `summary.json.failure_reports.unresolved_process_failure_count == 0` 和 unresolved CSV 没有数据行。`failed > 0` 表示模块执行后存在失败，不等于漏跑。若 `failure_report.csv` 仍有 `<timeout>/<crash>` process-level 行，说明对应模块仍没有完整 case 级结论；本节的 `--timeout 0` 只避免外层主动截断，底层进程仍可能真实崩溃，此时必须查看模块日志并单独续跑或改用第 7 节完整队列。

### 5.8 distributed 续跑

不要加 `--fresh`：

```bash
source /home/tmp/python_and_sh/env.sh

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py run-test-resume \
  /workspace/pytorch \
  --work-dir /home/tmp/torch2.13/run_test_distributed_resume_nmz \
  --timeout 0 \
  --quiet-dry-run \
  -- \
  --distributed-tests \
  --continue-through-error \
  --verbose \
  > /home/tmp/torch2.13/run_test_distributed_resume_nmz/runner_resume.out 2>&1 &
```

续跑规则：

- PASS：跳过
- SKIP：跳过
- FAIL/TIMEOUT：默认重跑
- 加 `--skip-fail`：跳过旧 FAIL/TIMEOUT

### 5.9 run-test 与 run-test-resume 的区别

`run-test` 只执行一次完整官方命令，输出 `latest/run_test.log` 和 `summary.json`；它没有模块 checkpoint，也不生成当前的 case 级 failure report。它适合探索和单模块复现：

```bash
source /home/tmp/python_and_sh/env.sh

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_subset.py run-test \
  /workspace/pytorch \
  --work-dir /home/tmp/torch2.13/run_test_one_module \
  -- \
  --include distributed/test_store \
  --verbose
```

输出目录包含 `summary.json`、`latest -> <timestamp>` 和 `latest/run_test.log`。查看状态：

```bash
tail -f /home/tmp/torch2.13/run_test_one_module/latest/run_test.log
ps -ef | grep -E 'run_pytorch_subset.py run-test|run_test.py' | grep -v grep
```

进程结束后，`summary.json` 的 `returncode: 0` 表示官方命令成功，非 0 表示失败。该模式没有 checkpoint，中断后只能重新执行整个命令，不能从中断模块继续。长时间 distributed 全量应使用 `run-test-resume`，因为它逐模块落 checkpoint、支持中断恢复，并生成模块失败列表和 case 级报告。

## 6. 官方 run_test.py 队列模式：Normal 完整入口

### 6.1 功能、入口和与直接 pytest 的区别

入口：

```text
/workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh
/workspace/pytorch-pytest-ops/runners/run_official_run_test_queue.py
```

队列以官方 dry-run 模块名为单位，逐模块执行：

```text
python3 run_test.py --include <module> --exclude-jit-executor --exclude-distributed-tests --verbose
```

与第 2 节直接 pytest 相比，它保留官方 custom handler、模块参数、`-x`、pytest reruns、stepcurrent `--rs/--scs` 和 `--continue-through-error`。因此 `doctests`、autoload、cpp extension AOT 等没有同名 `.py` 文件的目标也能运行。normal 模式仍明确排除 distributed 和 JIT executor；完整类别覆盖还要执行第 7 节 distributed。

新增队列能力包括：正则子集、从失败 CSV 选择模块、文件级 timeout/crash 完整模块自动补跑、checkpoint、最终 case CSV、unresolved 独立报告、计划清单覆盖对账，以及 official 模式稳定失败重测。

旧的 `check_optest_results_v2.py` 通过扫描日志把模块分成 `ok/error/check/interrupted`，仍可能受日志格式和截断位置影响。当前队列用 `run_test_tests.txt` 与 `.run_test_progress.json` 逐项对账，并生成 `module_status.csv`、`coverage_report.json` 和 `incomplete_modules.txt`，已把这一步自动化。

### 6.2 dry-run 只生成清单

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh dry-run-normal

wc -l /home/tmp/torch2.13/run_test_official_nmz/run_test_tests.txt
sed -n '1,10p' /home/tmp/torch2.13/run_test_official_nmz/run_test_tests.txt
tail -10 /home/tmp/torch2.13/run_test_official_nmz/run_test_tests.txt
```

输出 `run_test_dry_run.out` 和 `run_test_tests.txt`。清单保留官方原始模块名，不追加 `.py`。空清单不能进入正式运行。

### 6.3 正式全量后台运行

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
PYTORCH_NUM_PYTEST_RERUNS=2 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-normal
```

shell 内部已使用 nohup 和 `--fresh`。每个 worker 绑定一张 GPU；每个模块由官方入口运行。当前默认使用两种相互独立的外层 watchdog：

- `IDLE_TIMEOUT=7200`：连续 2 小时没有收到子进程任何 stdout/stderr 才终止；正常 verbose case 输出会持续刷新该计时器。
- `TIMEOUT=259200`：无论是否持续输出，单次模块最多运行 72 小时，防止不断刷日志但永不结束的异常。
- `PROCESS_RERUN_IDLE_TIMEOUT=7200`、`PROCESS_RERUN_TIMEOUT=259200`：权威完整模块补跑采用相同的 2 小时 idle + 72 小时 hard 边界。
- 任一值设为 `0` 表示关闭对应边界；例如单个合法 case 本来就可能静默超过 2 小时时，应提高 `IDLE_TIMEOUT`，不能把输出沉默直接等同于 hang。

只要 checkpoint 终态为 `TIMEOUT`，无论日志是否碰巧抽到一个当前 case，该模块都会进入 `process_module_rerun/`。再次 timeout 会明确保留为 incomplete，而不会静默当作完整。

本地运行未传 `run_test.py --enable-timeout` 时，官方普通 Python 测试通常没有统一墙钟 timeout；该选项启用后也依赖 `.additional_ci_files/test-times.json` 等 CI timing 数据，本地源码树不一定存在与当前 ROCm 节点匹配的数据。官方通过 `--rs/--scs` 处理它自己观察到的 case crash，但无法保证一个永久无输出的外层进程自行结束，所以队列仍保留上述 watchdog。

`functorch/test_ops`、`test_decomp`、`inductor/test_torchinductor_opinfo*` 可能包含数千到两万多个 case，总时长超过 12 小时并不异常。新的 idle watchdog 允许它们只要持续输出就继续运行，避免旧版 12 小时绝对上限的误杀；72 小时 hard limit 仍是最后保险。

`.run_test_progress.json` 的 `elapsed` 是本队列在当前节点记录的实测时长，不是官方 CI test-time。PASS/FAIL 时它是完整样本；TIMEOUT 时只是运行时长下界。新版会保存 `attempts/history`，并在续跑或补跑时让历史耗时长的模块优先入队，改善多 GPU 尾部利用率，但不会拿单个 timeout elapsed 猜测剩余时间。

注意外层杀死整个 `run_test.py` 后，下一次官方调用会为各 shard 创建带随机后缀的新 stepcurrent key，当前队列不能可靠复用旧调用的全部 shard 位置，通常会从该模块开头重新执行。发现旧版截断后，使用 6.6 的 `rerun-incomplete-normal` 可以把模块重新跑完，但不是从日志百分比处原地续接。

### 6.4 输出目录和失败报告

```text
/home/tmp/torch2.13/run_test_official_nmz/
  runner.out
  run_test_dry_run.out
  run_test_tests.txt
  .run_test_progress.json
  summary.json
  module_status.csv
  coverage_report.json
  incomplete_modules.txt
  latest -> <timestamp>
  <timestamp>/
    run_test_gpu_*.log
    process_module_rerun/
      run_test_tests.txt
      run_test_gpu_*.log
      summary.json
    process_module_rerun_<timestamp>/
      ...                       # 历史目录再次 process-rerun-only 时
    failure_report.csv
    failure_report.json
    failure_report.md
    unresolved_process_failures.csv
    unresolved_process_failures.json
    unresolved_process_failures.md
    unresolved_process_failure_files.txt
```

各文件的准确含义：

- `run_test_dry_run.out`：官方 dry-run 原始输出；只有 runner 自己执行 dry-run 时生成。
- `run_test_tests.txt`：去重、筛选后的官方模块计划，保留模块名而不是 `.py` 文件名。
- `.run_test_progress.json`：模块 checkpoint。`tests.<module>` 保存最新 `status/elapsed/returncode/time/timeout_kind`，并在 `history` 中保留各次 attempt；`PASS/FAIL/TIMEOUT` 都会占一条 completed record。
- `latest`：指向首次/续跑创建的主 timestamp 目录。`process-rerun-only` 在该目录下面追加补跑子目录，不会把 `latest` 改指向子目录。
- `run_test_gpu_<id>.log`：normal 模式每个 GPU worker 的合并日志；同一文件依次包含该 worker 执行的多个模块，以 `START/PASS/FAIL/TIMEOUT` 标记分段，不是每模块一个文件。
- `process_module_rerun*/run_test_tests.txt`：该轮权威完整模块补跑选择清单。
- `process_module_rerun*/summary.json`：该轮补跑的模块、timeout 和各 worker PASS/FAIL/TIMEOUT 数。目录存在但没有 summary 表示补跑中断。
- `latest/failure_report.csv/json/md`：只保存能映射到具体 `file.py::Class::case` nodeid 的失败 case。补跑模块的旧部分日志行会被替换。
- `latest/unresolved_process_failures.*`：官方返回非零但无法映射到可靠 case 的模块级行，例如 custom-handler 编译失败或导入/收集阶段异常。
- `summary.json`：当前主计划、最终报告、补跑清单/结果、`progress_stats`、本次 `timeouts` 配置和 coverage 的总索引。
- `module_status.csv`：计划中的每个模块一行，列中除 `module,status,elapsed,returncode,time` 外还包含 `attempts,timeout_kind,hard_timeout,idle_timeout`。`PASS` 表示官方进程返回 0；`FAIL` 表示官方进程已经返回非零；`TIMEOUT` 表示被队列外层 watchdog 截断；`MISSING` 表示计划中存在但没有 checkpoint。
- `incomplete_modules.txt`：仍为 `TIMEOUT` 或缺少 checkpoint 的模块名；完整时为空。
- `coverage_report.json`：最终覆盖闭合结论，是官方队列验收的权威文件。

`coverage_report.json` 字段：

- `planned`：`run_test_tests.txt` 模块数。
- `terminal`：真正完成官方调用的 `PASS + FAIL` 数，不包含 `TIMEOUT`。
- `pass/fail/timeout/missing`：最终模块状态计数。
- `unresolved_process_failures`：仍只有模块级结论、没有可靠 case 的报告行数。
- `timeout_modules/missing_modules`：对应模块清单。
- `timeout_details`：每个 timeout 模块的 elapsed、idle/hard 类型、边界和尝试次数；即使主失败 CSV 只有真实 case，也能从这里确认模块级截断。
- `coverage_complete`：只有 `timeout == 0`、`missing == 0`、`unresolved_process_failures == 0` 才为 `true`。

因此 `summary.json.progress_stats.completed_records == planned` 仍不代表完整：`TIMEOUT` 也写入 checkpoint。普通 `FAIL` 是官方进程已返回的终态，但如果没有任何可解析 case，它仍会通过 unresolved 让 `coverage_complete` 保持 `false`。

直接查看和解释这三个文件：

```bash
sed -n '1,30p' /home/tmp/torch2.13/run_test_official_nmz/module_status.csv
cat /home/tmp/torch2.13/run_test_official_nmz/coverage_report.json
cat /home/tmp/torch2.13/run_test_official_nmz/incomplete_modules.txt
```

例如 `planned=640, terminal=636, timeout=4, missing=0` 表示 640 个计划模块都有 checkpoint，但其中只有 636 个真正从官方进程返回，另 4 个被外层 timeout 截断；此时即使 `completed_records=640`，覆盖仍不完整。`unresolved_process_failures=9` 表示另有 9 条非零模块结果没有可靠 case nodeid，它们会出现在 `unresolved_process_failures.csv`。主 `failure_report.csv` 保持为真实失败 case 表；模块覆盖缺口看另外三类诊断文件。只有 `incomplete_modules.txt` 为空、unresolved 为 0 且 `coverage_complete=true`，才得到完整的模块覆盖和尽量完整的 case 失败表。

### 6.5 timeout/crash 处理

官方 `run_test.py` 对普通 pytest 模块使用 `-x + --sc + --rs/--scs`：当前 case 新进程通过后继续后续 case；连续失败三次输出 `FAILED CONSISTENTLY`，并在 keep-going 模式下跳过该 case继续。队列外层使用非阻塞日志读取，timeout 时终止整个进程组并写 synthetic Timeout 行。

若官方进程在导入/收集、首个 item 前崩溃，官方也可能没有 stepcurrent case。队列会把“checkpoint 为 FAIL 且该模块没有任何可解析失败行”视为疑似进程级中断并完整重跑；最终仍无 case 时保留 `<process-failure>` unresolved 文件级记录，不会静默删除。常见来源包括 custom-handler 的扩展构建失败、模块导入 `ImportError/NameError`，以及特殊 handler 自身返回非零。这些情况客观上没有已执行的 pytest case，不能伪造 `Class::case`。

当前合成的 `<process-failure>` 行只保存模块名、returncode 和 elapsed，`source_log` 为空；原始内容实际位于 `process_module_rerun*/run_test_gpu_*.log` 的对应 `START` 到 `FAIL` 区段，可用模块名搜索。它表示报告可导航性仍有限，不表示日志没有写入。

若完整补跑仍被 idle 或 hard timeout 截断，checkpoint 保持 `TIMEOUT`，`coverage_complete` 为 `false`，模块一定进入 `timeout_modules`、`timeout_details` 和 `incomplete_modules.txt`。主失败 CSV 只保留截断前已经明确解析出的真实失败 case，不再追加模块级 `<timeout>` 占位行；因此必须以 coverage 和 incomplete 清单验收模块完整性。

### 6.6 查看状态、中断续跑和结束验收

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh status-normal

tail -f /home/tmp/torch2.13/run_test_official_nmz/latest/run_test_gpu_0.log

# 只补跑不完整模块时，主输出和原始日志分别看：
tail -f /home/tmp/torch2.13/run_test_official_nmz/runner_incomplete_rerun.out
tail -f /home/tmp/torch2.13/run_test_official_nmz/latest/process_module_rerun*/run_test_gpu_*.log
```

续跑参数必须与首次一致，只把命令改为：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PYTORCH_NUM_PYTEST_RERUNS=2 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh resume-normal
```

PASS 跳过，FAIL/TIMEOUT 默认重跑。确认模块不遗漏：

```bash
python3 - <<'PY'
import json
from pathlib import Path
work=Path('/home/tmp/torch2.13/run_test_official_nmz')
planned=[x for x in (work/'run_test_tests.txt').read_text().splitlines() if x]
done=json.loads((work/'.run_test_progress.json').read_text()).get('tests', {})
missing=[x for x in planned if x not in done]
print('planned modules:', len(planned))
print('checkpoint modules:', sum(x in done for x in planned))
print('missing modules:', len(missing))
print('statuses:', {s:sum(v.get('status')==s for v in done.values()) for s in ('PASS','FAIL','TIMEOUT')})
for x in missing[:20]: print('MISSING',x)
PY
```

正常结束要求：无相关进程；根目录和最后一次 process rerun（若触发）都有 `summary.json`；最终 failure/unresolved 报告存在；`coverage_report.json` 中 `coverage_complete` 为 `true`。runner 有普通失败时也返回非 0，所以后台任务是否完整应以 coverage 和报告为准，而不是只看退出码。任务进程已经结束但 `coverage_complete: false` 的准确含义是“命令已收尾并留下诊断，但测试覆盖未闭合”。

已有历史目录只补跑不完整/未定位模块并自动重建主报告；这也是使用新版默认参数补跑旧目录中被 12 小时上限截断模块的命令：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh rerun-incomplete-normal
```

该命令读取原目录的 `run_test_tests.txt`、checkpoint 和 `latest`。选择范围包括：checkpoint 为 `TIMEOUT` 的模块；最终 unresolved 报告里的文件级 Timeout/Crash；以及 checkpoint 为 `FAIL` 但完全没有可解析失败行的模块。已经有可靠 case 报告的普通 FAIL 不会因该模式再次补跑。它会从模块开头重跑，持续输出时可运行到 72 小时，完成后直接更新原 `latest/failure_report.csv`、根 `summary.json` 和 coverage 文件，无需手工合并。如果确认存在单个合法 case 静默超过 2 小时，可提高 `PROCESS_RERUN_IDLE_TIMEOUT`；只有在人工监控下才建议设为 `0`。不要与另一个修改同一 work-dir 的 runner 同时运行。

### 6.7 Normal 子集测试

只跑 inductor/dynamo 官方模块：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_ind_dyn_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
INCLUDE_REGEX='^(inductor|dynamo)/' \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh dry-run-normal

ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
NORMAL_WORK_DIR=/home/tmp/torch2.13/run_test_official_ind_dyn_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
INCLUDE_REGEX='^(inductor|dynamo)/' \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-normal
```

输出、状态、续跑和验收与 6.4/6.6 相同，续跑必须继续传相同 `INCLUDE_REGEX`。也可设置 `EXCLUDE_REGEX`。

### 6.8 从失败 CSV 补跑官方模块

该模式把 CSV 中 case 映射回官方模块，并重跑对应完整模块，使官方 `--rs/--scs` 有机会覆盖该 case 后续内容：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
FAILURE_CSV=/home/tmp/torch2.13/run_test_official_nmz/latest/failure_report.csv \
FAILURE_WORK_DIR=/home/tmp/torch2.13/run_test_official_failure_rerun_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-normal-failures
```

续跑把最后命令改为 `resume-normal-failures`，其余变量保持一致。输出结构和最终 CSV 与 normal 全量一致。

查看状态使用 `python3 /workspace/pytorch-pytest-ops/scripts/inspect_test_run.py /home/tmp/torch2.13/run_test_official_failure_rerun_nmz`；正常结束同样要求该新目录的 `coverage_report.json.coverage_complete=true`，不能只根据失败 CSV 已生成判断完成。

### 6.9 Official 稳定失败重测

对 case-level `failure_report.csv` 使用官方入口连续确认三次：

```bash
source /home/tmp/python_and_sh/env.sh
mkdir -p /home/tmp/torch2.13/run_test_official_nmz/stable_rerun_official_3x

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/run_test_official_nmz/latest/failure_report.csv \
  --runner official \
  --attempts 3 \
  --timeout 1800 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir /home/tmp/torch2.13/run_test_official_nmz/stable_rerun_official_3x \
  --fresh \
  > /home/tmp/torch2.13/run_test_official_nmz/stable_rerun_official_3x/runner.out 2>&1 &
```

每次实际执行 `run_test.py --include <module> --pytest-single-test <file.py::Class::case>`。输出、checkpoint、筛选、续跑和稳定失败判定与第 4 节一致；续跑去掉 `--fresh`。process-level 行默认跳过，应先用 6.8 补成 case。

### 6.10 查询某个 case

```bash
python3 - <<'PY'
import csv
p='/home/tmp/torch2.13/run_test_official_nmz/latest/failure_report.csv'
needle='test_aot_compile'
for r in csv.DictReader(open(p,newline='',encoding='utf-8')):
    if needle in ' '.join(r.values()):
        print(r['nodeid'],r['error_type'],r['error_message'],r['source_log'])
PY
```

失败 case 查最终 CSV；通过/跳过 case 查 `latest/run_test_gpu_*.log`。CSV 只记录失败，不是全 case 数据库。

## 7. 官方 run_test.py 队列模式：Distributed 完整入口

### 7.1 与 Normal 的关键差异

distributed 参数为：

```text
--distributed-tests --continue-through-error --verbose
```

新版 shell 包装器自动传 `--no-bind-gpu`：只启动一个外层 worker，不覆盖 `HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES`，每个官方模块继承 `env.sh` 的全部可见 GPU。这样避免旧实现“多个单卡 worker”造成的端口竞争和多 GPU case漏测。模块内部资源、backend、world size和进程启动仍由官方 custom handler 控制。

`GPU_IDS` 在该模式下不用于逐模块绑定；真正的可见设备由启动容器和 `HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES` 决定。为了得到稳定结果，同一节点的其他容器不应同时占用相同 GPU 跑 distributed 测试，否则可能引入 OOM、RCCL/端口竞争、VMFault、性能下降和额外 timeout。若必须并行，应在容器层划分互不重叠的可见 GPU，并接受 GPU 数量变化可能改变覆盖范围。

### 7.2 dry-run 和正式全量命令

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh dry-run-distributed

wc -l /home/tmp/torch2.13/run_test_official_distributed_nmz/run_test_tests.txt
```

正式后台运行：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
PYTORCH_NUM_PYTEST_RERUNS=2 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-distributed
```

distributed 正式无人值守运行同样采用 2 小时无输出 + 72 小时 hard watchdog。若补跑仍超时，`coverage_complete` 会变为 `false`，具体模块一定列入 `timeout_modules`、`timeout_details` 和 `incomplete_modules.txt`；主 CSV 继续只保存已定位的真实失败 case。distributed 官方 pytest 本身按源码设置 `--reruns=0`，因为该类测试不支持 pytest-rerunfailures；case 继续主要依赖官方 stepcurrent/handler。

### 7.3 输出、状态、续跑和结束验收

输出结构和字段语义与 6.4 相同，日志名为 `run_test_gpu_all.log`，根目录换为 distributed work-dir；补跑日志位于 `latest/process_module_rerun*/run_test_gpu_all.log`。

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_nmz \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh status-distributed

tail -f /home/tmp/torch2.13/run_test_official_distributed_nmz/latest/run_test_gpu_all.log
```

续跑：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_nmz \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh resume-distributed
```

把 6.6 对账脚本的 work-dir 改成 distributed 目录。正常结束必须 summary/report 完整，并且 `coverage_report.json` 同时满足 `missing: 0`、`timeout: 0`、`unresolved_process_failures: 0`、`coverage_complete: true`。只看到 `.run_test_progress.json` 已有全部模块记录仍不够，因为其中可能包含 `TIMEOUT`。

### 7.4 Distributed 子集

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_c10d_nmz \
INCLUDE_REGEX='^distributed/(test_c10d|test_store)' \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh dry-run-distributed

ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_c10d_nmz \
INCLUDE_REGEX='^distributed/(test_c10d|test_store)' \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-distributed
```

清单和输出都写入 `run_test_official_distributed_c10d_nmz/`，结构与 7.3 相同。查看状态使用 `status-distributed` 或只读检查器；中断续跑把动作改为 `resume-distributed`，并保持相同的 `DIST_WORK_DIR/INCLUDE_REGEX/TIMEOUT/IDLE_TIMEOUT/PROCESS_RERUN_TIMEOUT/PROCESS_RERUN_IDLE_TIMEOUT`。验收仍要求该子集目录 `coverage_complete=true`。

### 7.5 从失败 CSV 补跑 distributed 模块

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
FAILURE_CSV=/home/tmp/torch2.13/run_test_official_distributed_nmz/latest/failure_report.csv \
FAILURE_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_failure_rerun_nmz \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh run-distributed-failures
```

续跑使用 `resume-distributed-failures`。该模式仍是单 worker、继承全部 GPU。

该命令写入 `FAILURE_WORK_DIR`，输出结构与 7.3 相同；用只读检查器查看状态。正常结束要求新目录的 `coverage_complete=true`，且最终 `latest/failure_report.csv` 与 unresolved 报告存在。它重跑 CSV 映射出的完整 distributed 模块，不会把结果自动发布回原全量目录。

已有 distributed 历史目录只补跑不完整模块并重建原报告：

```bash
ENV_SH=/home/tmp/python_and_sh/env.sh \
PYTORCH_ROOT=/workspace/pytorch \
DIST_WORK_DIR=/home/tmp/torch2.13/run_test_official_distributed_nmz \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
bash /workspace/pytorch-pytest-ops/runners/run_test-2.13-official-queue.sh rerun-incomplete-distributed
```

旧目录中被绝对上限截断的模块使用这里的 2 小时 idle + 72 小时 hard 配置重新完整执行。若合法 distributed case 会静默超过 2 小时，可提高 `PROCESS_RERUN_IDLE_TIMEOUT`；设为 `0` 后没有 idle 保护，应独占资源并人工防范永久 hang。

### 7.6 Distributed 稳定失败重测和 case 查询

```bash
source /home/tmp/python_and_sh/env.sh
mkdir -p /home/tmp/torch2.13/run_test_official_distributed_nmz/stable_rerun_official_3x

nohup env PYTHONUNBUFFERED=1 python3 -u /workspace/pytorch-pytest-ops/runners/rerun_stable_failures.py \
  /workspace/pytorch \
  /home/tmp/torch2.13/run_test_official_distributed_nmz/latest/failure_report.csv \
  --runner official \
  --official-run-test-arg=--distributed-tests \
  --attempts 3 \
  --timeout 3600 \
  --output-dir /home/tmp/torch2.13/run_test_official_distributed_nmz/stable_rerun_official_3x \
  --fresh \
  > /home/tmp/torch2.13/run_test_official_distributed_nmz/stable_rerun_official_3x/runner.out 2>&1 &
```

这里不传 `--gpu-ids`，稳定重测进程继承全部可见 GPU并串行执行，避免多个 distributed case抢占端口/GPU。查询失败 case沿用 6.10，只替换 CSV 路径；查询原始执行过程查看 `run_test_gpu_all.log`。

## 8. 常用排查命令

优先使用仓库自带的只读检查器，它会同时读取计划、checkpoint、summary、最终报告、自动补跑和官方 coverage：

```bash
python3 /workspace/pytorch-pytest-ops/scripts/inspect_test_run.py \
  /home/tmp/torch2.13/run_test_official_nmz \
  --pytorch-root /workspace/pytorch
```

检查器首先输出：

- `Artifact verdict: COMPLETE`：清单、真实文件 checkpoint、summary、补跑 summary、最终报告、unresolved 和 official coverage 均闭合。
- `Artifact verdict: FINALIZED_INCOMPLETE`：已经生成根 summary/report，但仍存在 unresolved、TIMEOUT、真实缺项或补跑缺 summary。
- `Artifact verdict: NOT_FINALIZED`：缺少根 summary 或最终失败报告，通常是仍在运行、中断，或日志尚未同步完成。
- `Virtual targets` 与 `Missing real items`：普通 direct-pytest 的 5 个官方 custom-handler 目标不会再被误算成真实文件漏跑。
- `Completion issues`：直接列出阻止完整验收的每个条件。
- `Legacy/version hints`：提示旧 runner 缺少恢复参数、旧 summary 元数据，或报告在 summary 之后被重新分析。

检查器中的 `Local work-dir procs` 只代表当前机器。共享目录由其他节点执行时，远端进程仍需在实际节点检查；`Artifact verdict` 描述磁盘产物是否闭合，不声称远端进程一定不存在。

### 8.1 看 CSV 中是否还有 process-level 兜底行

```bash
python3 - <<'PY'
import csv
p="/home/tmp/torch2.13/run_test_distributed_resume_nmz/latest/failure_report.csv"
rows=list(csv.DictReader(open(p, newline="", encoding="utf-8")))
case_rows=[r for r in rows if "::" in r.get("nodeid","") and not r.get("case_name","").startswith("<")]
process_rows=[r for r in rows if r not in case_rows]
print("total:", len(rows))
print("case-level:", len(case_rows))
print("process-level:", len(process_rows))
for r in process_rows:
    print(r.get("source_log"), r.get("test_file"), r.get("case_name"), r.get("error_type"), r.get("error_message"))
PY
```

理想情况下，新的 distributed 报告里 `process-level` 应尽量为 `0`。

官方队列还必须单独检查模块覆盖；CSV 没有 process-level 行不能替代 coverage 验收：

```bash
cat /home/tmp/torch2.13/run_test_official_nmz/coverage_report.json
cat /home/tmp/torch2.13/run_test_official_nmz/incomplete_modules.txt
```

`coverage_complete: false` 时，即使 `failure_report.csv` 搜不到 `Timeout`，也要以 `timeout_modules`、`missing_modules` 和 `unresolved_process_failures` 为准。

### 8.2 重新分析已有日志生成 failure_report

通用分析器可以重新解析普通 pytest 或 official run_test 日志：

```bash
python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --analyze-only /home/tmp/torch2.13/run_test_distributed_resume_nmz/latest
```

对于由旧版进程启动、结束时尚未生成 `unresolved_process_failures.*` 的普通全量目录，应等 `process_file_rerun/summary.json` 已经存在后再执行：

```bash
source /home/tmp/python_and_sh/env.sh

python3 /workspace/pytorch-pytest-ops/runners/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --analyze-only /home/tmp/torch2.13/log-final/pytest_full_bw/latest
```

分析器只有看到补跑 `summary.json` 这个完成标志，才会用 `process_file_rerun/` 的结果替换初次报告里的旧 process-level 行；补跑仍在进行或已经中断时不会提前隐藏尚未补跑的文件。

对第 6/7 节官方队列，`--analyze-only` 只重建日志解析得到的 failure report，不会重建 `module_status.csv`/`coverage_report.json`，也不会补入 checkpoint 中“返回非零但日志没有 case”的 synthetic `ProcessFailure`。官方队列历史目录的完整修复入口应使用 `rerun-incomplete-normal` 或 `rerun-incomplete-distributed`；最后仍以根目录 coverage 文件验收。

### 8.3 停止跑错的后台任务

先检查 runner、pytest 及其父进程：

```bash
ps -eo pid,ppid,pgid,stat,lstart,cmd | grep -E 'run_pytorch|run_test.py|python3 -m pytest' | grep -v grep
fuser -v /dev/kfd 2>/dev/null
```

如果 pytest 的 PPID 已是 1、长时间休眠且原 runner 不存在，它是孤儿残留。应按 PGID 精确清理整个进程组，避免遗留 `addr2line`、编译器或测试子进程；把下面数字换成检查到的 PGID：

```bash
PGID_TO_KILL=12345
kill -TERM -- -"$PGID_TO_KILL"
sleep 5
kill -KILL -- -"$PGID_TO_KILL" 2>/dev/null || true
```

不要直接照抄其他机器的 PID/PGID，也不要在未确认归属时使用全局 `pkill python3`。

精确停止某个 output-dir 的稳定重测：

```bash
pkill -f 'rerun_stable_failures.py .*stable_rerun_3x_march_x86_64_cpp_compile_after_fix'
```

查看残留 pytest：

```bash
ps -ef | grep 'python3 -m pytest' | grep -v grep
```

确认属于本次任务后再杀：

```bash
pkill -f 'python3 -m pytest'
```
