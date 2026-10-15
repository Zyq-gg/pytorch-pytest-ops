# 命令模板

使用模板前，必须先检查当前脚本的 `--help`。先确定以下变量：

```bash
OPS_ROOT=/path/to/cloned/pytorch-pytest-ops
PYTORCH_ROOT=/path/to/pytorch
ENV_SH=/path/to/environment.sh  # 环境已经激活时可以留空
```

续跑必须保持 work-dir、测试选择参数、GPU、timeout 和自定义环境变量不变。新任务使用 `--fresh`；续跑同一目录时移除 `--fresh`。

## 直接 pytest 可选文件全量

默认全量基线是后文的官方 Normal 与 Distributed 两套完整队列。只有需要比较测试入口或统一使用 pytest 日志时，才使用下面的 direct pytest 文件全量；日常增量验证优先使用其前缀或清单子集模式。

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

WORK=/home/tmp/torch2.13/log-final/pytest_full_nmz_new
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir "$WORK" \
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

续跑时复用同一条命令和 work-dir，移除 `--fresh`，通常把输出改为 `runner_resume.out`，并用新的 `$!` 更新 `runner.pid`。当前 direct runner 默认会重试 checkpoint 中的 FAIL；只有显式传入 `--skip-fail` 才会跳过旧失败。

## 只生成 dry-run 清单

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

mkdir -p "$WORK"
python3 "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only
```

该命令只生成 `$WORK/test_files.txt`，不执行测试 case。direct runner 默认向官方 dry-run 追加 `--exclude-jit-executor --exclude-distributed-tests`。

## 按前缀运行子集

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

WORK=/path/to/prefix-work-dir
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --include-prefix inductor/,dynamo/ \
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

## 从已有清单运行子集

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

WORK=/path/to/subset-work-dir
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/run_pytorch_subset.py" pytest-list \
  "$PYTORCH_ROOT" \
  --test-list /path/to/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --fresh \
  > "$WORK/runner.out" 2>&1 &

echo $! > "$WORK/runner.pid"
```

`pytest-list` 还支持 `--exclude-prefix`、`--include-regex` 和 `--exclude-regex`。它复用 direct runner 的 GPU 队列、stepcurrent 和 crash 恢复实现。

## 补跑 process-level 失败文件并发布

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

BASE=/path/to/original-full-work-dir
WORK=/path/to/supplemental-rerun-work-dir
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/run_pytorch_subset.py" pytest-failure-files \
  "$PYTORCH_ROOT" \
  --failure-csv "$BASE/latest/unresolved_process_failures.csv" \
  --work-dir "$WORK" \
  --publish-to-work-dir "$BASE" \
  --error-type Timeout,Crash \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 14400 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --fresh \
  > "$WORK/runner.out" 2>&1 &

echo $! > "$WORK/runner.pid"
```

发布只会在所选文件都有终态 checkpoint、补跑 summary 存在且 unresolved 数为 0 时执行。发布成功后，原全量目录的 `latest/external_rerun_merge.json` 会记录合并来源，使以后运行 `--analyze-only` 时能够重新应用该结果。

如果补跑已经完成、只需重新发布，请复用相同参数，移除 `--fresh`，增加 `--skip-fail`，并以前台方式执行。

## 稳定失败 case 重测

输入是中文测试总结表时，先转成 `failure_report.csv` 格式：

```bash
python3 "$OPS_ROOT/scripts/convert_test_summary_csv.py" \
  /path/to/test-summary.csv \
  /path/to/converted_failure_report.csv \
  --pytorch-root "$PYTORCH_ROOT"
```

具体 case 会生成精确 nodeid；文件级行会生成 `<module-stage>` 记录并被下面的默认重测跳过。先使用 `--dry-run-list` 核对 `Unique rerun rows` 和 `Missing targets`。

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"

WORK=/path/to/stable-rerun-work-dir
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u "$OPS_ROOT/runners/rerun_stable_failures.py" \
  "$PYTORCH_ROOT" /path/to/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir "$WORK" \
  > "$WORK/runner.out" 2>&1 &

echo $! > "$WORK/runner.pid"
```

默认跳过没有具体 `::case` 的 process-level 行。某个 case 一旦 PASS 就停止继续尝试；只有连续达到 `--attempts` 次且从未 PASS，才写入 `stable_failures.csv`。需要按 CSV 任意列筛选时，先查看当前 `--help` 中的 `--filter-column`、`--filter-equals`、`--filter-contains` 和 `--filter-regex`。

混合 Normal/Distributed 并明确要用官方入口一起重测时，加 `--runner official` 且省略 `--gpu-ids`。这样会用一个 worker 串行运行，每个 case 都继承全部可见 GPU；不要全局添加 `--distributed-tests`。

## 轻量官方 Distributed 续跑

轻量兼容入口使用：

```bash
python3 "$OPS_ROOT/runners/run_pytorch_subset.py" run-test-resume --help
```

完整命令以 `$OPS_ROOT/docs/PYTORCH_PYTEST_WORKFLOW.md` 的 distributed 章节为准。官方 `run_test.py` 参数必须放在 `--` 之后。续跑时保持同一 work-dir 和相同官方参数，并移除 `--fresh`。

该入口是串行模块队列，没有完整官方队列的 idle watchdog、自动完整模块补跑、`module_status.csv` 和 `coverage_report.json`，不应作为新的无人值守全量首选。

## 完整官方 Normal 队列

```bash
ENV_SH="$ENV_SH" \
PYTORCH_ROOT="$PYTORCH_ROOT" \
NORMAL_WORK_DIR="$WORK" \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
bash "$OPS_ROOT/runners/run_test-2.13-official-queue.sh" run-normal
```

包装器会在后台启动 runner，并在终端打印 PID 和日志路径。每次启动都会写带时间戳和 shell pid 的输出及 pid 文件，并维护 `runner_run_latest.out`、`runner_resume_latest.out` 或 `runner_incomplete_rerun_latest.out` 符号链接，避免启动新任务覆盖旧日志。正常续跑使用 `resume-normal`，并保持上述环境变量不变。

验收至少要求：

- 根目录 `summary.json` 存在；
- 被触发的最后一个 `process_module_rerun*/summary.json` 存在；
- `latest/failure_report.csv` 和 unresolved 报告存在；
- unresolved 没有数据行；
- `coverage_report.json.coverage_complete` 为 `true`。

`completed_records == planned` 不能替代 coverage 验收，因为 TIMEOUT 也会写入 checkpoint。

官方队列 resume 会保留跨时间戳的可靠 FAIL 报告。启动输出中的 `Done FAIL: N (reuse X, retry Y)` 表示：已有具体 case 的 FAIL 直接复用；无可靠 case 的 FAIL、TIMEOUT 和 MISSING 进入 `Need run`。本轮执行过的模块会用新报告整体替换旧行，未重跑模块保留 checkpoint 中的历史 `source_log`。

普通 `resume-normal` 在 `Need run: 0` 时会跳过空执行队列，并从 checkpoint 的权威日志重建最终报告、coverage 和根 summary。只想重跑已有 checkpoint 中的不完整模块时，使用：

```bash
ENV_SH="$ENV_SH" \
PYTORCH_ROOT="$PYTORCH_ROOT" \
NORMAL_WORK_DIR="$WORK" \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
bash "$OPS_ROOT/runners/run_test-2.13-official-queue.sh" rerun-incomplete-normal
```

该动作对应 `--process-rerun-only`：它会从模块开头重跑已有 checkpoint 的 TIMEOUT、unresolved process failure，以及没有可靠 case 报告的 FAIL；它不会补跑完全缺少 checkpoint 的 MISSING 模块。没有符合补跑条件的模块时，它只重建最终制品。若目录仍有 MISSING，应使用普通 `resume-normal`。

## 完整官方 Distributed 队列

```bash
ENV_SH="$ENV_SH" \
PYTORCH_ROOT="$PYTORCH_ROOT" \
DIST_WORK_DIR="$WORK" \
GPU_IDS=0,1,2,3,4,5,6,7 \
TIMEOUT=259200 \
IDLE_TIMEOUT=7200 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
bash "$OPS_ROOT/runners/run_test-2.13-official-queue.sh" run-distributed
```

Distributed 动作使用单个 `--no-bind-gpu` worker，不改写 GPU 可见性，因此继承容器内可见的全部设备。需要稳定结果时，不要让其他容器同时使用同一批 GPU。

默认官方配置同时启用两个 watchdog：

- 连续 2 小时没有子进程输出时终止模块；
- 单模块 72 小时 hard wall-clock 上限。

合法的长时间静默模块可以调整 `IDLE_TIMEOUT`/`PROCESS_RERUN_IDLE_TIMEOUT`；绝对上限使用 `TIMEOUT`/`PROCESS_RERUN_TIMEOUT`。被外层 timeout 终止的历史模块会由 `rerun-incomplete-distributed` 从模块开头重新运行，不保证从旧的随机 stepcurrent shard 百分比继续。

普通续跑使用 `resume-distributed`。如果只需补跑不完整模块或重建报告，应使用：

```bash
ENV_SH="$ENV_SH" \
PYTORCH_ROOT="$PYTORCH_ROOT" \
DIST_WORK_DIR="$WORK" \
GPU_IDS=0,1,2,3,4,5,6,7 \
PROCESS_RERUN_TIMEOUT=259200 \
PROCESS_RERUN_IDLE_TIMEOUT=7200 \
PROCESS_RERUN_ERROR_TYPES=Timeout,Crash \
bash "$OPS_ROOT/runners/run_test-2.13-official-queue.sh" rerun-incomplete-distributed
```

它与普通 resume 的区别是只选择已有 checkpoint 中需要完整模块补跑的项，不补 MISSING；不是规避空队列错误的临时方案。

## 自定义环境变量

直接调用 Python runner 时，把变量放在 `nohup env` 后，例如：

```bash
nohup env PYTHONUNBUFFERED=1 TORCHINDUCTOR_CPP_MARCH=znver1 python3 -u ...
```

dry-run、首次运行、自动补跑和续跑必须使用完全相同的自定义变量。比较不同配置时必须使用不同 work-dir。

runner 会把计划、PyTorch 源码、关键环境和语义参数的指纹写入 `<work-dir>/.run_manifest.json`。续跑发生不一致时会拒绝复用 checkpoint，并提示变化属于 `source`、`plan`、`environment` 还是 `options`。清单只保存环境变量名和整体哈希，不保存变量原值。

`TORCHINDUCTOR_CPP_MARCH` 会改变 Inductor C++ 编译目标，例如从 `-march=native` 改为指定目标。它只应在明确命名的配置中使用，且不能在同一 work-dir 的续跑或发布过程中途改变。

当前 runner 会在继承外部环境后强制写入以下内置默认值，因此外部同名变量不会覆盖它们：

```text
PYTORCH_TEST_WITH_ROCM=1
CONTINUE_THROUGH_ERROR=True
```

`MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC` 不再由 runner 设置；未在启动环境中设置时保持未设置，显式设置时原样传递给 dry-run、正式运行、自动补跑和续跑。该变量属于 manifest 跟踪的语义环境，续跑期间不能改变。

这套 runner 当前面向 ROCm 测试环境；迁移到 CUDA 或 CPU 环境前应先检查并调整源码中的默认环境策略。
