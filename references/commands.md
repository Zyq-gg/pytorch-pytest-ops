# Command Templates

Always inspect current `--help` before using these templates. Resolve these variables first:

```bash
OPS_ROOT=/path/to/cloned/pytorch-pytest-ops
PYTORCH_ROOT=/path/to/pytorch
ENV_SH=/path/to/environment.sh  # optional when the environment is already active
```

## Ordinary full run

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

Resume with the same command and work directory, remove `--fresh`, and normally write to `runner_resume.out`.

## Dry-run only

```bash
test -z "${ENV_SH:-}" || source "$ENV_SH"
python3 "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
  "$PYTORCH_ROOT" \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only
```

This creates `$WORK/test_files.txt` and does not execute tests.

## Prefix subset

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  "$OPS_ROOT/runners/run_pytorch_tests_prefix.py" \
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
  --fresh > "$WORK/runner.out" 2>&1 &
```

## Existing-list subset

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  "$OPS_ROOT/runners/run_pytorch_subset.py" pytest-list \
  "$PYTORCH_ROOT" \
  --test-list /path/to/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --recovery-case-timeout 600 \
  --recovery-attempts 3 \
  --recovery-max-total-time 7200 \
  --fresh > "$WORK/runner.out" 2>&1 &
```

## Process-level failure files with publication

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  "$OPS_ROOT/runners/run_pytorch_subset.py" pytest-failure-files \
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
  --fresh > "$WORK/runner.out" 2>&1 &
```

For a completed supplemental run that only needs publication, reuse the same arguments, remove `--fresh`, add `--skip-fail`, and run in the foreground.

## Stable failures

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  "$OPS_ROOT/runners/rerun_stable_failures.py" \
  "$PYTORCH_ROOT" /path/to/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir "$WORK" \
  > "$WORK/runner.out" 2>&1 &
```

Inspect current `--help` for generic CSV column filters before supplying filter arguments.

## Official distributed resume

Use the current documented command from the distributed section of `$OPS_ROOT/docs/PYTORCH_PYTEST_WORKFLOW.md`, then verify flags against:

```bash
python3 "$OPS_ROOT/runners/run_pytorch_subset.py" run-test-resume --help
```

Keep official `run_test.py` arguments after `--`. Use the same work directory and remove the fresh/reset option on resume.

## Complete official normal queue

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

Resume with `resume-normal`. Require root `summary.json`, the latest completed `process_module_rerun*/summary.json` when selected, no unresolved rows, and `coverage_report.json.coverage_complete == true`. `completed_records == planned` is insufficient because TIMEOUT also has a checkpoint record.

Current official queue resume preserves reliable historical FAIL reports across timestamp directories. Startup reports `Done FAIL: N (reuse X, retry Y)`; `Need run` contains unreported FAIL, TIMEOUT, and MISSING modules, not every old FAIL. The final report keeps historical `source_log` values for reused modules and replaces every old row for modules executed during the resume.

## Complete official distributed queue

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

The distributed action uses one `--no-bind-gpu` worker and inherits every device visible inside the container. Do not run another container against the same GPUs when stable distributed results matter.

The default official profile terminates after 2 hours without subprocess output and also retains a 72-hour hard limit. Override `IDLE_TIMEOUT`/`PROCESS_RERUN_IDLE_TIMEOUT` for legitimate silent cases, and override `TIMEOUT`/`PROCESS_RERUN_TIMEOUT` for the absolute limit. A timed-out historical module is restarted from its beginning by `rerun-incomplete-normal` or `rerun-incomplete-distributed`.

## Custom environment variables

Put variables after `nohup env`, for example:

```bash
nohup env PYTHONUNBUFFERED=1 TORCHINDUCTOR_CPP_MARCH=znver1 python3 -u ...
```

Use identical values for discovery, initial execution, automatic reruns, and resumes.

`TORCHINDUCTOR_CPP_MARCH` changes Inductor C++ compilation (`-march=native` versus the requested target). Use it only for an explicitly named configuration, use a separate work directory, and never change it midway through resume/rerun publication.
