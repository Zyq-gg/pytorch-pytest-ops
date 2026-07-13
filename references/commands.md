# Command Templates

Always inspect current `--help` before using these templates.

## Ordinary full run

```bash
source /home/tmp/python_and_sh/env.sh

WORK=/home/tmp/torch2.13/log-final/pytest_full_nmz_new
mkdir -p "$WORK"

nohup env PYTHONUNBUFFERED=1 \
  python3 -u /workspace/torch_test/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir "$WORK" \
  --timeout 1800 \
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
source /home/tmp/python_and_sh/env.sh
python3 /workspace/torch_test/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --dry-run-only
```

This creates `$WORK/test_files.txt` and does not execute tests.

## Prefix subset

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  /workspace/torch_test/run_pytorch_tests_prefix.py \
  /workspace/pytorch \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --include-prefix inductor/,dynamo/ \
  --timeout 1800 \
  --process-rerun-timeout 14400 \
  --fresh > "$WORK/runner.out" 2>&1 &
```

## Existing-list subset

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  /workspace/torch_test/run_pytorch_subset.py pytest-list \
  /workspace/pytorch \
  --test-list /path/to/test_files.txt \
  --include-prefix inductor/,dynamo/ \
  --work-dir "$WORK" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 1800 \
  --fresh > "$WORK/runner.out" 2>&1 &
```

## Process-level failure files with publication

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  /workspace/torch_test/run_pytorch_subset.py pytest-failure-files \
  /workspace/pytorch \
  --failure-csv "$BASE/latest/unresolved_process_failures.csv" \
  --work-dir "$WORK" \
  --publish-to-work-dir "$BASE" \
  --error-type Timeout,Crash \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --timeout 14400 \
  --fresh > "$WORK/runner.out" 2>&1 &
```

For a completed supplemental run that only needs publication, reuse the same arguments, remove `--fresh`, add `--skip-fail`, and run in the foreground.

## Stable failures

```bash
nohup env PYTHONUNBUFFERED=1 python3 -u \
  /workspace/torch_test/rerun_stable_failures.py \
  /workspace/pytorch /path/to/failure_report.csv \
  --attempts 3 \
  --timeout 600 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir "$WORK" \
  > "$WORK/runner.out" 2>&1 &
```

Inspect current `--help` for generic CSV column filters before supplying filter arguments.

## Official distributed resume

Use the current documented command from the distributed section of `/workspace/torch_test/PYTORCH_PYTEST_WORKFLOW.md`, then verify flags against:

```bash
python3 /workspace/torch_test/run_pytorch_subset.py run-test-resume --help
```

Keep official `run_test.py` arguments after `--`. Use the same work directory and remove the fresh/reset option on resume.

## Custom environment variables

Put variables after `nohup env`, for example:

```bash
nohup env PYTHONUNBUFFERED=1 TORCHINDUCTOR_CPP_MARCH=znver1 python3 -u ...
```

Use identical values for discovery, initial execution, automatic reruns, and resumes.
