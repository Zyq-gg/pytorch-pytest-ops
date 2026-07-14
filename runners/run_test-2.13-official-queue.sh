#!/usr/bin/env bash
set -eo pipefail

# Official run_test.py queue workflow for this container.
#
# This script keeps /workspace/pytorch/test/run_test.py as the real test
# entrypoint, but uses the sibling run_official_run_test_queue.py
# to add dry-run list generation, dynamic GPU assignment, checkpoint/resume,
# and nohup-friendly logs.
#
# Usage:
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh dry-run-normal
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh run-normal
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh resume-normal
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh dry-run-distributed
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh run-distributed
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh status-normal
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh status-distributed
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh rerun-incomplete-normal
#   FAILURE_CSV=/path/failure_report.csv bash ... run-normal-failures
#   FAILURE_CSV=/path/failure_report.csv bash ... run-distributed-failures
#
# Override paths for another environment:
#   ENV_SH=/path/to/env.sh \
#   PYTORCH_ROOT=/path/to/pytorch \
#   NORMAL_WORK_DIR=/path/to/output \
#   GPU_IDS=0,1 \
#   bash /workspace/torch_test/run_test-2.13-official-queue.sh run-normal
#
# PyTorch run_test.py adds pytest reruns by default:
#   PYTORCH_NUM_PYTEST_RERUNS=2 -> pytest --reruns=2
# Override example:
#   PYTORCH_NUM_PYTEST_RERUNS=0 bash /workspace/torch_test/run_test-2.13-official-queue.sh run-normal

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

ENV_SH=${ENV_SH:-/home/tmp/python_and_sh/env.sh}
if [[ -n "$ENV_SH" ]]; then
  source "$ENV_SH"
fi

PYTORCH_ROOT=${PYTORCH_ROOT:-/workspace/pytorch}
QUEUE_RUNNER=${QUEUE_RUNNER:-$SCRIPT_DIR/run_official_run_test_queue.py}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
TIMEOUT=${TIMEOUT:-21600}
# Keep unattended recovery finite. A second timeout is reported explicitly as
# incomplete instead of leaving the queue blocked forever.
PROCESS_RERUN_TIMEOUT=${PROCESS_RERUN_TIMEOUT:-43200}
PROCESS_RERUN_ERROR_TYPES=${PROCESS_RERUN_ERROR_TYPES:-Timeout,Crash}
INCLUDE_REGEX=${INCLUDE_REGEX:-}
EXCLUDE_REGEX=${EXCLUDE_REGEX:-}
export PYTORCH_NUM_PYTEST_RERUNS=${PYTORCH_NUM_PYTEST_RERUNS:-2}

NORMAL_WORK_DIR=${NORMAL_WORK_DIR:-/home/tmp/torch2.13/run_test_official_nmz}
DIST_WORK_DIR=${DIST_WORK_DIR:-/home/tmp/torch2.13/run_test_official_distributed_nmz}
FAILURE_CSV=${FAILURE_CSV:-}
FAILURE_WORK_DIR=${FAILURE_WORK_DIR:-/home/tmp/torch2.13/run_test_official_failure_rerun_nmz}

QUEUE_FILTER_ARGS=()
if [[ -n "$INCLUDE_REGEX" ]]; then
  QUEUE_FILTER_ARGS+=(--include-regex "$INCLUDE_REGEX")
fi
if [[ -n "$EXCLUDE_REGEX" ]]; then
  QUEUE_FILTER_ARGS+=(--exclude-regex "$EXCLUDE_REGEX")
fi

PROCESS_RERUN_ARGS=(
  --process-rerun
  --process-rerun-error-types "$PROCESS_RERUN_ERROR_TYPES"
  --process-rerun-timeout "$PROCESS_RERUN_TIMEOUT"
)

COMMON_RUN_TEST_ARGS=(
  --exclude-jit-executor
  --exclude-distributed-tests
  --verbose
)

DIST_RUN_TEST_ARGS=(
  --distributed-tests
  --continue-through-error
  --verbose
)

cmd=${1:-help}

case "$cmd" in
  dry-run-normal)
    mkdir -p "$NORMAL_WORK_DIR"
    python3 "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$NORMAL_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --dry-run-only \
      "${QUEUE_FILTER_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}"
    ;;

  run-normal)
    mkdir -p "$NORMAL_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$NORMAL_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --timeout "$TIMEOUT" \
      --fresh \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}" \
      > "$NORMAL_WORK_DIR/runner.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $NORMAL_WORK_DIR/runner.out"
    ;;

  resume-normal)
    mkdir -p "$NORMAL_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$NORMAL_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --timeout "$TIMEOUT" \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}" \
      > "$NORMAL_WORK_DIR/runner_resume.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $NORMAL_WORK_DIR/runner_resume.out"
    ;;

  rerun-incomplete-normal)
    mkdir -p "$NORMAL_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$NORMAL_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --process-rerun-only \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}" \
      > "$NORMAL_WORK_DIR/runner_incomplete_rerun.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $NORMAL_WORK_DIR/runner_incomplete_rerun.out"
    ;;

  dry-run-distributed)
    mkdir -p "$DIST_WORK_DIR"
    python3 "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$DIST_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --dry-run-only \
      "${QUEUE_FILTER_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}"
    ;;

  run-distributed)
    mkdir -p "$DIST_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$DIST_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --timeout "$TIMEOUT" \
      --fresh \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}" \
      > "$DIST_WORK_DIR/runner.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $DIST_WORK_DIR/runner.out"
    ;;

  resume-distributed)
    mkdir -p "$DIST_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$DIST_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --timeout "$TIMEOUT" \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}" \
      > "$DIST_WORK_DIR/runner_resume.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $DIST_WORK_DIR/runner_resume.out"
    ;;

  rerun-incomplete-distributed)
    mkdir -p "$DIST_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$DIST_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --process-rerun-only \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}" \
      > "$DIST_WORK_DIR/runner_incomplete_rerun.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $DIST_WORK_DIR/runner_incomplete_rerun.out"
    ;;

  run-normal-failures)
    if [[ -z "$FAILURE_CSV" ]]; then
      echo "ERROR: set FAILURE_CSV=/path/to/failure_report.csv" >&2
      exit 2
    fi
    mkdir -p "$FAILURE_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$FAILURE_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --timeout "$TIMEOUT" \
      --failure-csv "$FAILURE_CSV" \
      --fresh \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}" \
      > "$FAILURE_WORK_DIR/runner.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $FAILURE_WORK_DIR/runner.out"
    ;;

  resume-normal-failures)
    if [[ -z "$FAILURE_CSV" ]]; then
      echo "ERROR: set FAILURE_CSV=/path/to/failure_report.csv" >&2
      exit 2
    fi
    mkdir -p "$FAILURE_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$FAILURE_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --timeout "$TIMEOUT" \
      --failure-csv "$FAILURE_CSV" \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${COMMON_RUN_TEST_ARGS[@]}" \
      > "$FAILURE_WORK_DIR/runner_resume.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $FAILURE_WORK_DIR/runner_resume.out"
    ;;

  run-distributed-failures)
    if [[ -z "$FAILURE_CSV" ]]; then
      echo "ERROR: set FAILURE_CSV=/path/to/failure_report.csv" >&2
      exit 2
    fi
    mkdir -p "$FAILURE_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$FAILURE_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --timeout "$TIMEOUT" \
      --failure-csv "$FAILURE_CSV" \
      --fresh \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}" \
      > "$FAILURE_WORK_DIR/runner.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $FAILURE_WORK_DIR/runner.out"
    ;;

  resume-distributed-failures)
    if [[ -z "$FAILURE_CSV" ]]; then
      echo "ERROR: set FAILURE_CSV=/path/to/failure_report.csv" >&2
      exit 2
    fi
    mkdir -p "$FAILURE_WORK_DIR"
    nohup env PYTHONUNBUFFERED=1 python3 -u "$QUEUE_RUNNER" "$PYTORCH_ROOT" \
      --work-dir "$FAILURE_WORK_DIR" \
      --gpu-ids "$GPU_IDS" \
      --no-bind-gpu \
      --timeout "$TIMEOUT" \
      --failure-csv "$FAILURE_CSV" \
      "${QUEUE_FILTER_ARGS[@]}" \
      "${PROCESS_RERUN_ARGS[@]}" \
      -- "${DIST_RUN_TEST_ARGS[@]}" \
      > "$FAILURE_WORK_DIR/runner_resume.out" 2>&1 &
    echo "started pid=$!"
    echo "runner: $FAILURE_WORK_DIR/runner_resume.out"
    ;;

  status-normal)
    echo "processes:"
    ps -ef | grep -E 'run_official_run_test_queue|run_test.py' | grep -v grep || true
    echo
    echo "runner tail:"
    tail -80 "$NORMAL_WORK_DIR/runner.out" 2>/dev/null || true
    echo
    echo "progress:"
    python3 - <<PY
import json, os
p="$NORMAL_WORK_DIR/.run_test_progress.json"
if not os.path.exists(p):
    print("progress missing")
else:
    d=json.load(open(p))
    print(json.dumps(d.get("stats", {}), indent=2))
PY
    echo
    echo "coverage:"
    cat "$NORMAL_WORK_DIR/coverage_report.json" 2>/dev/null || true
    ;;

  status-distributed)
    echo "processes:"
    ps -ef | grep -E 'run_official_run_test_queue|run_test.py' | grep -v grep || true
    echo
    echo "runner tail:"
    tail -80 "$DIST_WORK_DIR/runner.out" 2>/dev/null || true
    echo
    echo "progress:"
    python3 - <<PY
import json, os
p="$DIST_WORK_DIR/.run_test_progress.json"
if not os.path.exists(p):
    print("progress missing")
else:
    d=json.load(open(p))
    print(json.dumps(d.get("stats", {}), indent=2))
PY
    echo
    echo "coverage:"
    cat "$DIST_WORK_DIR/coverage_report.json" 2>/dev/null || true
    ;;

  *)
    sed -n '1,40p' "$0"
    exit 1
    ;;
esac
