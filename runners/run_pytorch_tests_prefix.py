#!/usr/bin/env python3
"""
PyTorch test runner: dry-run parsing, GPU assignment, parallel execution, resume.

Usage:
  python run_pytorch_tests.py <pytorch_root>
  python run_pytorch_tests.py <pytorch_root> --dry-run-only
  python run_pytorch_tests.py <pytorch_root> --fresh
  python run_pytorch_tests.py <pytorch_root> --retry-fail
  python run_pytorch_tests.py <pytorch_root> --num-gpus 4
  python run_pytorch_tests.py <pytorch_root> --gpu-ids 2,3,6,7
  python run_pytorch_tests.py <pytorch_root> --include-prefix dynamo/,inductor/
  python run_pytorch_tests.py <pytorch_root> --crash-chunk-size 16
  python run_pytorch_tests.py <pytorch_root> --analyze-only /path/to/test_runs/latest
  python run_pytorch_tests.py <pytorch_root> --work-dir ./my_runs
  python run_pytorch_tests.py <pytorch_root> --timeout 600
  python run_pytorch_tests.py <pytorch_root> -- -x --foo

Resume:
  Re-running the same command reads .test_progress.json and skips completed tests.
  Failed tests are retried by default unless --skip-fail is used.
"""

import argparse
import ast
import concurrent.futures
import csv
import fcntl
import json
import os
import pty
import queue
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from datetime import datetime, timedelta

DEFAULT_DRY_RUN_ARGS = ["--exclude-jit-executor", "--exclude-distributed-tests"]
DEFAULT_TEST_ENV = {
    "PYTORCH_TEST_WITH_ROCM": "1",
    "CONTINUE_THROUGH_ERROR": "True",
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": "1",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SHORT_SUMMARY_RE = re.compile(r"\bshort test summary info\b", re.IGNORECASE)
FAILED_RE = re.compile(r"^FAILED(?:\s+\[[^\]]+\])?\s+(.+?)(?:\s+-\s+(.+))?$")
FAILED_CONSISTENTLY_RE = re.compile(r"FAILED CONSISTENTLY:\s+(.+)$")
FAILED_CONSISTENTLY_LIST_MARKER = "The following tests failed consistently:"
RUNNING_ITEMS_RE = re.compile(r"Running\s+\d+\s+items?\s+in this shard:\s+(.+)$")
NODEID_IN_LINE_RE = re.compile(r"(?P<nodeid>(?:test/)?[\w./+-]+\.py::[^\s]+)")
DIVIDER_RE = re.compile(r"^[=\-_\s]+$")
STATS_RE = re.compile(
    r"^\d+\s+(failed|passed|skipped|xfailed|xpassed|error|errors|warnings?)\b",
    re.IGNORECASE,
)
START_RE = re.compile(r"^===== .* START: (.+?) \[\d+/\d+\] \(GPU (\d+)\) =====$")
END_RE = re.compile(r"^===== .* (PASS|FAIL|SKIP|TIMEOUT): (.+?)(?: \(|$)")
STEPCURRENT_RECOVERY_DONE_RE = re.compile(
    r"^===== STEPCURRENT RECOVERY DONE: (.+?), iterations=\d+, failed=(?:True|False) =====$"
)
FALLBACK_RECOVERY_DONE_RE = re.compile(
    r"^===== RECOVERY DONE: (.+?), cases=\d+, failed=(?:True|False) =====$"
)
CRASH_RE = re.compile(
    r"(Fatal Python error: .+|Segmentation fault|SIGSEGV|SIGABRT|core dumped|"
    r"Aborted|Bus error|Illegal instruction|Floating point exception|"
    r"HSA_STATUS_ERROR[^\s:]*|MEMORY_APERTURE_VIOLATION|KERNEL VMFault)",
    re.IGNORECASE,
)
PY_FILE_RE = re.compile(r"(?P<file>(?:[\w./+-]+/)?[\w.+-]+\.py)\b")
FAILURE_CSV_FIELDS = [
    "source_log",
    "gpu",
    "test_file",
    "class_name",
    "case_name",
    "case_params",
    "error_type",
    "error_message",
    "nodeid",
    "raw",
]

_progress_lock = threading.Lock()
_shutdown_flag = threading.Event()


def make_test_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_TEST_ENV)
    return env


def run_dry_run(test_dir: str, extra_args: list[str]) -> str:
    cmd = [sys.executable, "run_test.py", "--dry-run"] + DEFAULT_DRY_RUN_ARGS + extra_args
    print(f"=== test dir: {test_dir} ===")
    print(f"=== command: {' '.join(cmd)} ===")
    result = subprocess.run(cmd, cwd=test_dir, env=make_test_env(), capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_dry_run_output(text: str) -> list[str]:
    tests = []
    in_block = False
    in_excluded = False

    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("Name: excluded"):
            in_excluded = True
            in_block = False
            continue
        if line.startswith("Name:") and "excluded" not in line:
            in_excluded = False
        if in_excluded:
            continue

        if re.match(r"^\s*(Serial|Parallel) tests\s*\(\d+\)\s*:", line):
            in_block = True
            continue

        if in_block:
            stripped = line.strip()
            if not stripped:
                continue
            if not line.startswith("    ") and not line.startswith("\t"):
                in_block = False
                continue
            m = re.match(r"^(.+?)\s+\d+/\d+$", stripped)
            if m:
                name = m.group(1)
                if not name.endswith(".py"):
                    name += ".py"
                tests.append(name)
    return tests


def load_progress(progress_file: str) -> dict[str, dict]:
    if os.path.isfile(progress_file):
        try:
            with open(progress_file, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "tests" in data:
                return data["tests"]
            return {
                k: {"status": v, "time": "", "elapsed": 0} if isinstance(v, str) else v
                for k, v in data.items()
            }
        except (json.JSONDecodeError, KeyError):
            legacy = {}
            with open(progress_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "|" in line:
                        path, status = line.rsplit("|", 1)
                        legacy[path] = {"status": status, "time": "", "elapsed": 0}
            return legacy
    return {}


def save_progress(progress_file: str, test_path: str, status: str, elapsed: float = 0) -> None:
    with _progress_lock:
        data = {}
        if os.path.isfile(progress_file):
            try:
                with open(progress_file, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}

        if "tests" not in data:
            data["tests"] = {}

        data["tests"][test_path] = {
            "status": status,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed": round(elapsed, 1),
        }
        data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["stats"] = compute_stats(data["tests"])

        tmp = progress_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, progress_file)


def compute_stats(tests: dict) -> dict:
    total = len(tests)
    passed = sum(1 for v in tests.values() if v.get("status") == "PASS")
    failed = sum(1 for v in tests.values() if v.get("status") == "FAIL")
    timeout = sum(1 for v in tests.values() if v.get("status") == "TIMEOUT")
    skipped = sum(1 for v in tests.values() if v.get("status") == "SKIP")
    remaining = total - passed - failed - timeout - skipped
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "timeout": timeout,
        "skipped": skipped,
        "remaining": remaining,
    }


def parse_gpu_ids(raw: str | None, num_gpus: int) -> list[int]:
    if raw is None:
        if num_gpus <= 0:
            raise ValueError("--num-gpus must be greater than 0")
        return list(range(num_gpus))

    gpu_ids = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if not item.isdigit():
            raise ValueError(f"invalid GPU id: {item!r}")
        gpu_ids.append(int(item))

    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicate IDs: {raw}")
    return gpu_ids


def filter_existing(tests: list[str], test_dir: str) -> tuple[list[str], list[str]]:
    existing, skipped = [], []
    for name in tests:
        if os.path.isfile(os.path.join(test_dir, name)):
            existing.append(name)
        else:
            skipped.append(name)
    return existing, skipped


def sort_tests_by_history(tests: list[str], progress: dict[str, dict]) -> list[str]:
    def elapsed(test: str) -> float:
        item = progress.get(test, {})
        if isinstance(item, dict):
            try:
                return float(item.get("elapsed", 0) or 0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    return sorted(tests, key=elapsed, reverse=True)


def make_work_queue(tests: list[str]) -> queue.Queue[tuple[int, str]]:
    work_queue: queue.Queue[tuple[int, str]] = queue.Queue()
    for idx, test in enumerate(tests):
        work_queue.put((idx, test))
    return work_queue


def empty_case_stats() -> dict[str, int]:
    return {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "xfailed": 0,
        "xpassed": 0,
        "deselected": 0,
        "unknown": 0,
    }


def parse_pytest_case_stats(output: str, returncode: int) -> dict[str, int]:
    stats = empty_case_stats()
    key_map = {
        "passed": "passed",
        "failed": "failed",
        "skipped": "skipped",
        "error": "errors",
        "errors": "errors",
        "xfailed": "xfailed",
        "xpassed": "xpassed",
        "deselected": "deselected",
    }

    # Pytest usually emits a final line like:
    # ===== 2 failed, 10 passed, 3 skipped, 1 deselected in 12.34s =====
    summary_lines = [
        line.strip()
        for line in output.splitlines()
        if " in " in line
        and re.search(
            r"\b(passed|failed|skipped|errors?|xfailed|xpassed|deselected)\b",
            line,
        )
    ]

    if summary_lines:
        summary = summary_lines[-1]
        for count, raw_key in re.findall(
            r"(\d+)\s+(passed|failed|skipped|errors?|xfailed|xpassed|deselected)\b",
            summary,
        ):
            stats[key_map[raw_key]] += int(count)
        return stats

    # Return code 5 means no tests collected. Do not count it as a case.
    if returncode not in (0, 5):
        stats["unknown"] = 1
    return stats


def add_case_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def case_total(stats: dict[str, int]) -> int:
    return (
        stats.get("passed", 0)
        + stats.get("failed", 0)
        + stats.get("skipped", 0)
        + stats.get("errors", 0)
        + stats.get("xfailed", 0)
        + stats.get("xpassed", 0)
        + stats.get("unknown", 0)
    )


def case_pass_rate(stats: dict[str, int]) -> float:
    total = case_total(stats)
    return (stats.get("passed", 0) * 100.0 / total) if total else 0.0


def parse_csv_items(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def filter_by_prefix(tests: list[str], prefixes: list[str]) -> list[str]:
    if not prefixes:
        return tests
    return [test for test in tests if any(test.startswith(prefix) for prefix in prefixes)]


def run_with_pty(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: int = 0,
) -> tuple[int, str]:
    """
    Run *cmd* in a pseudo-terminal so the child process produces full pytest
    output, including detailed short-summary failure lines.
    """
    master_fd, slave_fd = pty.openpty()

    winsize = struct.pack("HHHH", 40, 300, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=slave_fd,
        stderr=slave_fd,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    os.close(slave_fd)

    output_parts: list[bytes] = []
    deadline = time.monotonic() + timeout if (timeout is not None and timeout > 0) else None

    while True:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                while True:
                    rlist, _, _ = select.select([master_fd], [], [], 0.1)
                    if not rlist:
                        break
                    try:
                        data = os.read(master_fd, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    output_parts.append(data)
                os.close(master_fd)
                partial_output = b"".join(output_parts).decode("utf-8", errors="replace")
                partial_output = partial_output.replace("\r", "\n")
                raise subprocess.TimeoutExpired(cmd, timeout, output=partial_output)

        if proc.poll() is not None:
            break

        rlist, _, _ = select.select([master_fd], [], [], 1.0)
        if rlist:
            try:
                data = os.read(master_fd, 65536)
            except OSError:
                break
            if data:
                output_parts.append(data)

    while True:
        rlist, _, _ = select.select([master_fd], [], [], 0.1)
        if not rlist:
            break
        try:
            data = os.read(master_fd, 65536)
        except OSError:
            break
        if not data:
            break
        output_parts.append(data)

    os.close(master_fd)
    proc.wait()

    output = b"".join(output_parts).decode("utf-8", errors="replace")
    output = output.replace("\r", "\n")
    return proc.returncode, output


def is_process_crash(returncode: int, output: str) -> bool:
    return returncode < 0 or bool(CRASH_RE.search(output))


def collect_test_nodeids(test_dir: str, test: str, env: dict[str, str], timeout: int) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--color=no", test]
    try:
        returncode, output = run_with_pty(
            cmd,
            cwd=test_dir,
            env=env,
            timeout=timeout if timeout > 0 else None,
        )
    except subprocess.TimeoutExpired:
        return []

    if returncode not in (0, 5):
        return []

    nodeids = []
    prefix = f"{test}::"
    repo_prefix = f"test/{test}::"
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == test or line.startswith(prefix):
            nodeids.append(line)
        elif line == f"test/{test}":
            nodeids.append(test)
        elif line.startswith(repo_prefix):
            # Commands run with cwd=<repo>/test, so normalize nodeids emitted
            # relative to the repository root back to test-dir-relative paths.
            nodeids.append(line[len("test/") :])
    return nodeids


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def crash_message(output: str, returncode: int) -> str:
    match = CRASH_RE.search(output)
    if match:
        return match.group(1)
    if returncode < 0:
        try:
            return f"terminated by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"terminated by signal {-returncode}"
    return "process crashed"


def write_synthetic_failed_summary(log, nodeid: str, error_type: str, message: str) -> None:
    log.write("=========================== short test summary info ===========================\n")
    log.write(f"FAILED {nodeid} - {error_type}: {message}\n")
    log.write("============================== 1 failed in 0.00s ==============================\n")


def make_stepcurrent_key(test: str, gpu_id: int, global_idx: int) -> str:
    safe_test = re.sub(r"[^A-Za-z0-9_.-]+", "_", test).strip("_") or "test"
    return f"torch_test_{safe_test}_{gpu_id}_{global_idx}_{os.urandom(4).hex()}"


def read_stepcurrent_lastrun(test_dir: str, key: str) -> str:
    # pytest chooses the repository root from pytest.ini, even though commands
    # run with cwd=<repo>/test. Match official run_test.py and check the repo
    # root first; retain the old location as a compatibility fallback.
    cache_roots = [os.path.dirname(os.path.abspath(test_dir)), os.path.abspath(test_dir)]
    raw = ""
    for cache_root in cache_roots:
        cache_file = os.path.join(
            cache_root, ".pytest_cache", "v", "cache", "stepcurrent", key, "lastrun"
        )
        try:
            with open(cache_file, encoding="utf-8") as f:
                raw = f.read().strip()
            break
        except FileNotFoundError:
            continue
    if not raw or raw == "null":
        return ""
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, str) else ""
    except json.JSONDecodeError:
        return raw.strip("'\"")


def run_recovery_targets(
    *,
    targets: list[str],
    test_dir: str,
    env: dict[str, str],
    log,
    timeout: int,
    chunk_size: int,
) -> tuple[dict[str, int], list[tuple[str, str, float]], bool]:
    stats = empty_case_stats()
    failures: list[tuple[str, str, float]] = []
    any_failed = False

    if not targets:
        return stats, failures, any_failed

    pytest_cmd = [sys.executable, "-m", "pytest", "--tb=long", "--color=no"] + targets

    start = time.time()
    try:
        returncode, output = run_with_pty(
            pytest_cmd,
            cwd=test_dir,
            env=env,
            timeout=timeout if timeout > 0 else None,
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        if len(targets) > 1:
            mid = max(1, len(targets) // 2)
            left_stats, left_failures, left_failed = run_recovery_targets(
                targets=targets[:mid],
                test_dir=test_dir,
                env=env,
                log=log,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            right_stats, right_failures, right_failed = run_recovery_targets(
                targets=targets[mid:],
                test_dir=test_dir,
                env=env,
                log=log,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            add_case_stats(stats, left_stats)
            add_case_stats(stats, right_stats)
            return stats, left_failures + right_failures, left_failed or right_failed

        nodeid = targets[0]
        stats["unknown"] += 1
        any_failed = True
        message = f"timeout (>{timeout}s)"
        failures.append((nodeid, message, elapsed))
        log.write(f"===== RECOVERY TIMEOUT: {nodeid} ({message}) =====\n")
        write_synthetic_failed_summary(log, nodeid, "Timeout", message)
        return stats, failures, any_failed

    log.write(
        f"===== RECOVERY CHUNK: {len(targets)} case(s), exit={returncode}, "
        f"elapsed={elapsed:.1f}s =====\n"
    )
    log.write(output)
    if not output.endswith("\n"):
        log.write("\n")

    if is_process_crash(returncode, output):
        if len(targets) > 1:
            mid = max(1, len(targets) // 2)
            left_stats, left_failures, left_failed = run_recovery_targets(
                targets=targets[:mid],
                test_dir=test_dir,
                env=env,
                log=log,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            right_stats, right_failures, right_failed = run_recovery_targets(
                targets=targets[mid:],
                test_dir=test_dir,
                env=env,
                log=log,
                timeout=timeout,
                chunk_size=chunk_size,
            )
            add_case_stats(stats, left_stats)
            add_case_stats(stats, right_stats)
            return stats, left_failures + right_failures, left_failed or right_failed

        nodeid = targets[0]
        stats["unknown"] += 1
        any_failed = True
        message = crash_message(output, returncode)
        failures.append((nodeid, message, elapsed))
        log.write(f"===== RECOVERY CRASH: {nodeid} ({message}) =====\n")
        write_synthetic_failed_summary(log, nodeid, "Crash", message)
        return stats, failures, any_failed

    current_stats = parse_pytest_case_stats(output, returncode)
    add_case_stats(stats, current_stats)
    if returncode not in (0, 5):
        any_failed = True
        failures.append((targets[0], extract_failure_summary(output), elapsed))
    return stats, failures, any_failed


def run_pytest_with_stepcurrent(
    *,
    test: str,
    test_dir: str,
    env: dict[str, str],
    stepcurrent_key: str,
    mode: str,
    log,
    timeout: int,
) -> tuple[int | None, str, bool]:
    assert mode in {"rs", "scs"}
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=long",
        "--color=no",
        f"--{mode}={stepcurrent_key}",
        test,
    ]
    log.write(f"===== STEPCURRENT {mode.upper()}: {' '.join(cmd)} =====\n")
    try:
        returncode, output = run_with_pty(
            cmd,
            cwd=test_dir,
            env=env,
            timeout=timeout if timeout > 0 else None,
        )
        log.write(output)
        if not output.endswith("\n"):
            log.write("\n")
        # Match test/run_test.py: pytest exit code 5 means this resumed shard
        # contains no tests, so --scs has reached the end successfully.
        return (0 if returncode == 5 else returncode), output, False
    except subprocess.TimeoutExpired:
        message = f"timeout (>{timeout}s)"
        log.write(f"===== STEPCURRENT {mode.upper()} TIMEOUT: {test} ({message}) =====\n")
        return None, message, True


def recover_with_stepcurrent_resume(
    *,
    test: str,
    test_dir: str,
    env: dict[str, str],
    log,
    timeout: int,
    stepcurrent_key: str,
    initial_error_type: str,
    initial_message: str,
    max_failures_per_case: int = 3,
    max_iterations: int = 200,
) -> tuple[dict[str, int], list[tuple[str, str, float]], bool, bool]:
    """Resume a crashed/timed-out pytest file using PyTorch's stepcurrent plugin.

    Returns (case_stats, failures, any_failed, completed). completed=False means
    the stepcurrent cache was unavailable or recovery hit a guardrail; callers
    should fall back to the older collect/chunk recovery path.
    """
    start = time.time()
    stats = empty_case_stats()
    failures: list[tuple[str, str, float]] = []
    num_failures: dict[str, int] = {}
    recorded: set[str] = set()

    current = read_stepcurrent_lastrun(test_dir, stepcurrent_key)
    if not current:
        log.write("===== STEPCURRENT RECOVERY UNAVAILABLE: no lastrun cache =====\n")
        return stats, failures, False, False

    log.write(
        f"===== STEPCURRENT RECOVERY START: {test}, initial={current}, "
        f"{initial_error_type}: {initial_message} =====\n"
    )
    mode = "rs"
    any_failed = False

    for iteration in range(1, max_iterations + 1):
        before = read_stepcurrent_lastrun(test_dir, stepcurrent_key) or current
        returncode, output, timed_out = run_pytest_with_stepcurrent(
            test=test,
            test_dir=test_dir,
            env=env,
            stepcurrent_key=stepcurrent_key,
            mode=mode,
            log=log,
            timeout=timeout,
        )
        elapsed = time.time() - start

        if returncode == 0 and not timed_out:
            add_case_stats(stats, parse_pytest_case_stats(output, 0))
            if mode == "rs":
                log.write(f"===== STEPCURRENT SINGLE PASS: {before}; continue after it =====\n")
                mode = "scs"
                continue
            log.write(
                f"===== STEPCURRENT RECOVERY DONE: {test}, iterations={iteration}, "
                f"failed={any_failed} =====\n"
            )
            return stats, failures, any_failed, True

        current = read_stepcurrent_lastrun(test_dir, stepcurrent_key) or before
        if timed_out:
            error_type = "Timeout"
            error_message = output
        elif returncode is not None and is_process_crash(returncode, output):
            error_type = "Crash"
            error_message = crash_message(output, returncode)
        else:
            error_type, error_message = split_error(extract_failure_summary(output))
            if not error_type:
                error_type = "Failure"

        num_failures[current] = num_failures.get(current, 0) + 1
        log.write(
            f"===== STEPCURRENT FAILURE: {current} attempt "
            f"{num_failures[current]}/{max_failures_per_case} "
            f"({error_type}: {error_message}) =====\n"
        )

        if num_failures[current] >= max_failures_per_case:
            any_failed = True
            if current not in recorded:
                write_synthetic_failed_summary(log, current, error_type, error_message)
                failures.append((current, f"{error_type}: {error_message}", elapsed))
                recorded.add(current)
            log.write(f"===== STEPCURRENT SKIP CONSISTENT FAILURE: {current} =====\n")
            mode = "scs"
        else:
            mode = "rs"

    log.write(
        f"===== STEPCURRENT RECOVERY ABORTED: hit max_iterations={max_iterations} =====\n"
    )
    return stats, failures, any_failed, False


def recover_crashed_test(
    *,
    test: str,
    test_dir: str,
    env: dict[str, str],
    log,
    timeout: int,
    chunk_size: int,
) -> tuple[dict[str, int], list[tuple[str, str, float]], bool]:
    log.write(f"===== RECOVERY COLLECT: {test} =====\n")
    nodeids = collect_test_nodeids(test_dir, test, env, timeout)
    if not nodeids:
        stats = empty_case_stats()
        stats["unknown"] = 1
        message = "failed to collect nodeids for crash recovery"
        log.write(f"===== RECOVERY UNAVAILABLE: {test} ({message}) =====\n")
        return stats, [(test, message, 0.0)], True

    log.write(f"===== RECOVERY START: {test}, collected {len(nodeids)} case(s) =====\n")
    stats = empty_case_stats()
    failures: list[tuple[str, str, float]] = []
    any_failed = False

    for targets in chunked(nodeids, max(1, chunk_size)):
        chunk_stats, chunk_failures, chunk_failed = run_recovery_targets(
            targets=targets,
            test_dir=test_dir,
            env=env,
            log=log,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        add_case_stats(stats, chunk_stats)
        failures.extend(chunk_failures)
        any_failed = any_failed or chunk_failed

    log.write(
        f"===== RECOVERY DONE: {test}, cases={case_total(stats)}, "
        f"failed={any_failed} =====\n"
    )
    return stats, failures, any_failed


def run_gpu_tests(
    worker_idx: int,
    gpu_id: int,
    work_queue: queue.Queue[tuple[int, str]],
    total_tests: int,
    test_dir: str,
    log_dir: str,
    progress_file: str,
    timeout: int = 0,
    crash_recovery: bool = True,
    crash_chunk_size: int = 16,
) -> dict:
    env = make_test_env()
    env["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    if worker_idx > 0:
        time.sleep(worker_idx * 5)

    log_file = os.path.join(log_dir, f"gpu_{gpu_id}.log")
    passed = 0
    failed = 0
    skipped = 0
    start_time = time.time()
    failures = []
    case_stats = empty_case_stats()
    assigned = 0

    with open(log_file, "w", buffering=1, encoding="utf-8") as log:
        log.write(f"===== worker {worker_idx}, GPU {gpu_id}, stagger {worker_idx * 5}s =====\n")
        while not _shutdown_flag.is_set():
            try:
                global_idx, test = work_queue.get_nowait()
            except queue.Empty:
                break

            assigned += 1
            if _shutdown_flag.is_set():
                log.write("===== shutdown signal, exiting early =====\n")
                work_queue.task_done()
                break

            ts = datetime.now().strftime("%H:%M:%S")
            progress_pct = f"[{global_idx + 1}/{total_tests}]"
            log.write(f"===== {ts} START: {test} {progress_pct} (GPU {gpu_id}) =====\n")
            log.flush()

            test_start = time.time()
            stepcurrent_key = make_stepcurrent_key(test, gpu_id, global_idx)
            try:
                pytest_cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--tb=long",
                    "--color=no",
                    f"--sc={stepcurrent_key}",
                    "--print-items",
                    test,
                ]
                returncode, output = run_with_pty(
                    pytest_cmd,
                    cwd=test_dir,
                    env=env,
                    timeout=timeout if timeout > 0 else None,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed = time.time() - test_start
                partial_output = exc.output if isinstance(exc.output, str) else ""
                if partial_output:
                    log.write(partial_output)
                    if not partial_output.endswith("\n"):
                        log.write("\n")
                log.write(f"===== {ts} TIMEOUT: {test} (>{timeout}s) =====\n")
                step_nodeid = read_stepcurrent_lastrun(test_dir, stepcurrent_key) or test
                if crash_recovery:
                    recovered_stats, recovered_failures, recovered_failed, completed = recover_with_stepcurrent_resume(
                        test=test,
                        test_dir=test_dir,
                        env=env,
                        log=log,
                        timeout=timeout,
                        stepcurrent_key=stepcurrent_key,
                        initial_error_type="Timeout",
                        initial_message=f"timeout (>{timeout}s)",
                    )
                    if not completed:
                        recovered_stats, recovered_failures, recovered_failed = recover_crashed_test(
                            test=test,
                            test_dir=test_dir,
                            env=env,
                            log=log,
                            timeout=timeout,
                            chunk_size=crash_chunk_size,
                        )
                    add_case_stats(case_stats, recovered_stats)
                    failures.extend(recovered_failures)
                    if recovered_failed:
                        failed += 1
                        if not completed:
                            write_synthetic_failed_summary(log, step_nodeid, "Timeout", f"timeout (>{timeout}s)")
                        save_progress(progress_file, test, "FAIL", elapsed)
                    else:
                        passed += 1
                        save_progress(progress_file, test, "PASS", elapsed)
                else:
                    failed += 1
                    write_synthetic_failed_summary(log, step_nodeid, "Timeout", f"timeout (>{timeout}s)")
                    failures.append((step_nodeid, f"timeout (>{timeout}s)", elapsed))
                    save_progress(progress_file, test, "FAIL", elapsed)
                print(f"  [GPU {gpu_id}] TIMEOUT: {test} (>{timeout}s)")
                work_queue.task_done()
                continue

            elapsed = time.time() - test_start
            ts = datetime.now().strftime("%H:%M:%S")

            log.write(output)
            if not output.endswith("\n"):
                log.write("\n")

            process_crashed = is_process_crash(returncode, output)
            if process_crashed and crash_recovery:
                log.write(f"===== {ts} CRASH DETECTED: {test} (exit={returncode}) =====\n")
                step_nodeid = read_stepcurrent_lastrun(test_dir, stepcurrent_key) or test
                error_msg = crash_message(output, returncode)
                recovered_stats, recovered_failures, recovered_failed, completed = recover_with_stepcurrent_resume(
                    test=test,
                    test_dir=test_dir,
                    env=env,
                    log=log,
                    timeout=timeout,
                    stepcurrent_key=stepcurrent_key,
                    initial_error_type="Crash",
                    initial_message=error_msg,
                )
                if not completed:
                    recovered_stats, recovered_failures, recovered_failed = recover_crashed_test(
                        test=test,
                        test_dir=test_dir,
                        env=env,
                        log=log,
                        timeout=timeout,
                        chunk_size=crash_chunk_size,
                    )
                add_case_stats(case_stats, recovered_stats)
                failures.extend(recovered_failures)
                if recovered_failed:
                    failed += 1
                    if not completed:
                        write_synthetic_failed_summary(log, step_nodeid, "Crash", error_msg)
                        failures.append((step_nodeid, error_msg, elapsed))
                    log.write(f"===== {ts} FAIL: {test} (crash recovered, {elapsed:.1f}s) =====\n")
                    print(f"  [GPU {gpu_id}] CRASH: {test}  recovered failing case(s)")
                    save_progress(progress_file, test, "FAIL", elapsed)
                else:
                    passed += 1
                    log.write(f"===== {ts} PASS: {test} (crash recovery passed, {elapsed:.1f}s) =====\n")
                    save_progress(progress_file, test, "PASS", elapsed)
                log.write("\n")
                work_queue.task_done()
                continue

            current_case_stats = parse_pytest_case_stats(output, returncode)
            add_case_stats(case_stats, current_case_stats)

            if returncode == 0:
                passed += 1
                log.write(f"===== {ts} PASS: {test} ({elapsed:.1f}s) =====\n")
                save_progress(progress_file, test, "PASS", elapsed)
            elif returncode == 5:
                skipped += 1
                log.write(f"===== {ts} SKIP: {test} (no tests collected, {elapsed:.1f}s) =====\n")
                save_progress(progress_file, test, "SKIP", elapsed)
            else:
                failed += 1
                error_msg = extract_failure_summary(output)
                failures.append((test, error_msg, elapsed))
                log.write(f"===== {ts} FAIL: {test} (exit={returncode}, {elapsed:.1f}s) =====\n")
                print(f"  [GPU {gpu_id}] FAIL: {test}  {error_msg[:120]}")
                save_progress(progress_file, test, "FAIL", elapsed)

            log.write("\n")
            work_queue.task_done()

        total_elapsed = time.time() - start_time
        log.write("=" * 60 + "\n")
        log.write(
            f"GPU {gpu_id} finished: PASS={passed} FAIL={failed} "
            f"SKIP={skipped} assigned={assigned} elapsed={timedelta(seconds=int(total_elapsed))}\n"
        )
        log.write(
            "Case stats: "
            f"total={case_total(case_stats)} "
            f"passed={case_stats['passed']} "
            f"failed={case_stats['failed']} "
            f"skipped={case_stats['skipped']} "
            f"errors={case_stats['errors']} "
            f"xfailed={case_stats['xfailed']} "
            f"xpassed={case_stats['xpassed']} "
            f"deselected={case_stats['deselected']} "
            f"unknown={case_stats['unknown']} "
            f"pass_rate={case_pass_rate(case_stats):.2f}%\n"
        )
        log.write("=" * 60 + "\n")

    return {
        "worker_idx": worker_idx,
        "gpu_id": gpu_id,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "assigned": assigned,
        "elapsed": total_elapsed,
        "failures": failures,
        "case_stats": case_stats,
    }


def extract_failure_summary(output: str) -> str:
    for line in output.splitlines():
        line = line.strip()
        if line and any(
            kw in line
            for kw in [
                "Error: ",
                "error: ",
                "AssertionError",
                "assert ",
                "Assertion `",
                "Fatal Python error",
                "SIGABRT",
                "SIGSEGV",
            ]
        ):
            return line[:200]

    for line in output.splitlines():
        if line.strip().startswith("FAILED"):
            return line.strip()[:200]

    tail = output.strip().split("\n")
    return "\n".join(tail[-5:])[:200]


def setup_signal_handler() -> None:
    def handler(signum, frame):
        print(f"\nreceived signal {signal.Signals(signum).name}, exiting gracefully...")
        _shutdown_flag.set()

    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def iter_log_files(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        direct_logs = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".log")
        )
        nested_logs = []
        for root, dirs, files in os.walk(path):
            if root == path:
                continue
            for name in files:
                if name.endswith(".log"):
                    nested_logs.append(os.path.join(root, name))
        return direct_logs + sorted(nested_logs)
    raise FileNotFoundError(path)


def parse_gpu_from_log(source_log: str) -> str:
    match = re.search(r"gpu_(\d+)\.log$", os.path.basename(source_log))
    return match.group(1) if match else ""


def extract_failed_lines(text: str) -> list[str]:
    lines = strip_ansi(text).splitlines()
    in_summary = False
    failures: list[str] = []
    current: str | None = None

    for raw_line in lines:
        line = raw_line.strip()

        if not in_summary:
            if SHORT_SUMMARY_RE.search(line):
                in_summary = True
            continue

        if not line:
            continue

        if line.startswith("FAILED "):
            if current:
                failures.append(current)
            current = line
            continue

        normalized = line.strip("= -_")
        if STATS_RE.match(line) or STATS_RE.match(normalized):
            if current:
                failures.append(current)
                current = None
            in_summary = False
            continue

        if DIVIDER_RE.match(line):
            continue

        if current:
            current = f"{current} {line}"

    if current:
        failures.append(current)

    return failures


def split_nodeid_parts(nodeid: str) -> list[str]:
    parts: list[str] = []
    start = 0
    bracket_depth = 0
    i = 0

    while i < len(nodeid):
        char = nodeid[i]
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif nodeid.startswith("::", i) and bracket_depth == 0:
            parts.append(nodeid[start:i])
            i += 2
            start = i
            continue
        i += 1

    parts.append(nodeid[start:])
    return parts


def split_nodeid(nodeid: str) -> tuple[str, str, str, str]:
    parts = split_nodeid_parts(nodeid)
    test_file = parts[0]
    class_name = ""
    case_full = ""

    if len(parts) >= 3:
        class_name = parts[-2]
        case_full = parts[-1]
    elif len(parts) == 2:
        case_full = parts[-1]

    case_name = case_full
    case_params = ""
    match = re.match(r"^(.+?)(?:\[(.*)\])?$", case_full)
    if match:
        case_name = match.group(1)
        case_params = match.group(2) or ""

    return test_file, class_name, case_name, case_params


def split_error(error: str) -> tuple[str, str]:
    if ": " not in error:
        return "", error
    error_type, error_message = error.split(": ", 1)
    return error_type, error_message


def make_failure_row(
    *,
    source_log: str,
    gpu: str,
    test_file: str,
    error_type: str,
    error_message: str,
    nodeid: str,
    raw: str,
) -> dict[str, str]:
    process_case_name = "<process>"
    normalized = error_type.strip().lower()
    if normalized in {"crash", "timeout"}:
        process_case_name = f"<{normalized}>"

    return {
        "source_log": source_log,
        "gpu": gpu,
        "test_file": test_file,
        "class_name": "",
        "case_name": process_case_name,
        "case_params": "",
        "error_type": error_type,
        "error_message": error_message,
        "nodeid": nodeid,
        "raw": raw,
    }


def parse_failed_line(line: str, source_log: str) -> dict[str, str]:
    match = FAILED_RE.match(line)
    if not match:
        return make_failure_row(
            source_log=source_log,
            gpu=parse_gpu_from_log(source_log),
            test_file="",
            error_type="Unparsed",
            error_message=line,
            nodeid="",
            raw=line,
        )

    nodeid = normalize_run_test_nodeid(match.group(1))
    error = match.group(2) or ""
    test_file, class_name, case_name, case_params = split_nodeid(nodeid)
    error_type, error_message = split_error(error)

    return {
        "source_log": source_log,
        "gpu": parse_gpu_from_log(source_log),
        "test_file": test_file,
        "class_name": class_name,
        "case_name": case_name,
        "case_params": case_params,
        "error_type": error_type,
        "error_message": error_message,
        "nodeid": nodeid,
        "raw": line,
    }


def make_case_failure_row(
    *,
    source_log: str,
    nodeid: str,
    error_type: str,
    error_message: str,
    raw: str,
) -> dict[str, str]:
    nodeid = normalize_run_test_nodeid(nodeid)
    test_file, class_name, case_name, case_params = split_nodeid(nodeid)
    return {
        "source_log": source_log,
        "gpu": parse_gpu_from_log(source_log),
        "test_file": test_file,
        "class_name": class_name,
        "case_name": case_name,
        "case_params": case_params,
        "error_type": error_type,
        "error_message": error_message,
        "nodeid": nodeid,
        "raw": raw,
    }


def normalize_run_test_nodeid(nodeid: str) -> str:
    nodeid = strip_ansi(nodeid).strip().strip("'\"")
    nodeid = re.sub(r"\s*!{5,}.*$", "", nodeid).strip()
    if nodeid.startswith("test/"):
        nodeid = nodeid[len("test/") :]
    return nodeid


def parse_failed_consistently_tests(text: str) -> list[str]:
    tests: list[str] = []
    seen: set[str] = set()

    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        if FAILED_CONSISTENTLY_LIST_MARKER in line:
            payload = line.split(FAILED_CONSISTENTLY_LIST_MARKER, 1)[1].strip()
            if payload.startswith(":"):
                payload = payload[1:].strip()
            try:
                parsed = ast.literal_eval(payload)
            except Exception:
                match = re.search(r"\[(.*)\]", payload)
                parsed = [item.strip().strip("'\"") for item in match.group(1).split(",")] if match else []
            for item in parsed:
                nodeid = normalize_run_test_nodeid(str(item))
                if "::" in nodeid and nodeid not in seen:
                    seen.add(nodeid)
                    tests.append(nodeid)
            continue

        match = FAILED_CONSISTENTLY_RE.search(line)
        if match:
            nodeid = normalize_run_test_nodeid(match.group(1))
            if "::" in nodeid and nodeid not in seen:
                seen.add(nodeid)
                tests.append(nodeid)

    return tests


def strip_distributed_log_prefix(line: str) -> str:
    line = strip_ansi(line).strip()
    line = re.sub(r"^\[rank\d+\]:[EWI]\d{4}\s+\S+\s+\d+\s+[^\]]+\]\s*", "", line)
    line = re.sub(r"^[IWE]\d{4}\s+\S+\s+\d+\s+[^\]]+\]\s*", "", line)
    return line.strip()


def shorten_text(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def concise_run_test_error(window: str) -> str:
    lines = [strip_distributed_log_prefix(line) for line in window.splitlines()]
    lines = [line for line in lines if line]
    noise_re = re.compile(
        r"(stopping after \d+ failures|=+ \d+ failed|=+ test session starts|"
        r"^-+ Captured|^FAILED \[.*\]\s+[\w./+-]+\.py::)"
    )
    signal_lines = [line for line in lines if not noise_re.search(line)]

    priority_markers = [
        "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION",
        "KERNEL VMFault",
        "Fatal Python error",
        "Segmentation fault",
        "SIGSEGV",
        "SIGABRT",
        "core dumped",
    ]
    for marker in priority_markers:
        for line in signal_lines:
            if marker in line:
                return shorten_text(line)

    for line in signal_lines:
        if "Unexpected success" in line:
            return shorten_text(line)

    for i, line in enumerate(signal_lines):
        if "HSACOError" in line or "Cannot select:" in line:
            nearby = " ".join(signal_lines[i : min(len(signal_lines), i + 30)])
            return shorten_text(nearby)

    for line in reversed(signal_lines):
        if " - " in line and ("FAILED " in line or "ERROR " in line):
            tail = line.split(" - ", 1)[1].strip()
            if re.search(
                r"(AssertionError|RuntimeError|TypeError|ValueError|ImportError|"
                r"ModuleNotFoundError|TraceError|AttributeError|torch\.[\w.]*Error):",
                tail,
            ):
                return shorten_text(tail)

    exception_re = re.compile(
        r"((?:(?:torch|triton)\.[\w.]+|[A-Za-z_][\w.]*)?"
        r"(?:[A-Za-z_]\w*Error|Exception|Unsupported):\s*.*|AssertionError$)"
    )
    candidates: list[str] = []
    for line in signal_lines:
        match = exception_re.search(line)
        if match:
            candidates.append(match.group(1))
    if candidates:
        precise = [item for item in candidates if not item.startswith("Exception: Caused by sample input")]
        return shorten_text((precise or candidates)[-1])

    for line in signal_lines:
        if "OutOfResources" in line or "No valid triton configs" in line:
            return shorten_text(line)

    for line in reversed(signal_lines):
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            return shorten_text(line)

    return ""


def split_error_loose(error: str) -> tuple[str, str]:
    error = error.strip()
    if not error:
        return "", ""
    if re.search(
        r"(Fatal Python error|SIG[A-Z]+|core dumped|Segmentation fault|Aborted|KERNEL VMFault)",
        error,
    ):
        return "Crash", error
    match = re.match(
        r"^((?:(?:torch|triton)\.[\w.]+|[A-Za-z_][\w.]*)?"
        r"(?:[A-Za-z_]\w*Error|Exception|Unsupported)):\s*(.*)$",
        error,
    )
    if match:
        return match.group(1), match.group(2)
    if error == "AssertionError":
        return "AssertionError", ""
    return "", error


def find_run_test_failure_window(lines: list[str], nodeid: str) -> tuple[int, str]:
    variants = [nodeid, f"test/{nodeid}"]
    best_idx = -1

    for idx, line in enumerate(lines):
        if "FAILED CONSISTENTLY:" in line and any(variant in line for variant in variants):
            best_idx = idx
            break

    if best_idx < 0:
        for idx, line in enumerate(lines):
            if line.startswith("FAILED ") and any(variant in line for variant in variants):
                best_idx = idx

    if best_idx < 0:
        for idx, line in enumerate(lines):
            if any(variant in line for variant in variants):
                best_idx = idx
                break

    if best_idx < 0:
        return -1, ""

    start = max(0, best_idx - 260)
    end = min(len(lines), best_idx + 40)
    return best_idx + 1, "\n".join(lines[start:end])


def parse_official_run_test_failure(nodeid: str, source_log: str, text: str) -> dict[str, str]:
    lines = strip_ansi(text).splitlines()
    line_no, window = find_run_test_failure_window(lines, nodeid)
    error = concise_run_test_error(window)
    error_type, error_message = split_error_loose(error)
    test_file, class_name, case_name, case_params = split_nodeid(nodeid)
    raw = f"FAILED: test/{nodeid}"
    if line_no > 0:
        raw = f"{raw} (near line {line_no})"

    return {
        "source_log": source_log,
        "gpu": parse_gpu_from_log(source_log),
        "test_file": test_file,
        "class_name": class_name,
        "case_name": case_name,
        "case_params": case_params,
        "error_type": error_type,
        "error_message": error_message,
        "nodeid": nodeid,
        "raw": raw,
    }


def extract_official_run_test_failures(text: str, source_log: str) -> list[dict[str, str]]:
    return [
        parse_official_run_test_failure(nodeid, source_log, text)
        for nodeid in parse_failed_consistently_tests(text)
    ]


def extract_pytest_summary_failures(text: str, source_log: str, *, official: bool = False) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if official:
        seen: set[str] = set()
        for raw_line in strip_ansi(text).splitlines():
            line = raw_line.strip()
            if not line.startswith("FAILED"):
                continue
            match = NODEID_IN_LINE_RE.search(line)
            if not match:
                continue
            nodeid = normalize_run_test_nodeid(match.group("nodeid"))
            if nodeid in seen:
                continue
            rows.append(parse_official_run_test_failure(nodeid, source_log, text))
            seen.add(nodeid)
        return rows

    for line in extract_failed_lines(text):
        parsed = parse_failed_line(line, source_log)
        nodeid = parsed.get("nodeid", "")
        if not nodeid or "::" not in nodeid:
            continue
        rows.append(parsed)
    return rows


def extract_official_timeout_current_failures(text: str, source_log: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    last_nodeid = ""
    seen: set[str] = set()

    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()

        running = RUNNING_ITEMS_RE.search(line)
        if running:
            matches = NODEID_IN_LINE_RE.findall(running.group(1))
            if matches:
                last_nodeid = normalize_run_test_nodeid(matches[-1])

        node_match = NODEID_IN_LINE_RE.match(line)
        if node_match:
            last_nodeid = normalize_run_test_nodeid(node_match.group("nodeid"))

        if line.startswith("===== TIMEOUT after ") and last_nodeid and last_nodeid not in seen:
            rows.append(
                make_case_failure_row(
                    source_log=source_log,
                    nodeid=last_nodeid,
                    error_type="Timeout",
                    error_message=line.strip("= "),
                    raw=line,
                )
            )
            seen.add(last_nodeid)

    return rows


def is_official_run_test_log(text: str) -> bool:
    markers = [
        "Running test batch 'tests to run'",
        "The following tests failed consistently:",
        "The following tests failed and then succeeded when run in a new process",
        "FAILED CONSISTENTLY:",
        "===== command: ",
    ]
    return any(marker in text for marker in markers) and "run_test.py" in text[:1000]


def extract_process_failures(text: str, source_log: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_test = ""
    current_gpu = parse_gpu_from_log(source_log)
    recorded_for_current = False

    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()

        start = START_RE.match(line)
        if start:
            current_test = start.group(1)
            current_gpu = start.group(2)
            recorded_for_current = False
            continue

        end = END_RE.match(line)
        if end:
            status, test_name = end.groups()
            if status == "TIMEOUT" and not recorded_for_current:
                rows.append(
                    make_failure_row(
                        source_log=source_log,
                        gpu=current_gpu,
                        test_file=test_name,
                        error_type="Timeout",
                        error_message=line,
                        nodeid=test_name,
                        raw=line,
                    )
                )
            current_test = ""
            recorded_for_current = False
            continue

        if not current_test or recorded_for_current:
            continue

        crash = CRASH_RE.search(line)
        if crash:
            rows.append(
                make_failure_row(
                    source_log=source_log,
                    gpu=current_gpu,
                    test_file=current_test,
                    error_type="Crash",
                    error_message=crash.group(1),
                    nodeid=current_test,
                    raw=line,
                )
            )
            recorded_for_current = True

    return rows


def extract_global_crashes(text: str, source_log: str) -> list[dict[str, str]]:
    current_gpu = parse_gpu_from_log(source_log)
    current_test_file = ""

    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        file_match = PY_FILE_RE.search(line)
        if file_match:
            current_test_file = file_match.group("file")

        crash = CRASH_RE.search(line)
        if crash:
            test_file = current_test_file or "<process>"
            return [
                make_failure_row(
                    source_log=source_log,
                    gpu=current_gpu,
                    test_file=test_file,
                    error_type="Crash",
                    error_message=crash.group(1),
                    nodeid=test_file,
                    raw=line,
                )
            ]

    return []


def failure_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["source_log"],
        row["gpu"],
        row["nodeid"],
        row["error_type"],
        row["error_message"],
    )


def canonical_test_name(name: str) -> str:
    name = name.strip()
    return name[len("test/") :] if name.startswith("test/") else name


def canonicalize_failure_row(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    nodeid = normalize_run_test_nodeid(result.get("nodeid", ""))
    result["nodeid"] = nodeid
    result["test_file"] = canonical_test_name(
        nodeid.split("::", 1)[0] if nodeid else result.get("test_file", "")
    )
    return result


def deduplicate_failure_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the last observation for each canonical case/process nodeid."""
    ordered: dict[str, dict[str, str]] = {}
    anonymous: list[dict[str, str]] = []
    for raw_row in rows:
        row = canonicalize_failure_row(raw_row)
        key = row.get("nodeid", "")
        if not key:
            anonymous.append(row)
            continue
        if key in ordered:
            del ordered[key]
        ordered[key] = row
    return anonymous + list(ordered.values())


def completed_recovery_files(text: str) -> set[str]:
    """Return files whose timeout/crash recovery reached a terminal state."""
    completed: set[str] = set()
    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        end = END_RE.match(line)
        if end and end.group(1) == "PASS":
            completed.add(end.group(2))
            continue
        for pattern in (STEPCURRENT_RECOVERY_DONE_RE, FALLBACK_RECOVERY_DONE_RE):
            match = pattern.match(line)
            if match:
                completed.add(match.group(1))
                break
    return completed


def collect_failures_from_logs(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for log_file in iter_log_files(path):
        with open(log_file, encoding="utf-8", errors="replace") as f:
            text = f.read()
        is_official = is_official_run_test_log(text)
        official_rows = extract_official_run_test_failures(text, log_file)
        summary_rows = extract_pytest_summary_failures(text, log_file, official=is_official)
        timeout_rows = extract_official_timeout_current_failures(text, log_file) if is_official else []
        if is_official:
            parsed_rows = official_rows if official_rows else summary_rows
            parsed_rows = parsed_rows + timeout_rows
        else:
            parsed_rows = summary_rows
        has_case_rows = any("::" in row.get("nodeid", "") for row in parsed_rows)
        raw_process_rows = extract_process_failures(text, log_file)
        recovered_files_in_log = completed_recovery_files(text)
        process_rows = [
            row
            for row in raw_process_rows
            if row["test_file"] not in recovered_files_in_log
        ]
        if not (is_official and has_case_rows):
            parsed_rows.extend(process_rows)
        if (
            not raw_process_rows
            and not recovered_files_in_log
            and not official_rows
            and not summary_rows
            and not timeout_rows
        ):
            parsed_rows.extend(extract_global_crashes(text, log_file))

        for row in parsed_rows:
            key = failure_row_key(row)
            if key not in seen:
                rows.append(row)
                seen.add(key)

    recovered_files = {
        row["test_file"]
        for row in rows
        if row["nodeid"].startswith(f"{row['test_file']}::")
    }
    return [
        row
        for row in rows
        if not (
            row["error_type"] in {"Crash", "Timeout"}
            and row["test_file"] in recovered_files
            and row["nodeid"] == row["test_file"]
        )
    ]


def write_failure_csv(rows: list[dict[str, str]], output_file: str) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_failure_json(rows: list[dict[str, str]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_failure_markdown(rows: list[dict[str, str]], output_file: str) -> None:
    by_file: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_file.setdefault(row["test_file"] or "<unparsed>", []).append(row)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Pytest Failure Report\n\n")
        f.write(f"Total failures: {len(rows)}\n\n")

        for test_file, items in sorted(by_file.items(), key=lambda item: (-len(item[1]), item[0])):
            f.write(f"## {test_file} ({len(items)})\n\n")
            for row in items:
                case = row["case_name"] or row["nodeid"] or "<unparsed>"
                if row["case_params"]:
                    case = f"{case}[{row['case_params']}]"
                if row["class_name"]:
                    case = f"{row['class_name']}::{case}"

                error = row["error_message"]
                if row["error_type"] and row["error_message"]:
                    error = f"{row['error_type']}: {row['error_message']}"
                elif row["error_type"]:
                    error = row["error_type"]
                elif not error:
                    error = row["raw"]

                f.write(f"- `{case}`")
                if row["gpu"]:
                    f.write(f" GPU {row['gpu']}")
                f.write("\n")
                f.write(f"  - `{error}`\n")
            f.write("\n")


def print_failure_summary(rows: list[dict[str, str]]) -> None:
    print(f"Total failures: {len(rows)}")

    by_file: dict[str, int] = {}
    by_error: dict[str, int] = {}
    for row in rows:
        by_file[row["test_file"] or "<unparsed>"] = by_file.get(row["test_file"] or "<unparsed>", 0) + 1
        error_key = row["error_type"] or "<unknown>"
        by_error[error_key] = by_error.get(error_key, 0) + 1

    if by_file:
        print("\nTop files:")
        for test_file, count in sorted(by_file.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"{count:5d}  {test_file}")

    if by_error:
        print("\nError types:")
        for error_type, count in sorted(by_error.items(), key=lambda item: (-item[1], item[0]))[:20]:
            print(f"{count:5d}  {error_type}")


def generate_failure_reports(log_dir: str) -> dict[str, str | int]:
    rows = collect_failures_from_logs(log_dir)
    rerun_dir = os.path.join(log_dir, "process_file_rerun")
    rerun_list = os.path.join(rerun_dir, "test_files.txt")
    rerun_summary = os.path.join(rerun_dir, "summary.json")
    # Only replace initial process-level rows after the rerun stage has a
    # completion marker. During an active/interrupted rerun, dropping those
    # rows would hide files that have not been retried yet.
    if os.path.isfile(rerun_list) and os.path.isfile(rerun_summary):
        with open(rerun_list, encoding="utf-8") as f:
            rerun_files = {line.strip() for line in f if line.strip()}
        rows = filter_stale_process_rows_after_file_rerun(rows, rerun_files, rerun_dir)
    external_merge = os.path.join(log_dir, "external_rerun_merge.json")
    if os.path.isfile(external_merge):
        try:
            with open(external_merge, encoding="utf-8") as f:
                metadata = json.load(f)
            selected = {
                canonical_test_name(test)
                for test in metadata.get("selected_files", [])
                if test
            }
            source_work_dir = metadata.get("source_work_dir", "")
            source_report = os.path.join(source_work_dir, "latest", "failure_report.csv")
            source_summary = os.path.join(source_work_dir, "summary.json")
            with open(source_summary, encoding="utf-8") as f:
                supplemental_summary = json.load(f)
            unresolved = supplemental_summary.get("failure_reports", {}).get(
                "unresolved_process_failure_count"
            )
            if selected and unresolved == 0 and os.path.isfile(source_report):
                with open(source_report, newline="", encoding="utf-8") as f:
                    supplemental_rows = list(csv.DictReader(f))
                rows = [
                    row
                    for row in rows
                    if canonical_test_name(row.get("test_file", "")) not in selected
                ] + [
                    row
                    for row in supplemental_rows
                    if canonical_test_name(row.get("test_file", "")) in selected
                ]
                rows = deduplicate_failure_rows(rows)
                print(
                    f"Applied external rerun merge: {len(selected)} files from {source_work_dir}"
                )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"WARNING: cannot apply external rerun merge {external_merge}: {exc}")
    return generate_failure_reports_from_rows(log_dir, rows)


def update_current_summary_reports(log_dir: str, reports: dict[str, str | int]) -> bool:
    """Sync analyze-only results when log_dir is the work directory's latest run."""
    work_dir = os.path.dirname(os.path.normpath(log_dir))
    latest = os.path.join(work_dir, "latest")
    summary_file = os.path.join(work_dir, "summary.json")
    if not os.path.exists(latest) or not os.path.isfile(summary_file):
        return False
    try:
        if not os.path.samefile(log_dir, latest):
            return False
        with open(summary_file, encoding="utf-8") as f:
            summary = json.load(f)
        summary["failure_reports"] = reports
        tmp_file = summary_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, summary_file)
        return True
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def generate_failure_reports_from_rows(log_dir: str, rows: list[dict[str, str]]) -> dict[str, str | int]:
    csv_file = os.path.join(log_dir, "failure_report.csv")
    json_file = os.path.join(log_dir, "failure_report.json")
    md_file = os.path.join(log_dir, "failure_report.md")
    unresolved_rows = [row for row in rows if is_process_level_failure(row)]
    unresolved_csv = os.path.join(log_dir, "unresolved_process_failures.csv")
    unresolved_json = os.path.join(log_dir, "unresolved_process_failures.json")
    unresolved_md = os.path.join(log_dir, "unresolved_process_failures.md")
    unresolved_files = os.path.join(log_dir, "unresolved_process_failure_files.txt")

    write_failure_csv(rows, csv_file)
    write_failure_json(rows, json_file)
    write_failure_markdown(rows, md_file)
    write_failure_csv(unresolved_rows, unresolved_csv)
    write_failure_json(unresolved_rows, unresolved_json)
    write_failure_markdown(unresolved_rows, unresolved_md)
    with open(unresolved_files, "w", encoding="utf-8") as f:
        seen: set[str] = set()
        for row in unresolved_rows:
            test_file = row.get("test_file", "")
            if test_file and test_file not in seen:
                f.write(test_file + "\n")
                seen.add(test_file)

    print_failure_summary(rows)
    print(f"\nFailure CSV:      {csv_file}")
    print(f"Failure JSON:     {json_file}")
    print(f"Failure Markdown: {md_file}")
    print(f"Unresolved process-level failures: {len(unresolved_rows)}")
    print(f"Unresolved CSV:   {unresolved_csv}")
    print(f"Unresolved files: {unresolved_files}")

    return {
        "failure_count": len(rows),
        "failure_csv": csv_file,
        "failure_json": json_file,
        "failure_markdown": md_file,
        "unresolved_process_failure_count": len(unresolved_rows),
        "unresolved_process_failure_csv": unresolved_csv,
        "unresolved_process_failure_json": unresolved_json,
        "unresolved_process_failure_markdown": unresolved_md,
        "unresolved_process_failure_files": unresolved_files,
    }


def is_process_level_failure(row: dict[str, str]) -> bool:
    return "::" not in row.get("nodeid", "") or row.get("case_name", "").startswith("<")


def select_process_failure_files(
    rows: list[dict[str, str]],
    test_dir: str,
    error_types: set[str],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for row in rows:
        test_file = row.get("test_file", "")
        if not test_file.endswith(".py"):
            continue
        if not is_process_level_failure(row):
            continue
        if row.get("error_type", "") not in error_types:
            continue
        if not os.path.isfile(os.path.join(test_dir, test_file)):
            continue
        if test_file not in seen:
            selected.append(test_file)
            seen.add(test_file)
    return selected


def filter_stale_process_rows_after_file_rerun(
    rows: list[dict[str, str]],
    rerun_files: set[str],
    rerun_dir: str,
) -> list[dict[str, str]]:
    if not rerun_files:
        return rows
    rerun_dir_abs = os.path.abspath(rerun_dir)
    filtered: list[dict[str, str]] = []
    for row in rows:
        source_log = os.path.abspath(row.get("source_log", ""))
        is_from_rerun = source_log.startswith(rerun_dir_abs + os.sep)
        if (
            row.get("test_file") in rerun_files
            and is_process_level_failure(row)
            and not is_from_rerun
        ):
            continue
        filtered.append(row)
    return filtered


def rerun_process_failure_files(
    *,
    tests: list[str],
    test_dir: str,
    log_dir: str,
    progress_file: str,
    gpu_ids: list[int],
    timeout: int,
    crash_recovery: bool,
    crash_chunk_size: int,
) -> tuple[list[dict], float, str]:
    rerun_dir = os.path.join(log_dir, "process_file_rerun")
    os.makedirs(rerun_dir, exist_ok=True)
    with open(os.path.join(rerun_dir, "test_files.txt"), "w", encoding="utf-8") as f:
        for test in tests:
            f.write(test + "\n")
    rerun_queue = make_work_queue(tests)
    results: list[dict] = []
    total_elapsed = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(tests))) as pool:
        futures = {
            pool.submit(
                run_gpu_tests,
                worker_idx,
                gpu_ids[worker_idx],
                rerun_queue,
                len(tests),
                test_dir,
                rerun_dir,
                progress_file,
                timeout,
                crash_recovery,
                crash_chunk_size,
            ): gpu_ids[worker_idx]
            for worker_idx in range(min(len(gpu_ids), len(tests)))
        }
        for fut in concurrent.futures.as_completed(futures):
            gpu_id = futures[fut]
            result = fut.result()
            results.append(result)
            total_elapsed = max(total_elapsed, result["elapsed"])
            print(
                f"[process-rerun GPU {gpu_id}] PASS={result['passed']:>4} "
                f"FAIL={result['failed']:>4} SKIP={result['skipped']:>4} "
                f"FILES={result.get('assigned', 0):>4} "
                f"elapsed={timedelta(seconds=int(result['elapsed']))}"
            )
    rerun_summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timeout": timeout,
        "tests": tests,
        "results": results,
        "elapsed": round(total_elapsed, 1),
        "elapsed_str": str(timedelta(seconds=int(total_elapsed))),
    }
    with open(os.path.join(rerun_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(rerun_summary, f, indent=2, ensure_ascii=False)
    return results, total_elapsed, rerun_dir


def generate_summary(
    work_dir: str,
    log_dir: str,
    results: list[dict],
    gpu_ids: list[int],
    total_elapsed: float,
    failure_reports: dict[str, str | int] | None = None,
) -> dict:
    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "log_dir": log_dir,
        "elapsed": round(total_elapsed, 1),
        "elapsed_str": str(timedelta(seconds=int(total_elapsed))),
        "num_gpus": len(gpu_ids),
        "gpu_ids": gpu_ids,
        "total": {"passed": 0, "failed": 0, "skipped": 0, "elapsed": 0},
        "case_stats": empty_case_stats(),
        "per_gpu": results,
        "all_failures": [],
        "failure_reports": failure_reports or {},
    }

    for result in results:
        summary["total"]["passed"] += result["passed"]
        summary["total"]["failed"] += result["failed"]
        summary["total"]["skipped"] += result["skipped"]
        summary["total"]["elapsed"] += result["elapsed"]
        summary["all_failures"].extend(result["failures"])
        add_case_stats(summary["case_stats"], result.get("case_stats", empty_case_stats()))

    failures_by_file = {}
    for name, error, _ in summary["all_failures"]:
        file_name = name.split("/")[-1] if "/" in name else name
        failures_by_file.setdefault(file_name, []).append(error)
    summary["failures_by_file"] = failures_by_file

    with open(os.path.join(work_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def print_final_report(summary: dict, progress: dict[str, dict]) -> None:
    total = summary["total"]
    pstats = compute_stats(progress) if progress else {"passed": 0, "failed": 0, "skipped": 0}

    print()
    print("=" * 70)
    print("Run summary")
    print("=" * 70)
    print(f"Current PASS: {total['passed']}")
    print(f"Current FAIL: {total['failed']}")
    print(f"Current SKIP: {total['skipped']}")
    print(f"Cumulative PASS: {pstats.get('passed', 0)}")
    print(f"Cumulative FAIL: {pstats.get('failed', 0)}")
    print(f"Cumulative SKIP: {pstats.get('skipped', 0)}")
    print(f"Elapsed: {timedelta(seconds=int(summary['elapsed']))}")

    case_stats = summary.get("case_stats", empty_case_stats())
    print()
    print("Case summary")
    print(f"Case TOTAL: {case_total(case_stats)}")
    print(f"Case PASS:  {case_stats.get('passed', 0)}")
    print(f"Case FAIL:  {case_stats.get('failed', 0)}")
    print(f"Case ERROR: {case_stats.get('errors', 0)}")
    print(f"Case SKIP:  {case_stats.get('skipped', 0)}")
    print(f"Case XFAIL: {case_stats.get('xfailed', 0)}")
    print(f"Case XPASS: {case_stats.get('xpassed', 0)}")
    print(f"Deselected: {case_stats.get('deselected', 0)}")
    print(f"Unknown:    {case_stats.get('unknown', 0)}")
    print(f"Pass rate:  {case_pass_rate(case_stats):.2f}%")

    failures = summary.get("all_failures", [])
    if failures:
        print(f"\nCurrent failures: {len(failures)}")
        by_file = summary.get("failures_by_file", {})
        top = sorted(by_file.items(), key=lambda x: -len(x[1]))[:8]
        for file_name, errors in top:
            print(f"  {file_name}: {len(errors)} failures")

    pending_fails = [k for k, v in progress.items() if v.get("status") == "FAIL"]
    if pending_fails:
        print(f"\nCumulative FAIL still pending retry: {len(pending_fails)}")

    log_dir = summary.get("log_dir", "")
    print(f"\nLog dir: {log_dir}")
    if log_dir:
        print(f"Summary: {os.path.join(os.path.dirname(log_dir), 'summary.json')}")

    reports = summary.get("failure_reports", {})
    if reports:
        print(f"Failure CSV: {reports.get('failure_csv', '')}")
        print(f"Failure Markdown: {reports.get('failure_markdown', '')}")
        print(f"Failure JSON: {reports.get('failure_json', '')}")


def main() -> None:
    setup_signal_handler()

    pass_through = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        pass_through = sys.argv[idx + 1 :]
        known = sys.argv[1:idx]
    else:
        known = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="PyTorch test runner: dry-run, GPU assignment, parallel execution, resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pytorch_root", help="PyTorch repo root")
    parser.add_argument("--fresh", action="store_true", help="ignore previous progress")
    parser.add_argument("--retry-fail", action="store_true", help="retry previous failures explicitly")
    parser.add_argument("--skip-fail", action="store_true", help="skip previous failures")
    parser.add_argument("--num-gpus", type=int, default=8, help="GPU count when --gpu-ids is not set")
    parser.add_argument("--gpu-ids", default=None, help="comma-separated GPU IDs, e.g. 0,2,5,7")
    parser.add_argument(
        "--include-prefix",
        default=None,
        help="comma-separated test path prefixes to keep, e.g. dynamo/,inductor/",
    )
    parser.add_argument(
        "--no-crash-recovery",
        action="store_true",
        help="disable adaptive chunk/bisect recovery for crashed test files",
    )
    parser.add_argument(
        "--crash-chunk-size",
        type=int,
        default=16,
        help="initial case chunk size used during crash recovery",
    )
    parser.add_argument("--dry-run-only", action="store_true", help="only generate test_files.txt")
    parser.add_argument(
        "--analyze-only",
        default=None,
        help="only analyze an existing log file or log directory, then exit",
    )
    parser.add_argument("--work-dir", default=None, help="work dir, default: <pytorch_root>/test_runs")
    parser.add_argument("--timeout", type=int, default=0, help="per-test timeout seconds, 0 means unlimited")
    parser.add_argument(
        "--process-rerun",
        dest="process_rerun",
        action="store_true",
        help="rerun process-level timeout/crash files after initial failure analysis",
    )
    parser.add_argument(
        "--no-process-rerun",
        dest="process_rerun",
        action="store_false",
        help="disable automatic rerun for process-level timeout/crash files",
    )
    parser.add_argument(
        "--process-rerun-error-types",
        default="Timeout,Crash",
        help="comma-separated process-level error types to rerun, e.g. Timeout or Timeout,Crash",
    )
    parser.add_argument(
        "--process-rerun-timeout",
        type=int,
        default=None,
        help="timeout seconds for automatic process-level file rerun; default max(7200, --timeout * 4)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print more details")
    parser.add_argument("--analyze", dest="analyze", action="store_true", help="analyze failure logs after run")
    parser.add_argument("--no-analyze", dest="analyze", action="store_false", help="disable failure log analysis")
    parser.set_defaults(analyze=True)
    parser.set_defaults(process_rerun=True)

    args, unknown = parser.parse_known_args(known)
    dry_run_extra = unknown + pass_through

    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids, args.num_gpus)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    if args.analyze_only:
        try:
            analyze_path = os.path.abspath(args.analyze_only)
            reports = generate_failure_reports(analyze_path)
            if update_current_summary_reports(analyze_path, reports):
                print(
                    "Updated summary: "
                    + os.path.join(os.path.dirname(os.path.normpath(analyze_path)), "summary.json")
                )
        except FileNotFoundError:
            print(f"ERROR: analyze path does not exist: {args.analyze_only}", file=sys.stderr)
            sys.exit(1)
        return

    pytorch_root = os.path.abspath(args.pytorch_root)
    test_dir = os.path.join(pytorch_root, "test")

    if not os.path.isdir(test_dir):
        print(f"ERROR: test dir does not exist: {test_dir}")
        sys.exit(1)

    work_dir = args.work_dir or os.path.join(pytorch_root, "test_runs")
    os.makedirs(work_dir, exist_ok=True)

    test_list_file = os.path.join(work_dir, "test_files.txt")
    progress_file = os.path.join(work_dir, ".test_progress.json")

    print(f"PyTorch root: {pytorch_root}")
    print(f"test dir:     {test_dir}")
    print(f"work dir:     {work_dir}")
    print(f"GPU IDs:      {','.join(str(x) for x in gpu_ids)}")
    print("Test env:     " + ", ".join(f"{k}={v}" for k, v in DEFAULT_TEST_ENV.items()))
    print(f"Crash recovery: {'off' if args.no_crash_recovery else 'on'}")
    print()

    print("=" * 60)
    print("Step 1: parse dry-run test list")
    print("=" * 60)

    combined = run_dry_run(test_dir, dry_run_extra)
    tests = parse_dry_run_output(combined)

    if not tests:
        print("ERROR: no tests parsed from dry-run output")
        print("--- raw output, first 2000 chars ---")
        print(combined[:2000])
        sys.exit(1)

    print(f"Parsed tests: {len(tests)}")

    include_prefixes = parse_csv_items(args.include_prefix)
    if include_prefixes:
        before_filter = len(tests)
        tests = filter_by_prefix(tests, include_prefixes)
        print(f"Include prefixes: {','.join(include_prefixes)}")
        print(f"Filtered tests: {len(tests)} / {before_filter}")
        if not tests:
            print("ERROR: no tests left after --include-prefix filtering")
            sys.exit(1)

    with open(test_list_file, "w", encoding="utf-8") as f:
        for test in tests:
            f.write(test + "\n")
    print(f"Test list saved: {test_list_file}")

    if args.dry_run_only:
        print("--dry-run-only complete.")
        return

    print()
    print("=" * 60)
    print("Step 2: load progress and filter")
    print("=" * 60)

    if args.fresh and os.path.isfile(progress_file):
        os.remove(progress_file)
        print("[--fresh] removed old progress file")

    progress = load_progress(progress_file)
    existing, not_found = filter_existing(tests, test_dir)
    if not_found and args.verbose:
        for skipped in not_found:
            print(f"  [not found] {skipped}")

    done_pass = []
    done_skip = []
    done_fail = []
    remaining = []

    for test in existing:
        if test in progress:
            status = progress[test].get("status", progress[test]) if isinstance(progress[test], dict) else progress[test]
            if status == "PASS":
                done_pass.append(test)
            elif status == "SKIP":
                done_skip.append(test)
            elif status == "FAIL":
                done_fail.append(test)
            else:
                remaining.append(test)
        else:
            remaining.append(test)

    if not args.skip_fail and done_fail:
        remaining.extend(done_fail)
        retry_count = len(done_fail)
    else:
        retry_count = 0

    total = len(existing)
    done_count = len(done_pass) + len(done_skip) + (len(done_fail) if args.skip_fail else 0)
    pct = done_count * 100 // total if total > 0 else 0

    print(f"Total parsed:      {len(tests)}")
    print(f"File not found:    {len(not_found)}")
    print(f"Done PASS:         {len(done_pass)}")
    print(f"Done SKIP:         {len(done_skip)}")
    print(f"Done FAIL:         {len(done_fail)}" + (" retry" if retry_count else " skip (--skip-fail)"))
    print(f"Need run:          {len(remaining)}")
    print(f"Overall progress:  {done_count}/{total} ({pct}%)")
    print(f"GPU IDs:           {','.join(str(x) for x in gpu_ids)}")
    if remaining:
        print(f"Dynamic queue:     on")
        print(f"Estimated per GPU: ~{len(remaining) // len(gpu_ids)} tests")

    if not remaining:
        print("\nAll tests completed.")
        print_final_report(
            {
                "total": {"passed": 0, "failed": 0, "skipped": 0, "elapsed": 0},
                "all_failures": [],
                "failures_by_file": {},
                "elapsed": 0,
                "log_dir": "",
            },
            progress,
        )
        return

    remaining = sort_tests_by_history(remaining, progress)
    work_queue = make_work_queue(remaining)
    if args.verbose:
        print("  dynamic queue order, first 20:")
        for test in remaining[:20]:
            elapsed = progress.get(test, {}).get("elapsed", 0) if isinstance(progress.get(test, {}), dict) else 0
            print(f"    {elapsed:>8}s  {test}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(work_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    latest_link = os.path.join(work_dir, "latest")
    try:
        if os.path.exists(latest_link):
            os.remove(latest_link)
        if sys.platform == "win32":
            with open(latest_link + ".txt", "w", encoding="utf-8") as f:
                f.write(log_dir)
        else:
            os.symlink(timestamp, latest_link)
    except OSError:
        pass

    print(f"\nLog dir:       {log_dir}")
    print(f"Progress file: {progress_file}")
    if args.timeout > 0:
        print(f"Per-test timeout: {args.timeout}s")
    print()
    print("=" * 60)
    print("Step 3: run tests in parallel")
    print("=" * 60)
    print()

    total_elapsed = 0.0
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(
                run_gpu_tests,
                worker_idx,
                gpu_ids[worker_idx],
                work_queue,
                len(remaining),
                test_dir,
                log_dir,
                progress_file,
                args.timeout,
                not args.no_crash_recovery,
                args.crash_chunk_size,
            ): gpu_ids[worker_idx]
            for worker_idx in range(min(len(gpu_ids), len(remaining)))
        }
        for fut in concurrent.futures.as_completed(futures):
            gpu_id = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                total_elapsed = max(total_elapsed, result["elapsed"])
                print(
                    f"[GPU {gpu_id}] PASS={result['passed']:>4} "
                    f"FAIL={result['failed']:>4} SKIP={result['skipped']:>4} "
                    f"FILES={result.get('assigned', 0):>4} "
                    f"cases={case_total(result.get('case_stats', empty_case_stats())):>5} "
                    f"elapsed={timedelta(seconds=int(result['elapsed']))}"
                )
            except Exception as exc:
                print(f"[GPU {gpu_id}] exception: {exc}")

    print()
    print("=" * 60)
    print("Step 4: analyze failure logs")
    print("=" * 60)

    failure_reports = {}
    process_rerun_results: list[dict] = []
    process_rerun_elapsed = 0.0
    if args.analyze:
        initial_rows = collect_failures_from_logs(log_dir)
        failure_reports = generate_failure_reports_from_rows(log_dir, initial_rows)

        if args.process_rerun:
            process_error_types = set(parse_csv_items(args.process_rerun_error_types))
            process_tests = select_process_failure_files(initial_rows, test_dir, process_error_types)
            if process_tests:
                process_timeout = (
                    args.process_rerun_timeout
                    if args.process_rerun_timeout is not None
                    else max(7200, args.timeout * 4 if args.timeout > 0 else 7200)
                )
                print()
                print("=" * 60)
                print("Step 4b: rerun process-level failure files")
                print("=" * 60)
                print(f"Process error types: {','.join(sorted(process_error_types))}")
                print(f"Files to rerun:       {len(process_tests)}")
                print(f"Rerun timeout:        {process_timeout}s")
                for test in process_tests:
                    print(f"  {test}")
                process_rerun_results, process_rerun_elapsed, process_rerun_dir = rerun_process_failure_files(
                    tests=process_tests,
                    test_dir=test_dir,
                    log_dir=log_dir,
                    progress_file=progress_file,
                    gpu_ids=gpu_ids,
                    timeout=process_timeout,
                    crash_recovery=not args.no_crash_recovery,
                    crash_chunk_size=args.crash_chunk_size,
                )
                final_rows = collect_failures_from_logs(log_dir)
                final_rows = filter_stale_process_rows_after_file_rerun(
                    final_rows,
                    set(process_tests),
                    process_rerun_dir,
                )
                failure_reports = generate_failure_reports_from_rows(log_dir, final_rows)
            else:
                print("No process-level failure files selected for rerun.")
        else:
            print("Process-level failure file rerun disabled.")
    else:
        print("Failure log analysis disabled.")

    print()
    print("=" * 60)
    print("Step 5: generate summary")
    print("=" * 60)

    if process_rerun_results:
        results.extend(process_rerun_results)
        total_elapsed += process_rerun_elapsed
    summary = generate_summary(work_dir, log_dir, results, gpu_ids, total_elapsed, failure_reports)
    progress = load_progress(progress_file)
    print_final_report(summary, progress)
    print()

    if args.skip_fail and done_fail:
        print(f"Tip: {len(done_fail)} previous failures were skipped. Use --retry-fail to rerun them.")

    if args.verbose:
        print(f"Work dir:      {work_dir}")
        print(f"Test list:     {test_list_file}")
        print(f"Progress file: {progress_file}")
        print(f"Summary file:  {os.path.join(work_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
