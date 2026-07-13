#!/usr/bin/env python3
"""
Rerun failed pytest cases from a failure_report.csv and emit stable failures.

The input CSV is the failure_report.csv produced by run_pytorch_tests_prefix.py.
Each unique nodeid is rerun until it passes or reaches --attempts consecutive
failures. Only cases that fail every attempt are written to stable_failures.csv.
"""

from __future__ import annotations

import argparse
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
from pathlib import Path


DEFAULT_TEST_ENV = {
    "PYTORCH_TEST_WITH_ROCM": "1",
    "CONTINUE_THROUGH_ERROR": "True",
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": "1",
}

CRASH_RE = re.compile(
    r"(Fatal Python error: .+|Segmentation fault|SIGSEGV|SIGABRT|core dumped|"
    r"Aborted|Bus error|Illegal instruction|Floating point exception|"
    r"HSA_STATUS_ERROR[^\s:]*|MEMORY_APERTURE_VIOLATION|KERNEL VMFault)",
    re.IGNORECASE,
)
FAILED_RE = re.compile(r"^FAILED(?:\s+\[[^\]]+\])?\s+(.+?)(?:\s+-\s+(.+))?$")
SHORT_SUMMARY_RE = re.compile(r"\bshort test summary info\b", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

INPUT_FIELDS = [
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

RESULT_FIELDS = [
    "nodeid",
    "test_file",
    "class_name",
    "case_name",
    "case_params",
    "stable_failed",
    "attempts_run",
    "statuses",
    "returncodes",
    "error_types",
    "error_messages",
    "rerun_gpu",
    "rerun_log",
    "original_error_type",
    "original_error_message",
    "original_source_log",
    "original_raw",
]


def make_env(gpu_id: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_TEST_ENV)
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def parse_gpu_ids(raw: str | None) -> list[int | None]:
    if raw is None or raw.strip() == "":
        return [None]

    gpu_ids: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            gpu_ids.extend(range(int(start), int(end) + 1))
        else:
            gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("--gpu-ids did not contain any GPU id")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpu-ids contains duplicates: {raw}")
    return gpu_ids


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def run_with_pty(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str]:
    """Run a command in its own process group and capture merged stdout/stderr."""
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
    deadline = time.monotonic() + timeout if timeout > 0 else None
    status = "EXIT"

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            status = "TIMEOUT"
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            break

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
    returncode = proc.wait()
    output = b"".join(output_parts).decode("utf-8", errors="replace").replace("\r", "\n")
    return returncode, output, status


def extract_failure(output: str, returncode: int, run_status: str) -> tuple[str, str, str]:
    clean = strip_ansi(output)
    if run_status == "TIMEOUT":
        return "TIMEOUT", "Timeout", "pytest attempt timed out"

    crash = CRASH_RE.search(clean)
    if returncode < 0 or crash:
        message = crash.group(1) if crash else f"terminated by signal {-returncode}"
        return "CRASH", "Crash", message

    if returncode == 0:
        return "PASS", "", ""

    in_summary = False
    for raw_line in clean.splitlines():
        line = raw_line.strip()
        if SHORT_SUMMARY_RE.search(line):
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.startswith("FAILED "):
            match = FAILED_RE.match(line)
            if match:
                error = match.group(2) or ""
                if ": " in error:
                    error_type, error_message = error.split(": ", 1)
                else:
                    error_type, error_message = "", error
                return "FAIL", error_type, error_message
            return "FAIL", "Failed", line

    for raw_line in reversed(clean.splitlines()):
        line = raw_line.strip()
        if any(key in line for key in ["Error", "Exception", "Assertion", "assert "]):
            if ": " in line:
                error_type, error_message = line.split(": ", 1)
                return "FAIL", error_type[-120:], error_message
            return "FAIL", "Failed", line

    tail = "\n".join(clean.strip().splitlines()[-5:])
    return "FAIL", "Failed", tail[:500]


def row_matches_filter(
    row: dict[str, str],
    *,
    column: str | None,
    equals: str | None,
    contains: str | None,
    regex: str | None,
) -> bool:
    if not column:
        return True
    value = row.get(column, "")
    if equals is not None and value != equals:
        return False
    if contains is not None and contains not in value:
        return False
    if regex is not None and re.search(regex, value) is None:
        return False
    return True


def normalize_input_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "source_log": row.get("source_log", "") or row.get("original_source_log", ""),
        "gpu": row.get("gpu", "") or row.get("rerun_gpu", ""),
        "test_file": row.get("test_file", ""),
        "class_name": row.get("class_name", ""),
        "case_name": row.get("case_name", ""),
        "case_params": row.get("case_params", ""),
        "error_type": row.get("error_type", "") or row.get("original_error_type", "") or row.get("error_types", ""),
        "error_message": (
            row.get("error_message", "")
            or row.get("original_error_message", "")
            or row.get("error_messages", "")
        ),
        "nodeid": row.get("nodeid", ""),
        "raw": row.get("raw", "") or row.get("original_raw", ""),
    }


def load_failure_rows(
    csv_file: Path,
    include_process_level: bool,
    *,
    filter_column: str | None = None,
    filter_equals: str | None = None,
    filter_contains: str | None = None,
    filter_regex: str | None = None,
) -> list[dict[str, str]]:
    with csv_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if filter_column and rows and filter_column not in rows[0]:
        available = ", ".join(rows[0].keys())
        raise SystemExit(f"filter column {filter_column!r} not found. Available columns: {available}")

    by_nodeid: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row_matches_filter(
            row,
            column=filter_column,
            equals=filter_equals,
            contains=filter_contains,
            regex=filter_regex,
        ):
            continue
        nodeid = (row.get("nodeid") or "").strip()
        if not nodeid:
            continue
        case_name = row.get("case_name", "")
        if not include_process_level and ("::" not in nodeid or case_name.startswith("<")):
            continue
        by_nodeid.setdefault(nodeid, normalize_input_row(row))
    return list(by_nodeid.values())


def target_exists(test_dir: Path, nodeid: str) -> bool:
    test_file = nodeid.split("::", 1)[0].removeprefix("test/")
    return (test_dir / test_file).is_file()


def rerun_one(
    row: dict[str, str],
    *,
    test_dir: Path,
    gpu_id: int | None,
    attempts: int,
    timeout: int,
    log,
    extra_pytest_args: list[str],
    runner: str,
    official_run_test_args: list[str],
) -> dict[str, str]:
    nodeid = row["nodeid"]
    statuses: list[str] = []
    returncodes: list[str] = []
    error_types: list[str] = []
    error_messages: list[str] = []

    for attempt in range(1, attempts + 1):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"===== {ts} START attempt {attempt}/{attempts}: {nodeid} =====\n")
        log.flush()

        if runner == "official":
            test_file, _ = nodeid.split("::", 1)
            module = test_file.removeprefix("test/")
            if module.endswith(".py"):
                module = module[:-3]
            selector = nodeid.removeprefix("test/")
            cmd = [
                sys.executable,
                "run_test.py",
                "--include",
                module,
                "--continue-through-error",
                "--verbose",
            ] + official_run_test_args + ["--pytest-single-test", selector]
        else:
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "--tb=long",
                "--color=no",
                "-q",
                nodeid,
            ] + extra_pytest_args

        start = time.time()
        returncode, output, run_status = run_with_pty(
            cmd,
            cwd=str(test_dir),
            env=make_env(gpu_id),
            timeout=timeout,
        )
        elapsed = time.time() - start
        status, error_type, error_message = extract_failure(output, returncode, run_status)

        log.write(output)
        if output and not output.endswith("\n"):
            log.write("\n")
        log.write(
            f"===== END attempt {attempt}/{attempts}: status={status} "
            f"returncode={returncode} elapsed={elapsed:.1f}s =====\n\n"
        )
        log.flush()

        statuses.append(status)
        returncodes.append(str(returncode))
        error_types.append(error_type)
        error_messages.append(error_message)

        if status == "PASS":
            break

    stable_failed = len(statuses) == attempts and all(status != "PASS" for status in statuses)
    return {
        "nodeid": nodeid,
        "test_file": row.get("test_file", ""),
        "class_name": row.get("class_name", ""),
        "case_name": row.get("case_name", ""),
        "case_params": row.get("case_params", ""),
        "stable_failed": "yes" if stable_failed else "no",
        "attempts_run": str(len(statuses)),
        "statuses": "|".join(statuses),
        "returncodes": "|".join(returncodes),
        "error_types": "|".join(error_types),
        "error_messages": " || ".join(error_messages),
        "rerun_gpu": "" if gpu_id is None else str(gpu_id),
        "rerun_log": getattr(log, "name", ""),
        "original_error_type": row.get("error_type", ""),
        "original_error_message": row.get("error_message", ""),
        "original_source_log": row.get("source_log", ""),
        "original_raw": row.get("raw", ""),
    }


def worker(
    *,
    worker_idx: int,
    gpu_id: int | None,
    work_queue: queue.Queue[dict[str, str]],
    test_dir: Path,
    log_dir: Path,
    attempts: int,
    timeout: int,
    extra_pytest_args: list[str],
    runner: str,
    official_run_test_args: list[str],
    results: list[dict[str, str]],
    lock: threading.Lock,
) -> None:
    log_name = f"rerun_gpu_{gpu_id}.log" if gpu_id is not None else f"rerun_worker_{worker_idx}.log"
    log_path = log_dir / log_name
    with log_path.open("w", buffering=1, encoding="utf-8") as log:
        while True:
            try:
                row = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = rerun_one(
                    row,
                    test_dir=test_dir,
                    gpu_id=gpu_id,
                    attempts=attempts,
                    timeout=timeout,
                    log=log,
                    extra_pytest_args=extra_pytest_args,
                    runner=runner,
                    official_run_test_args=official_run_test_args,
                )
                with lock:
                    results.append(result)
                    done = len(results)
                verdict = "STABLE_FAIL" if result["stable_failed"] == "yes" else "NOT_STABLE"
                print(f"[{done}] GPU {gpu_id}: {verdict} {row['nodeid']}", flush=True)
            finally:
                work_queue.task_done()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {field: row.get(field, "") for field in RESULT_FIELDS}
            for row in reader
            if row.get("nodeid")
        ]


def append_checkpoint(path: Path, row: dict[str, str], lock: threading.Lock) -> None:
    with lock:
        exists = path.is_file() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun failure_report.csv cases and write cases that fail N consecutive times."
    )
    parser.add_argument("pytorch_root", help="PyTorch repo root, e.g. /workspace/pytorch")
    parser.add_argument("failure_csv", help="failure_report.csv from the original run")
    parser.add_argument("--attempts", type=int, default=3, help="consecutive failures required")
    parser.add_argument("--timeout", type=int, default=600, help="timeout seconds per case attempt")
    parser.add_argument("--gpu-ids", default=None, help="comma/range GPU ids, e.g. 0,1,2 or 0-7")
    parser.add_argument("--output-dir", default=None, help="default: <failure_csv_dir>/stable_rerun_<timestamp>")
    parser.add_argument("--stable-csv", default="stable_failures.csv", help="stable failure CSV filename")
    parser.add_argument("--all-results-csv", default="rerun_all_results.csv", help="all rerun results CSV filename")
    parser.add_argument("--include-process-level", action="store_true", help="rerun rows whose nodeid is only a file")
    parser.add_argument("--limit", type=int, default=0, help="only rerun first N unique rows")
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoint in output dir")
    parser.add_argument("--dry-run-list", action="store_true", help="only parse and print planned reruns")
    parser.add_argument("--pytest-arg", action="append", default=[], help="extra pytest arg, repeatable")
    parser.add_argument(
        "--runner",
        choices=["pytest", "official"],
        default="pytest",
        help="execute cases with direct pytest or official test/run_test.py",
    )
    parser.add_argument(
        "--official-run-test-arg",
        action="append",
        default=[],
        help="extra run_test.py argument in --runner official mode; repeatable",
    )
    parser.add_argument("--filter-column", default=None, help="input CSV column used for row filtering")
    parser.add_argument("--filter-equals", default=None, help="keep rows whose filter column exactly equals this value")
    parser.add_argument("--filter-contains", default=None, help="keep rows whose filter column contains this value")
    parser.add_argument("--filter-regex", default=None, help="keep rows whose filter column matches this regex")
    args = parser.parse_args()

    if args.attempts <= 0:
        raise SystemExit("--attempts must be > 0")
    active_filters = [args.filter_equals is not None, args.filter_contains is not None, args.filter_regex is not None]
    if any(active_filters) and not args.filter_column:
        raise SystemExit("--filter-column is required when using --filter-equals/--filter-contains/--filter-regex")
    if sum(active_filters) > 1:
        raise SystemExit("use only one of --filter-equals, --filter-contains, --filter-regex")

    pytorch_root = Path(args.pytorch_root).resolve()
    test_dir = pytorch_root / "test"
    failure_csv = Path(args.failure_csv).resolve()
    if not test_dir.is_dir():
        raise SystemExit(f"test dir does not exist: {test_dir}")
    if not failure_csv.is_file():
        raise SystemExit(f"failure csv does not exist: {failure_csv}")

    rows = load_failure_rows(
        failure_csv,
        include_process_level=args.include_process_level,
        filter_column=args.filter_column,
        filter_equals=args.filter_equals,
        filter_contains=args.filter_contains,
        filter_regex=args.filter_regex,
    )
    missing = [row for row in rows if not target_exists(test_dir, row["nodeid"])]
    rows = [row for row in rows if row not in missing]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Failure CSV:       {failure_csv}")
    print(f"PyTorch test dir:  {test_dir}")
    print(f"Unique rerun rows: {len(rows)}")
    print(f"Missing targets:   {len(missing)}")
    print(f"Attempts:          {args.attempts}")
    print(f"Timeout:           {args.timeout}s")
    print(f"Runner:            {args.runner}")
    if args.filter_column:
        print(f"Filter column:     {args.filter_column}")
        if args.filter_equals is not None:
            print(f"Filter equals:     {args.filter_equals}")
        if args.filter_contains is not None:
            print(f"Filter contains:   {args.filter_contains}")
        if args.filter_regex is not None:
            print(f"Filter regex:      {args.filter_regex}")

    if args.dry_run_list:
        for row in rows[:50]:
            print(row["nodeid"])
        if len(rows) > 50:
            print(f"... {len(rows) - 50} more")
        return

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else failure_csv.parent / f"stable_rerun_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = output_dir / "rerun_checkpoint.csv"

    checkpoint_rows: list[dict[str, str]] = []
    if args.fresh and checkpoint_csv.exists():
        checkpoint_csv.unlink()
    elif not args.fresh:
        checkpoint_rows = load_checkpoint(checkpoint_csv)
        completed = {row["nodeid"] for row in checkpoint_rows}
        if completed:
            before = len(rows)
            rows = [row for row in rows if row["nodeid"] not in completed]
            print(f"Checkpoint:        {len(completed)} completed, {before - len(rows)} skipped")

    work_queue: queue.Queue[dict[str, str]] = queue.Queue()
    for row in rows:
        work_queue.put(row)

    results: list[dict[str, str]] = list(checkpoint_rows)
    lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    start = time.time()

    def checkpointing_worker(**kwargs) -> None:
        original_results = kwargs.pop("results")
        local_results: list[dict[str, str]] = []

        class ResultProxy(list):
            def append(self, item):  # type: ignore[no-untyped-def]
                append_checkpoint(checkpoint_csv, item, checkpoint_lock)
                original_results.append(item)
                local_results.append(item)

            def __len__(self):  # type: ignore[no-untyped-def]
                return len(original_results)

        kwargs["results"] = ResultProxy()
        worker(**kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [
            pool.submit(
                checkpointing_worker,
                worker_idx=idx,
                gpu_id=gpu_id,
                work_queue=work_queue,
                test_dir=test_dir,
                log_dir=output_dir,
                attempts=args.attempts,
                timeout=args.timeout,
                extra_pytest_args=args.pytest_arg,
                runner=args.runner,
                official_run_test_args=args.official_run_test_arg,
                results=results,
                lock=lock,
            )
            for idx, gpu_id in enumerate(gpu_ids)
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    stable_rows = [row for row in results if row["stable_failed"] == "yes"]
    all_csv = output_dir / args.all_results_csv
    stable_csv = output_dir / args.stable_csv
    write_csv(all_csv, sorted(results, key=lambda row: row["nodeid"]))
    write_csv(stable_csv, sorted(stable_rows, key=lambda row: row["nodeid"]))

    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": str(timedelta(seconds=int(time.time() - start))),
        "failure_csv": str(failure_csv),
        "output_dir": str(output_dir),
        "attempts": args.attempts,
        "timeout": args.timeout,
        "gpu_ids": gpu_ids,
        "runner": args.runner,
        "official_run_test_args": args.official_run_test_arg,
        "filter_column": args.filter_column,
        "filter_equals": args.filter_equals,
        "filter_contains": args.filter_contains,
        "filter_regex": args.filter_regex,
        "total_rerun": len(results),
        "stable_failures": len(stable_rows),
        "not_stable_or_passed": len(results) - len(stable_rows),
        "missing_targets": len(missing),
        "stable_csv": str(stable_csv),
        "all_results_csv": str(all_csv),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Stable rerun complete")
    print(f"Elapsed:       {summary['elapsed']}")
    print(f"Total rerun:   {len(results)}")
    print(f"Stable failed: {len(stable_rows)}")
    print(f"Stable CSV:    {stable_csv}")
    print(f"All results:   {all_csv}")
    print(f"Output dir:    {output_dir}")


if __name__ == "__main__":
    main()
