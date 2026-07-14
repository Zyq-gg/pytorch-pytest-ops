#!/usr/bin/env python3
"""
Queue-based runner for PyTorch's official test/run_test.py.

This keeps run_test.py as the execution entrypoint, but adds:
  * dry-run list generation
  * per-module progress checkpointing
  * dynamic GPU assignment
  * nohup-friendly logs and summary
  * resume after interruption
  * regex/failure-CSV subset selection
  * process-level timeout/crash module reruns
  * case-level and unresolved failure reports
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import queue
import re
import select
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from run_pytorch_tests_prefix import (
    collect_failures_from_logs,
    generate_failure_reports,
    generate_failure_reports_from_rows,
    is_process_level_failure,
)


DEFAULT_TEST_ENV = {
    "PYTORCH_TEST_WITH_ROCM": "1",
    "CONTINUE_THROUGH_ERROR": "True",
    "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC": "1",
}

TEST_LINE_RE = re.compile(r"^\s{2,}(.+?)\s+\d+/\d+\s*$")
BLOCK_RE = re.compile(r"^\s*(Serial|Parallel) tests\s*\(\d+\)\s*:")

_progress_lock = threading.Lock()


def make_env(gpu_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_TEST_ENV)
    if gpu_id is not None:
        env["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def parse_csv_items(raw: str | None) -> set[str]:
    return {item.strip() for item in (raw or "").split(",") if item.strip()}


def module_to_test_file(module: str) -> str:
    return module if module.endswith(".py") else module + ".py"


def test_file_to_module(test_file: str) -> str:
    module = test_file.removeprefix("test/")
    return module[:-3] if module.endswith(".py") else module


def load_failure_modules(
    csv_file: Path,
    *,
    process_only: bool,
    error_types: set[str],
) -> list[str]:
    modules: list[str] = []
    seen: set[str] = set()
    with csv_file.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if process_only and not is_process_level_failure(row):
                continue
            if error_types and row.get("error_type", "") not in error_types:
                continue
            test_file = row.get("test_file", "") or row.get("nodeid", "").split("::", 1)[0]
            if not test_file:
                continue
            module = test_file_to_module(test_file)
            if module and module not in seen:
                modules.append(module)
                seen.add(module)
    return modules


def filter_modules(
    tests: list[str],
    *,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[str]:
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    selected: list[str] = []
    seen: set[str] = set()
    for test in tests:
        if include and include.search(test) is None:
            continue
        if exclude and exclude.search(test) is not None:
            continue
        if test not in seen:
            selected.append(test)
            seen.add(test)
    return selected


def replace_rerun_module_rows(
    rows: list[dict[str, str]],
    rerun_modules: set[str],
    rerun_dir: Path,
) -> list[dict[str, str]]:
    """Use a complete module rerun as the sole source for that module."""
    rerun_dir_abs = str(rerun_dir.resolve())
    filtered: list[dict[str, str]] = []
    for row in rows:
        source_log = os.path.abspath(row.get("source_log", ""))
        module = test_file_to_module(row.get("test_file", ""))
        is_from_rerun = source_log.startswith(rerun_dir_abs + os.sep)
        if module in rerun_modules and not is_from_rerun:
            continue
        filtered.append(row)
    return filtered


def select_process_rerun_modules(
    tests: list[str],
    progress: dict[str, dict],
    rows: list[dict[str, str]],
    wanted_types: set[str],
) -> list[str]:
    """Select incomplete modules without relying on case-report granularity."""
    selected: set[str] = set()
    reported_modules = {
        test_file_to_module(row.get("test_file", "")) for row in rows if row.get("test_file")
    }
    if not wanted_types or "Timeout" in wanted_types:
        selected.update(
            module for module in tests if progress.get(module, {}).get("status") == "TIMEOUT"
        )
    for row in rows:
        if not is_process_level_failure(row):
            continue
        if wanted_types and row.get("error_type", "") not in wanted_types:
            continue
        module = test_file_to_module(row.get("test_file", ""))
        if module in tests:
            selected.add(module)
    if not wanted_types or "Crash" in wanted_types:
        selected.update(
            module
            for module in tests
            if progress.get(module, {}).get("status") == "FAIL"
            and module not in reported_modules
        )
    return [module for module in tests if module in selected]


def append_unreported_terminal_rows(
    rows: list[dict[str, str]], tests: list[str], progress: dict[str, dict]
) -> list[dict[str, str]]:
    """Ensure terminal process failures are visible as unresolved report rows."""
    result = list(rows)
    existing = {test_file_to_module(row.get("test_file", "")) for row in rows}
    for module in tests:
        item = progress.get(module, {})
        status = item.get("status")
        if status not in {"TIMEOUT", "FAIL"} or module in existing:
            continue
        if status == "TIMEOUT":
            case_name = "<timeout>"
            error_type = "Timeout"
            message = f"module did not reach a terminal run_test.py result ({item.get('elapsed', 0)}s)"
        else:
            case_name = "<process-failure>"
            error_type = "ProcessFailure"
            message = (
                "run_test.py returned nonzero without a parsable case failure "
                f"(rc={item.get('returncode', '')}, elapsed={item.get('elapsed', 0)}s)"
            )
        result.append(
            {
                "source_log": "",
                "gpu": "",
                "test_file": module_to_test_file(module),
                "class_name": "",
                "case_name": case_name,
                "case_params": "",
                "error_type": error_type,
                "error_message": message,
                "nodeid": module_to_test_file(module),
                "raw": message,
            }
        )
    return result


def write_module_coverage(
    work_dir: Path,
    tests: list[str],
    progress: dict[str, dict],
    unresolved_count: int,
) -> dict:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    timeout_modules: list[str] = []
    for module in tests:
        item = progress.get(module)
        status = item.get("status", "MISSING") if item else "MISSING"
        if status not in {"PASS", "FAIL", "TIMEOUT"}:
            missing.append(module)
        if status == "TIMEOUT":
            timeout_modules.append(module)
        rows.append(
            {
                "module": module,
                "status": status,
                "elapsed": item.get("elapsed", "") if item else "",
                "returncode": item.get("returncode", "") if item else "",
                "time": item.get("time", "") if item else "",
            }
        )
    csv_path = work_dir / "module_status.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "status", "elapsed", "returncode", "time"])
        writer.writeheader()
        writer.writerows(rows)
    incomplete = missing + timeout_modules
    (work_dir / "incomplete_modules.txt").write_text(
        "\n".join(incomplete) + ("\n" if incomplete else ""), encoding="utf-8"
    )
    coverage = {
        "planned": len(tests),
        "terminal": sum(1 for row in rows if row["status"] in {"PASS", "FAIL"}),
        "pass": sum(1 for row in rows if row["status"] == "PASS"),
        "fail": sum(1 for row in rows if row["status"] == "FAIL"),
        "timeout": len(timeout_modules),
        "missing": len(missing),
        "unresolved_process_failures": unresolved_count,
        "coverage_complete": not incomplete and unresolved_count == 0,
        "missing_modules": missing,
        "timeout_modules": timeout_modules,
        "module_status_csv": str(csv_path),
        "incomplete_modules_file": str(work_dir / "incomplete_modules.txt"),
    }
    (work_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return coverage


def parse_gpu_ids(raw: str) -> list[int]:
    gpu_ids = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"duplicate GPU IDs: {raw}")
    return gpu_ids


def strip_args_for_single_run(args: list[str]) -> list[str]:
    """Remove selection/dry-run args that conflict with --include <one test>."""
    out: list[str] = []
    i = 0
    while i < len(args):
        item = args[i]
        if item == "--dry-run":
            i += 1
            continue
        if item in ("--include", "-i"):
            i += 1
            while i < len(args) and not args[i].startswith("-"):
                i += 1
            continue
        out.append(item)
        i += 1
    return out


def run_dry_run(test_dir: Path, run_test_args: list[str]) -> str:
    cmd = [sys.executable, "run_test.py", "--dry-run"] + run_test_args
    result = subprocess.run(
        cmd,
        cwd=str(test_dir),
        env=make_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_test.py --dry-run failed with return code {result.returncode}\n{result.stdout[-4000:]}"
        )
    return result.stdout


def parse_dry_run_tests(text: str) -> list[str]:
    tests: list[str] = []
    in_tests = False
    in_excluded = False
    for line in text.splitlines():
        if line.startswith("Name: excluded"):
            in_excluded = True
            in_tests = False
            continue
        if line.startswith("Name:") and "excluded" not in line:
            in_excluded = False
            in_tests = False
            continue
        if in_excluded:
            continue
        if BLOCK_RE.match(line):
            in_tests = True
            continue
        if not in_tests:
            continue
        if line and not line.startswith((" ", "\t")):
            in_tests = False
            continue
        match = TEST_LINE_RE.match(line)
        if match:
            tests.append(match.group(1).strip())
    return tests


def write_latest(work_dir: Path, log_dir: Path) -> None:
    latest = work_dir / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_dir.name)
    except OSError:
        (work_dir / "latest.txt").write_text(str(log_dir), encoding="utf-8")


def load_progress(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get("tests", {}) if isinstance(data, dict) else {}


def save_progress(path: Path, test_name: str, status: str, elapsed: float, returncode: int | None) -> None:
    with _progress_lock:
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        tests = data.setdefault("tests", {})
        tests[test_name] = {
            "status": status,
            "elapsed": round(elapsed, 1),
            "returncode": returncode,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        stats = {
            "total": len(tests),
            "passed": sum(1 for v in tests.values() if v.get("status") == "PASS"),
            "failed": sum(1 for v in tests.values() if v.get("status") == "FAIL"),
            "timeout": sum(1 for v in tests.values() if v.get("status") == "TIMEOUT"),
        }
        stats["done"] = stats["passed"] + stats["failed"] + stats["timeout"]
        data["stats"] = stats
        data["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def run_one(
    *,
    worker_idx: int,
    gpu_id: int | None,
    work_queue: queue.Queue[tuple[int, str]],
    total: int,
    test_dir: Path,
    log_dir: Path,
    progress_file: Path,
    timeout: int,
    run_test_args: list[str],
) -> dict:
    passed = failed = timed_out = 0
    assigned = 0
    gpu_label = "all" if gpu_id is None else str(gpu_id)
    log_path = log_dir / f"run_test_gpu_{gpu_label}.log"
    single_args = strip_args_for_single_run(run_test_args)

    with log_path.open("a", buffering=1, encoding="utf-8") as log:
        log.write(f"===== worker {worker_idx}, GPU {gpu_label} =====\n")
        while True:
            try:
                index, test_name = work_queue.get_nowait()
            except queue.Empty:
                break
            assigned += 1
            start = time.time()
            ts = datetime.now().strftime("%H:%M:%S")
            cmd = [sys.executable, "run_test.py", "--include", test_name] + single_args
            log.write(f"===== {ts} START: {test_name} [{index + 1}/{total}] (GPU {gpu_label}) =====\n")
            log.write(f"===== command: {' '.join(cmd)} =====\n")
            returncode: int | None = None
            status = "FAIL"
            proc = subprocess.Popen(
                cmd,
                cwd=str(test_dir),
                env=make_env(gpu_id),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            try:
                assert proc.stdout is not None
                fd = proc.stdout.fileno()
                os.set_blocking(fd, False)
                deadline = time.time() + timeout if timeout > 0 else None
                while True:
                    if deadline is not None and time.time() > deadline:
                        status = "TIMEOUT"
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                    ready, _, _ = select.select([fd], [], [], 1)
                    if ready:
                        chunk = proc.stdout.read()
                        if chunk:
                            log.write(chunk)
                        elif proc.poll() is not None:
                            break
                    elif proc.poll() is not None:
                        break
                rest = proc.stdout.read()
                if rest:
                    log.write(rest)
                returncode = proc.wait()
                if status != "TIMEOUT":
                    status = "PASS" if returncode == 0 else "FAIL"
            except Exception as exc:  # noqa: BLE001
                log.write(f"\n===== runner exception: {type(exc).__name__}: {exc} =====\n")
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
                status = "FAIL"

            if status == "TIMEOUT":
                test_file = module_to_test_file(test_name)
                log.write("=========================== short test summary info ===========================\n")
                log.write(f"FAILED {test_file} - Timeout: exceeded {timeout}s\n")
                log.write("============================== 1 failed in 0.00s ==============================\n")

            elapsed = time.time() - start
            if status == "PASS":
                passed += 1
            elif status == "TIMEOUT":
                timed_out += 1
            else:
                failed += 1
            save_progress(progress_file, test_name, status, elapsed, returncode)
            log.write(
                f"===== {datetime.now().strftime('%H:%M:%S')} {status}: {test_name} "
                f"({elapsed:.1f}s, rc={returncode}) =====\n"
            )
            work_queue.task_done()

    return {
        "gpu": gpu_id,
        "assigned": assigned,
        "passed": passed,
        "failed": failed,
        "timeout": timed_out,
    }


def execute_queue(
    *,
    tests: list[str],
    gpu_ids: list[int | None],
    test_dir: Path,
    log_dir: Path,
    progress_file: Path,
    timeout: int,
    run_test_args: list[str],
) -> list[dict]:
    work_queue: queue.Queue[tuple[int, str]] = queue.Queue()
    for index, test in enumerate(tests):
        work_queue.put((index, test))
    results: list[dict] = []
    worker_ids = gpu_ids[: len(tests)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_ids)) as pool:
        futures = [
            pool.submit(
                run_one,
                worker_idx=i,
                gpu_id=gpu_id,
                work_queue=work_queue,
                total=len(tests),
                test_dir=test_dir,
                log_dir=log_dir,
                progress_file=progress_file,
                timeout=timeout,
                run_test_args=run_test_args,
            )
            for i, gpu_id in enumerate(worker_ids)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[GPU {result['gpu']}] PASS={result['passed']} FAIL={result['failed']} "
                f"TIMEOUT={result['timeout']} TESTS={result['assigned']}",
                flush=True,
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytorch_root")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument(
        "--no-bind-gpu",
        action="store_true",
        help="run one worker without overriding GPU visibility; useful for multi-GPU distributed modules",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-fail", action="store_true")
    parser.add_argument("--dry-run-only", action="store_true")
    parser.add_argument("--no-analyze", action="store_true", help="Do not generate failure_report.csv/json/md")
    parser.add_argument("--test-list", help="Reuse an existing official test list instead of dry-run")
    parser.add_argument("--include-regex", help="Keep official modules matching this regex")
    parser.add_argument("--exclude-regex", help="Exclude official modules matching this regex")
    parser.add_argument("--failure-csv", help="Select official modules referenced by an existing failure CSV")
    parser.add_argument(
        "--failure-process-only",
        action="store_true",
        help="with --failure-csv, select only process-level rows",
    )
    parser.add_argument(
        "--failure-error-types",
        default="",
        help="with --failure-csv, keep comma-separated error types; empty means all",
    )
    parser.add_argument("--process-rerun", dest="process_rerun", action="store_true")
    parser.add_argument("--no-process-rerun", dest="process_rerun", action="store_false")
    parser.add_argument("--process-rerun-error-types", default="Timeout,Crash")
    parser.add_argument("--process-rerun-timeout", type=int, default=43200)
    parser.add_argument(
        "--process-rerun-only",
        action="store_true",
        help="rerun incomplete modules in an existing work directory and rebuild its reports",
    )
    parser.add_argument("--allow-fail", action="store_true", help="exit 0 even if modules fail")
    parser.set_defaults(process_rerun=True)
    args, run_test_args = parser.parse_known_args()

    if run_test_args and run_test_args[0] == "--":
        run_test_args = run_test_args[1:]

    pytorch_root = Path(args.pytorch_root).resolve()
    test_dir = pytorch_root / "test"
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not test_dir.is_dir():
        raise SystemExit(f"test dir does not exist: {test_dir}")

    raw_dry_run = ""
    existing_test_list = work_dir / "run_test_tests.txt"
    if args.process_rerun_only:
        if args.fresh:
            raise SystemExit("--process-rerun-only cannot be combined with --fresh")
        if not existing_test_list.is_file():
            raise SystemExit(f"existing test list does not exist: {existing_test_list}")
        tests = [
            line.strip()
            for line in existing_test_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.failure_csv:
        failure_csv = Path(args.failure_csv).resolve()
        if not failure_csv.is_file():
            raise SystemExit(f"failure CSV does not exist: {failure_csv}")
        tests = load_failure_modules(
            failure_csv,
            process_only=args.failure_process_only,
            error_types=parse_csv_items(args.failure_error_types),
        )
    elif args.test_list:
        tests = [
            line.strip()
            for line in Path(args.test_list).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        raw_dry_run = run_dry_run(test_dir, run_test_args)
        (work_dir / "run_test_dry_run.out").write_text(raw_dry_run, encoding="utf-8")
        tests = parse_dry_run_tests(raw_dry_run)

    tests = filter_modules(
        tests,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    test_list_file = work_dir / "run_test_tests.txt"
    if not args.process_rerun_only:
        test_list_file.write_text("\n".join(tests) + ("\n" if tests else ""), encoding="utf-8")

    print(f"PyTorch root: {pytorch_root}", flush=True)
    print(f"test dir:     {test_dir}", flush=True)
    print(f"work dir:     {work_dir}", flush=True)
    print(f"test list:    {test_list_file}", flush=True)
    print(f"tests:        {len(tests)}", flush=True)
    print(f"run args:     {' '.join(run_test_args)}", flush=True)
    if args.failure_csv:
        print(f"failure CSV:  {Path(args.failure_csv).resolve()}", flush=True)
    if args.dry_run_only:
        return 0
    if not tests:
        raise SystemExit("no tests selected")

    progress_file = work_dir / ".run_test_progress.json"
    if args.fresh and progress_file.exists():
        progress_file.unlink()
    progress = load_progress(progress_file)

    remaining: list[str] = []
    done_pass = done_fail = done_timeout = 0
    for test in tests:
        item = progress.get(test, {})
        status = item.get("status")
        if status == "PASS":
            done_pass += 1
        elif status == "FAIL":
            done_fail += 1
            if not args.skip_fail:
                remaining.append(test)
        elif status == "TIMEOUT":
            done_timeout += 1
            if not args.skip_fail:
                remaining.append(test)
        else:
            remaining.append(test)

    gpu_ids: list[int | None] = [None] if args.no_bind_gpu else parse_gpu_ids(args.gpu_ids)
    print(f"Done PASS:    {done_pass}", flush=True)
    print(f"Done FAIL:    {done_fail}" + (" skip" if args.skip_fail else " retry"), flush=True)
    print(f"Done TIMEOUT: {done_timeout}" + (" skip" if args.skip_fail else " retry"), flush=True)
    print(f"Need run:     {len(remaining)}", flush=True)
    print(
        "GPU IDs:      " + ("inherited/all (single worker)" if args.no_bind_gpu else ",".join(str(x) for x in gpu_ids)),
        flush=True,
    )
    if not remaining and not args.process_rerun_only:
        print("All selected tests completed.", flush=True)
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.process_rerun_only:
        latest = work_dir / "latest"
        if latest.is_symlink() or latest.is_dir():
            log_dir = latest.resolve()
        elif (work_dir / "latest.txt").is_file():
            log_dir = Path((work_dir / "latest.txt").read_text(encoding="utf-8").strip())
        else:
            raise SystemExit(f"existing latest result does not exist under {work_dir}")
    else:
        log_dir = work_dir / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        write_latest(work_dir, log_dir)
    print(f"log dir:      {log_dir}", flush=True)
    print(f"progress:     {progress_file}", flush=True)

    start = time.time()
    results = []
    if not args.process_rerun_only:
        results = execute_queue(
            tests=remaining,
            gpu_ids=gpu_ids,
            test_dir=test_dir,
            log_dir=log_dir,
            progress_file=progress_file,
            timeout=args.timeout,
            run_test_args=run_test_args,
        )

    progress = load_progress(progress_file)
    failure_reports: dict = {}
    if not args.no_analyze:
        failure_reports = generate_failure_reports(str(log_dir))

    process_rerun_modules: list[str] = []
    process_rerun_results: list[dict] = []
    process_rerun_dir: Path | None = None
    if args.process_rerun and not args.no_analyze:
        initial_rows = collect_failures_from_logs(str(log_dir))
        wanted_types = parse_csv_items(args.process_rerun_error_types)
        progress = load_progress(progress_file)
        process_rerun_modules = select_process_rerun_modules(
            tests, progress, initial_rows, wanted_types
        )
        if process_rerun_modules:
            process_rerun_name = "process_module_rerun"
            if args.process_rerun_only and (log_dir / process_rerun_name).exists():
                process_rerun_name += "_" + timestamp
            process_rerun_dir = log_dir / process_rerun_name
            process_rerun_dir.mkdir(parents=True, exist_ok=True)
            (process_rerun_dir / "run_test_tests.txt").write_text(
                "\n".join(process_rerun_modules) + "\n", encoding="utf-8"
            )
            print("\n===== process-level module rerun =====", flush=True)
            print(f"Modules: {len(process_rerun_modules)}", flush=True)
            print(f"Timeout: {args.process_rerun_timeout}s", flush=True)
            process_rerun_results = execute_queue(
                tests=process_rerun_modules,
                gpu_ids=gpu_ids,
                test_dir=test_dir,
                log_dir=process_rerun_dir,
                progress_file=progress_file,
                timeout=args.process_rerun_timeout,
                run_test_args=run_test_args,
            )
            (process_rerun_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "modules": process_rerun_modules,
                        "timeout": args.process_rerun_timeout,
                        "results": process_rerun_results,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            progress = load_progress(progress_file)
            final_rows = collect_failures_from_logs(str(log_dir))
            final_rows = replace_rerun_module_rows(
                final_rows, set(process_rerun_modules), process_rerun_dir
            )
            final_rows = append_unreported_terminal_rows(final_rows, tests, progress)
            failure_reports = generate_failure_reports_from_rows(str(log_dir), final_rows)

    progress = load_progress(progress_file)
    if not args.no_analyze and not process_rerun_modules:
        final_rows = collect_failures_from_logs(str(log_dir))
        final_rows = append_unreported_terminal_rows(final_rows, tests, progress)
        failure_reports = generate_failure_reports_from_rows(str(log_dir), final_rows)
    unresolved_count = failure_reports.get("unresolved_process_failure_count", 0)
    coverage = write_module_coverage(work_dir, tests, progress, unresolved_count)

    summary = {
        "pytorch_root": str(pytorch_root),
        "work_dir": str(work_dir),
        "log_dir": str(log_dir),
        "test_list": str(test_list_file),
        "progress_file": str(progress_file),
        "run_test_args": run_test_args,
        "gpu_ids": gpu_ids,
        "elapsed_seconds": round(time.time() - start, 1),
        "worker_results": results,
        "process_rerun_modules": process_rerun_modules,
        "process_rerun_results": process_rerun_results,
        "process_rerun_dir": str(process_rerun_dir) if process_rerun_dir else "",
        "failure_reports": failure_reports,
        "coverage": coverage,
        "coverage_complete": coverage["coverage_complete"],
        "progress_stats": {
            "total_selected": len(tests),
            "completed_records": len(progress),
            "pass": sum(1 for v in progress.values() if v.get("status") == "PASS"),
            "fail": sum(1 for v in progress.values() if v.get("status") == "FAIL"),
            "timeout": sum(1 for v in progress.values() if v.get("status") == "TIMEOUT"),
        },
    }
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("summary:", work_dir / "summary.json", flush=True)
    print(json.dumps(summary["progress_stats"], indent=2), flush=True)
    print("coverage:", json.dumps(coverage, indent=2), flush=True)
    has_failures = summary["progress_stats"]["fail"] > 0 or summary["progress_stats"]["timeout"] > 0
    return 0 if args.allow_fail or not has_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
