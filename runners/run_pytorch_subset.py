#!/usr/bin/env python3
"""
Run selected PyTorch tests without disturbing the full-test runner.

Subcommands:
  pytest-list  Run pytest files selected from an existing test_files.txt.
  pytest-failure-files
               Run pytest files from process-level rows in failure_report.csv.
  run-test     Run PyTorch's official test/run_test.py with arbitrary args.
  run-test-resume
               Run official run_test.py module-by-module with checkpoint resume.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_pytorch_tests_prefix import (  # noqa: E402
    DEFAULT_TEST_ENV,
    canonical_test_name,
    compute_stats,
    deduplicate_failure_rows,
    empty_case_stats,
    filter_existing,
    generate_failure_reports,
    generate_failure_reports_from_rows,
    generate_summary,
    load_progress,
    make_work_queue,
    parse_csv_items,
    parse_gpu_ids,
    print_final_report,
    run_gpu_tests,
    save_progress,
    sort_tests_by_history,
    update_current_summary_reports,
)


def make_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_TEST_ENV)
    return env


def read_test_list(path: Path) -> list[str]:
    tests = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if not item.endswith(".py"):
            item += ".py"
        tests.append(item)
    return tests


def filter_tests(
    tests: list[str],
    *,
    include_prefix: str | None,
    exclude_prefix: str | None,
    include_regex: str | None,
    exclude_regex: str | None,
) -> list[str]:
    include_prefixes = parse_csv_items(include_prefix)
    exclude_prefixes = parse_csv_items(exclude_prefix)
    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    selected = []
    for test in tests:
        if include_prefixes and not any(test.startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes and any(test.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if include_re and not include_re.search(test):
            continue
        if exclude_re and exclude_re.search(test):
            continue
        selected.append(test)
    return selected


def read_process_level_failure_files(
    failure_csv: Path,
    *,
    error_types: str | None,
) -> list[str]:
    wanted = {item.strip() for item in parse_csv_items(error_types)}
    use_all_errors = not wanted or any(item.lower() == "all" for item in wanted)

    tests: list[str] = []
    seen: set[str] = set()
    with failure_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_file = (row.get("test_file") or row.get("nodeid") or "").strip()
            nodeid = (row.get("nodeid") or "").strip()
            case_name = (row.get("case_name") or "").strip()
            error_type = (row.get("error_type") or "").strip()

            is_process_level = "::" not in nodeid or case_name.startswith("<")
            if not is_process_level:
                continue
            if not use_all_errors and error_type not in wanted:
                continue
            if not test_file.endswith(".py"):
                continue
            if test_file not in seen:
                tests.append(test_file)
                seen.add(test_file)
    return tests


def read_failure_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def publish_pytest_failure_file_results(
    *,
    source_work_dir: Path,
    target_work_dir: Path,
    selected_tests: list[str],
) -> None:
    """Replace selected files in a full run's report with completed rerun results."""
    source_latest = source_work_dir / "latest"
    source_summary = source_work_dir / "summary.json"
    source_progress = source_work_dir / ".test_progress.json"
    source_report = source_latest / "failure_report.csv"
    target_latest = target_work_dir / "latest"
    target_report = target_latest / "failure_report.csv"

    required = [source_summary, source_progress, source_report, target_report]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("cannot publish; required completed output is missing: " + ", ".join(missing))

    progress = load_progress(str(source_progress))
    incomplete = []
    for test in selected_tests:
        item = progress.get(test)
        status = item.get("status") if isinstance(item, dict) else item
        if status not in {"PASS", "FAIL", "SKIP"}:
            incomplete.append(test)
    if incomplete:
        raise SystemExit(
            f"cannot publish; {len(incomplete)} selected files have no terminal checkpoint: "
            + ", ".join(incomplete[:20])
        )

    with source_summary.open(encoding="utf-8") as f:
        summary = json.load(f)
    unresolved = summary.get("failure_reports", {}).get("unresolved_process_failure_count")
    if unresolved != 0:
        raise SystemExit(
            "cannot publish; supplemental report still has "
            f"{unresolved!r} unresolved process-level failures"
        )

    selected = {canonical_test_name(test) for test in selected_tests}
    base_rows = [
        row
        for row in read_failure_rows(target_report)
        if canonical_test_name(row.get("test_file", "")) not in selected
    ]
    rerun_rows = [
        row
        for row in read_failure_rows(source_report)
        if canonical_test_name(row.get("test_file", "")) in selected
    ]
    merged_rows = deduplicate_failure_rows(base_rows + rerun_rows)
    reports = generate_failure_reports_from_rows(str(target_latest), merged_rows)
    if not update_current_summary_reports(str(target_latest), reports):
        raise SystemExit("published reports, but failed to update target summary.json")

    metadata = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "source_work_dir": str(source_work_dir),
        "target_work_dir": str(target_work_dir),
        "selected_files": sorted(selected),
        "base_rows_kept": len(base_rows),
        "supplemental_rows": len(rerun_rows),
        "final_rows": len(merged_rows),
        "failure_reports": reports,
    }
    metadata_file = target_latest / "external_rerun_merge.json"
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Published final report: {target_latest / 'failure_report.csv'}")
    print(f"Published unresolved:   {reports['unresolved_process_failure_count']}")
    print(f"Merge metadata:         {metadata_file}")


def update_latest(work_dir: Path, log_dir: Path) -> None:
    latest = work_dir / "latest"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_dir.name)
    except OSError:
        (work_dir / "latest.txt").write_text(str(log_dir), encoding="utf-8")


RUN_TEST_ITEM_RE = re.compile(r"^\s{4}(.+?)\s+\d+/\d+\s*$")


def parse_run_test_dry_run_output(text: str) -> list[str]:
    tests: list[str] = []
    in_excluded = False
    in_test_block = False

    for line in text.splitlines():
        if line.startswith("Name: excluded"):
            in_excluded = True
            in_test_block = False
            continue
        if line.startswith("Name:") and "excluded" not in line:
            in_excluded = False
            in_test_block = False
            continue
        if in_excluded:
            continue
        if re.match(r"^\s*(Serial|Parallel) tests\s*\(\d+\)\s*:", line):
            in_test_block = True
            continue
        if not in_test_block:
            continue

        match = RUN_TEST_ITEM_RE.match(line)
        if match:
            tests.append(match.group(1))
            continue
        stripped = line.strip()
        if stripped and not line.startswith(" "):
            in_test_block = False

    return tests


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "test"


def remove_dry_run_arg(args: list[str]) -> list[str]:
    return [arg for arg in args if arg != "--dry-run"]


def ensure_no_include_arg(args: list[str]) -> None:
    if "--include" in args or "-i" in args:
        raise SystemExit("run-test-resume manages --include itself; pass filters such as --distributed-tests instead")


def run_command_to_log(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    timeout: int,
    echo: bool = True,
    failure_nodeid: str | None = None,
) -> tuple[int, float, str]:
    start = time.time()
    timed_out = False
    with log_file.open("w", buffering=1, encoding="utf-8") as log:
        log.write(f"===== command: {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        while True:
            if timeout > 0 and time.time() - start > timeout:
                timed_out = True
                break

            ready, _, _ = select.select([fd], [], [], 1)
            if ready:
                chunk = proc.stdout.read()
                if chunk:
                    if echo:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    log.write(chunk)
                elif proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    if echo:
                        sys.stdout.write(rest)
                        sys.stdout.flush()
                    log.write(rest)
                break

        if timed_out:
            timed_out = True
            log.write(f"\n===== TIMEOUT after {timeout}s; killing process group =====\n")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
            if failure_nodeid:
                log.write("=========================== short test summary info ===========================\n")
                log.write(f"FAILED {failure_nodeid} - Timeout: exceeded {timeout}s\n")
                log.write("============================== 1 failed in 0.00s ==============================\n")
        else:
            returncode = proc.wait()

    elapsed = time.time() - start
    return returncode, elapsed, "TIMEOUT" if timed_out else ("PASS" if returncode == 0 else "FAIL")


def run_pytest_list(args: argparse.Namespace) -> None:
    pytorch_root = Path(args.pytorch_root).resolve()
    test_dir = pytorch_root / "test"
    test_list = Path(args.test_list).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.is_dir():
        raise SystemExit(f"test dir does not exist: {test_dir}")
    if not test_list.is_file():
        raise SystemExit(f"test list does not exist: {test_list}")

    tests = read_test_list(test_list)
    tests = filter_tests(
        tests,
        include_prefix=args.include_prefix,
        exclude_prefix=args.exclude_prefix,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    selected_file = work_dir / "selected_test_files.txt"
    selected_file.write_text("\n".join(tests) + ("\n" if tests else ""), encoding="utf-8")

    print(f"PyTorch root:   {pytorch_root}")
    print(f"test dir:       {test_dir}")
    print(f"source list:    {test_list}")
    print(f"work dir:       {work_dir}")
    print(f"selected tests: {len(tests)}")
    print(f"selected list:  {selected_file}")
    if args.dry_run_only:
        return
    if not tests:
        raise SystemExit("no tests selected")

    gpu_ids = parse_gpu_ids(args.gpu_ids, args.num_gpus)
    progress_file = work_dir / ".test_progress.json"
    if args.fresh and progress_file.exists():
        progress_file.unlink()

    progress = load_progress(str(progress_file))
    existing, missing = filter_existing(tests, str(test_dir))
    remaining = []
    done_fail = []
    done_pass = []
    done_skip = []
    for test in existing:
        item = progress.get(test)
        status = item.get("status") if isinstance(item, dict) else item
        if status == "PASS":
            done_pass.append(test)
        elif status == "SKIP":
            done_skip.append(test)
        elif status == "FAIL":
            done_fail.append(test)
        else:
            remaining.append(test)
    if args.skip_fail:
        pass
    else:
        remaining.extend(done_fail)

    print(f"Existing files: {len(existing)}")
    print(f"Missing files:  {len(missing)}")
    print(f"Done PASS:      {len(done_pass)}")
    print(f"Done SKIP:      {len(done_skip)}")
    print(f"Done FAIL:      {len(done_fail)}" + (" skip" if args.skip_fail else " retry"))
    print(f"Need run:       {len(remaining)}")
    print(f"GPU IDs:        {','.join(str(x) for x in gpu_ids)}")

    if not remaining:
        print("All selected tests completed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = work_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    update_latest(work_dir, log_dir)
    print(f"Log dir:        {log_dir}")
    print(f"Progress file:  {progress_file}")

    work_queue = make_work_queue(sort_tests_by_history(remaining, progress))
    start = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {
            pool.submit(
                run_gpu_tests,
                worker_idx,
                gpu_ids[worker_idx],
                work_queue,
                len(remaining),
                str(test_dir),
                str(log_dir),
                str(progress_file),
                args.timeout,
                not args.no_crash_recovery,
                args.crash_chunk_size,
            ): gpu_ids[worker_idx]
            for worker_idx in range(min(len(gpu_ids), len(remaining)))
        }
        for fut in concurrent.futures.as_completed(futures):
            gpu_id = futures[fut]
            result = fut.result()
            results.append(result)
            print(
                f"[GPU {gpu_id}] PASS={result['passed']} FAIL={result['failed']} "
                f"SKIP={result['skipped']} FILES={result.get('assigned', 0)}",
                flush=True,
            )

    failure_reports = generate_failure_reports(str(log_dir)) if args.analyze else {}
    summary = generate_summary(
        str(work_dir),
        str(log_dir),
        results,
        gpu_ids,
        max((r["elapsed"] for r in results), default=time.time() - start),
        failure_reports,
    )
    progress = load_progress(str(progress_file))
    print_final_report(summary, progress)


def run_pytest_failure_files(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    failure_csv = Path(args.failure_csv).resolve()
    if not failure_csv.is_file():
        raise SystemExit(f"failure csv does not exist: {failure_csv}")

    tests = read_process_level_failure_files(
        failure_csv,
        error_types=args.error_type,
    )
    tests = filter_tests(
        tests,
        include_prefix=args.include_prefix,
        exclude_prefix=args.exclude_prefix,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
    )

    generated_list = work_dir / "failure_process_test_files.txt"
    generated_list.write_text("\n".join(tests) + ("\n" if tests else ""), encoding="utf-8")

    print(f"Failure CSV:         {failure_csv}")
    print(f"Process error types: {args.error_type}")
    print(f"Generated test list: {generated_list}")
    print(f"Selected test files: {len(tests)}")
    print(f"File rerun timeout:  {args.timeout}s")

    delegated_args = argparse.Namespace(**vars(args))
    delegated_args.test_list = str(generated_list)
    delegated_args.include_prefix = None
    delegated_args.exclude_prefix = None
    delegated_args.include_regex = None
    delegated_args.exclude_regex = None
    run_pytest_list(delegated_args)
    if args.publish_to_work_dir and not args.dry_run_only:
        publish_pytest_failure_file_results(
            source_work_dir=work_dir,
            target_work_dir=Path(args.publish_to_work_dir).resolve(),
            selected_tests=tests,
        )


def run_official_run_test(args: argparse.Namespace, run_test_args: list[str]) -> None:
    pytorch_root = Path(args.pytorch_root).resolve()
    test_dir = pytorch_root / "test"
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.is_dir():
        raise SystemExit(f"test dir does not exist: {test_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = work_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    update_latest(work_dir, log_dir)

    log_file = log_dir / "run_test.log"
    summary_file = work_dir / "summary.json"
    cmd = [sys.executable, "run_test.py"] + run_test_args

    print(f"PyTorch root: {pytorch_root}")
    print(f"test dir:     {test_dir}")
    print(f"work dir:     {work_dir}")
    print(f"log dir:      {log_dir}")
    print(f"log file:     {log_file}")
    print(f"command:      {' '.join(cmd)}")

    start = time.time()
    with log_file.open("w", buffering=1, encoding="utf-8") as log:
        log.write(f"===== command: {' '.join(cmd)} =====\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(test_dir),
            env=make_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = proc.wait()

    elapsed = time.time() - start
    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": round(elapsed, 1),
        "elapsed_str": str(timedelta(seconds=int(elapsed))),
        "command": cmd,
        "returncode": returncode,
        "log_dir": str(log_dir),
        "log_file": str(log_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nrun_test.py finished: returncode={returncode}, elapsed={summary['elapsed_str']}")
    print(f"Summary: {summary_file}")
    if returncode != 0:
        raise SystemExit(returncode)


def run_official_run_test_resume(args: argparse.Namespace, run_test_args: list[str]) -> None:
    ensure_no_include_arg(run_test_args)

    pytorch_root = Path(args.pytorch_root).resolve()
    test_dir = pytorch_root / "test"
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.is_dir():
        raise SystemExit(f"test dir does not exist: {test_dir}")

    progress_file = work_dir / ".run_test_progress.json"
    if args.fresh and progress_file.exists():
        progress_file.unlink()

    dry_run_args = list(run_test_args)
    if "--dry-run" not in dry_run_args:
        dry_run_args = ["--dry-run"] + dry_run_args
    dry_run_cmd = [sys.executable, "run_test.py"] + dry_run_args

    dry_run_log = work_dir / "run_test_dry_run.log"
    print(f"PyTorch root:     {pytorch_root}")
    print(f"test dir:         {test_dir}")
    print(f"work dir:         {work_dir}")
    print(f"dry-run command:  {' '.join(dry_run_cmd)}")
    returncode, elapsed, status = run_command_to_log(
        dry_run_cmd,
        cwd=test_dir,
        env=make_env(),
        log_file=dry_run_log,
        timeout=args.dry_run_timeout,
        echo=not args.quiet_dry_run,
    )
    if returncode != 0:
        raise SystemExit(f"dry-run failed: returncode={returncode}, log={dry_run_log}")

    dry_run_text = dry_run_log.read_text(encoding="utf-8", errors="replace")
    tests = parse_run_test_dry_run_output(dry_run_text)
    if args.include_regex:
        include_re = re.compile(args.include_regex)
        tests = [test for test in tests if include_re.search(test)]
    if args.exclude_regex:
        exclude_re = re.compile(args.exclude_regex)
        tests = [test for test in tests if not exclude_re.search(test)]
    if args.limit is not None:
        tests = tests[: args.limit]
    tests_file = work_dir / "run_test_modules.txt"
    tests_file.write_text("\n".join(tests) + ("\n" if tests else ""), encoding="utf-8")

    print(f"dry-run status:   {status}, elapsed={timedelta(seconds=int(elapsed))}")
    print(f"selected modules: {len(tests)}")
    print(f"module list:      {tests_file}")
    if args.dry_run_only:
        return
    if not tests:
        raise SystemExit("no run_test.py modules selected")

    progress = load_progress(str(progress_file))
    remaining: list[str] = []
    done_pass: list[str] = []
    done_fail: list[str] = []
    done_skip: list[str] = []
    for test in tests:
        item = progress.get(test)
        item_status = item.get("status") if isinstance(item, dict) else item
        if item_status == "PASS":
            done_pass.append(test)
        elif item_status == "SKIP":
            done_skip.append(test)
        elif item_status in {"FAIL", "TIMEOUT"}:
            done_fail.append(test)
        else:
            remaining.append(test)
    if not args.skip_fail:
        remaining.extend(done_fail)

    print(f"Done PASS:        {len(done_pass)}")
    print(f"Done SKIP:        {len(done_skip)}")
    print(f"Done FAIL/TIMEOUT:{len(done_fail)}" + (" skip" if args.skip_fail else " retry"))
    print(f"Need run:         {len(remaining)}")
    print(f"Progress file:    {progress_file}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = work_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    update_latest(work_dir, log_dir)
    print(f"log dir:          {log_dir}")

    run_args = remove_dry_run_arg(run_test_args)
    results = []
    start = time.time()
    for index, test in enumerate(remaining, 1):
        log_file = log_dir / f"{index:04d}_{sanitize_name(test)}.log"
        cmd = [sys.executable, "run_test.py", "--include", test] + run_args
        print(f"===== START {index}/{len(remaining)}: {test} =====", flush=True)
        returncode, elapsed, status = run_command_to_log(
            cmd,
            cwd=test_dir,
            env=make_env(),
            log_file=log_file,
            timeout=args.timeout,
            echo=True,
            failure_nodeid=test,
        )
        print(
            f"===== {status}: {test} returncode={returncode} "
            f"elapsed={timedelta(seconds=int(elapsed))} log={log_file} =====",
            flush=True,
        )
        save_progress(str(progress_file), test, status, elapsed)
        results.append(
            {
                "test": test,
                "status": status,
                "returncode": returncode,
                "elapsed": round(elapsed, 1),
                "elapsed_str": str(timedelta(seconds=int(elapsed))),
                "log_file": str(log_file),
            }
        )
        if args.analyze and status != "PASS":
            generate_failure_reports(str(log_dir))

    progress = load_progress(str(progress_file))
    tests_progress = progress
    passed = sum(1 for t in tests if tests_progress.get(t, {}).get("status") == "PASS")
    failed_tests = [
        t
        for t in tests
        if tests_progress.get(t, {}).get("status") in {"FAIL", "TIMEOUT"}
    ]
    skipped = sum(1 for t in tests if tests_progress.get(t, {}).get("status") == "SKIP")
    remaining_count = len(tests) - passed - len(failed_tests) - skipped

    failures_csv = log_dir / "run_test_failures.csv"
    with failures_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "status", "elapsed", "log_file"])
        writer.writeheader()
        for test in failed_tests:
            item = tests_progress.get(test, {})
            writer.writerow(
                {
                    "test": test,
                    "status": item.get("status", ""),
                    "elapsed": item.get("elapsed", ""),
                    "log_file": next((r["log_file"] for r in results if r["test"] == test), ""),
                }
            )

    failure_reports = generate_failure_reports(str(log_dir)) if args.analyze else {}

    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed": round(time.time() - start, 1),
        "elapsed_str": str(timedelta(seconds=int(time.time() - start))),
        "dry_run_command": dry_run_cmd,
        "run_args": run_args,
        "total": len(tests),
        "passed": passed,
        "failed": len(failed_tests),
        "skipped": skipped,
        "remaining": remaining_count,
        "log_dir": str(log_dir),
        "progress_file": str(progress_file),
        "module_list": str(tests_file),
        "failures_csv": str(failures_csv),
        "failure_reports": failure_reports,
        "results_this_run": results,
    }
    summary_file = work_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== run-test-resume summary =====")
    print(f"Total:     {len(tests)}")
    print(f"PASS:      {passed}")
    print(f"FAIL:      {len(failed_tests)}")
    print(f"SKIP:      {skipped}")
    print(f"Remaining: {remaining_count}")
    print(f"Summary:   {summary_file}")
    print(f"Failures:  {failures_csv}")
    if failure_reports:
        print(f"Failure CSV: {failure_reports.get('failure_csv', '')}")
    if failed_tests and not args.allow_fail:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected PyTorch test subsets.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    pytest_parser = subparsers.add_parser("pytest-list", help="Run pytest files from test_files.txt")
    pytest_parser.add_argument("pytorch_root")
    pytest_parser.add_argument("--test-list", required=True)
    pytest_parser.add_argument("--work-dir", required=True)
    pytest_parser.add_argument("--include-prefix", default=None, help="comma-separated prefixes, e.g. inductor/,dynamo/")
    pytest_parser.add_argument("--exclude-prefix", default=None)
    pytest_parser.add_argument("--include-regex", default=None)
    pytest_parser.add_argument("--exclude-regex", default=None)
    pytest_parser.add_argument("--gpu-ids", default=None)
    pytest_parser.add_argument("--num-gpus", type=int, default=8)
    pytest_parser.add_argument("--timeout", type=int, default=0)
    pytest_parser.add_argument("--fresh", action="store_true")
    pytest_parser.add_argument("--skip-fail", action="store_true")
    pytest_parser.add_argument("--dry-run-only", action="store_true")
    pytest_parser.add_argument("--no-crash-recovery", action="store_true")
    pytest_parser.add_argument("--crash-chunk-size", type=int, default=16)
    pytest_parser.add_argument("--analyze", dest="analyze", action="store_true")
    pytest_parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    pytest_parser.set_defaults(analyze=True)

    failure_files_parser = subparsers.add_parser(
        "pytest-failure-files",
        help="Run pytest files referenced by process-level failure_report.csv rows",
    )
    failure_files_parser.add_argument("pytorch_root")
    failure_files_parser.add_argument("--failure-csv", required=True)
    failure_files_parser.add_argument("--work-dir", required=True)
    failure_files_parser.add_argument(
        "--publish-to-work-dir",
        default=None,
        help=(
            "after a complete unresolved-free rerun, replace these files in the "
            "target full-run failure report and update its summary.json"
        ),
    )
    failure_files_parser.add_argument(
        "--error-type",
        default="Timeout,Crash",
        help="comma-separated process error types to keep, e.g. Timeout,Crash; use all for every type",
    )
    failure_files_parser.add_argument("--include-prefix", default=None)
    failure_files_parser.add_argument("--exclude-prefix", default=None)
    failure_files_parser.add_argument("--include-regex", default=None)
    failure_files_parser.add_argument("--exclude-regex", default=None)
    failure_files_parser.add_argument("--gpu-ids", default=None)
    failure_files_parser.add_argument("--num-gpus", type=int, default=8)
    failure_files_parser.add_argument("--timeout", type=int, default=7200)
    failure_files_parser.add_argument("--fresh", action="store_true")
    failure_files_parser.add_argument("--skip-fail", action="store_true")
    failure_files_parser.add_argument("--dry-run-only", action="store_true")
    failure_files_parser.add_argument("--no-crash-recovery", action="store_true")
    failure_files_parser.add_argument("--crash-chunk-size", type=int, default=16)
    failure_files_parser.add_argument("--analyze", dest="analyze", action="store_true")
    failure_files_parser.add_argument("--no-analyze", dest="analyze", action="store_false")
    failure_files_parser.set_defaults(analyze=True)

    run_test_parser = subparsers.add_parser("run-test", help="Run official test/run_test.py")
    run_test_parser.add_argument("pytorch_root")
    run_test_parser.add_argument("--work-dir", required=True)

    run_test_resume_parser = subparsers.add_parser(
        "run-test-resume",
        help="Run official test/run_test.py module-by-module with checkpoint resume",
    )
    run_test_resume_parser.add_argument("pytorch_root")
    run_test_resume_parser.add_argument("--work-dir", required=True)
    run_test_resume_parser.add_argument("--timeout", type=int, default=0, help="per-module timeout seconds; 0 disables")
    run_test_resume_parser.add_argument("--dry-run-timeout", type=int, default=0, help="dry-run timeout seconds; 0 disables")
    run_test_resume_parser.add_argument("--fresh", action="store_true", help="remove existing checkpoint before running")
    run_test_resume_parser.add_argument("--skip-fail", action="store_true", help="skip previously failed/timed-out modules")
    run_test_resume_parser.add_argument("--dry-run-only", action="store_true", help="only generate module list")
    run_test_resume_parser.add_argument("--quiet-dry-run", action="store_true", help="do not echo dry-run output to stdout")
    run_test_resume_parser.add_argument("--allow-fail", action="store_true", help="exit 0 even if some modules fail")
    run_test_resume_parser.add_argument("--include-regex", default=None, help="filter dry-run modules by regex")
    run_test_resume_parser.add_argument("--exclude-regex", default=None, help="exclude dry-run modules by regex")
    run_test_resume_parser.add_argument("--limit", type=int, default=None, help="run only the first N selected modules")
    run_test_resume_parser.add_argument("--analyze", dest="analyze", action="store_true", help="generate case-level failure_report.csv/json/md")
    run_test_resume_parser.add_argument("--no-analyze", dest="analyze", action="store_false", help="skip case-level failure report generation")
    run_test_resume_parser.set_defaults(analyze=True)

    if len(sys.argv) > 1 and sys.argv[1] in {"run-test", "run-test-resume"}:
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            known = sys.argv[1:idx]
            run_test_args = sys.argv[idx + 1 :]
        else:
            known = sys.argv[1:]
            run_test_args = []
        args = parser.parse_args(known)
        if args.cmd == "run-test":
            run_official_run_test(args, run_test_args)
        else:
            run_official_run_test_resume(args, run_test_args)
        return

    args = parser.parse_args()
    if args.cmd == "pytest-list":
        run_pytest_list(args)
    elif args.cmd == "pytest-failure-files":
        run_pytest_failure_files(args)
    else:
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
