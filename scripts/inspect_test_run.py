#!/usr/bin/env python3
"""Read-only summary for PyTorch pytest and official run_test work directories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


TERMINAL = {"PASS", "FAIL", "SKIP", "TIMEOUT"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_plan(path: Path) -> list[str]:
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []


def read_failure_counts(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "rows": None,
        "unique_nodeids": None,
        "process_level": None,
        "error_types": {},
    }
    if not path.is_file():
        return result
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return result
    result["rows"] = len(rows)
    result["unique_nodeids"] = len({row.get("nodeid", "") for row in rows if row.get("nodeid")})
    result["process_level"] = sum(
        "::" not in row.get("nodeid", "") or row.get("case_name", "").startswith("<")
        for row in rows
    )
    result["error_types"] = dict(Counter(row.get("error_type", "") or "<unknown>" for row in rows))
    return result


def process_lines(work_dir: Path) -> tuple[list[str], list[str]]:
    try:
        output = subprocess.check_output(["ps", "-ef"], text=True, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return [], []
    needle = str(work_dir)
    work_dir_arg = re.compile(rf"(?<!\S){re.escape(needle)}(?!\S)")
    names = (
        "run_pytorch_tests_prefix.py",
        "run_pytorch_subset.py",
        "rerun_stable_failures.py",
        "run_official_run_test_queue.py",
        "run_test_official",
    )
    all_runners = [
        line
        for line in output.splitlines()
        if any(name in line for name in names)
        and "inspect_test_run.py" not in line
    ]
    return [line for line in all_runners if work_dir_arg.search(line)], all_runners


def inspect(work_dir: Path) -> dict[str, Any]:
    work_dir = work_dir.resolve()
    latest = work_dir / "latest"
    progress_path = work_dir / ".test_progress.json"
    mode = "pytest"
    if not progress_path.is_file():
        progress_path = work_dir / ".run_test_progress.json"
        mode = "official-run-test" if progress_path.is_file() else "unknown"

    plan_candidates = (
        ["test_files.txt", "selected_test_files.txt", "failure_process_test_files.txt"]
        if mode == "pytest"
        else ["run_test_tests.txt", "run_test_modules.txt"]
    )
    plan_path = next((work_dir / name for name in plan_candidates if (work_dir / name).is_file()), None)
    plan = read_plan(plan_path) if plan_path else []

    progress = load_json(progress_path) or {}
    tests = progress.get("tests", progress)
    if not isinstance(tests, dict):
        tests = {}
    statuses = Counter()
    for value in tests.values():
        status = value.get("status") if isinstance(value, dict) else value
        statuses[str(status or "UNKNOWN").upper()] += 1
    missing = [item for item in plan if item not in tests]
    nonterminal = []
    for item, value in tests.items():
        status = value.get("status") if isinstance(value, dict) else value
        if str(status or "").upper() not in TERMINAL:
            nonterminal.append(item)

    summary = load_json(work_dir / "summary.json")
    latest_target = str(latest.resolve()) if latest.exists() else None
    report = read_failure_counts(latest / "failure_report.csv")
    unresolved = read_failure_counts(latest / "unresolved_process_failures.csv")
    reports = summary.get("failure_reports", {}) if summary else {}
    coverage = load_json(work_dir / "coverage_report.json")
    nested_rerun = latest / "process_file_rerun"
    official_reruns = sorted(latest.glob("process_module_rerun*")) if latest.exists() else []

    workdir_processes, all_runner_processes = process_lines(work_dir)
    return {
        "work_dir": str(work_dir),
        "mode": mode,
        "plan_file": str(plan_path) if plan_path else None,
        "plan_count": len(plan),
        "progress_file": str(progress_path) if progress_path.is_file() else None,
        "progress_count": len(tests),
        "progress_updated": progress.get("updated"),
        "statuses": dict(statuses),
        "missing_count": len(missing),
        "missing_first_20": missing[:20],
        "nonterminal_count": len(nonterminal),
        "latest": latest_target,
        "summary_exists": summary is not None,
        "summary_unresolved": reports.get("unresolved_process_failure_count"),
        "coverage": coverage,
        "failure_report": report,
        "unresolved_report": unresolved,
        "process_rerun_exists": nested_rerun.is_dir(),
        "process_rerun_summary_exists": (nested_rerun / "summary.json").is_file(),
        "official_process_reruns": [str(path) for path in official_reruns],
        "official_process_rerun_summaries": sum(
            (path / "summary.json").is_file() for path in official_reruns
        ),
        "local_workdir_processes": workdir_processes,
        "local_all_test_runner_processes": all_runner_processes,
    }


def print_text(data: dict[str, Any]) -> None:
    print(f"Work dir:             {data['work_dir']}")
    print(f"Mode:                 {data['mode']}")
    print(f"Plan:                 {data['plan_file']} ({data['plan_count']})")
    print(f"Progress:             {data['progress_file']} ({data['progress_count']})")
    print(f"Progress updated:     {data['progress_updated']}")
    print(f"Statuses:             {data['statuses']}")
    print(f"Missing from progress:{data['missing_count']:>6}")
    for item in data["missing_first_20"]:
        print(f"  MISSING {item}")
    print(f"Nonterminal entries:  {data['nonterminal_count']}")
    print(f"Latest:               {data['latest']}")
    print(f"Summary exists:       {data['summary_exists']}")
    print(f"Summary unresolved:   {data['summary_unresolved']}")
    coverage = data["coverage"]
    if coverage:
        print(
            "Coverage:             "
            f"complete={coverage.get('coverage_complete')} planned={coverage.get('planned')} "
            f"terminal={coverage.get('terminal')} timeout={coverage.get('timeout')} "
            f"missing={coverage.get('missing')}"
        )
    report = data["failure_report"]
    print(
        "Failure report:        "
        f"exists={report['exists']} rows={report['rows']} "
        f"unique={report['unique_nodeids']} process_level={report['process_level']}"
    )
    unresolved = data["unresolved_report"]
    print(
        "Unresolved report:     "
        f"exists={unresolved['exists']} rows={unresolved['rows']}"
    )
    print(
        "Process file rerun:    "
        f"exists={data['process_rerun_exists']} summary={data['process_rerun_summary_exists']}"
    )
    print(
        "Official module rerun: "
        f"runs={len(data['official_process_reruns'])} "
        f"summaries={data['official_process_rerun_summaries']}"
    )
    processes = data["local_workdir_processes"]
    print(f"Local work-dir procs: {len(processes)}")
    for line in processes[:20]:
        print(f"  {line}")
    all_processes = data["local_all_test_runner_processes"]
    print(f"Local all runners:    {len(all_processes)}")
    if not processes and all_processes:
        print("  Other local test runners exist, but none contains this work-dir.")
    print("Note: process inspection only covers this machine.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    data = inspect(args.work_dir)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_text(data)


if __name__ == "__main__":
    main()
