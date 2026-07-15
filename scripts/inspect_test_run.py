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
from datetime import datetime
from pathlib import Path
from typing import Any


TERMINAL = {"PASS", "FAIL", "SKIP", "TIMEOUT"}
KNOWN_DIRECT_VIRTUAL_TARGETS = {
    "doctests",
    "test_autoload_disable",
    "test_autoload_enable",
    "test_cpp_extensions_aot_ninja",
    "test_cpp_extensions_aot_no_ninja",
}


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


def timestamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")
    except OSError:
        return None


def is_known_direct_virtual_target(item: str) -> bool:
    name = item.removeprefix("test/")
    if name.endswith(".py"):
        name = name[:-3]
    return name in KNOWN_DIRECT_VIRTUAL_TARGETS


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


def inspect(work_dir: Path, pytorch_root: Path | None = None) -> dict[str, Any]:
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
    if mode == "official-run-test":
        mode = (
            "official-run-test-lightweight"
            if plan_path and plan_path.name == "run_test_modules.txt"
            else "official-queue"
        )
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
    virtual_missing: list[str] = []
    missing_real = list(missing)
    if mode == "pytest":
        virtual_missing = [item for item in missing if is_known_direct_virtual_target(item)]
        missing_real = [item for item in missing if item not in virtual_missing]
    nonterminal = []
    for item, value in tests.items():
        status = value.get("status") if isinstance(value, dict) else value
        if str(status or "").upper() not in TERMINAL:
            nonterminal.append(item)

    summary_path = work_dir / "summary.json"
    summary = load_json(summary_path)
    if pytorch_root is None:
        candidate = (summary or {}).get("pytorch_root") or os.environ.get("PYTORCH_ROOT")
        if candidate:
            pytorch_root = Path(candidate)
        elif Path("/workspace/pytorch/test").is_dir():
            pytorch_root = Path("/workspace/pytorch")
    latest_target = str(latest.resolve()) if latest.exists() else None
    report = read_failure_counts(latest / "failure_report.csv")
    unresolved = read_failure_counts(latest / "unresolved_process_failures.csv")
    reports = summary.get("failure_reports", {}) if summary else {}
    coverage = load_json(work_dir / "coverage_report.json")
    nested_rerun = latest / "process_file_rerun"
    official_reruns = sorted(latest.glob("process_module_rerun*")) if latest.exists() else []
    nested_rerun_summary = load_json(nested_rerun / "summary.json")

    completion_issues: list[str] = []
    if not plan:
        completion_issues.append("planned test/module list is missing or empty")
    if missing_real:
        completion_issues.append(f"{len(missing_real)} real planned items have no checkpoint")
    if nonterminal:
        completion_issues.append(f"{len(nonterminal)} checkpoint entries are nonterminal")
    if summary is None:
        completion_issues.append("root summary.json is missing or invalid")
    if not report["exists"]:
        completion_issues.append("latest/failure_report.csv is missing")
    if not unresolved["exists"]:
        completion_issues.append("latest/unresolved_process_failures.csv is missing")
    if nested_rerun.is_dir() and nested_rerun_summary is None:
        completion_issues.append("process_file_rerun exists without a valid summary.json")
    incomplete_official_reruns = [
        str(path) for path in official_reruns if load_json(path / "summary.json") is None
    ]
    if incomplete_official_reruns:
        completion_issues.append(
            f"{len(incomplete_official_reruns)} official process rerun directories lack summary.json"
        )
    unresolved_rows = unresolved.get("rows")
    if unresolved_rows is None:
        unresolved_rows = reports.get("unresolved_process_failure_count")
    if unresolved_rows is None:
        completion_issues.append("unresolved process-failure count is unavailable")
    elif unresolved_rows:
        completion_issues.append(f"{unresolved_rows} unresolved process-level failures remain")
    if mode == "official-queue":
        if coverage is None:
            completion_issues.append("official coverage_report.json is missing or invalid")
        elif not coverage.get("coverage_complete"):
            completion_issues.append("official coverage_complete is false")

    legacy_hints: list[str] = []
    runner_out = work_dir / "runner.out"
    try:
        runner_text = runner_out.read_text(encoding="utf-8", errors="replace")
    except OSError:
        runner_text = ""
    if mode == "pytest" and runner_text and "Recovery case timeout:" not in runner_text:
        legacy_hints.append("runner output predates explicit case-timeout/attempt/total recovery settings")
    if summary is not None and "unresolved_process_failure_count" not in reports:
        legacy_hints.append("root summary lacks current unresolved failure-report metadata")
    report_time = timestamp(latest / "failure_report.csv")
    summary_time = timestamp(summary_path)
    if report_time and summary_time and report_time > summary_time:
        legacy_hints.append("failure reports were rebuilt after root summary; summary report metadata may be stale")

    artifacts_finalized = summary is not None and report["exists"]
    artifacts_complete = not completion_issues
    if artifacts_complete:
        artifact_verdict = "COMPLETE"
    elif artifacts_finalized:
        artifact_verdict = "FINALIZED_INCOMPLETE"
    else:
        artifact_verdict = "NOT_FINALIZED"

    workdir_processes, all_runner_processes = process_lines(work_dir)
    return {
        "work_dir": str(work_dir),
        "mode": mode,
        "plan_file": str(plan_path) if plan_path else None,
        "plan_count": len(plan),
        "pytorch_root": str(pytorch_root.resolve()) if pytorch_root else None,
        "progress_file": str(progress_path) if progress_path.is_file() else None,
        "progress_count": len(tests),
        "progress_updated": progress.get("updated"),
        "statuses": dict(statuses),
        "missing_count": len(missing),
        "missing_first_20": missing[:20],
        "virtual_plan_count": len(virtual_missing),
        "virtual_plan_first_20": virtual_missing[:20],
        "missing_real_count": len(missing_real),
        "missing_real_first_20": missing_real[:20],
        "nonterminal_count": len(nonterminal),
        "latest": latest_target,
        "summary_exists": summary is not None,
        "summary_time": summary_time,
        "summary_unresolved": reports.get("unresolved_process_failure_count"),
        "coverage": coverage,
        "failure_report": report,
        "unresolved_report": unresolved,
        "process_rerun_exists": nested_rerun.is_dir(),
        "process_rerun_summary_exists": (nested_rerun / "summary.json").is_file(),
        "process_rerun_plan_count": len(read_plan(nested_rerun / "test_files.txt")),
        "process_rerun_timeout": (nested_rerun_summary or {}).get("timeout"),
        "official_process_reruns": [str(path) for path in official_reruns],
        "official_process_rerun_summaries": sum(
            (path / "summary.json").is_file() for path in official_reruns
        ),
        "incomplete_official_process_reruns": incomplete_official_reruns,
        "failure_report_time": report_time,
        "artifact_verdict": artifact_verdict,
        "artifacts_complete": artifacts_complete,
        "completion_issues": completion_issues,
        "legacy_hints": legacy_hints,
        "local_workdir_processes": workdir_processes,
        "local_all_test_runner_processes": all_runner_processes,
    }


def print_text(data: dict[str, Any]) -> None:
    print(f"Artifact verdict:      {data['artifact_verdict']}")
    print(f"Work dir:             {data['work_dir']}")
    print(f"Mode:                 {data['mode']}")
    print(f"Plan:                 {data['plan_file']} ({data['plan_count']})")
    print(f"Progress:             {data['progress_file']} ({data['progress_count']})")
    print(f"Progress updated:     {data['progress_updated']}")
    print(f"Statuses:             {data['statuses']}")
    print(f"Missing from progress:{data['missing_count']:>6}")
    print(f"  Virtual targets:    {data['virtual_plan_count']:>6}")
    for item in data["virtual_plan_first_20"]:
        print(f"    VIRTUAL {item}")
    print(f"  Missing real items: {data['missing_real_count']:>6}")
    for item in data["missing_real_first_20"]:
        print(f"    MISSING {item}")
    print(f"Nonterminal entries:  {data['nonterminal_count']}")
    print(f"Latest:               {data['latest']}")
    print(f"Summary exists:       {data['summary_exists']}")
    print(f"Summary time:         {data['summary_time']}")
    print(f"Summary unresolved:   {data['summary_unresolved']}")
    coverage = data["coverage"]
    if coverage:
        print(
            "Coverage:             "
            f"complete={coverage.get('coverage_complete')} planned={coverage.get('planned')} "
            f"terminal={coverage.get('terminal')} timeout={coverage.get('timeout')} "
            f"missing={coverage.get('missing')}"
        )
        for item in coverage.get("timeout_details", [])[:20]:
            print(
                "  TIMEOUT "
                f"{item.get('module')} kind={item.get('timeout_kind') or 'unknown'} "
                f"elapsed={item.get('elapsed')}s attempts={item.get('attempts')}"
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
        f"exists={data['process_rerun_exists']} summary={data['process_rerun_summary_exists']} "
        f"planned={data['process_rerun_plan_count']} timeout={data['process_rerun_timeout']}"
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
    if data["completion_issues"]:
        print("Completion issues:")
        for issue in data["completion_issues"]:
            print(f"  - {issue}")
    if data["legacy_hints"]:
        print("Legacy/version hints:")
        for hint in data["legacy_hints"]:
            print(f"  - {hint}")
    print("Note: process inspection only covers this machine.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--pytorch-root", type=Path, help="source tree used to classify real files")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    data = inspect(args.work_dir, args.pytorch_root)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print_text(data)


if __name__ == "__main__":
    main()
