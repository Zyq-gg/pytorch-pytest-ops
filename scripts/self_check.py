#!/usr/bin/env python3
"""Validate a cloned pytorch-pytest-ops repository without external packages."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "SKILL.md",
    "agents/openai.yaml",
    "references/commands.md",
    "references/runner-selection.md",
    "references/status-and-reports.md",
    "runners/run_pytorch_tests_prefix.py",
    "runners/run_pytorch_subset.py",
    "runners/rerun_stable_failures.py",
    "runners/run_official_run_test_queue.py",
    "runners/run_test-2.13-official-queue.sh",
    "scripts/install_skill.sh",
    "scripts/inspect_test_run.py",
    "scripts/self_check.py",
]


def run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout[-4000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-root", type=Path, help="optionally validate a PyTorch source tree")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("missing bundled files:\n  " + "\n  ".join(missing))

    for path in list((ROOT / "runners").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for relative in (
        "runners/run_pytorch_tests_prefix.py",
        "runners/run_pytorch_subset.py",
        "runners/rerun_stable_failures.py",
        "runners/run_official_run_test_queue.py",
        "scripts/inspect_test_run.py",
    ):
        run([sys.executable, str(ROOT / relative), "--help"])
    run(["bash", "-n", str(ROOT / "runners/run_test-2.13-official-queue.sh")])

    if args.pytorch_root:
        run_test = args.pytorch_root.resolve() / "test" / "run_test.py"
        if not run_test.is_file():
            raise SystemExit(f"PyTorch run_test.py not found: {run_test}")
        print(f"PyTorch source: OK ({args.pytorch_root.resolve()})")

    print(f"Repository root: {ROOT}")
    print("Bundled skill, runners, parser, inspector, and references: OK")
    print("External Python packages required by this repository: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
