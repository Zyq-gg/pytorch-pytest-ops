#!/usr/bin/env python3
"""Convert a localized PyTorch test summary CSV to failure_report.csv format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_FIELDS = [
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

DEFAULT_COLUMNS = {
    "test_file": "py 文件",
    "class_name": "类名",
    "case_name": "case 名",
    "error": "错误类型及信息",
}


def normalize_test_file(value: str) -> str:
    name = value.strip().replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    if not name:
        raise ValueError("empty test file")
    if name.startswith("/") or ".." in Path(name).parts:
        raise ValueError(f"test file must be relative and remain under test/: {value!r}")
    return name if name.startswith("test/") else f"test/{name}"


def split_error(value: str, *, module_stage: bool) -> tuple[str, str]:
    error = value.strip()
    head, separator, tail = error.partition(":")
    if separator and head.strip() and len(head.strip()) <= 160:
        return head.strip(), tail.strip() or error
    return ("ModuleStageFailure" if module_stage else "Failure"), error


def convert_rows(
    rows: list[dict[str, str]],
    *,
    columns: dict[str, str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    names = columns or DEFAULT_COLUMNS
    converted: list[dict[str, str]] = []
    seen: set[str] = set()
    concrete = 0
    module_stage = 0
    duplicates = 0

    for source_row, row in enumerate(rows, 2):
        raw_file = (row.get(names["test_file"]) or "").strip()
        class_name = (row.get(names["class_name"]) or "").strip()
        case_name = (row.get(names["case_name"]) or "").strip()
        error = (row.get(names["error"]) or "").strip()
        if not any((raw_file, class_name, case_name, error)):
            continue
        if not raw_file:
            raise ValueError(f"row {source_row}: empty {names['test_file']!r}")

        test_file = normalize_test_file(raw_file)
        is_module_stage = class_name == "(文件级)" or not case_name or case_name.startswith("(")
        if is_module_stage:
            output_class = ""
            output_case = "<module-stage>"
            nodeid = test_file
        else:
            output_class = class_name
            output_case = case_name
            parts = [test_file]
            if class_name:
                parts.append(class_name)
            parts.append(case_name)
            nodeid = "::".join(parts)

        if nodeid in seen:
            duplicates += 1
            continue
        seen.add(nodeid)
        if is_module_stage:
            module_stage += 1
        else:
            concrete += 1
        error_type, error_message = split_error(error, module_stage=is_module_stage)
        converted.append(
            {
                "source_log": "",
                "gpu": "",
                "test_file": test_file,
                "class_name": output_class,
                "case_name": output_case,
                "case_params": "",
                "error_type": error_type,
                "error_message": error_message,
                "nodeid": nodeid,
                "raw": error,
            }
        )

    return converted, {
        "input_rows": len(rows),
        "output_rows": len(converted),
        "concrete_cases": concrete,
        "module_stage_rows": module_stage,
        "duplicate_nodeids": duplicates,
    }


def validate_concrete_targets(rows: list[dict[str, str]], pytorch_root: Path) -> list[str]:
    test_dir = pytorch_root.resolve() / "test"
    if not test_dir.is_dir():
        raise ValueError(f"PyTorch test directory does not exist: {test_dir}")
    missing = []
    for row in rows:
        if "::" not in row["nodeid"]:
            continue
        relative = row["test_file"].removeprefix("test/")
        if not (test_dir / relative).is_file():
            missing.append(row["test_file"])
    return sorted(set(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path, help="localized test summary CSV")
    parser.add_argument("output_csv", type=Path, help="failure_report.csv-compatible output")
    parser.add_argument(
        "--pytorch-root",
        type=Path,
        help="optionally verify that every concrete case's test file exists",
    )
    parser.add_argument("--file-column", default=DEFAULT_COLUMNS["test_file"])
    parser.add_argument("--class-column", default=DEFAULT_COLUMNS["class_name"])
    parser.add_argument("--case-column", default=DEFAULT_COLUMNS["case_name"])
    parser.add_argument("--error-column", default=DEFAULT_COLUMNS["error"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = {
        "test_file": args.file_column,
        "class_name": args.class_column,
        "case_name": args.case_column,
        "error": args.error_column,
    }
    with args.input_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing_columns = [value for value in columns.values() if value not in fieldnames]
        if missing_columns:
            available = ", ".join(repr(value) for value in fieldnames)
            raise SystemExit(
                f"missing input columns: {missing_columns}. Available columns: {available}"
            )
        converted, stats = convert_rows(list(reader), columns=columns)

    missing_targets: list[str] = []
    if args.pytorch_root:
        try:
            missing_targets = validate_concrete_targets(converted, args.pytorch_root)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        if missing_targets:
            preview = "\n  ".join(missing_targets[:20])
            raise SystemExit(
                f"missing concrete test files: {len(missing_targets)}\n  {preview}"
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(converted)

    print(f"Input CSV:          {args.input_csv.resolve()}")
    print(f"Output CSV:         {args.output_csv.resolve()}")
    print(f"Input rows:         {stats['input_rows']}")
    print(f"Output rows:        {stats['output_rows']}")
    print(f"Concrete cases:     {stats['concrete_cases']}")
    print(f"Module-stage rows:  {stats['module_stage_rows']}")
    print(f"Duplicate nodeids:  {stats['duplicate_nodeids']}")
    if args.pytorch_root:
        print("Concrete targets:   OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
