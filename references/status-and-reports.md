# Status And Reports

## Read-only first pass

```bash
python3 "$OPS_ROOT/scripts/inspect_test_run.py" <work-dir>
tail -100 <work-dir>/runner.out
ps -ef | grep -E 'run_pytorch|run_test.py|python3 -m pytest' | grep -v grep
```

The process query only describes the current machine. If storage is shared and execution occurred on another node, use files for progress and inspect processes on that node.

## Ordinary pytest outputs

- `test_files.txt`: official dry-run plan before filtering nonexistent virtual targets.
- `.test_progress.json`: file checkpoint; use `tests`, `stats`, and `updated`.
- `<timestamp>/gpu_*.log`: worker logs.
- `latest`: symlink to the active/latest timestamp.
- `latest/process_file_rerun/`: automatic large-timeout rerun of unresolved files.
- `latest/failure_report.csv`: final located failures.
- `latest/unresolved_process_failures.csv`: only process-level failures without a reliable case nodeid.
- `summary.json`: root completion summary and report paths.
- `latest/external_rerun_merge.json`: persistent publication metadata for a separate historical rerun.

## Official queue outputs

- `run_test_tests.txt`: official queue module plan (`run_test_modules.txt` is used by the lightweight `run-test-resume` entry).
- `.run_test_progress.json`: module checkpoint; TIMEOUT is recorded but does not count as coverage terminal.
- `<timestamp>/run_test_gpu_*.log`: normal combined worker logs; distributed uses `run_test_gpu_all.log`.
- `<timestamp>/process_module_rerun*/`: authoritative complete-module rerun logs and summary.
- `latest/failure_report.csv`: parsed case failures. A timed-out module with partial case rows may not have an extra module-level Timeout row.
- `latest/unresolved_process_failures.csv`: nonzero modules without a reliable case nodeid.
- `module_status.csv`: one row per plan module with status, elapsed, return code, and timestamp. `PASS` and `FAIL` are terminal official returns; `TIMEOUT` was killed by the outer watchdog; `MISSING` has no checkpoint.
- `incomplete_modules.txt`: missing and TIMEOUT modules.
- `coverage_report.json`: authoritative official completion result; `terminal` counts only `PASS + FAIL`, and completion requires zero timeout, missing, and unresolved rows plus `coverage_complete: true`.
- `summary.json`: report, rerun, progress, and coverage index.

## Completion interpretation

No process plus no summary means interrupted or incomplete, not successful. A summary plus missing checkpoint entries is also incomplete. A nonzero unresolved count means the runner finished but failed to obtain complete case-level conclusions for those files/modules. For the official queue, a summary with `coverage_complete: false` means the command ended with explicit diagnostics but coverage did not close.

For ordinary direct pytest, a plan/progress difference can be caused by official virtual targets that have no matching file. Verify every missing item with `test -f "$PYTORCH_ROOT/test/<item>"` before classifying it as omitted.

## Query one case

Search the final CSV by exact nodeid first, then search logs:

```bash
python3 - <<'PY'
import csv
p = "/path/to/latest/failure_report.csv"
needle = "test_file.py::ClassName::test_case"
with open(p, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if needle in row.get("nodeid", ""):
            print(row)
PY

grep -R -F "test_file.py::ClassName::test_case" /path/to/latest --include='*.log'
```

`failure_report.csv` contains failures, not every passing case. Absence from the CSV only means no failure row was extracted; confirm pass/skip from logs when needed.
