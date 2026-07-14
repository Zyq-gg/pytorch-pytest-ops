---
name: pytorch-pytest-ops
description: Operate and diagnose the local PyTorch pytest infrastructure under /workspace/pytorch-pytest-ops/runners. Use when asked to generate exact dry-run, full-run, subset, distributed, resume, rerun, stable-failure, nohup, environment-variable, status-inspection, completion-validation, case-query, or failure-report commands; inspect PyTorch test result directories or processes; explain timeout/crash recovery and reports; or maintain the testing workflow documentation.
---

# PyTorch Pytest Operations

Use the bundled runner source as the authority and give commands that are directly runnable in the user's environment.

## Establish Context

1. Resolve the current paths. In this environment, start with:
   - PyTorch source: `/workspace/pytorch`
   - runners: `/workspace/pytorch-pytest-ops/runners`
   - environment: `/home/tmp/python_and_sh/env.sh`
   - workflow: `/workspace/pytorch-pytest-ops/docs/PYTORCH_PYTEST_WORKFLOW.md`
2. Inspect the relevant bundled runner source or `--help` before composing commands. Do not rely on remembered flags when the source is available.
3. Read only the relevant workflow section. Use `grep -n`, `sed`, or `rg` when installed.
4. Preserve user-provided work directories, environment variables, GPU IDs, timeout values, and runner choice across resume commands.

Read [references/runner-selection.md](references/runner-selection.md) when choosing an entry point. Read [references/commands.md](references/commands.md) for command templates. Read [references/status-and-reports.md](references/status-and-reports.md) when inspecting a run or explaining outputs.

## Handle Requests

### Give a run command

1. Identify the entry point and category: ordinary pytest, pytest subset, process-level file rerun, stable case rerun, official normal queue, or official distributed queue.
2. Include `source /home/tmp/python_and_sh/env.sh` and `mkdir -p "$WORK"`.
3. For a long run, use `nohup env PYTHONUNBUFFERED=1 python3 -u ... > "$WORK/runner.out" 2>&1 &` and save `$!` to `runner.pid`.
4. For a new run, include `--fresh`. For resume, use the same work directory and semantic parameters but remove `--fresh`.
5. Keep custom environment variables identical across dry-run, run, automatic rerun, and resume. Use a different work directory when comparing configurations.
6. After every run command, provide output paths, status commands, resume behavior, and completion checks.

### Inspect a directory

Run the bundled read-only inspector first:

```bash
python3 /workspace/pytorch-pytest-ops/scripts/inspect_test_run.py <work-dir>
```

Then inspect `runner.out`, active timestamp logs, and relevant JSON only as needed. Never conclude completion from `ps` alone. If the process ran on another node, explicitly say local process inspection is not authoritative.

### Decide whether a run is complete

Require all applicable conditions:

- The planned list is nonempty.
- Every real planned file/module has a terminal checkpoint.
- The root `summary.json` exists.
- For official queue runs, `coverage_report.json.coverage_complete` is `true`.
- The final `latest/failure_report.csv` exists.
- `summary.json.failure_reports.unresolved_process_failure_count` is `0` and the unresolved CSV has no data rows.
- Any automatic `process_file_rerun/` that was triggered has its own `summary.json`.
- No relevant process remains on the machine that actually ran the command.

Treat a concrete `file.py::Class::case` with `error_type=Crash` or `Timeout` as a located case failure, not process-level unresolved. Do not delete it merely to make error-type counts zero.

### Diagnose count mismatches

Compare the plan with `progress["tests"]`, not the top-level JSON key count. Ordinary direct-pytest dry-run currently includes five official virtual/custom-handler targets without matching `.py` files; they are intentionally absent from the direct file checkpoint unless the local source has changed. Verify names from current files before explaining.

### Publish an independent historical rerun

For `pytest-failure-files`, include `--publish-to-work-dir <original-full-work-dir>` when the user wants the completed supplemental result to replace stale rows in the main report. Publishing must occur only after all selected files have terminal checkpoints and the supplemental unresolved count is zero. `external_rerun_merge.json` makes later `--analyze-only` rebuilds reapply that merge.

## Safety And Accuracy

- Do not start, stop, kill, resume, or overwrite a test run unless the user requests it. Status requests are read-only.
- Before suggesting `--fresh`, distinguish a new work directory from a resume; `--fresh` discards checkpoint use.
- Do not claim all PyTorch categories were covered by ordinary pytest. Distributed and official custom handlers use the official entry point.
- Do not claim “no omissions” solely because a runner exited. Report the exact plan/checkpoint/missing/unresolved counts.
- Prefer `grep -E` fallbacks because `rg` may be unavailable in runtime containers.
- Treat runner source as newer than bundled examples. If flags differ, follow source and mention the discrepancy.
- When diagnosing stepcurrent recovery, account for both PyTorch 2.9 `stepcurrent/<key>` and newer `stepcurrent/<key>/lastrun` cache layouts.
- Keep `/workspace/pytorch-pytest-ops/docs/PYTORCH_PYTEST_WORKFLOW.md` synchronized when changing runner behavior or durable operating procedures.

## Response Shape

For commands, provide one complete copy-runnable block, followed by concise output, status, resume, and completion sections. For inspections, lead with the verdict, then evidence and the exact next command only when needed.
