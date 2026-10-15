---
name: pytorch-pytest-ops
description: Operate and diagnose PyTorch pytest and official run_test.py workflows using the runners bundled with this skill. Use for exact dry-run, full-run, subset, distributed, resume, rerun, stable-failure, localized summary CSV conversion, nohup, environment-variable, status-inspection, completion-validation, case-query, log parsing, or failure-report commands; explain timeout/crash recovery and reports; or maintain the workflow documentation.
---

# PyTorch Pytest Operations

Use the bundled runner source as the authority and give commands that are directly runnable in the user's environment. This skill is self-contained for orchestration and log parsing; do not depend on `/workspace/torch_test`, `check_optest_results_v2.py`, or other scripts outside this skill checkout.

## Establish Context

1. Resolve `OPS_ROOT` from the directory containing this `SKILL.md`; never assume the repository was cloned to `/workspace`. Use `$OPS_ROOT/runners`, `$OPS_ROOT/scripts`, and `$OPS_ROOT/docs` in generated commands.
2. Discover the target-specific paths separately:
   - `PYTORCH_ROOT`: a source tree containing `test/run_test.py`
   - `ENV_SH`: an optional environment activation script; omit it when the environment is already active
   - `WORK`: the user-selected result directory
3. Inspect the relevant bundled runner source or `--help` before composing commands. Do not rely on remembered flags when the source is available.
4. Read only the relevant workflow section. Use `grep -n`, `sed`, or `rg` when installed.
5. Preserve user-provided work directories, environment variables, GPU IDs, timeout values, and runner choice across resume commands.
6. When the input is a localized test-summary CSV instead of a native runner report, inspect its headers and use `$OPS_ROOT/scripts/convert_test_summary_csv.py`. Validate the converted plan with the selected runner's dry-run before starting tests.

Read [references/runner-selection.md](references/runner-selection.md) when choosing an entry point. Read [references/commands.md](references/commands.md) for command templates. Read [references/status-and-reports.md](references/status-and-reports.md) when inspecting a run or explaining outputs.

## Handle Requests

### Give a run command

1. Identify the entry point and category: ordinary pytest, pytest subset, process-level file rerun, stable case rerun, official normal queue, or official distributed queue.
   For a new full-suite baseline, default to both complete official queues: Normal and Distributed, with separate work directories and separate completion checks. Use direct pytest primarily for selected modules/files/cases, code-change incremental validation, failure reproduction, and stable-failure analysis; only present its full-file mode as an optional alternative. Use lightweight `run-test-resume` only for exploration or compatibility with its existing work directories.
2. Include `source "$ENV_SH"` only when an environment script exists, and always create `$WORK`.
3. For a long run, use `nohup env PYTHONUNBUFFERED=1 python3 -u ... > "$WORK/runner.out" 2>&1 &` and save `$!` to `runner.pid`.
4. For a new run, include `--fresh`. For resume, use the same work directory and semantic parameters but remove `--fresh`.
5. Keep custom environment variables identical across dry-run, run, automatic rerun, and resume. Use a different work directory when comparing configurations.
   Current runners persist these semantics in `<work-dir>/.run_manifest.json` and reject mismatched resumes before replacing the saved plan. Do not delete the manifest to bypass this check. A legacy checkpoint adopted without an older manifest remains explicitly unverified.
6. For the official queue, preserve both hard and idle timeout settings. The current shell defaults are 72 hours hard and 2 hours without subprocess output for initial and process-rerun modules.
7. After every run command, provide output paths, status commands, resume behavior, and completion checks.

### Convert a localized failure summary

Use `scripts/convert_test_summary_csv.py` for tables whose rows contain a test file, class, case, and error but no native `nodeid`. Its default columns are `py 文件`, `类名`, `case 名`, and `错误类型及信息`; use the column override flags for other headers. Concrete rows become exact `test/file.py::Class::case` nodeids. File-level rows become diagnosed `<module-stage>` rows and remain excluded from stable case reruns unless the user explicitly requests process-level execution. Pass `--pytorch-root` to validate concrete test-file targets, then require the stable runner's `Unique rerun rows` and `Missing targets` dry-run counts before a long run.

For a mixed Normal/Distributed CSV that the user explicitly wants to run as one official stable-rerun queue, omit `--gpu-ids` so a single worker inherits all visible devices. Do not add `--distributed-tests` globally: explicit `--include <module>` selects distributed modules, while the global flag can change selection semantics for ordinary modules. This mixed mode is serial and should not overlap other GPU jobs.

### Inspect a directory

Run the bundled read-only inspector first:

```bash
python3 "$OPS_ROOT/scripts/inspect_test_run.py" <work-dir> --pytorch-root "$PYTORCH_ROOT"
```

Then inspect `runner.out`, active timestamp logs, and relevant JSON only as needed. Never conclude completion from `ps` alone. If the process ran on another node, explicitly say local process inspection is not authoritative.
Distinguish `Run verdict` from `Artifact verdict`: the former becomes `RUNNING` for a matching local process, while the latter describes only on-disk completion artifacts.

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

Treat a concrete `file.py::Class::case` with `error_type=Crash` or `Timeout` as a located case failure, not process-level unresolved. Also treat a parsed import/collection/build/runner error with `case_name=<module-stage>` as a diagnosed module failure: it belongs in the main report without inventing a pytest case. Reserve unresolved rows for failures that have neither a reliable case nor a concrete module-stage diagnostic.

For the complete official queue, `failure_report.csv` contains concrete case nodeids plus diagnosed `<module-stage>` failures, and it is recall-first: official stable summaries and ordinary pytest FAILED lines are merged, so a flaky case that later passed may remain. Read timeout/missing/truly unresolved process failures from `module_status.csv`, `coverage_report.json`, `incomplete_modules.txt`, and `unresolved_process_failures.csv`; use stable case reruns when the user needs a precision-oriented consistently-failing set.

### Diagnose count mismatches

Compare the plan with `progress["tests"]`, not the top-level JSON key count. Ordinary direct-pytest dry-run currently includes five official virtual/custom-handler targets without matching `.py` files; they are intentionally absent from the direct file checkpoint unless the local source has changed. Verify names from current files before explaining.

### Publish an independent historical rerun

For `pytest-failure-files`, include `--publish-to-work-dir <original-full-work-dir>` when the user wants the completed supplemental result to replace stale rows in the main report. Publishing must occur only after all selected files have terminal checkpoints and the supplemental unresolved count is zero. `external_rerun_merge.json` makes later `--analyze-only` rebuilds reapply that merge.

## Safety And Accuracy

- Do not start, stop, kill, resume, or overwrite a test run unless the user requests it. Status requests are read-only.
- Before suggesting `--fresh`, distinguish a new work directory from a resume; `--fresh` discards checkpoint use.
- Do not claim all PyTorch categories were covered by ordinary pytest. Distributed and official custom handlers use the official entry point.
- Do not present direct pytest and the official queue as equal default full-suite choices. The official Normal plus Distributed queues are the default coverage baseline; direct pytest is the preferred precision/incremental tool and an optional ordinary-file full run.
- Do not claim “no omissions” solely because a runner exited. Report the exact plan/checkpoint/missing/unresolved counts.
- Prefer `grep -E` fallbacks because `rg` may be unavailable in runtime containers.
- Treat runner source as newer than bundled examples. If flags differ, follow source and mention the discrepancy.
- When diagnosing stepcurrent recovery, account for both PyTorch 2.9 `stepcurrent/<key>` and newer `stepcurrent/<key>/lastrun` cache layouts.
- An outer official-queue timeout kills the whole `run_test.py` invocation. A later incomplete-module rerun starts that module again; it does not reliably continue every randomly keyed official shard from the prior percentage.
- Keep `$OPS_ROOT/docs/PYTORCH_PYTEST_WORKFLOW.md` synchronized when changing runner behavior or durable operating procedures.

## Response Shape

For commands, provide one complete copy-runnable block, followed by concise output, status, resume, and completion sections. For inspections, lead with the verdict, then evidence and the exact next command only when needed.
