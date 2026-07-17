# Runner Selection

## Default full-suite baseline

Use `$OPS_ROOT/runners/run_test-2.13-official-queue.sh` and its Python queue runner as the default full-suite entry point. Run the official Normal and Distributed queues separately, with separate work directories and separate coverage validation. This preserves official custom handlers and module behavior while adding checkpointing, process-level module reruns, case reports, `module_status.csv`, `coverage_report.json`, and `incomplete_modules.txt`.

Official case reports are recall-first candidates: merge official stable-failure summaries with ordinary pytest `FAILED nodeid` lines, then use stable case reruns when a precision-oriented consistently-failing set is required.

## Direct pytest files: incremental and optional full run

Use `$OPS_ROOT/runners/run_pytorch_tests_prefix.py` primarily for prefix-selected modules, code-change incremental validation, entry-point comparison, and failure reproduction. It also retains an optional ordinary file-based full suite discovered from official `run_test.py --dry-run`: each real file runs with `python3 -m pytest`, files are distributed over GPUs, checkpointed by file, recovered with stepcurrent `--rs/--scs`, and process-level Timeout/Crash files are rerun automatically.

This path excludes official distributed tests and JIT executor tests by default. It also cannot directly execute official virtual/custom-handler targets that have no matching `.py` file, so it is not the default complete-coverage claim.

## Ordinary pytest subsets

Use `$OPS_ROOT/runners/run_pytorch_subset.py pytest-list` to filter an existing test file list by prefix or regex while retaining the ordinary GPU queue and recovery logic.

Use `pytest-failure-files` to select only process-level rows from an existing failure CSV and rerun their complete files with a larger timeout. Add `--publish-to-work-dir` to merge a completed unresolved-free supplemental run into the original full report.

## Stable case reruns

Use `$OPS_ROOT/runners/rerun_stable_failures.py` when the input already contains concrete nodeids. `--attempts N` controls the number of attempts required for stable failure classification. This runner retries exact cases, so it does not need file-level `--rs/--scs` continuation.

## Lightweight official run_test.py

Use `$OPS_ROOT/runners/run_pytorch_subset.py run-test-resume` only for lightweight serial official module execution, distributed exploration, and compatibility with existing work directories. It does not provide the official queue idle watchdog, coverage files, or automatic second complete-module rerun, so new unattended distributed full runs should use the complete queue below.

## Coverage statement

Describe complete coverage as separate categories. The default baseline is official Normal plus official Distributed; excluded JIT executor categories still require separate planning. Direct pytest, official normal custom handlers, and distributed tests are not interchangeable. State exactly which categories were run.
