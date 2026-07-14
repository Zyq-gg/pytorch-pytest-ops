# Runner Selection

## Ordinary direct pytest

Use `$OPS_ROOT/runners/run_pytorch_tests_prefix.py` for the ordinary file-based full suite discovered from official `run_test.py --dry-run`. It executes each real file with `python3 -m pytest`, distributes files over GPUs, checkpoints by file, applies stepcurrent `--rs/--scs` recovery, automatically reruns process-level Timeout/Crash files, and creates final case reports.

This path excludes official distributed tests and JIT executor tests by default. It also cannot directly execute official virtual/custom-handler targets that have no matching `.py` file.

## Ordinary pytest subsets

Use `$OPS_ROOT/runners/run_pytorch_subset.py pytest-list` to filter an existing test file list by prefix or regex while retaining the ordinary GPU queue and recovery logic.

Use `pytest-failure-files` to select only process-level rows from an existing failure CSV and rerun their complete files with a larger timeout. Add `--publish-to-work-dir` to merge a completed unresolved-free supplemental run into the original full report.

## Stable case reruns

Use `$OPS_ROOT/runners/rerun_stable_failures.py` when the input already contains concrete nodeids. `--attempts N` controls the number of attempts required for stable failure classification. This runner retries exact cases, so it does not need file-level `--rs/--scs` continuation.

## Official run_test.py

Use `$OPS_ROOT/runners/run_pytorch_subset.py run-test-resume` for official module-by-module execution with checkpoint resume, especially distributed tests and custom handlers.

Use `$OPS_ROOT/runners/run_test-2.13-official-queue.sh` and its Python queue runner only when the user specifically requests the official normal/distributed queue workflow or is continuing an existing queue work directory. Inspect the shell variables and current `--help` before issuing commands because paths and modes are configurable.

## Coverage statement

Describe complete coverage as separate categories. Ordinary direct pytest, official normal custom handlers, distributed tests, and excluded JIT executor categories are not interchangeable. State exactly which categories were run.
