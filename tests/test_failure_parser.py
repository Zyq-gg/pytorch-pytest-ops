import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
SPEC = importlib.util.spec_from_file_location(
    "failure_parser", RUNNERS / "run_pytorch_tests_prefix.py"
)
PARSER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSER)


class FailureParserTest(unittest.TestCase):
    def test_direct_dry_run_rejects_nonzero_partial_output(self):
        result = subprocess.CompletedProcess(
            ["python3", "run_test.py"],
            1,
            stdout="Serial tests (1):\n    test_partial 1/1\n",
            stderr="fatal after partial list\n",
        )
        with mock.patch.object(PARSER.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "return code 1"):
                PARSER.run_dry_run("/tmp", [])

    def test_manifest_mismatch_does_not_overwrite_existing_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pytorch"
            work = Path(tmp) / "work"
            (root / "test").mkdir(parents=True)
            argv = [
                "run_pytorch_tests_prefix.py",
                str(root),
                "--work-dir",
                str(work),
                "--dry-run-only",
            ]
            old_dry_run = "Serial tests (1):\n    test_old 1/1\n"
            new_dry_run = "Serial tests (1):\n    test_new 1/1\n"
            with mock.patch.object(PARSER.sys, "argv", argv + ["--fresh"]):
                with mock.patch.object(PARSER, "run_dry_run", return_value=old_dry_run):
                    PARSER.main()
            with mock.patch.object(PARSER.sys, "argv", argv):
                with mock.patch.object(PARSER, "run_dry_run", return_value=new_dry_run):
                    with self.assertRaisesRegex(RuntimeError, "plan"):
                        PARSER.main()
            self.assertEqual(
                (work / "test_files.txt").read_text(encoding="utf-8"),
                "test_old.py\n",
            )

    def test_finalize_only_rebuilds_reports_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            log_dir = work / "20260728_120000"
            log_dir.mkdir()
            (work / "latest").symlink_to(log_dir.name)
            progress = {"test_complete.py": {"status": "PASS"}}

            summary = PARSER.finalize_existing_pytest_run(
                work_dir=str(work),
                planned_tests=["test_complete.py"],
                progress=progress,
                gpu_ids=[0],
                analyze=True,
            )

            self.assertTrue(summary["finalize_only"])
            self.assertEqual(summary["progress_stats"]["passed"], 1)
            self.assertEqual(summary["missing_checkpoints"], [])
            self.assertTrue((work / "summary.json").is_file())
            self.assertTrue((log_dir / "failure_report.csv").is_file())
            self.assertTrue((log_dir / "unresolved_process_failures.csv").is_file())

    def test_finalize_only_refuses_failed_checkpoint_without_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "without an existing latest"):
                PARSER.finalize_existing_pytest_run(
                    work_dir=tmp,
                    planned_tests=["test_failed.py"],
                    progress={"test_failed.py": {"status": "FAIL"}},
                    gpu_ids=[0],
                    analyze=True,
                )

    def test_latest_txt_fallback_wins_over_a_real_latest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "latest").mkdir()
            newer = work / "20260728_130000"
            newer.mkdir()
            (work / "latest.txt").write_text(str(newer), encoding="utf-8")
            self.assertEqual(
                PARSER.resolve_latest_log_dir(str(work)),
                str(newer),
            )

    def test_official_log_keeps_stable_and_pytest_fallback_nodeids(self):
        text = """===== command: python3 run_test.py --include test_stable =====
============================= test session starts ==============================
Running 1 items in this shard: test/test_stable.py::Tests::test_stable
RuntimeError: stable failure
FAILED test/test_stable.py::Tests::test_stable - RuntimeError: stable failure
FAILED CONSISTENTLY: test/test_stable.py::Tests::test_stable
The following tests failed consistently: ['test/test_stable.py::Tests::test_stable']
===== command: python3 run_test.py --include test_incomplete =====
============================= test session starts ==============================
Running 1 items in this shard: test/test_incomplete.py::Tests::test_seen_before_exit
FAILED [1.0000s] test/test_incomplete.py::Tests::test_seen_before_exit
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_0.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual(
            {row["nodeid"] for row in rows},
            {
                "test_stable.py::Tests::test_stable",
                "test_incomplete.py::Tests::test_seen_before_exit",
            },
        )

    def test_official_error_prefers_matching_pytest_session(self):
        text = """===== command: python3 run_test.py --include test_target =====
============================= test session starts ==============================
Running 1 items in this shard: test/test_other.py::Tests::test_other
KERNEL VMFault from an earlier session
FAILED test/test_other.py::Tests::test_other
============================= test session starts ==============================
Running 1 items in this shard: test/test_target.py::Tests::test_target
RuntimeError: target-specific detail
FAILED test/test_target.py::Tests::test_target - RuntimeError: target-specific detail
FAILED CONSISTENTLY: test/test_target.py::Tests::test_target
The following tests failed consistently: ['test/test_target.py::Tests::test_target']
"""
        row = PARSER.parse_official_run_test_failure(
            "test_target.py::Tests::test_target", "run_test_gpu_0.log", text
        )
        self.assertEqual(row["error_type"], "RuntimeError")
        self.assertEqual(row["error_message"], "target-specific detail")

    def test_official_flaky_failed_line_is_retained_for_recall(self):
        text = """===== command: python3 run_test.py --include test_flaky =====
============================= test session starts ==============================
Running 1 items in this shard: test/test_flaky.py::Tests::test_flaky
FAILED [1.0000s] test/test_flaky.py::Tests::test_flaky
The following tests failed and then succeeded when run in a new process['test/test_flaky.py::Tests::test_flaky']
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_0.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual([row["nodeid"] for row in rows], ["test_flaky.py::Tests::test_flaky"])

    def test_official_structured_single_test_timeout_is_a_concrete_case(self):
        nodeid = (
            "distributed/test_c10d_nccl.py::NCCLTraceTest::"
            "test_allgather_uneven_timing_enabled_False"
        )
        text = f"""===== worker 0, GPU all =====
===== 12:00:00 START: distributed/test_nccl [1/2] (GPU all) =====
===== 12:00:01 PASS: distributed/test_nccl (1.0s, rc=0) =====
===== 12:00:02 START: distributed/test_c10d_nccl [2/2] (GPU all) =====
===== command: python3 run_test.py --include distributed/test_c10d_nccl =====
Running 1 items in this shard: test/{nodeid}
Command took >15min, returning 124
Test exited with non-zero exitcode 124. Command to reproduce: python3 distributed/test_c10d_nccl.py --pytest-single-test {nodeid}
AssertionError: 1 unit test(s) failed:
\t{nodeid}
>>>>>>>> KERNEL VMFault !!!! <<<<<<
===== 12:15:03 FAIL: distributed/test_c10d_nccl (901.0s, rc=1) =====
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_all.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual([row["nodeid"] for row in rows], [nodeid])
        self.assertEqual(rows[0]["error_type"], "Timeout")
        self.assertIn("code 124", rows[0]["error_message"])
        self.assertEqual(rows[0]["gpu"], "all")

    def test_official_shared_log_keeps_process_crash_for_another_module(self):
        text = """===== worker 0, GPU all =====
===== 12:00:00 START: distributed/test_case [1/2] (GPU all) =====
===== command: python3 run_test.py --include distributed/test_case =====
Running 1 items in this shard: test/distributed/test_case.py::Tests::test_failure
FAILED distributed/test_case.py::Tests::test_failure - AssertionError: bad value
===== 12:00:01 FAIL: distributed/test_case (1.0s, rc=1) =====
===== 12:00:02 START: distributed/test_import [2/2] (GPU all) =====
Fatal Python error: Segmentation fault
===== 12:00:03 FAIL: distributed/test_import (1.0s, rc=1) =====
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_all.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual(
            {row["nodeid"] for row in rows},
            {
                "distributed/test_case.py::Tests::test_failure",
                "distributed/test_import",
            },
        )
        process_row = next(row for row in rows if row["nodeid"].endswith("test_import"))
        self.assertEqual(process_row["error_type"], "Crash")
        self.assertEqual(process_row["gpu"], "all")

    def test_official_rank_prefixed_error_line_is_retained(self):
        text = """===== command: python3 run_test.py --include distributed/test_error =====
[rank0]: ============================= test session starts ==============================
[rank0]: Running 1 items in this shard: test/distributed/test_error.py::Tests::test_setup
[rank0]: ERROR distributed/test_error.py::Tests::test_setup - RuntimeError: setup failed
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_all.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual(
            [row["nodeid"] for row in rows],
            ["distributed/test_error.py::Tests::test_setup"],
        )

    def test_official_crash_during_known_active_case_is_not_process_level(self):
        nodeid = "distributed/test_crash.py::Tests::test_active"
        text = f"""===== worker 0, GPU all =====
===== 12:00:00 START: distributed/test_crash [1/1] (GPU all) =====
===== command: python3 run_test.py --include distributed/test_crash =====
[rank0]: Running 1 item in this shard: test/{nodeid}
Fatal Python error: Segmentation fault
===== 12:00:01 FAIL: distributed/test_crash (1.0s, rc=1) =====
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_all.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual([row["nodeid"] for row in rows], [nodeid])
        self.assertEqual(rows[0]["error_type"], "Crash")

    def test_official_outer_timeout_uses_single_active_case_not_plan_list(self):
        text = """===== worker 0, GPU all =====
===== 12:00:00 START: distributed/test_timeout [1/1] (GPU all) =====
===== command: python3 run_test.py --include distributed/test_timeout =====
Running 224 items in this shard: test/wrong.py::Tests::test_wrong, test/also_wrong.py::Tests::test_wrong
Running 1 item in this shard: test/distributed/test_timeout.py::Tests::test_active
===== 12:15:00 TIMEOUT: distributed/test_timeout (900.0s, idle) =====
"""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "run_test_gpu_all.log"
            log.write_text(text, encoding="utf-8")
            rows = PARSER.collect_failures_from_logs(str(log))

        self.assertEqual(
            [row["nodeid"] for row in rows],
            ["distributed/test_timeout.py::Tests::test_active"],
        )
        self.assertEqual(rows[0]["error_type"], "Timeout")


if __name__ == "__main__":
    unittest.main()
