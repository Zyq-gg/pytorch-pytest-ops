import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
SPEC = importlib.util.spec_from_file_location(
    "failure_parser", RUNNERS / "run_pytorch_tests_prefix.py"
)
PARSER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PARSER)


class FailureParserTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
