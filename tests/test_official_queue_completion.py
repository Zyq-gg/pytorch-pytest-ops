import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))
SPEC = importlib.util.spec_from_file_location(
    "official_queue", RUNNERS / "run_official_run_test_queue.py"
)
QUEUE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(QUEUE)


class OfficialQueueCompletionTest(unittest.TestCase):
    def test_checkpoint_timeout_is_rerun_even_when_parser_located_a_case(self):
        tests = ["test_ok", "inductor/test_slow"]
        progress = {
            "test_ok": {"status": "PASS"},
            "inductor/test_slow": {"status": "TIMEOUT"},
        }
        rows = [
            {
                "test_file": "inductor/test_slow.py",
                "nodeid": "inductor/test_slow.py::Tests::test_active",
                "case_name": "test_active",
                "error_type": "Timeout",
            }
        ]
        self.assertEqual(
            QUEUE.select_process_rerun_modules(tests, progress, rows, {"Timeout", "Crash"}),
            ["inductor/test_slow"],
        )

    def test_complete_rerun_replaces_every_old_row_for_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            rerun_dir = Path(tmp) / "rerun"
            rerun_dir.mkdir()
            rows = [
                {
                    "source_log": str(Path(tmp) / "old.log"),
                    "test_file": "test_slow.py",
                    "case_name": "old_partial_failure",
                },
                {
                    "source_log": str(rerun_dir / "new.log"),
                    "test_file": "test_slow.py",
                    "case_name": "new_authoritative_failure",
                },
                {
                    "source_log": str(Path(tmp) / "old.log"),
                    "test_file": "test_other.py",
                    "case_name": "other_failure",
                },
            ]
            result = QUEUE.replace_rerun_module_rows(rows, {"test_slow"}, rerun_dir)
            self.assertEqual(
                [row["case_name"] for row in result],
                ["new_authoritative_failure", "other_failure"],
            )

    def test_coverage_requires_no_missing_timeout_or_unresolved_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            coverage = QUEUE.write_module_coverage(
                work,
                ["pass", "ordinary_failure", "timed_out", "missing"],
                {
                    "pass": {"status": "PASS", "elapsed": 1},
                    "ordinary_failure": {"status": "FAIL", "elapsed": 2},
                    "timed_out": {"status": "TIMEOUT", "elapsed": 3},
                },
                1,
            )
            self.assertFalse(coverage["coverage_complete"])
            self.assertEqual(coverage["terminal"], 2)
            self.assertEqual(coverage["timeout_modules"], ["timed_out"])
            self.assertEqual(coverage["missing_modules"], ["missing"])
            self.assertEqual(
                (work / "incomplete_modules.txt").read_text().splitlines(),
                ["missing", "timed_out"],
            )
            self.assertEqual(
                json.loads((work / "coverage_report.json").read_text())["planned"], 4
            )

    def test_terminal_timeout_gets_explicit_unresolved_row(self):
        rows = QUEUE.append_terminal_timeout_rows(
            [], ["test_slow"], {"test_slow": {"status": "TIMEOUT", "elapsed": 90}}
        )
        self.assertEqual(rows[0]["test_file"], "test_slow.py")
        self.assertEqual(rows[0]["case_name"], "<timeout>")
        self.assertEqual(rows[0]["error_type"], "Timeout")


if __name__ == "__main__":
    unittest.main()
