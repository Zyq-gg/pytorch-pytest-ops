import csv
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
    def test_idle_and_hard_timeout_are_independent(self):
        self.assertEqual(
            QUEUE.timeout_kind(
                now=7200,
                started_at=0,
                last_output_at=0,
                hard_timeout=259200,
                idle_timeout=7200,
            ),
            "idle",
        )
        self.assertEqual(
            QUEUE.timeout_kind(
                now=259200,
                started_at=0,
                last_output_at=259199,
                hard_timeout=259200,
                idle_timeout=7200,
            ),
            "hard",
        )
        self.assertEqual(
            QUEUE.timeout_kind(
                now=20000,
                started_at=0,
                last_output_at=19999,
                hard_timeout=259200,
                idle_timeout=7200,
            ),
            "",
        )

    def test_progress_preserves_attempt_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".run_test_progress.json"
            QUEUE.save_progress(
                path,
                "test_slow",
                "TIMEOUT",
                10,
                15,
                timeout_kind="idle",
                hard_timeout=100,
                idle_timeout=10,
            )
            QUEUE.save_progress(path, "test_slow", "PASS", 20, 0)
            item = json.loads(path.read_text())["tests"]["test_slow"]
            self.assertEqual(item["attempts"], 2)
            self.assertEqual([entry["status"] for entry in item["history"]], ["TIMEOUT", "PASS"])

    def test_slow_modules_are_scheduled_from_longest_history(self):
        self.assertEqual(
            QUEUE.sort_modules_by_history(
                ["short", "unknown", "long"],
                {"short": {"elapsed": 2}, "long": {"elapsed": 20}},
            ),
            ["long", "short", "unknown"],
        )

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

    def test_unreported_nonzero_module_is_rerun_as_possible_process_crash(self):
        self.assertEqual(
            QUEUE.select_process_rerun_modules(
                ["test_import"], {"test_import": {"status": "FAIL"}}, [], {"Timeout", "Crash"}
            ),
            ["test_import"],
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
            self.assertEqual(coverage["timeout_details"][0]["module"], "timed_out")
            self.assertEqual(coverage["missing_modules"], ["missing"])
            self.assertEqual(
                (work / "incomplete_modules.txt").read_text().splitlines(),
                ["missing", "timed_out"],
            )
            self.assertEqual(
                json.loads((work / "coverage_report.json").read_text())["planned"], 4
            )

    def test_terminal_timeout_gets_explicit_unresolved_row(self):
        rows = QUEUE.append_unreported_terminal_rows(
            [], ["test_slow"], {"test_slow": {"status": "TIMEOUT", "elapsed": 90}}
        )
        self.assertEqual(rows[0]["test_file"], "test_slow.py")
        self.assertEqual(rows[0]["case_name"], "<timeout>")
        self.assertEqual(rows[0]["error_type"], "Timeout")

    def test_unreported_nonzero_result_gets_process_failure_row(self):
        rows = QUEUE.append_unreported_terminal_rows(
            [],
            ["test_import"],
            {"test_import": {"status": "FAIL", "elapsed": 2, "returncode": 1}},
        )
        self.assertEqual(rows[0]["case_name"], "<process-failure>")
        self.assertEqual(rows[0]["error_type"], "ProcessFailure")

    def test_official_main_report_can_exclude_process_level_rows(self):
        keys = (
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
        )
        case_row = dict.fromkeys(keys, "")
        case_row.update(
            test_file="test_real.py",
            class_name="Tests",
            case_name="test_failure",
            error_type="AssertionError",
            nodeid="test_real.py::Tests::test_failure",
        )
        process_row = dict.fromkeys(keys, "")
        process_row.update(
            test_file="test_import.py",
            case_name="<process-failure>",
            error_type="ProcessFailure",
            nodeid="test_import.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            reports = QUEUE.generate_failure_reports_from_rows(
                tmp,
                [case_row, process_row],
                include_process_rows_in_main=False,
            )
            with open(reports["failure_csv"], newline="", encoding="utf-8") as f:
                main_rows = list(csv.DictReader(f))
            with open(
                reports["unresolved_process_failure_csv"], newline="", encoding="utf-8"
            ) as f:
                unresolved_rows = list(csv.DictReader(f))
            self.assertEqual([row["nodeid"] for row in main_rows], [case_row["nodeid"]])
            self.assertEqual([row["nodeid"] for row in unresolved_rows], [process_row["nodeid"]])


if __name__ == "__main__":
    unittest.main()
