import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "inspect_test_run", ROOT / "scripts" / "inspect_test_run.py"
)
INSPECTOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(INSPECTOR)


class InspectTestRunTest(unittest.TestCase):
    def make_direct_run(self, root: Path, unresolved_rows: int = 0) -> Path:
        work = root / "work"
        latest = work / "20260715_120000"
        latest.mkdir(parents=True)
        (work / "latest").symlink_to(latest.name)
        plan = ["test_real.py"] + [
            f"{name}.py" for name in sorted(INSPECTOR.KNOWN_DIRECT_VIRTUAL_TARGETS)
        ]
        (work / "test_files.txt").write_text("\n".join(plan) + "\n")
        (work / ".test_progress.json").write_text(
            json.dumps({"tests": {"test_real.py": {"status": "PASS"}}})
        )
        (work / "summary.json").write_text(
            json.dumps(
                {
                    "failure_reports": {
                        "unresolved_process_failure_count": unresolved_rows,
                    }
                }
            )
        )
        fields = [
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
        for name in ("failure_report.csv", "unresolved_process_failures.csv"):
            with (latest / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                if name.startswith("unresolved"):
                    for index in range(unresolved_rows):
                        writer.writerow(
                            {
                                "test_file": f"test_timeout_{index}.py",
                                "case_name": "<timeout>",
                                "error_type": "Timeout",
                                "nodeid": f"test_timeout_{index}.py",
                            }
                        )
        return work

    def test_virtual_targets_do_not_count_as_missing_real_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.make_direct_run(Path(tmp))
            result = INSPECTOR.inspect(work)
            self.assertEqual(result["virtual_plan_count"], 5)
            self.assertEqual(result["missing_real_count"], 0)
            self.assertEqual(result["artifact_verdict"], "COMPLETE")

    def test_unresolved_rows_make_finalized_run_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.make_direct_run(Path(tmp), unresolved_rows=2)
            result = INSPECTOR.inspect(work)
            self.assertEqual(result["artifact_verdict"], "FINALIZED_INCOMPLETE")
            self.assertIn(
                "2 unresolved process-level failures remain", result["completion_issues"]
            )

    def test_lightweight_official_run_does_not_require_queue_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.make_direct_run(Path(tmp))
            (work / ".test_progress.json").rename(work / ".run_test_progress.json")
            (work / "test_files.txt").rename(work / "run_test_modules.txt")
            (work / "run_test_modules.txt").write_text("test_real.py\n")
            result = INSPECTOR.inspect(work)
            self.assertEqual(result["mode"], "official-run-test-lightweight")
            self.assertEqual(result["artifact_verdict"], "COMPLETE")

    def test_complete_official_queue_requires_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.make_direct_run(Path(tmp))
            (work / ".test_progress.json").rename(work / ".run_test_progress.json")
            (work / "test_files.txt").rename(work / "run_test_tests.txt")
            (work / "run_test_tests.txt").write_text("test_real.py\n")
            result = INSPECTOR.inspect(work)
            self.assertEqual(result["mode"], "official-queue")
            self.assertIn(
                "official coverage_report.json is missing or invalid",
                result["completion_issues"],
            )


if __name__ == "__main__":
    unittest.main()
