from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER = Path(__file__).resolve().parents[1] / "runners" / "run_pytorch_tests_prefix.py"
SPEC = importlib.util.spec_from_file_location("run_pytorch_tests_prefix_budget", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecoveryBudgetTest(unittest.TestCase):
    def test_timeout_is_capped_by_remaining_budget(self) -> None:
        with mock.patch.object(MODULE.time, "monotonic", return_value=100.0):
            self.assertEqual(MODULE.bounded_recovery_timeout(5, 110.0), 5)
            self.assertEqual(MODULE.bounded_recovery_timeout(30, 110.0), 10)
            self.assertEqual(MODULE.bounded_recovery_timeout(0, 110.0), 10)

    def test_expired_budget_raises(self) -> None:
        with mock.patch.object(MODULE.time, "monotonic", return_value=100.0):
            with self.assertRaises(MODULE.RecoveryBudgetExceeded):
                MODULE.bounded_recovery_timeout(5, 100.0)

    def test_single_crash_is_retried_then_can_pass(self) -> None:
        outcomes = [(-6, "Aborted"), (0, "1 passed in 0.01s")]

        def fake_run(*args, **kwargs):
            return outcomes.pop(0)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            MODULE, "run_with_pty", side_effect=fake_run
        ):
            stats, failures, failed = MODULE.run_recovery_targets(
                targets=["test_file.py::TestCase::test_value"],
                test_dir=tmp,
                env={},
                log=io.StringIO(),
                timeout=60,
                chunk_size=1,
                max_attempts=3,
            )

        self.assertFalse(failed)
        self.assertEqual(failures, [])
        self.assertEqual(outcomes, [])
        self.assertEqual(stats["passed"], 1)


if __name__ == "__main__":
    unittest.main()
