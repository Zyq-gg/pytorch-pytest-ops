import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))

import rerun_stable_failures  # noqa: E402
import run_official_run_test_queue  # noqa: E402
import run_pytorch_subset  # noqa: E402
import run_pytorch_tests_prefix  # noqa: E402


MIOPEN_DETERMINISTIC = "MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC"


class EnvironmentInheritanceTest(unittest.TestCase):
    def generated_environments(self):
        return (
            run_pytorch_tests_prefix.make_test_env(),
            run_pytorch_subset.make_env(),
            rerun_stable_failures.make_env(None),
            run_official_run_test_queue.make_env(),
        )

    def test_miopen_deterministic_remains_unset_when_not_in_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            for env in self.generated_environments():
                self.assertNotIn(MIOPEN_DETERMINISTIC, env)

    def test_miopen_deterministic_is_inherited_without_override(self):
        with patch.dict(os.environ, {MIOPEN_DETERMINISTIC: "0"}, clear=True):
            for env in self.generated_environments():
                self.assertEqual(env[MIOPEN_DETERMINISTIC], "0")


if __name__ == "__main__":
    unittest.main()
