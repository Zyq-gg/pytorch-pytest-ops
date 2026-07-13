from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "runners" / "run_pytorch_tests_prefix.py"
SPEC = importlib.util.spec_from_file_location("run_pytorch_tests_prefix", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StepcurrentCacheCompatibilityTest(unittest.TestCase):
    def make_test_dir(self, root: Path) -> Path:
        test_dir = root / "test"
        test_dir.mkdir()
        return test_dir

    def cache_entry(self, root: Path, key: str) -> Path:
        return root / ".pytest_cache" / "v" / "cache" / "stepcurrent" / key

    def test_reads_pytorch_29_flat_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = self.make_test_dir(root)
            entry = self.cache_entry(root, "flat-key")
            entry.parent.mkdir(parents=True)
            entry.write_text(json.dumps("test_file.py::TestCase::test_value"), encoding="utf-8")

            self.assertEqual(
                MODULE.read_stepcurrent_lastrun(str(test_dir), "flat-key"),
                "test_file.py::TestCase::test_value",
            )

    def test_reads_new_directory_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = self.make_test_dir(root)
            entry = self.cache_entry(root, "directory-key")
            entry.mkdir(parents=True)
            (entry / "lastrun").write_text(
                json.dumps("test_file.py::TestCase::test_value"), encoding="utf-8"
            )

            self.assertEqual(
                MODULE.read_stepcurrent_lastrun(str(test_dir), "directory-key"),
                "test_file.py::TestCase::test_value",
            )

    def test_incomplete_directory_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = self.make_test_dir(root)
            self.cache_entry(root, "incomplete-key").mkdir(parents=True)

            self.assertEqual(
                MODULE.read_stepcurrent_lastrun(str(test_dir), "incomplete-key"), ""
            )


if __name__ == "__main__":
    unittest.main()
