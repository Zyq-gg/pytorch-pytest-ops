import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNNERS = Path(__file__).resolve().parents[1] / "runners"
sys.path.insert(0, str(RUNNERS))

from run_manifest import build_run_manifest, ensure_run_manifest  # noqa: E402


class RunManifestTest(unittest.TestCase):
    def make_manifest(self, root: Path, *, timeout: int = 10):
        return build_run_manifest(
            runner="test-runner",
            pytorch_root=root,
            planned_items=["test_one.py"],
            options={"timeout": timeout, "mode": "normal"},
            env={
                "PATH": "/usr/bin",
                "TORCHINDUCTOR_CPP_MARCH": "znver1",
                "UNRELATED_SECRET": "must-not-be-recorded",
            },
        )

    def test_manifest_verifies_same_semantic_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            first = self.make_manifest(root)
            ensure_run_manifest(
                work, first, fresh=True, checkpoint_exists=False
            )
            ensure_run_manifest(
                work,
                self.make_manifest(root),
                fresh=False,
                checkpoint_exists=True,
            )

    def test_manifest_rejects_changed_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            ensure_run_manifest(
                work,
                self.make_manifest(root, timeout=10),
                fresh=True,
                checkpoint_exists=False,
            )
            with self.assertRaisesRegex(RuntimeError, "options"):
                ensure_run_manifest(
                    work,
                    self.make_manifest(root, timeout=20),
                    fresh=False,
                    checkpoint_exists=True,
                )

    def test_manifest_can_ignore_process_only_timeout_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            ensure_run_manifest(
                work,
                self.make_manifest(root, timeout=10),
                fresh=True,
                checkpoint_exists=False,
            )
            ensure_run_manifest(
                work,
                self.make_manifest(root, timeout=20),
                fresh=False,
                checkpoint_exists=True,
                ignored_paths={"options.timeout"},
            )

    def test_legacy_checkpoint_is_marked_as_unverified_adoption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            path = ensure_run_manifest(
                work,
                self.make_manifest(root),
                fresh=False,
                checkpoint_exists=True,
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(data["legacy_checkpoint_adopted_without_verification"])

    def test_legacy_checkpoint_rejects_a_changed_saved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work"
            work.mkdir()
            saved_plan = work / "test_files.txt"
            saved_plan.write_text("test_old.py\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "legacy checkpoint plan"):
                ensure_run_manifest(
                    work,
                    self.make_manifest(root),
                    fresh=False,
                    checkpoint_exists=True,
                    saved_plan_path=saved_plan,
                )
            self.assertFalse((work / ".run_manifest.json").exists())

    def test_manifest_does_not_store_environment_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.make_manifest(Path(tmp))
            serialized = json.dumps(manifest)
            self.assertNotIn("znver1", serialized)
            self.assertNotIn("must-not-be-recorded", serialized)
            self.assertEqual(
                manifest["details"]["environment"]["keys"],
                ["PATH", "TORCHINDUCTOR_CPP_MARCH"],
            )


if __name__ == "__main__":
    unittest.main()
