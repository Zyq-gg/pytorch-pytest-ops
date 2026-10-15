import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_test_summary_csv.py"
SPEC = importlib.util.spec_from_file_location("convert_test_summary_csv", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CONVERTER)


class ConvertTestSummaryCsvTest(unittest.TestCase):
    def test_converts_concrete_parameterized_and_module_stage_rows(self):
        rows = [
            {
                "py 文件": "inductor/test_example.py",
                "类名": "TestExample",
                "case 名": "test_value[dtype-float32]",
                "错误类型及信息": "AssertionError: values differ",
            },
            {
                "py 文件": "test_import.py",
                "类名": "(文件级)",
                "case 名": "(收集/初始化阶段失败, 未执行任何 case)",
                "错误类型及信息": "ImportError: missing symbol",
            },
            {
                "py 文件": "test_function.py",
                "类名": "",
                "case 名": "test_top_level",
                "错误类型及信息": "plain failure",
            },
        ]

        converted, stats = CONVERTER.convert_rows(rows)

        self.assertEqual(stats["concrete_cases"], 2)
        self.assertEqual(stats["module_stage_rows"], 1)
        self.assertEqual(
            converted[0]["nodeid"],
            "test/inductor/test_example.py::TestExample::test_value[dtype-float32]",
        )
        self.assertEqual(converted[0]["error_type"], "AssertionError")
        self.assertEqual(converted[1]["case_name"], "<module-stage>")
        self.assertEqual(converted[1]["nodeid"], "test/test_import.py")
        self.assertEqual(converted[2]["nodeid"], "test/test_function.py::test_top_level")

    def test_deduplicates_nodeids(self):
        row = {
            "py 文件": "test_example.py",
            "类名": "Tests",
            "case 名": "test_case",
            "错误类型及信息": "AssertionError: bad",
        }
        converted, stats = CONVERTER.convert_rows([row, dict(row)])
        self.assertEqual(len(converted), 1)
        self.assertEqual(stats["concrete_cases"], 1)
        self.assertEqual(stats["duplicate_nodeids"], 1)

    def test_validates_only_concrete_targets(self):
        rows = [
            {"test_file": "test/existing.py", "nodeid": "test/existing.py::Tests::test_case"},
            {"test_file": "test/virtual.py", "nodeid": "test/virtual.py"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test").mkdir()
            (root / "test" / "existing.py").touch()
            self.assertEqual(CONVERTER.validate_concrete_targets(rows, root), [])

    def test_output_fields_match_stable_rerun_input(self):
        expected = {
            "source_log", "gpu", "test_file", "class_name", "case_name",
            "case_params", "error_type", "error_message", "nodeid", "raw",
        }
        self.assertEqual(set(CONVERTER.OUTPUT_FIELDS), expected)


if __name__ == "__main__":
    unittest.main()
