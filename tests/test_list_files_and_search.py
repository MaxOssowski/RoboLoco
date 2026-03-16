from __future__ import annotations

import unittest

from orchestrator.policies import WORKSPACE
from tools.filesystem import ListFilesTool
from tools.search import SearchInFilesTool


class ListFilesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_list_files"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "alpha.py").write_text("x = 1", encoding="utf-8")
        (self.test_dir / "beta.py").write_text("y = 2", encoding="utf-8")

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_list_files_output_is_newline_separated(self) -> None:
        result = ListFilesTool.run({"path": str(self.test_dir.relative_to(WORKSPACE))})

        self.assertTrue(result.ok)
        lines = result.output.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(all(line.startswith("[FILE]") for line in lines))

    def test_list_files_empty_directory(self) -> None:
        empty_dir = WORKSPACE / "test_list_files_empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = ListFilesTool.run({"path": str(empty_dir.relative_to(WORKSPACE))})
            self.assertTrue(result.ok)
            self.assertEqual(result.output, "<empty directory>")
        finally:
            empty_dir.rmdir()


class SearchInFilesToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_search"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "sample.py").write_text(
            "def hello():\n    pass\n\ndef world():\n    pass\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_search_literal_string(self) -> None:
        result = SearchInFilesTool.run({"pattern": "hello", "path": str(self.test_dir.relative_to(WORKSPACE))})
        self.assertTrue(result.ok)
        self.assertIn("hello", result.output)

    def test_search_regex_pattern(self) -> None:
        result = SearchInFilesTool.run(
            {"pattern": r"def \w+\(", "path": str(self.test_dir.relative_to(WORKSPACE))}
        )
        self.assertTrue(result.ok)
        lines = result.output.splitlines()
        self.assertEqual(len(lines), 2)

    def test_search_output_is_newline_separated(self) -> None:
        result = SearchInFilesTool.run(
            {"pattern": r"def \w+\(", "path": str(self.test_dir.relative_to(WORKSPACE))}
        )
        self.assertTrue(result.ok)
        # Each match is on its own line
        for line in result.output.splitlines():
            self.assertIn(":", line)

    def test_search_invalid_regex_returns_error(self) -> None:
        result = SearchInFilesTool.run({"pattern": "[invalid", "path": "."})
        self.assertFalse(result.ok)
        self.assertIn("Invalid regex pattern", result.output)

    def test_search_no_matches(self) -> None:
        result = SearchInFilesTool.run({"pattern": "zzz_no_match_zzz", "path": str(self.test_dir.relative_to(WORKSPACE))})
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "<no matches>")


if __name__ == "__main__":
    unittest.main()
