from __future__ import annotations

import json
import unittest

from orchestrator.main import Orchestrator
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from tools.filesystem import SummarizeFileTool


class SummarizeFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_summarize_file_tool"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_summarize_python_file_returns_structured_summary(self) -> None:
        file_path = self.test_dir / "sample.py"
        file_path.write_text(
            "import json\n\nclass Demo:\n    pass\n\ndef main():\n    return 1\n",
            encoding="utf-8",
        )

        result = SummarizeFileTool.run({"path": str(file_path.relative_to(WORKSPACE))})
        payload = json.loads(result.output)

        self.assertTrue(result.ok)
        self.assertEqual(payload["path"], "test_summarize_file_tool/sample.py")
        self.assertEqual(payload["extension"], ".py")
        self.assertIn("import json", payload["python_symbols"]["imports"])
        self.assertIn("Demo", payload["python_symbols"]["classes"])
        self.assertIn("main", payload["python_symbols"]["functions"])

    def test_summarize_file_fails_for_missing_file(self) -> None:
        result = SummarizeFileTool.run({"path": "test_summarize_file_tool/missing.py"})

        self.assertFalse(result.ok)
        self.assertIn("File not found:", result.output)

    def test_registry_and_permissions_expose_summarize_file(self) -> None:
        orchestrator = Orchestrator(model="test-model")

        self.assertIn("summarize_file", orchestrator.tool_registry)
        self.assertIn("summarize_file", AGENT_PERMISSIONS["researcher"])


if __name__ == "__main__":
    unittest.main()
