from __future__ import annotations

import unittest

from orchestrator.main import Orchestrator
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from tools.filesystem import FileExistsTool


class FileExistsToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_file_exists_tool"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_file_exists_succeeds_for_existing_file(self) -> None:
        file_path = self.test_dir / "artifact.txt"
        file_path.write_text("ok", encoding="utf-8")

        result = FileExistsTool.run({"path": str(file_path.relative_to(WORKSPACE))})

        self.assertTrue(result.ok)
        self.assertEqual(result.tool, "file_exists")
        self.assertIn("File exists:", result.output)

    def test_file_exists_fails_for_missing_file(self) -> None:
        result = FileExistsTool.run({"path": "test_file_exists_tool/missing.txt"})

        self.assertFalse(result.ok)
        self.assertIn("File not found:", result.output)

    def test_file_exists_fails_for_directory_path(self) -> None:
        result = FileExistsTool.run({"path": str(self.test_dir.relative_to(WORKSPACE))})

        self.assertFalse(result.ok)
        self.assertIn("Path is not a file:", result.output)

    def test_registry_and_permissions_expose_file_exists(self) -> None:
        orchestrator = Orchestrator(model="test-model", strict_verifier=True)

        self.assertIn("file_exists", orchestrator.tool_registry)
        self.assertIn("file_exists", AGENT_PERMISSIONS["verifier"])
        self.assertIn("file_exists", AGENT_PERMISSIONS["researcher"])


if __name__ == "__main__":
    unittest.main()
