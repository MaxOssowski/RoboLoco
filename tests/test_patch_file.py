from __future__ import annotations

import unittest

from agents.planner import PlannerAgent
from orchestrator.main import Orchestrator
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from tools.filesystem import PatchFileTool


class PatchFileToolTests(unittest.TestCase):
    """Unit tests for the PatchFileTool."""

    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_patch_file"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def _write(self, name: str, content: str):
        p = self.test_dir / name
        p.write_text(content, encoding="utf-8")
        return p

    # ── happy path ────────────────────────────────────────────────────────────

    def test_applies_patch_to_existing_file(self) -> None:
        self._write("greet.py", "def greet():\n    return 'hello'\n")
        result = PatchFileTool.run({
            "path": "test_patch_file/greet.py",
            "old_lines": "    return 'hello'",
            "new_lines": "    return 'hi there'",
        })
        self.assertTrue(result.ok)
        self.assertIn("Patched file:", result.output)
        content = (self.test_dir / "greet.py").read_text(encoding="utf-8")
        self.assertIn("hi there", content)
        self.assertNotIn("'hello'", content)

    def test_replaces_only_first_occurrence(self) -> None:
        self._write("dup.txt", "foo\nfoo\nfoo\n")
        result = PatchFileTool.run({
            "path": "test_patch_file/dup.txt",
            "old_lines": "foo",
            "new_lines": "bar",
        })
        self.assertTrue(result.ok)
        content = (self.test_dir / "dup.txt").read_text(encoding="utf-8")
        # exactly one replacement
        self.assertEqual(content.count("foo"), 2)
        self.assertEqual(content.count("bar"), 1)

    def test_output_contains_relative_path(self) -> None:
        self._write("out.py", "x = 1\n")
        result = PatchFileTool.run({
            "path": "test_patch_file/out.py",
            "old_lines": "x = 1",
            "new_lines": "x = 2",
        })
        self.assertTrue(result.ok)
        self.assertIn("test_patch_file/out.py", result.output)

    def test_multiline_patch(self) -> None:
        original = "def foo():\n    x = 1\n    return x\n"
        self._write("multi.py", original)
        result = PatchFileTool.run({
            "path": "test_patch_file/multi.py",
            "old_lines": "    x = 1\n    return x",
            "new_lines": "    x = 42\n    return x * 2",
        })
        self.assertTrue(result.ok)
        content = (self.test_dir / "multi.py").read_text(encoding="utf-8")
        self.assertIn("42", content)
        self.assertIn("return x * 2", content)

    # ── failure: file does not exist ──────────────────────────────────────────

    def test_fails_when_file_does_not_exist(self) -> None:
        result = PatchFileTool.run({
            "path": "test_patch_file/nonexistent.py",
            "old_lines": "something",
            "new_lines": "other",
        })
        self.assertFalse(result.ok)
        self.assertIn("File not found", result.output)

    def test_error_message_mentions_create_alternative(self) -> None:
        result = PatchFileTool.run({
            "path": "test_patch_file/missing.py",
            "old_lines": "x",
            "new_lines": "y",
        })
        self.assertFalse(result.ok)
        # error should guide user toward creation tools
        lower = result.output.lower()
        self.assertTrue(
            "write_file" in lower or "code_file" in lower,
            f"Expected mention of creation tool, got: {result.output}",
        )

    # ── failure: context mismatch ─────────────────────────────────────────────

    def test_fails_when_old_lines_not_in_file(self) -> None:
        self._write("src.py", "x = 1\n")
        result = PatchFileTool.run({
            "path": "test_patch_file/src.py",
            "old_lines": "y = 999",
            "new_lines": "y = 0",
        })
        self.assertFalse(result.ok)
        self.assertIn("context mismatch", result.output)

    def test_file_unchanged_after_failed_patch(self) -> None:
        original = "x = 1\n"
        self._write("safe.py", original)
        PatchFileTool.run({
            "path": "test_patch_file/safe.py",
            "old_lines": "this does not exist",
            "new_lines": "replacement",
        })
        content = (self.test_dir / "safe.py").read_text(encoding="utf-8")
        self.assertEqual(content, original)

    def test_error_suggests_reading_file_first(self) -> None:
        self._write("ctx.py", "a = 1\n")
        result = PatchFileTool.run({
            "path": "test_patch_file/ctx.py",
            "old_lines": "b = 2",
            "new_lines": "b = 3",
        })
        self.assertFalse(result.ok)
        self.assertIn("Read the file first", result.output)

    # ── validation ────────────────────────────────────────────────────────────

    def test_fails_when_old_lines_is_empty(self) -> None:
        self._write("v.py", "x = 1\n")
        from orchestrator.policies import ValidationError
        with self.assertRaises(ValidationError):
            PatchFileTool.run({
                "path": "test_patch_file/v.py",
                "old_lines": "",
                "new_lines": "something",
            })

    def test_fails_when_old_lines_is_not_string(self) -> None:
        from orchestrator.policies import ValidationError
        with self.assertRaises(ValidationError):
            PatchFileTool.run({
                "path": "test_patch_file/v.py",
                "old_lines": 42,
                "new_lines": "something",
            })

    def test_fails_when_new_lines_is_not_string(self) -> None:
        from orchestrator.policies import ValidationError
        with self.assertRaises(ValidationError):
            PatchFileTool.run({
                "path": "test_patch_file/v.py",
                "old_lines": "x",
                "new_lines": None,
            })


class PatchFilePermissionsAndRegistryTests(unittest.TestCase):
    """Verify patch_file is wired into permissions and the tool registry."""

    def test_patch_file_in_coder_permissions(self) -> None:
        self.assertIn("patch_file", AGENT_PERMISSIONS["coder"])

    def test_patch_file_not_in_verifier_permissions(self) -> None:
        self.assertNotIn("patch_file", AGENT_PERMISSIONS["verifier"])

    def test_patch_file_not_in_researcher_permissions(self) -> None:
        self.assertNotIn("patch_file", AGENT_PERMISSIONS["researcher"])

    def test_patch_file_in_orchestrator_registry(self) -> None:
        orchestrator = Orchestrator(model="test-model")
        self.assertIn("patch_file", orchestrator.tool_registry)

    def test_planner_permissions_advertise_patch_file(self) -> None:
        self.assertIn("patch_file", PlannerAgent._PERMISSIONS)

    def test_planner_rules_mention_patch_file(self) -> None:
        self.assertIn("patch_file", PlannerAgent._RULES)

    def test_planner_schema_example_uses_patch_file(self) -> None:
        self.assertIn("patch_file", PlannerAgent._SCHEMA)
        self.assertIn("old_lines", PlannerAgent._SCHEMA)
        self.assertIn("new_lines", PlannerAgent._SCHEMA)

    def test_planner_validates_patch_file_action_shape(self) -> None:
        """_validate_plan should accept a patch_file action for coder."""
        orchestrator = Orchestrator(model="test-model")
        data = {
            "reasoning_summary": "Patch a file.",
            "status": "ready",
            "success_criteria": ["file out.py contains updated greeting"],
            "actions": [
                {
                    "agent": "coder",
                    "tool": "patch_file",
                    "args": {
                        "path": "out.py",
                        "old_lines": "print('hello')",
                        "new_lines": "print('hi')",
                    },
                }
            ],
        }
        result = orchestrator.planner._validate_plan(data, orchestrator.tool_registry)
        self.assertEqual(result.status, "ready")
        self.assertEqual(result.actions[0].tool, "patch_file")


class FileCreationVsModificationTests(unittest.TestCase):
    """Verify the intended role separation between creation and modification tools."""

    def test_write_file_creates_new_file(self) -> None:
        from tools.filesystem import FilesystemTool
        test_path = WORKSPACE / "test_patch_file_creation"
        test_path.mkdir(parents=True, exist_ok=True)
        target = test_path / "brand_new.txt"
        try:
            result = FilesystemTool.run({
                "path": "test_patch_file_creation/brand_new.txt",
                "content": "created",
            })
            self.assertTrue(result.ok)
            self.assertTrue(target.exists())
        finally:
            if target.exists():
                target.unlink()
            if test_path.exists():
                test_path.rmdir()

    def test_patch_file_refuses_to_create_missing_file(self) -> None:
        """patch_file must not create files; creation is write_file / code_file's job."""
        result = PatchFileTool.run({
            "path": "test_patch_file/absolutely_new_file.py",
            "old_lines": "anything",
            "new_lines": "something else",
        })
        self.assertFalse(result.ok)
        # file must NOT have been created
        self.assertFalse((WORKSPACE / "test_patch_file" / "absolutely_new_file.py").exists())


if __name__ == "__main__":
    unittest.main()
