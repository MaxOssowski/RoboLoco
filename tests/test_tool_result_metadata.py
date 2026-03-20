"""Tests for structured ToolResult.data metadata across all file-writing tools.

Verifies that:
- ToolResult carries a ``data`` dict with machine-readable fields.
- write_file, replace_in_file, patch_file, file_exists, code_file and
  modify_file all populate ``data`` correctly.
- Orchestrator._build_summary consumes ``data`` to build files_touched without
  relying on string-prefix parsing.
- VerifierAgent._written_files_exist consumes ``data`` and falls back
  gracefully to the legacy string format when ``data`` is absent.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.coder import CoderAgent
from agents.verifier import VerifierAgent
from orchestrator.main import Orchestrator
from orchestrator.policies import WORKSPACE
from orchestrator.state import AgentResult, TaskState, ToolCall, ToolResult
from tools.filesystem import (
    FileExistsTool,
    FilesystemTool,
    PatchFileTool,
    ReplaceInFileTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verifier(strict: bool = False) -> VerifierAgent:
    return VerifierAgent(model="test-model", tool_registry={}, strict_verifier=strict)


def _tool_event(payload: dict) -> dict:
    return {"event_type": "tool_result", "payload": payload}


# ---------------------------------------------------------------------------
# ToolResult default
# ---------------------------------------------------------------------------

class ToolResultDefaultDataTest(unittest.TestCase):
    def test_data_defaults_to_empty_dict(self) -> None:
        r = ToolResult(ok=True, tool="some_tool", output="done")
        self.assertEqual(r.data, {})

    def test_data_is_independent_between_instances(self) -> None:
        r1 = ToolResult(ok=True, tool="t", output="x")
        r2 = ToolResult(ok=True, tool="t", output="y")
        r1.data["foo"] = "bar"
        self.assertNotIn("foo", r2.data)


# ---------------------------------------------------------------------------
# write_file metadata
# ---------------------------------------------------------------------------

class WriteFileMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_write"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_write_file_emits_file_write_kind(self) -> None:
        result = FilesystemTool.run({
            "path": "test_metadata_write/out.txt",
            "content": "hello",
        })
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_write")

    def test_write_file_data_path_is_relative(self) -> None:
        result = FilesystemTool.run({
            "path": "test_metadata_write/out.txt",
            "content": "hello",
        })
        self.assertTrue(result.ok)
        p = result.data.get("path", "")
        self.assertFalse(p.startswith("/"), f"Expected relative path, got: {p}")
        self.assertIn("test_metadata_write/out.txt", p)

    def test_write_file_data_path_matches_output(self) -> None:
        result = FilesystemTool.run({
            "path": "test_metadata_write/match.txt",
            "content": "x",
        })
        self.assertTrue(result.ok)
        # The path in data should appear in the human-readable output too
        self.assertIn(result.data["path"], result.output)


# ---------------------------------------------------------------------------
# replace_in_file metadata
# ---------------------------------------------------------------------------

class ReplaceInFileMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_replace"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "src.txt").write_text("hello world\n", encoding="utf-8")

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_replace_in_file_emits_file_replace_kind(self) -> None:
        result = ReplaceInFileTool.run({
            "path": "test_metadata_replace/src.txt",
            "old_text": "hello",
            "new_text": "hi",
        })
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_replace")

    def test_replace_in_file_data_path_is_relative(self) -> None:
        result = ReplaceInFileTool.run({
            "path": "test_metadata_replace/src.txt",
            "old_text": "world",
            "new_text": "there",
        })
        self.assertTrue(result.ok)
        p = result.data.get("path", "")
        self.assertFalse(p.startswith("/"), f"Expected relative path, got: {p}")


# ---------------------------------------------------------------------------
# patch_file metadata
# ---------------------------------------------------------------------------

class PatchFileMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_patch"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "code.py").write_text("x = 1\n", encoding="utf-8")

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_patch_file_emits_file_patch_kind(self) -> None:
        result = PatchFileTool.run({
            "path": "test_metadata_patch/code.py",
            "old_lines": "x = 1",
            "new_lines": "x = 2",
        })
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_patch")

    def test_patch_file_data_path_is_relative(self) -> None:
        # Reset the file so this test is independent of ordering.
        (self.test_dir / "code.py").write_text("x = 10\n", encoding="utf-8")
        result = PatchFileTool.run({
            "path": "test_metadata_patch/code.py",
            "old_lines": "x = 10",
            "new_lines": "x = 20",
        })
        self.assertTrue(result.ok)
        p = result.data.get("path", "")
        self.assertFalse(p.startswith("/"), f"Expected relative path, got: {p}")

    def test_patch_file_failure_has_empty_data(self) -> None:
        result = PatchFileTool.run({
            "path": "test_metadata_patch/nonexistent.py",
            "old_lines": "anything",
            "new_lines": "other",
        })
        self.assertFalse(result.ok)
        self.assertEqual(result.data, {})


# ---------------------------------------------------------------------------
# file_exists metadata
# ---------------------------------------------------------------------------

class FileExistsMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_exists"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        (self.test_dir / "present.txt").write_text("here", encoding="utf-8")

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_file_exists_emits_file_exists_kind(self) -> None:
        result = FileExistsTool.run({"path": "test_metadata_exists/present.txt"})
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_exists")

    def test_file_exists_data_path_is_relative(self) -> None:
        result = FileExistsTool.run({"path": "test_metadata_exists/present.txt"})
        self.assertTrue(result.ok)
        p = result.data.get("path", "")
        self.assertFalse(p.startswith("/"), f"Expected relative path, got: {p}")

    def test_file_exists_data_has_exists_true(self) -> None:
        result = FileExistsTool.run({"path": "test_metadata_exists/present.txt"})
        self.assertTrue(result.ok)
        self.assertIs(result.data.get("exists"), True)

    def test_file_exists_missing_has_empty_data(self) -> None:
        result = FileExistsTool.run({"path": "test_metadata_exists/ghost.txt"})
        self.assertFalse(result.ok)
        self.assertEqual(result.data, {})


# ---------------------------------------------------------------------------
# code_file / modify_file metadata (via CoderAgent)
# ---------------------------------------------------------------------------

class CoderFileMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_coder"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.state = TaskState(goal="Test metadata.")
        self.coder = CoderAgent(tool_registry={}, model="test-model")

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_code_file_emits_file_write_kind(self) -> None:
        call = ToolCall(
            agent="coder", tool="code_file",
            args={"path": "test_metadata_coder/new.py", "specification": "Print 1."},
        )
        with patch.object(self.coder._llm, "generate", return_value="print(1)"):
            result = self.coder.execute(call, self.state)
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_write")

    def test_code_file_data_path_is_relative(self) -> None:
        call = ToolCall(
            agent="coder", tool="code_file",
            args={"path": "test_metadata_coder/new2.py", "specification": "Print 2."},
        )
        with patch.object(self.coder._llm, "generate", return_value="print(2)"):
            result = self.coder.execute(call, self.state)
        self.assertTrue(result.ok)
        p = result.data.get("path", "")
        self.assertFalse(p.startswith("/"), f"Expected relative path, got: {p}")

    def test_modify_file_emits_file_write_kind(self) -> None:
        src = self.test_dir / "edit.py"
        src.write_text("x = 1\n", encoding="utf-8")
        call = ToolCall(
            agent="coder", tool="modify_file",
            args={"path": "test_metadata_coder/edit.py", "specification": "Change x to 2."},
        )
        with patch.object(self.coder._llm, "generate", return_value="x = 2\n"):
            result = self.coder.execute(call, self.state)
        self.assertTrue(result.ok)
        self.assertEqual(result.data.get("kind"), "file_write")

    def test_code_file_data_logged_in_state(self) -> None:
        call = ToolCall(
            agent="coder", tool="code_file",
            args={"path": "test_metadata_coder/logged.py", "specification": "Print 3."},
        )
        with patch.object(self.coder._llm, "generate", return_value="print(3)"):
            self.coder.execute(call, self.state)
        events = [e for e in self.state.history if e["event_type"] == "tool_result"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["payload"]["data"]["kind"], "file_write")


# ---------------------------------------------------------------------------
# _build_summary consumes structured data
# ---------------------------------------------------------------------------

class BuildSummaryMetadataTest(unittest.TestCase):
    """_build_summary must use data["path"] for files_touched, not string parsing."""

    def _make_orchestrator(self) -> Orchestrator:
        return Orchestrator(model="test-model")

    def _make_plan(self):
        from orchestrator.state import AgentResult
        return AgentResult(agent="planner", reasoning_summary="plan", success_criteria=["done"])

    def _summary(self, tool_results: list) -> dict:
        orc = self._make_orchestrator()
        result = {"verification_status": "passed", "verification_summary": "ok"}
        import time
        return orc._build_summary(
            goal="test goal",
            result=result,
            all_tool_results=tool_results,
            plan=self._make_plan(),
            retry_count=0,
            start_time=time.monotonic() - 1,
        )

    def test_write_file_data_adds_to_files_touched(self) -> None:
        tool_results = [
            {"tool": "write_file", "ok": True, "output": "Wrote file: foo.py",
             "data": {"kind": "file_write", "path": "foo.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertIn("foo.py", summary["files_touched"])

    def test_patch_file_data_adds_to_files_touched(self) -> None:
        tool_results = [
            {"tool": "patch_file", "ok": True, "output": "Patched file: bar.py",
             "data": {"kind": "file_patch", "path": "bar.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertIn("bar.py", summary["files_touched"])

    def test_replace_in_file_data_adds_to_files_touched(self) -> None:
        tool_results = [
            {"tool": "replace_in_file", "ok": True, "output": "Replaced text in: baz.py",
             "data": {"kind": "file_replace", "path": "baz.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertIn("baz.py", summary["files_touched"])

    def test_code_file_data_adds_to_files_touched(self) -> None:
        tool_results = [
            {"tool": "code_file", "ok": True, "output": "Wrote file: gen.py",
             "data": {"kind": "file_write", "path": "gen.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertIn("gen.py", summary["files_touched"])

    def test_legacy_write_file_no_data_still_adds_to_files_touched(self) -> None:
        """Entries without data dict must still work via string fallback."""
        tool_results = [
            {"tool": "write_file", "ok": True, "output": "Wrote file: legacy.py", "data": {}},
        ]
        summary = self._summary(tool_results)
        self.assertIn("legacy.py", summary["files_touched"])

    def test_files_touched_deduplicated(self) -> None:
        tool_results = [
            {"tool": "write_file", "ok": True, "output": "Wrote file: dup.py",
             "data": {"kind": "file_write", "path": "dup.py"}},
            {"tool": "patch_file", "ok": True, "output": "Patched file: dup.py",
             "data": {"kind": "file_patch", "path": "dup.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertEqual(summary["files_touched"].count("dup.py"), 1)

    def test_failed_tool_not_added_to_files_touched(self) -> None:
        tool_results = [
            {"tool": "write_file", "ok": False, "output": "error",
             "data": {"kind": "file_write", "path": "should_not_appear.py"}},
        ]
        summary = self._summary(tool_results)
        self.assertNotIn("should_not_appear.py", summary["files_touched"])


# ---------------------------------------------------------------------------
# _written_files_exist consumes structured data
# ---------------------------------------------------------------------------

class WrittenFilesExistMetadataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_metadata_verifier"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.verifier = _make_verifier()

    def tearDown(self) -> None:
        for p in sorted(self.test_dir.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def _write(self, name: str) -> str:
        rel = f"test_metadata_verifier/{name}"
        (WORKSPACE / rel).write_text("content", encoding="utf-8")
        return rel

    def test_passes_when_file_exists_via_data(self) -> None:
        rel = self._write("present.py")
        events = [_tool_event(
            {"tool": "write_file", "ok": True, "output": f"Wrote file: {rel}",
             "data": {"kind": "file_write", "path": rel}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_fails_when_file_missing_via_data(self) -> None:
        rel = "test_metadata_verifier/ghost.py"
        events = [_tool_event(
            {"tool": "write_file", "ok": True, "output": f"Wrote file: {rel}",
             "data": {"kind": "file_write", "path": rel}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertFalse(ok)
        self.assertIn(rel, missing)

    def test_legacy_fallback_passes_when_file_exists(self) -> None:
        """data={} triggers legacy string parsing and still works correctly."""
        rel = self._write("legacy.py")
        events = [_tool_event(
            {"tool": "write_file", "ok": True, "output": f"Wrote file: {rel}", "data": {}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_legacy_fallback_fails_when_file_missing(self) -> None:
        rel = "test_metadata_verifier/missing_legacy.py"
        events = [_tool_event(
            {"tool": "write_file", "ok": True, "output": f"Wrote file: {rel}", "data": {}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertFalse(ok)
        self.assertIn(rel, missing)

    def test_no_data_key_at_all_falls_back_to_legacy(self) -> None:
        """Payload without 'data' key entirely is treated as legacy."""
        rel = self._write("no_data_key.py")
        events = [_tool_event(
            {"tool": "write_file", "ok": True, "output": f"Wrote file: {rel}"}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_code_file_with_data_checked(self) -> None:
        rel = self._write("codefile.py")
        events = [_tool_event(
            {"tool": "code_file", "ok": True, "output": f"Wrote file: {rel}",
             "data": {"kind": "file_write", "path": rel}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertTrue(ok)

    def test_modify_file_with_data_missing_detected(self) -> None:
        rel = "test_metadata_verifier/no_modify.py"
        events = [_tool_event(
            {"tool": "modify_file", "ok": True, "output": f"Wrote file: {rel}",
             "data": {"kind": "file_write", "path": rel}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertFalse(ok)
        self.assertIn(rel, missing)

    def test_failed_tool_events_ignored(self) -> None:
        rel = "test_metadata_verifier/failed.py"
        events = [_tool_event(
            {"tool": "write_file", "ok": False, "output": "error",
             "data": {"kind": "file_write", "path": rel}}
        )]
        ok, missing = self.verifier._written_files_exist(events)
        self.assertTrue(ok)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
