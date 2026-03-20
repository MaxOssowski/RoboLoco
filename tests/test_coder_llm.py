from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.coder import CoderAgent
from agents.planner import PlannerAgent
from orchestrator.main import Orchestrator
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from orchestrator.state import TaskState, ToolCall


class CoderLLMTests(unittest.TestCase):
    """Tests for CoderAgent's LLM-backed code_file and modify_file actions."""

    def setUp(self) -> None:
        self.test_dir = WORKSPACE / "test_coder_llm"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.state = TaskState(goal="Create a hello world script.")
        self.coder = CoderAgent(tool_registry={}, model="test-model")

    def tearDown(self) -> None:
        for path in sorted(self.test_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    # ── code_file ─────────────────────────────────────────────────────────────

    def test_code_file_creates_file_with_llm_content(self) -> None:
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={
                "path": "test_coder_llm/hello.py",
                "specification": "Print 'hello world' to stdout.",
            },
        )
        with patch.object(self.coder._llm, "generate", return_value="print('hello world')"):
            result = self.coder.execute(call, self.state)

        self.assertTrue(result.ok)
        self.assertEqual(result.tool, "code_file")
        self.assertIn("Wrote file:", result.output)
        written = (WORKSPACE / "test_coder_llm" / "hello.py").read_text(encoding="utf-8")
        self.assertIn("hello world", written)

    # ── _extract_code: chain-of-thought / thinking model output ──────────────

    def test_extract_code_strips_think_block(self) -> None:
        raw = "<think>\nOkay, the user wants hello world.\n</think>\nprint('hello world')"
        result = self.coder._extract_code(raw)
        self.assertEqual(result, "print('hello world')")

    def test_extract_code_strips_multiline_think_block(self) -> None:
        raw = (
            "<think>\n"
            "Let me reason through this.\n"
            "Step 1: import sys\n"
            "Step 2: call print\n"
            "</think>\n"
            "print('hello world')"
        )
        result = self.coder._extract_code(raw)
        self.assertEqual(result, "print('hello world')")
        self.assertNotIn("<think>", result)
        self.assertNotIn("Let me reason", result)

    def test_extract_code_strips_think_block_before_fence(self) -> None:
        raw = (
            "<think>user wants fenced code</think>\n"
            "```python\nprint('hi')\n```"
        )
        result = self.coder._extract_code(raw)
        self.assertNotIn("<think>", result)
        self.assertNotIn("```", result)
        self.assertEqual(result, "print('hi')")

    def test_extract_code_strips_bare_close_think_tag(self) -> None:
        # Some model variants emit </think> without an opening tag
        raw = "</think>\nprint('hi')"
        result = self.coder._extract_code(raw)
        self.assertNotIn("</think>", result)
        self.assertEqual(result, "print('hi')")

    def test_extract_code_no_contamination_unchanged(self) -> None:
        raw = "print('hello world')"
        self.assertEqual(self.coder._extract_code(raw), raw)

    def test_code_file_with_think_block_writes_clean_code(self) -> None:
        """End-to-end: LLM response with <think> block produces a clean file."""
        llm_response = (
            "<think>\nOkay, I need to print hello world.\n</think>\n"
            "print('hello world')"
        )
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={"path": "test_coder_llm/think.py", "specification": "Print hello world."},
        )
        with patch.object(self.coder._llm, "generate", return_value=llm_response):
            result = self.coder.execute(call, self.state)

        self.assertTrue(result.ok)
        written = (WORKSPACE / "test_coder_llm" / "think.py").read_text(encoding="utf-8")
        self.assertNotIn("<think>", written)
        self.assertNotIn("Okay, I need", written)
        self.assertIn("print", written)

    def test_code_file_strips_markdown_fences(self) -> None:
        llm_response = "```python\nprint('hello')\n```"
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={"path": "test_coder_llm/fenced.py", "specification": "Print hello."},
        )
        with patch.object(self.coder._llm, "generate", return_value=llm_response):
            result = self.coder.execute(call, self.state)

        self.assertTrue(result.ok)
        written = (WORKSPACE / "test_coder_llm" / "fenced.py").read_text(encoding="utf-8")
        self.assertNotIn("```", written)
        self.assertIn("print", written)

    def test_code_file_fails_on_llm_error(self) -> None:
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={"path": "test_coder_llm/fail.py", "specification": "Something."},
        )
        with patch.object(self.coder._llm, "generate", side_effect=RuntimeError("ollama crashed")):
            result = self.coder.execute(call, self.state)

        self.assertFalse(result.ok)
        self.assertIn("LLM generation failed", result.output)
        self.assertIn("ollama crashed", result.output)

    def test_code_file_fails_on_empty_llm_response(self) -> None:
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={"path": "test_coder_llm/empty.py", "specification": "Something."},
        )
        with patch.object(self.coder._llm, "generate", return_value=""):
            result = self.coder.execute(call, self.state)

        self.assertFalse(result.ok)
        self.assertIn("empty", result.output.lower())

    def test_code_file_fails_without_path(self) -> None:
        call = ToolCall(
            agent="coder", tool="code_file", args={"specification": "Do something."}
        )
        result = self.coder.execute(call, self.state)
        self.assertFalse(result.ok)
        self.assertIn("path", result.output.lower())

    def test_code_file_fails_without_specification(self) -> None:
        call = ToolCall(
            agent="coder", tool="code_file", args={"path": "test_coder_llm/x.py"}
        )
        result = self.coder.execute(call, self.state)
        self.assertFalse(result.ok)
        self.assertIn("specification", result.output.lower())

    # ── modify_file ───────────────────────────────────────────────────────────

    def test_modify_file_updates_existing_file(self) -> None:
        src = self.test_dir / "greet.py"
        src.write_text("print('hello')\n", encoding="utf-8")

        call = ToolCall(
            agent="coder",
            tool="modify_file",
            args={
                "path": "test_coder_llm/greet.py",
                "specification": "Change the greeting to 'hi there'.",
            },
        )
        with patch.object(self.coder._llm, "generate", return_value="print('hi there')\n"):
            result = self.coder.execute(call, self.state)

        self.assertTrue(result.ok)
        self.assertEqual(result.tool, "modify_file")
        updated = src.read_text(encoding="utf-8")
        self.assertIn("hi there", updated)

    def test_modify_file_fails_when_source_missing(self) -> None:
        call = ToolCall(
            agent="coder",
            tool="modify_file",
            args={
                "path": "test_coder_llm/nonexistent.py",
                "specification": "Add a function.",
            },
        )
        result = self.coder.execute(call, self.state)
        self.assertFalse(result.ok)
        self.assertIn("Cannot read source file", result.output)

    def test_modify_file_fails_on_llm_error(self) -> None:
        src = self.test_dir / "target.py"
        src.write_text("x = 1\n", encoding="utf-8")

        call = ToolCall(
            agent="coder",
            tool="modify_file",
            args={"path": "test_coder_llm/target.py", "specification": "Rename x to y."},
        )
        with patch.object(self.coder._llm, "generate", side_effect=RuntimeError("timeout")):
            result = self.coder.execute(call, self.state)

        self.assertFalse(result.ok)
        self.assertIn("LLM generation failed", result.output)

    # ── tool result logging ───────────────────────────────────────────────────

    def test_execute_logs_tool_result_to_state(self) -> None:
        call = ToolCall(
            agent="coder",
            tool="code_file",
            args={"path": "test_coder_llm/logged.py", "specification": "Print 42."},
        )
        with patch.object(self.coder._llm, "generate", return_value="print(42)"):
            self.coder.execute(call, self.state)

        tool_events = [e for e in self.state.history if e["event_type"] == "tool_result"]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0]["payload"]["tool"], "code_file")
        self.assertTrue(tool_events[0]["payload"]["ok"])

    # ── permissions and registry ──────────────────────────────────────────────

    def test_code_file_and_modify_file_in_coder_permissions(self) -> None:
        self.assertIn("code_file", AGENT_PERMISSIONS["coder"])
        self.assertIn("modify_file", AGENT_PERMISSIONS["coder"])

    def test_code_file_and_modify_file_in_orchestrator_registry(self) -> None:
        orchestrator = Orchestrator(model="test-model")
        self.assertIn("code_file", orchestrator.tool_registry)
        self.assertIn("modify_file", orchestrator.tool_registry)

    # ── planner schema alignment ──────────────────────────────────────────────

    def test_planner_permissions_advertise_code_file_and_modify_file(self) -> None:
        self.assertIn("code_file", PlannerAgent._PERMISSIONS)
        self.assertIn("modify_file", PlannerAgent._PERMISSIONS)

    def test_planner_rules_instruct_coder_not_to_write_code_itself(self) -> None:
        # Planner rules should mention code_file for new files
        self.assertIn("code_file", PlannerAgent._RULES)
        # And must not instruct the planner to write full code bodies
        self.assertNotIn("write_file", PlannerAgent._RULES.split("code_file")[0].lower()
                         if "code_file" in PlannerAgent._RULES else "")

    def test_planner_schema_example_uses_code_file(self) -> None:
        self.assertIn("code_file", PlannerAgent._SCHEMA)
        self.assertIn("specification", PlannerAgent._SCHEMA)

    def test_planner_validates_code_file_action_shape(self) -> None:
        """_validate_plan should accept a code_file action for coder without error."""
        orchestrator = Orchestrator(model="test-model")
        planner = orchestrator.planner
        data = {
            "reasoning_summary": "Create a script.",
            "status": "ready",
            "success_criteria": ["file out.py exists", "python3 out.py exits with code 0"],
            "actions": [
                {
                    "agent": "coder",
                    "tool": "code_file",
                    "args": {"path": "out.py", "specification": "Print 1."},
                }
            ],
        }
        result = planner._validate_plan(data, orchestrator.tool_registry)
        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].tool, "code_file")


if __name__ == "__main__":
    unittest.main()
