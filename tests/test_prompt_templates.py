"""Tests for the prompt template system (prompts/loader.py) and migrated prompt builders.

Covers:
- render_prompt() core mechanics: loading, substitution, error handling.
- PlannerAgent prompt builders: critical content is rendered correctly.
- VerifierAgent prompt builders: critical content is rendered correctly.

Tests assert on *targeted content* (key instructions, interpolated values,
required structural markers) rather than exact full-string snapshots, so they
remain stable under minor wording tweaks to the template files while still
catching regressions in content or interpolation.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.planner import PlannerAgent
from agents.verifier import VerifierAgent
from orchestrator.repair import RepairDiagnostic, fingerprint_failed_action
from orchestrator.state import TaskState
from prompts.loader import render_prompt


# ---------------------------------------------------------------------------
# render_prompt() unit tests
# ---------------------------------------------------------------------------


class RenderPromptLoadingTest(unittest.TestCase):
    """render_prompt loads templates from the prompts/ directory."""

    def test_raises_file_not_found_for_unknown_template(self) -> None:
        with self.assertRaises(FileNotFoundError):
            render_prompt("does_not_exist.md", foo="bar")

    def test_raises_key_error_for_missing_placeholder(self) -> None:
        # Write a temp template to the prompts directory isn't practical here,
        # so we test indirectly: a template that uses {{x}} without supplying x.
        # We verify that existing templates raise KeyError when required kwargs
        # are omitted (using planner_json_repair.md which needs {{malformed_output}}).
        with self.assertRaises(KeyError):
            render_prompt("planner_json_repair.md")  # malformed_output missing

    def test_extra_kwargs_are_silently_ignored(self) -> None:
        # planner_json_repair.md only needs {{malformed_output}}; extra keys must not error.
        result = render_prompt(
            "planner_json_repair.md",
            malformed_output="bad json",
            extra_unused_key="should be ignored",
        )
        self.assertIn("bad json", result)

    def test_single_braces_in_template_are_left_unchanged(self) -> None:
        """JSON literals in templates (single braces) must pass through untouched."""
        result = render_prompt("planner_json_repair.md", malformed_output="x")
        # The template itself has no JSON, but verifier_semantic does.
        result2 = render_prompt(
            "verifier_semantic.md",
            goal="g",
            criteria_section="",
            tool_text="[]",
        )
        # The JSON shape example uses single braces — they must appear verbatim.
        self.assertIn('{"status":', result2.replace(" ", "").replace("\n", ""))

    def test_rendered_output_is_stripped(self) -> None:
        result = render_prompt("planner_json_repair.md", malformed_output="x")
        self.assertEqual(result, result.strip())

    def test_placeholder_value_containing_double_braces_is_not_resubstituted(self) -> None:
        """Values that look like placeholders must NOT trigger a second substitution."""
        tricky = "{{malformed_output}}"  # looks like a placeholder
        result = render_prompt("planner_json_repair.md", malformed_output=tricky)
        # The value should appear verbatim, not cause a KeyError or infinite loop.
        self.assertIn(tricky, result)


# ---------------------------------------------------------------------------
# PlannerAgent prompt builders
# ---------------------------------------------------------------------------


class PlannerPlanPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PlannerAgent(model="test-model")

    def _prompt(self, goal: str = "create hello.py") -> str:
        state = TaskState(goal=goal)
        return self.planner._build_prompt(state)

    def test_contains_json_only_instruction(self) -> None:
        self.assertIn("Return ONLY valid JSON", self._prompt())

    def test_contains_planner_agent_role_description(self) -> None:
        self.assertIn("planner agent", self._prompt())

    def test_goal_is_interpolated(self) -> None:
        prompt = self._prompt(goal="write a snake game")
        self.assertIn("write a snake game", prompt)

    def test_contains_permissions(self) -> None:
        prompt = self._prompt()
        self.assertIn("coder", prompt)
        self.assertIn("verifier", prompt)

    def test_contains_rules(self) -> None:
        prompt = self._prompt()
        self.assertIn("code_file", prompt)
        self.assertIn("patch_file", prompt)

    def test_contains_json_schema_shape(self) -> None:
        prompt = self._prompt()
        self.assertIn("reasoning_summary", prompt)
        self.assertIn("success_criteria", prompt)
        self.assertIn("actions", prompt)

    def test_no_repair_mode_text_in_normal_prompt(self) -> None:
        self.assertNotIn("repair mode", self._prompt())

    def test_memory_section_absent_when_no_recent_summaries(self) -> None:
        with patch("agents.planner.load_recent_summaries", return_value=[]):
            prompt = self._prompt()
        self.assertNotIn("Prior completed tasks", prompt)

    def test_memory_section_present_when_recent_summaries_exist(self) -> None:
        fake_summary = [{"goal": "old task", "status": "passed", "files_touched": ["a.py"], "key_outputs": []}]
        with patch("agents.planner.load_recent_summaries", return_value=fake_summary):
            prompt = self._prompt()
        self.assertIn("Prior completed tasks", prompt)
        self.assertIn("old task", prompt)


class PlannerJsonRepairPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PlannerAgent(model="test-model")

    def test_contains_json_only_instruction(self) -> None:
        prompt = self.planner._build_json_repair_prompt("broken stuff")
        self.assertIn("Return ONLY valid JSON", prompt)

    def test_malformed_output_is_interpolated(self) -> None:
        prompt = self.planner._build_json_repair_prompt('{"bad": true, trailing_comma,}')
        self.assertIn('{"bad": true, trailing_comma,}', prompt)

    def test_instructs_not_to_invent_actions(self) -> None:
        prompt = self.planner._build_json_repair_prompt("junk")
        self.assertIn("Do not invent new actions", prompt)


class PlannerRepairPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = PlannerAgent(model="test-model")

    def _state_with_tool_diag(self, **diag_overrides) -> TaskState:
        state = TaskState(goal="create foo.py")
        fp = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "Permission denied", 0)
        defaults = {
            "attempt": 0,
            "success_criteria": ["foo.py exists"],
            "verification_reasoning": "Tool failed.",
            "failed_fingerprint": fp,
            "prior_fingerprints": [],
        }
        defaults.update(diag_overrides)
        diag = RepairDiagnostic(**defaults)
        state.log("repair_diagnostic", diag.as_dict())
        return state

    def _state_with_semantic_diag(self) -> TaskState:
        state = TaskState(goal="run tests")
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria=["exit code 0"],
            verification_reasoning="Tests failed — 3 errors.",
            failed_fingerprint=None,
        )
        state.log("repair_diagnostic", diag.as_dict())
        return state

    def _prompt(self, state: TaskState) -> str:
        diag = [e for e in state.history if e["event_type"] == "repair_diagnostic"][-1]["payload"]
        return self.planner._build_repair_prompt(state, diag)

    def test_contains_repair_mode_text(self) -> None:
        self.assertIn("repair mode", self._prompt(self._state_with_tool_diag()))

    def test_contains_json_only_instruction(self) -> None:
        self.assertIn("Return ONLY valid JSON", self._prompt(self._state_with_tool_diag()))

    def test_goal_is_interpolated(self) -> None:
        state = self._state_with_tool_diag()
        prompt = self._prompt(state)
        self.assertIn("create foo.py", prompt)

    def test_success_criteria_interpolated(self) -> None:
        state = self._state_with_tool_diag()
        prompt = self._prompt(state)
        self.assertIn("foo.py exists", prompt)

    def test_tool_failure_section_present(self) -> None:
        prompt = self._prompt(self._state_with_tool_diag())
        self.assertIn("Tool failure", prompt)
        self.assertIn("coder.write_file(path)", prompt)
        self.assertIn("Permission denied", prompt)

    def test_semantic_failure_section_present(self) -> None:
        prompt = self._prompt(self._state_with_semantic_diag())
        self.assertIn("Semantic failure", prompt)
        self.assertIn("Tests failed", prompt)

    def test_prior_signatures_listed_when_present(self) -> None:
        fp0 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 0)
        fp1 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 1)
        state = self._state_with_tool_diag(
            attempt=1,
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        prompt = self._prompt(state)
        self.assertIn("Previously failed action fingerprints", prompt)
        self.assertIn(fp0.signature, prompt)

    def test_repeated_failure_warning_present_when_repeated(self) -> None:
        fp0 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 0)
        fp1 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 1)
        state = self._state_with_tool_diag(
            attempt=1,
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        prompt = self._prompt(state)
        self.assertIn("WARNING", prompt)

    def test_no_warning_on_first_failure(self) -> None:
        prompt = self._prompt(self._state_with_tool_diag())
        self.assertNotIn("WARNING", prompt)

    def test_contains_permissions_and_schema(self) -> None:
        prompt = self._prompt(self._state_with_tool_diag())
        self.assertIn("coder", prompt)
        self.assertIn("reasoning_summary", prompt)


# ---------------------------------------------------------------------------
# VerifierAgent prompt builders
# ---------------------------------------------------------------------------


class VerifierSemanticPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = VerifierAgent(model="test-model", tool_registry={})

    def _state_with_tools(self, goal: str = "create foo.py", criteria: list[str] | None = None) -> TaskState:
        state = TaskState(goal=goal)
        if criteria:
            state.log(
                "agent_result",
                {"agent": "planner", "success_criteria": criteria},
            )
        state.log("tool_result", {"tool": "write_file", "ok": True, "output": "Wrote file: foo.py", "data": {}})
        return state

    def test_contains_json_only_instruction(self) -> None:
        state = self._state_with_tools()
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("Return ONLY valid JSON", prompt)

    def test_goal_is_interpolated(self) -> None:
        state = self._state_with_tools(goal="run the test suite")
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("run the test suite", prompt)

    def test_tool_results_are_included(self) -> None:
        state = self._state_with_tools()
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("write_file", prompt)

    def test_success_criteria_section_present_when_provided(self) -> None:
        state = self._state_with_tools(criteria=["file foo.py exists"])
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("Success criteria", prompt)
        self.assertIn("file foo.py exists", prompt)

    def test_success_criteria_section_absent_when_none(self) -> None:
        state = self._state_with_tools(criteria=None)
        prompt = self.verifier._build_verification_prompt(state)
        self.assertNotIn("Success criteria", prompt)

    def test_contains_allowed_statuses(self) -> None:
        state = self._state_with_tools()
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("passed", prompt)
        self.assertIn("failed", prompt)
        self.assertIn("inconclusive", prompt)

    def test_json_shape_example_in_prompt(self) -> None:
        state = self._state_with_tools()
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("reasoning_summary", prompt)

    def test_does_not_resubstitute_tool_output_content(self) -> None:
        """Tool output that looks like a placeholder must not cause a KeyError."""
        state = TaskState(goal="test")
        state.log("agent_result", {"agent": "planner", "success_criteria": []})
        # Simulate a tool result whose output contains {{something}}
        state.log("tool_result", {"tool": "run_shell", "ok": True, "output": "echo {{hello}}", "data": {}})
        # Must not raise
        prompt = self.verifier._build_verification_prompt(state)
        self.assertIn("{{hello}}", prompt)


class VerifierSourceStaticPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = VerifierAgent(model="test-model", tool_registry={})
        self.state = TaskState(goal="write a GUI app with pygame")

    def _build(
        self,
        target_file: str = "game.py",
        file_content: str = "import pygame\n",
        compile_ok: bool = True,
        compile_output: str = "",
        success_criteria: list[str] | None = None,
    ) -> str:
        return self.verifier._build_source_verification_prompt(
            self.state,
            target_file=target_file,
            file_content=file_content,
            compile_ok=compile_ok,
            compile_output=compile_output,
            success_criteria=success_criteria or [],
        )

    def test_contains_json_only_instruction(self) -> None:
        self.assertIn("Return ONLY valid JSON", self._build())

    def test_goal_is_interpolated(self) -> None:
        self.assertIn("write a GUI app with pygame", self._build())

    def test_target_file_is_interpolated(self) -> None:
        self.assertIn("game.py", self._build(target_file="game.py"))

    def test_file_content_is_interpolated(self) -> None:
        self.assertIn("import pygame", self._build(file_content="import pygame\nprint('hi')"))

    def test_compile_ok_shows_passed(self) -> None:
        self.assertIn("PASSED", self._build(compile_ok=True))

    def test_compile_fail_shows_failed_and_output(self) -> None:
        prompt = self._build(compile_ok=False, compile_output="SyntaxError line 3")
        self.assertIn("FAILED", prompt)
        self.assertIn("SyntaxError line 3", prompt)

    def test_success_criteria_interpolated(self) -> None:
        prompt = self._build(success_criteria=["window opens", "no crash on start"])
        self.assertIn("window opens", prompt)
        self.assertIn("no crash on start", prompt)

    def test_none_specified_when_no_criteria(self) -> None:
        self.assertIn("none specified", self._build(success_criteria=[]))

    def test_static_inspection_instruction_present(self) -> None:
        self.assertIn("static source inspection", self._build())

    def test_contains_allowed_statuses(self) -> None:
        prompt = self._build()
        self.assertIn("passed", prompt)
        self.assertIn("failed", prompt)
        self.assertIn("inconclusive", prompt)

    def test_file_content_with_double_braces_does_not_error(self) -> None:
        """Python source with f-string escapes like {{count}} must not cause KeyError."""
        tricky_content = "result = f'{{count}} items'\n"
        prompt = self._build(file_content=tricky_content)
        self.assertIn("{{count}}", prompt)


if __name__ == "__main__":
    unittest.main()
