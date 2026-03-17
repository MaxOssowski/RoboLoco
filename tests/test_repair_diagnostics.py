from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.planner import PlannerAgent
from orchestrator.main import Orchestrator
from orchestrator.repair import (
    FailedActionFingerprint,
    RepairDiagnostic,
    fingerprint_failed_action,
)
from orchestrator.state import AgentResult, TaskState, ToolCall


# ---------------------------------------------------------------------------
# FailedActionFingerprint
# ---------------------------------------------------------------------------


class FailedActionFingerprintTests(unittest.TestCase):
    def test_arg_keys_are_sorted(self) -> None:
        fp = fingerprint_failed_action(
            agent="coder",
            tool="write_file",
            args={"content": "x", "path": "foo.py"},
            error_output="err",
            attempt=0,
        )
        self.assertEqual(fp.arg_keys, ["content", "path"])

    def test_signature_format(self) -> None:
        fp = fingerprint_failed_action("coder", "write_file", {"path": "x", "content": "y"}, "e", 0)
        self.assertEqual(fp.signature, "coder.write_file(content, path)")

    def test_empty_args_signature(self) -> None:
        fp = fingerprint_failed_action("verifier", "file_exists", {}, "not found", 0)
        self.assertEqual(fp.signature, "verifier.file_exists()")

    def test_as_dict_includes_signature(self) -> None:
        fp = fingerprint_failed_action("coder", "write_file", {"path": "x"}, "err", 1)
        d = fp.as_dict()
        self.assertEqual(d["signature"], "coder.write_file(path)")
        self.assertEqual(d["attempt"], 1)
        self.assertEqual(d["arg_keys"], ["path"])
        self.assertEqual(d["arg_snapshot"], {"path": "x"})

    def test_arg_snapshot_is_a_copy(self) -> None:
        args = {"path": "foo.py"}
        fp = fingerprint_failed_action("coder", "write_file", args, "err", 0)
        args["path"] = "mutated"
        self.assertEqual(fp.arg_snapshot["path"], "foo.py")


# ---------------------------------------------------------------------------
# RepairDiagnostic
# ---------------------------------------------------------------------------


class RepairDiagnosticTests(unittest.TestCase):
    def _fp(self, attempt: int = 0, tool: str = "write_file") -> FailedActionFingerprint:
        return fingerprint_failed_action("coder", tool, {"path": "x"}, "err", attempt)

    def test_not_repeated_when_no_prior_fingerprints(self) -> None:
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="file exists",
            verification_reasoning="tool failed",
            failed_fingerprint=self._fp(0),
            prior_fingerprints=[],
        )
        self.assertFalse(diag.is_repeated_failure)

    def test_repeated_when_same_signature_in_prior(self) -> None:
        fp0 = self._fp(0)
        fp1 = self._fp(1)  # identical signature
        diag = RepairDiagnostic(
            attempt=1,
            success_criteria="file exists",
            verification_reasoning="tool failed",
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        self.assertTrue(diag.is_repeated_failure)

    def test_not_repeated_when_different_tool(self) -> None:
        fp0 = self._fp(0, tool="write_file")
        fp1 = self._fp(1, tool="replace_in_file")
        diag = RepairDiagnostic(
            attempt=1,
            success_criteria="ok",
            verification_reasoning="fail",
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        self.assertFalse(diag.is_repeated_failure)

    def test_not_repeated_when_no_failed_fingerprint(self) -> None:
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="",
            verification_reasoning="semantic fail",
            failed_fingerprint=None,
        )
        self.assertFalse(diag.is_repeated_failure)

    def test_as_dict_structure(self) -> None:
        fp = self._fp(0)
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="file written",
            verification_reasoning="fail",
            failed_fingerprint=fp,
            prior_fingerprints=[],
        )
        d = diag.as_dict()
        self.assertIn("attempt", d)
        self.assertIn("success_criteria", d)
        self.assertIn("verification_reasoning", d)
        self.assertIn("failed_fingerprint", d)
        self.assertIn("is_repeated_failure", d)
        self.assertIn("prior_signatures", d)
        self.assertEqual(d["prior_signatures"], [])
        self.assertFalse(d["is_repeated_failure"])

    def test_as_dict_prior_signatures_populated(self) -> None:
        fp0 = self._fp(0)
        fp1 = self._fp(1)
        diag = RepairDiagnostic(
            attempt=1,
            success_criteria="ok",
            verification_reasoning="fail",
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        d = diag.as_dict()
        self.assertEqual(d["prior_signatures"], [fp0.signature])
        self.assertTrue(d["is_repeated_failure"])

    def test_as_dict_none_fingerprint(self) -> None:
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="ok",
            verification_reasoning="verifier said no",
            failed_fingerprint=None,
        )
        d = diag.as_dict()
        self.assertIsNone(d["failed_fingerprint"])


# ---------------------------------------------------------------------------
# Orchestrator integration — repair_diagnostic event emission
# ---------------------------------------------------------------------------


class OrchestratorRepairDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator(model="test-model")

    def _plan_with(self, actions, success_criteria=""):
        def fake_plan(state, tool_registry):
            return AgentResult(
                agent="planner",
                reasoning_summary="plan",
                status="ready",
                success_criteria=success_criteria,
                actions=actions,
            )
        return fake_plan

    def test_tool_failure_emits_repair_diagnostic_with_fingerprint(self) -> None:
        """file_exists on a nonexistent path → tool failure → repair_diagnostic with fingerprint."""
        plan = self._plan_with(
            actions=[ToolCall(agent="verifier", tool="file_exists", args={"path": "__nonexistent_diag_test__.py"})],
            success_criteria="The nonexistent file exists.",
        )
        with patch.object(self.orchestrator.planner, "act", side_effect=plan):
            result = self.orchestrator.run("check file", max_retries=0)

        diag_events = [e for e in result["history"] if e["event_type"] == "repair_diagnostic"]
        self.assertEqual(len(diag_events), 1)
        payload = diag_events[0]["payload"]
        self.assertIsNotNone(payload["failed_fingerprint"])
        self.assertEqual(payload["failed_fingerprint"]["tool"], "file_exists")
        self.assertEqual(payload["failed_fingerprint"]["agent"], "verifier")
        self.assertEqual(payload["failed_fingerprint"]["arg_keys"], ["path"])
        self.assertEqual(payload["success_criteria"], "The nonexistent file exists.")
        self.assertFalse(payload["is_repeated_failure"])

    def test_semantic_failure_emits_repair_diagnostic_without_fingerprint(self) -> None:
        """All tools pass but verifier rejects → repair_diagnostic with no fingerprint."""
        plan = self._plan_with(
            actions=[ToolCall(agent="verifier", tool="run_shell", args={"command": "pwd"})],
            success_criteria="Output matches expected.",
        )
        with patch.object(self.orchestrator.planner, "act", side_effect=plan):
            with patch.object(
                self.orchestrator.verifier,
                "_semantic_verify",
                return_value=AgentResult(
                    agent="verifier", reasoning_summary="Goal not achieved.", status="failed"
                ),
            ):
                result = self.orchestrator.run("run check", max_retries=0)

        diag_events = [e for e in result["history"] if e["event_type"] == "repair_diagnostic"]
        self.assertEqual(len(diag_events), 1)
        payload = diag_events[0]["payload"]
        self.assertIsNone(payload["failed_fingerprint"])
        self.assertEqual(payload["verification_reasoning"], "Goal not achieved.")
        self.assertEqual(payload["success_criteria"], "Output matches expected.")

    def test_repeated_fingerprint_detected_across_retries(self) -> None:
        """Same failing pattern across attempts sets is_repeated_failure on 2nd+ diagnostics."""
        plan = self._plan_with(
            actions=[ToolCall(agent="verifier", tool="file_exists", args={"path": "__nonexistent_diag_test__.py"})],
            success_criteria="File exists.",
        )
        with patch.object(self.orchestrator.planner, "act", side_effect=plan):
            result = self.orchestrator.run("check file", max_retries=2)

        diag_events = [e for e in result["history"] if e["event_type"] == "repair_diagnostic"]
        self.assertEqual(len(diag_events), 3)
        # attempt 0: no prior fingerprints
        self.assertFalse(diag_events[0]["payload"]["is_repeated_failure"])
        self.assertEqual(diag_events[0]["payload"]["prior_signatures"], [])
        # attempt 1: fp from attempt 0 is a prior
        self.assertTrue(diag_events[1]["payload"]["is_repeated_failure"])
        self.assertEqual(len(diag_events[1]["payload"]["prior_signatures"]), 1)
        # attempt 2: fp from attempts 0 and 1 are priors
        self.assertTrue(diag_events[2]["payload"]["is_repeated_failure"])
        self.assertEqual(len(diag_events[2]["payload"]["prior_signatures"]), 2)

    def test_no_repair_context_events_remain(self) -> None:
        """The old repair_context event type is no longer emitted."""
        plan = self._plan_with(
            actions=[ToolCall(agent="verifier", tool="file_exists", args={"path": "__nonexistent_diag_test__.py"})],
        )
        with patch.object(self.orchestrator.planner, "act", side_effect=plan):
            result = self.orchestrator.run("check file", max_retries=1)

        repair_ctx_events = [e for e in result["history"] if e["event_type"] == "repair_context"]
        self.assertEqual(len(repair_ctx_events), 0)


# ---------------------------------------------------------------------------
# Planner repair prompt routing
# ---------------------------------------------------------------------------


class PlannerRepairPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator(model="test-model")
        self.planner = self.orchestrator.planner

    def _capture_prompt(self, state: TaskState) -> str:
        captured = {}

        def fake_get_plan_data(prompt):
            captured["value"] = prompt
            return {
                "reasoning_summary": "fixed",
                "status": "ready",
                "success_criteria": "ok",
                "actions": [],
            }

        with patch.object(self.planner, "_get_plan_data", side_effect=fake_get_plan_data):
            self.planner.act(state, self.orchestrator.tool_registry)

        return captured["value"]

    def test_uses_normal_prompt_without_repair_diagnostic(self) -> None:
        state = TaskState(goal="create foo.py")
        prompt = self._capture_prompt(state)
        self.assertNotIn("repair mode", prompt)
        self.assertIn("planner agent", prompt)

    def test_uses_repair_prompt_with_repair_diagnostic(self) -> None:
        state = TaskState(goal="create foo.py")
        fp = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "Permission denied", 0)
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="foo.py is created with correct content.",
            verification_reasoning="Tool failed.",
            failed_fingerprint=fp,
            prior_fingerprints=[],
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertIn("repair mode", prompt)
        self.assertIn("coder.write_file(path)", prompt)
        self.assertIn("foo.py is created with correct content.", prompt)
        self.assertIn("Permission denied", prompt)

    def test_repair_prompt_includes_unmet_success_criteria(self) -> None:
        state = TaskState(goal="run tests")
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="All tests pass with exit code 0.",
            verification_reasoning="Tests failed.",
            failed_fingerprint=None,
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertIn("All tests pass with exit code 0.", prompt)

    def test_repair_prompt_shows_prior_signatures(self) -> None:
        state = TaskState(goal="create foo.py")
        fp0 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 0)
        fp1 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 1)
        diag = RepairDiagnostic(
            attempt=1,
            success_criteria="File created.",
            verification_reasoning="Tool failed.",
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertIn("Previously failed action fingerprints", prompt)
        self.assertIn(fp0.signature, prompt)

    def test_repair_prompt_warns_on_repeated_failure(self) -> None:
        state = TaskState(goal="create foo.py")
        fp0 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 0)
        fp1 = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 1)
        diag = RepairDiagnostic(
            attempt=1,
            success_criteria="File created.",
            verification_reasoning="Tool failed.",
            failed_fingerprint=fp1,
            prior_fingerprints=[fp0],
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertIn("WARNING", prompt)

    def test_repair_prompt_no_warning_on_first_failure(self) -> None:
        state = TaskState(goal="create foo.py")
        fp = fingerprint_failed_action("coder", "write_file", {"path": "foo.py"}, "err", 0)
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="File created.",
            verification_reasoning="Tool failed.",
            failed_fingerprint=fp,
            prior_fingerprints=[],
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertNotIn("WARNING", prompt)

    def test_semantic_failure_repair_prompt_no_fingerprint_section(self) -> None:
        state = TaskState(goal="run script")
        diag = RepairDiagnostic(
            attempt=0,
            success_criteria="Output contains 'hello'.",
            verification_reasoning="Output was 'world', not 'hello'.",
            failed_fingerprint=None,
        )
        state.log("repair_diagnostic", diag.as_dict())

        prompt = self._capture_prompt(state)
        self.assertIn("Semantic failure", prompt)
        self.assertIn("Output was 'world', not 'hello'.", prompt)
        self.assertNotIn("Tool failure", prompt)


if __name__ == "__main__":
    unittest.main()
