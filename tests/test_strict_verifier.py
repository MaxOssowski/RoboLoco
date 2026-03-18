from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.verifier import VerifierAgent
from orchestrator.main import Orchestrator
from orchestrator.state import AgentResult, TaskState, ToolCall


class StrictVerifierTests(unittest.TestCase):
    def test_verifier_allows_inconclusive_without_strict_mode(self) -> None:
        state = TaskState(goal="verify output")
        state.log("tool_result", {"ok": True, "tool": "run_shell", "output": "done"})
        verifier = VerifierAgent(model="test-model", tool_registry={}, strict_verifier=False)

        with patch.object(
            verifier,
            "_semantic_verify",
            return_value=AgentResult(
                agent="verifier",
                reasoning_summary="Not enough evidence.",
                status="inconclusive",
            ),
        ):
            result = verifier.act(state)

        self.assertEqual(result.status, "inconclusive")
        self.assertIn("inconclusive", result.reasoning_summary.lower())

    def test_verifier_fails_closed_in_strict_mode(self) -> None:
        state = TaskState(goal="verify output")
        state.log("tool_result", {"ok": True, "tool": "run_shell", "output": "done"})
        verifier = VerifierAgent(model="test-model", tool_registry={}, strict_verifier=True)

        with patch.object(
            verifier,
            "_semantic_verify",
            return_value=AgentResult(
                agent="verifier",
                reasoning_summary="Not enough evidence.",
                status="inconclusive",
            ),
        ):
            result = verifier.act(state)

        self.assertEqual(result.status, "failed")
        self.assertIn("strict verifier mode", result.reasoning_summary.lower())

    def test_orchestrator_retries_when_strict_verifier_is_inconclusive(self) -> None:
        orchestrator = Orchestrator(model="test-model", strict_verifier=True)
        planner_calls = {"count": 0}

        def fake_plan(state, tool_registry):
            planner_calls["count"] += 1
            return AgentResult(
                agent="planner",
                reasoning_summary="Run a single check.",
                status="ready",
                actions=[ToolCall(agent="verifier", tool="run_shell", args={"command": "pwd"})],
            )

        with patch.object(orchestrator.planner, "act", side_effect=fake_plan):
            with patch.object(
                orchestrator.verifier,
                "_semantic_verify",
                return_value=AgentResult(
                    agent="verifier",
                    reasoning_summary="Still not enough evidence.",
                    status="inconclusive",
                ),
            ):
                result = orchestrator.run("verify output strictly", max_retries=1)

        self.assertEqual(planner_calls["count"], 2)
        self.assertEqual(result["planner_status"], "failed")
        self.assertEqual(result["verification_status"], "failed")
        self.assertEqual(
            result["verification_summary"],
            "Agent could not repair the task within retry limit.",
        )


if __name__ == "__main__":
    unittest.main()
