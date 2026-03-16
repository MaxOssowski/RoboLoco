from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.planner import PlannerAgent
from orchestrator.main import Orchestrator
from orchestrator.state import TaskState


class PlannerJsonRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator(model="test-model")
        self.planner = PlannerAgent(model="test-model")

    def test_planner_repairs_malformed_json_output(self) -> None:
        malformed = """```json
{"reasoning_summary":"Write file","status":"ready","actions":[
  {"agent":"coder","tool":"write_file","args":{"path":"example.py","content":"print('hi')"}}
"""
        repaired = """
{
  "reasoning_summary": "Write file",
  "status": "ready",
  "actions": [
    {"agent": "coder", "tool": "write_file", "args": {"path": "example.py", "content": "print('hi')"}}
  ]
}
"""

        with patch.object(self.planner, "_call_ollama", side_effect=[malformed, repaired]) as mocked_call:
            plan = self.planner.act(
                TaskState(goal="create example.py"),
                self.orchestrator.tool_registry,
            )

        self.assertEqual(mocked_call.call_count, 2)
        self.assertEqual(plan.status, "ready")
        self.assertEqual(len(plan.actions), 1)
        self.assertEqual(plan.actions[0].tool, "write_file")

    def test_planner_fails_after_json_repair_attempt(self) -> None:
        malformed = "not json at all"
        still_bad = "still not json"

        # 1 initial call + 2 repair attempts (max_repair_attempts = 2)
        with patch.object(self.planner, "_call_ollama", side_effect=[malformed, still_bad, still_bad]) as mocked_call:
            plan = self.planner.act(
                TaskState(goal="do something"),
                self.orchestrator.tool_registry,
            )

        self.assertEqual(mocked_call.call_count, 3)
        self.assertEqual(plan.status, "failed")
        self.assertIn("Planner JSON repair failed", plan.reasoning_summary)


if __name__ == "__main__":
    unittest.main()
