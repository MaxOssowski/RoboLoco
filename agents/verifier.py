
from __future__ import annotations

import json
import subprocess

from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from orchestrator.state import AgentResult, TaskState, ToolCall, ToolResult


class VerifierAgent:
    name = "verifier"

    def __init__(
        self,
        model: str,
        tool_registry: dict[str, object],
        strict_verifier: bool = False,
    ) -> None:
        self.model = model
        self.tool_registry = tool_registry
        self.strict_verifier = strict_verifier

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in AGENT_PERMISSIONS[self.name]

    def execute(self, call: ToolCall, state: TaskState) -> ToolResult:
        if not self.can_execute(call.tool):
            result = ToolResult(
                ok=False,
                tool=call.tool,
                output=f"Permission denied for verifier agent: {call.tool}",
            )
            state.log("tool_result", result.__dict__)
            return result

        tool_cls = self.tool_registry.get(call.tool)
        if tool_cls is None:
            result = ToolResult(
                ok=False,
                tool=call.tool,
                output=f"Unknown tool: {call.tool}",
            )
            state.log("tool_result", result.__dict__)
            return result

        try:
            result = tool_cls.run(call.args)
        except Exception as exc:
            result = ToolResult(
                ok=False,
                tool=call.tool,
                output=f"{type(exc).__name__}: {exc}",
            )

        state.log("tool_result", result.__dict__)
        return result

    def _extract_json(self, text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end != -1:
                return json.loads(text[start:end].strip())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("No valid JSON found in verifier response.")

    def _call_ollama(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Ollama call failed").strip())
        return (result.stdout or "").strip()

    def _build_verification_prompt(self, state: TaskState) -> str:
        recent_tools = [e["payload"] for e in state.history if e["event_type"] == "tool_result"][-6:]
        tool_text = json.dumps(recent_tools, indent=2)

        return f"""
You are the verifier agent in a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

Evaluate whether the task goal was actually achieved based on the recent tool results.
Do not assume success just because a command exited successfully.

Goal:
{state.goal}

Recent tool results:
{tool_text}

Return JSON in exactly this shape:
{{
  "status": "passed",
  "reasoning_summary": "Goal achieved because ..."
}}

Allowed statuses:
- passed
- failed
- inconclusive
""".strip()

    def _semantic_verify(self, state: TaskState) -> AgentResult:
        try:
            raw = self._call_ollama(self._build_verification_prompt(state))
            data = self._extract_json(raw)

            status = data.get("status", "inconclusive")
            reasoning_summary = data.get("reasoning_summary", "Semantic verification completed.")

            if status not in {"passed", "failed", "inconclusive"}:
                status = "inconclusive"

            if not isinstance(reasoning_summary, str):
                reasoning_summary = "Semantic verification completed."

            return AgentResult(
                agent=self.name,
                reasoning_summary=reasoning_summary,
                status=status,
            )
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Semantic verification unavailable: {type(exc).__name__}",
                status="inconclusive",
            )

    def _current_attempt_tool_results(self, state: TaskState) -> list:
        """Return only tool results logged after the most recent planner agent_result."""
        last_planner_idx = -1
        for i, e in enumerate(state.history):
            if e["event_type"] == "agent_result" and e["payload"].get("agent") == "planner":
                last_planner_idx = i
        return [
            e for e in state.history[last_planner_idx + 1:]
            if e["event_type"] == "tool_result"
        ]

    def act(self, state: TaskState) -> AgentResult:
        last_tool_events = self._current_attempt_tool_results(state)

        if not last_tool_events:
            return AgentResult(
                agent=self.name,
                reasoning_summary="Nothing to verify yet.",
                status="idle",
            )

        failed = [e for e in last_tool_events if not e["payload"].get("ok")]
        if failed:
            return AgentResult(
                agent=self.name,
                reasoning_summary="One or more tool executions failed. Check logs and repair the plan.",
                status="failed",
            )

        semantic_result = self._semantic_verify(state)
        if semantic_result.status in {"passed", "failed"}:
            return semantic_result

        if self.strict_verifier:
            return AgentResult(
                agent=self.name,
                reasoning_summary=(
                    "Semantic verification was inconclusive. Strict verifier mode requires"
                    " explicit proof before counting the task as successful."
                ),
                status="failed",
            )

        return AgentResult(
            agent=self.name,
            reasoning_summary="All executed tools succeeded, but semantic verification was inconclusive.",
            status="passed",
        )
