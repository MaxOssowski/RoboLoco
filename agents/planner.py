
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE, ValidationError
from orchestrator.state import AgentResult, TaskState, ToolCall


class PlannerAgent:
    name = "planner"
    max_repair_attempts = 1

    def __init__(self, model: str = "llama3.1:8b") -> None:
        self.model = model

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

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            if end != -1:
                candidate = text[start:end].strip()
                return json.loads(candidate)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)

        raise ValidationError("Planner did not return valid JSON.")

    def _build_json_repair_prompt(self, malformed_output: str) -> str:
        return f"""
You are repairing malformed planner output for a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

The previous planner output was malformed or wrapped in extra text.
Re-emit the same plan as valid JSON only.
Do not invent new actions unless needed to preserve the original intent.

Malformed planner output:
{malformed_output}
""".strip()

    def _get_plan_data(self, prompt: str) -> Dict[str, Any]:
        raw = self._call_ollama(prompt)

        try:
            return self._extract_json(raw)
        except Exception as first_exc:
            repair_error = first_exc
            repair_raw = raw

            for _ in range(self.max_repair_attempts):
                repair_prompt = self._build_json_repair_prompt(repair_raw)
                repair_raw = self._call_ollama(repair_prompt)
                try:
                    return self._extract_json(repair_raw)
                except Exception as exc:
                    repair_error = exc

            raise ValidationError(f"Planner JSON repair failed: {repair_error}") from repair_error

    def _validate_plan(self, data: Dict[str, Any], tool_registry: Dict[str, object]) -> AgentResult:
        if not isinstance(data, dict):
            raise ValidationError("Planner output must be a JSON object.")

        reasoning_summary = data.get("reasoning_summary", "")
        actions_raw = data.get("actions", [])
        status = data.get("status", "ready")

        if not isinstance(reasoning_summary, str):
            raise ValidationError("reasoning_summary must be a string.")
        if not isinstance(actions_raw, list):
            raise ValidationError("actions must be a list.")
        if not isinstance(status, str):
            raise ValidationError("status must be a string.")

        actions: List[ToolCall] = []
        for item in actions_raw:
            if not isinstance(item, dict):
                raise ValidationError("Each action must be an object.")

            agent = item.get("agent")
            tool = item.get("tool")
            args = item.get("args", {})

            if not isinstance(agent, str):
                raise ValidationError("Action agent must be a string.")
            if agent not in AGENT_PERMISSIONS:
                raise ValidationError(f"Unknown agent requested by planner: {agent}")

            if not isinstance(tool, str):
                raise ValidationError("Action tool must be a string.")
            if tool not in tool_registry:
                raise ValidationError(f"Unknown tool requested by planner: {tool}")
            if tool not in AGENT_PERMISSIONS[agent]:
                raise ValidationError(f"Tool {tool} is not allowed for agent {agent}")

            if not isinstance(args, dict):
                raise ValidationError("Action args must be an object.")

            actions.append(ToolCall(agent=agent, tool=tool, args=args))

        return AgentResult(
            agent=self.name,
            reasoning_summary=reasoning_summary or "LLM produced a structured plan.",
            actions=actions,
            status=status,
        )

    def _build_prompt(self, state: TaskState) -> str:
        repair_events = [e for e in state.history if e["event_type"] == "repair_context"]
        repair_text = ""

        if repair_events:
            last = repair_events[-1]["payload"]
            repair_text = f"""

Previous attempt failed.
Failed tool: {last.get('failed_tool')}
Failed agent: {last.get('failed_agent')}
Failed args: {json.dumps(last.get('failed_args', {}))}
Error output:
{last.get('error_output')}

Please repair the plan and avoid repeating the same mistake.
"""

        return f"""
You are the planner agent in a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

You do not execute tools yourself. You only create an action plan.

Available agents and permissions:
- coder: write_file, read_file, replace_in_file
- verifier: run_shell, read_file, file_exists
- researcher: read_file, list_files, search_in_files, file_exists, summarize_file

Rules:
- All paths must be relative.
- Prefer python3 over python.
- Do not use shell chaining, pipes, redirects, or multiple commands.
- Use replace_in_file when changing part of an existing file instead of rewriting the whole file.
- If a task requires creating and running a Python file, first assign write_file to coder, then run_shell to verifier.
- Use file_exists when the goal depends on confirming an artifact was created.
- Use summarize_file when you need a concise understanding of a file instead of raw contents.
- Use researcher for inspection tasks like listing files, searching in files, or reading existing files.
- Never assign write_file or run_shell to researcher.
- Keep reasoning_summary short.
- status should usually be "ready".
- Produce the smallest viable action list.

Goal:
{state.goal}
{repair_text}

Return JSON in exactly this shape:
{{
  "reasoning_summary": "...",
  "status": "ready",
  "actions": [
    {{"agent": "researcher", "tool": "search_in_files", "args": {{"pattern": "ToolCall", "path": "."}}}},
    {{"agent": "researcher", "tool": "summarize_file", "args": {{"path": "agents/planner.py"}}}},
    {{"agent": "coder", "tool": "replace_in_file", "args": {{"path": "example.py", "old_text": "print('hi')", "new_text": "print('hello')"}}}},
    {{"agent": "verifier", "tool": "run_shell", "args": {{"command": "python3 example.py"}}}},
    {{"agent": "verifier", "tool": "file_exists", "args": {{"path": "example.py"}}}}
  ]
}}
""".strip()

    def act(self, state: TaskState, tool_registry: Dict[str, object]) -> AgentResult:
        prompt = self._build_prompt(state)

        try:
            data = self._get_plan_data(prompt)
            return self._validate_plan(data, tool_registry)
        except Exception as exc:
            goal = state.goal.lower()

            if "random_numbers.py" in goal and "5 random numbers" in goal:
                code = """import random

def print_random_numbers():
    for _ in range(5):
        print(random.randint(1, 100))

if __name__ == '__main__':
    print_random_numbers()
"""
                return AgentResult(
                    agent=self.name,
                    reasoning_summary=f"LLM planning failed, using deterministic fallback: {type(exc).__name__}",
                    actions=[
                        ToolCall(agent="coder", tool="write_file", args={"path": "random_numbers.py", "content": code}),
                        ToolCall(agent="verifier", tool="run_shell", args={"command": "python3 random_numbers.py"}),
                    ],
                    status="ready",
                )

            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Planning failed: {type(exc).__name__}: {exc}",
                actions=[],
                status="failed",
            )
