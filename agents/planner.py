
from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List

from memory.summary_store import load_recent_summaries
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE, ValidationError
from orchestrator.state import AgentResult, TaskState, ToolCall


class PlannerAgent:
    name = "planner"
    max_repair_attempts = 2

    def __init__(self, model: str = "llama3.1:8b", timeout: int = 60) -> None:
        self.model = model
        self.timeout = timeout

    def _call_ollama(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=self.timeout,
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

        # success_criteria is required and must be a non-empty list of non-empty strings
        sc_raw = data.get("success_criteria")
        if not isinstance(sc_raw, list):
            raise ValidationError(
                f"success_criteria must be a list of strings, got {type(sc_raw).__name__}. "
                "Provide at least one concrete, testable condition."
            )
        if not sc_raw:
            raise ValidationError(
                "success_criteria must not be empty. "
                "Provide at least one concrete, testable condition."
            )
        if not all(isinstance(c, str) and c.strip() for c in sc_raw):
            raise ValidationError(
                "Every success criterion must be a non-empty string."
            )
        success_criteria = [c.strip() for c in sc_raw]

        return AgentResult(
            agent=self.name,
            reasoning_summary=reasoning_summary or "LLM produced a structured plan.",
            actions=actions,
            status=status,
            success_criteria=success_criteria,
        )

    def _build_memory_text(self) -> str:
        recent = load_recent_summaries(limit=3)
        if not recent:
            return ""
        lines = ["Prior completed tasks (use as context if relevant):"]
        for s in recent:
            files = ", ".join(s.get("files_touched", [])) or "none"
            key_outputs = s.get("key_outputs", [])
            out_snippet = key_outputs[0][:80].strip() if key_outputs else ""
            line = f'- "{s["goal"]}" → {s["status"]} | files: {files}'
            if out_snippet:
                line += f' | output: {out_snippet}'
            lines.append(line)
        return "\n" + "\n".join(lines) + "\n"

    _RULES = """\
- All paths must be relative.
- Prefer python3 over python.
- Do not use shell chaining, pipes, redirects, or multiple commands.
- To create a new code file, use code_file (coder). Provide a clear natural-language specification — do NOT write the code yourself.
- To modify a specific section of an existing file, use patch_file (coder). Provide old_lines (the exact text to replace, copied verbatim from the file) and new_lines (the replacement). This is the primary tool for surgical edits to existing files.
- To perform a larger modification of an existing code file that requires rewriting most of it, use modify_file (coder). Describe the required change as a specification — do NOT write the new code yourself.
- Use replace_in_file (coder) only as a last resort for tiny literal substitutions when patch_file is not appropriate.
- Use write_file (coder) only for creating new non-code content such as plain-text files or config files.
- After every code_file or write_file action, always follow it with a file_exists action (verifier agent) to confirm the file was created. This is mandatory, not optional.
- If a task requires running a Python file, follow the coder action with run_shell (verifier).
- If the task produces an interactive or GUI Python app (e.g. pygame, tkinter, a game with a window), do NOT add a run_shell action to the plan. Static verification will be performed automatically by the verifier.
- Use summarize_file when you need a concise understanding of a file instead of raw contents.
- Use researcher for inspection tasks like listing files, searching in files, or reading existing files.
- Never assign write_file, code_file, modify_file, patch_file, or run_shell to researcher.
- success_criteria must be a non-empty JSON array of strings. Each string must be one concrete, testable condition. Good: "file foo.py exists", "python3 foo.py exits with code 0", "stdout contains 'hello world'". Bad: "task is completed", "works correctly". An empty array is not allowed.
- Keep reasoning_summary short.
- status should usually be "ready".
- Produce the smallest viable action list."""

    _PERMISSIONS = """\
- coder: code_file, modify_file, patch_file, write_file, read_file, replace_in_file
- verifier: run_shell, read_file, file_exists
- researcher: read_file, list_files, search_in_files, file_exists, summarize_file"""

    _SCHEMA = """\
{
  "reasoning_summary": "...",
  "status": "ready",
  "success_criteria": [
    "file hello.py exists",
    "python3 hello.py exits with code 0",
    "stdout contains 'hello world'"
  ],
  "actions": [
    {"agent": "researcher", "tool": "summarize_file", "args": {"path": "agents/planner.py"}},
    {"agent": "coder", "tool": "code_file", "args": {"path": "hello.py", "specification": "Print 'hello world' to stdout."}},
    {"agent": "verifier", "tool": "file_exists", "args": {"path": "hello.py"}},
    {"agent": "verifier", "tool": "run_shell", "args": {"command": "python3 hello.py"}},
    {"agent": "coder", "tool": "patch_file", "args": {"path": "hello.py", "old_lines": "print('hello world')", "new_lines": "print('hello universe')"}}
  ]
}"""

    def _build_prompt(self, state: TaskState) -> str:
        memory_text = self._build_memory_text()

        return f"""
You are the planner agent in a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

You do not execute tools yourself. You only create an action plan.

Available agents and permissions:
{self._PERMISSIONS}

Rules:
{self._RULES}
{memory_text}
Goal:
{state.goal}

Return JSON in exactly this shape:
{self._SCHEMA}
""".strip()

    def _build_repair_prompt(self, state: TaskState, diagnostic: dict) -> str:
        attempt = diagnostic["attempt"]
        sc_raw = diagnostic.get("success_criteria", [])
        # Handle both list (new) and string (legacy) shapes gracefully
        if isinstance(sc_raw, list) and sc_raw:
            criteria_text = "\n".join(f"  - {c}" for c in sc_raw)
        elif isinstance(sc_raw, str) and sc_raw:
            criteria_text = f"  - {sc_raw}"
        else:
            criteria_text = "  (none specified)"
        verification_reasoning = diagnostic.get("verification_reasoning", "")
        failed_fp = diagnostic.get("failed_fingerprint")
        is_repeated = diagnostic.get("is_repeated_failure", False)
        prior_signatures = diagnostic.get("prior_signatures", [])

        if failed_fp:
            repeated_warning = (
                "\n\nWARNING: This fingerprint has already failed in a prior attempt. "
                "You MUST use a fundamentally different approach — "
                "do not retry the same tool with a similar argument shape."
            ) if is_repeated else ""
            failure_section = (
                f"--- Tool failure on attempt {attempt} ---\n"
                f"  Fingerprint : {failed_fp['signature']}\n"
                f"  Agent       : {failed_fp['agent']}\n"
                f"  Tool        : {failed_fp['tool']}\n"
                f"  Args        : {json.dumps(failed_fp['arg_snapshot'])}\n"
                f"  Error       : {failed_fp['error_output']}"
                f"{repeated_warning}"
            )
        else:
            failure_section = (
                f"--- Semantic failure on attempt {attempt} ---\n"
                "All tool calls succeeded but the verifier determined the goal was NOT achieved.\n"
                f"Verifier reasoning: {verification_reasoning}"
            )

        prior_section = ""
        if prior_signatures:
            lines = "\n".join(f"  - {sig}" for sig in prior_signatures)
            prior_section = (
                f"\nPreviously failed action fingerprints (do not repeat these patterns):\n{lines}\n"
            )

        memory_text = self._build_memory_text()

        return f"""
You are the planner agent in repair mode for a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

A previous action plan failed. Produce a corrected plan that directly addresses the failure below.
{prior_section}
Goal:
{state.goal}

Unmet success criteria:
{criteria_text}

{failure_section}

Available agents and permissions:
{self._PERMISSIONS}

Rules:
{self._RULES}
{memory_text}
Return JSON in exactly this shape:
{self._SCHEMA}
""".strip()

    def act(self, state: TaskState, tool_registry: Dict[str, object]) -> AgentResult:
        repair_events = [e for e in state.history if e["event_type"] == "repair_diagnostic"]
        if repair_events:
            prompt = self._build_repair_prompt(state, repair_events[-1]["payload"])
        else:
            prompt = self._build_prompt(state)

        try:
            data = self._get_plan_data(prompt)
            return self._validate_plan(data, tool_registry)
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Planning failed: {type(exc).__name__}: {exc}",
                actions=[],
                status="failed",
            )
