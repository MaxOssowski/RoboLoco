from __future__ import annotations

import json
from typing import Any, Dict, List

from memory.summary_store import load_recent_summaries
from orchestrator.llm import OllamaClient
from orchestrator.policies import AGENT_PERMISSIONS, ValidationError
from orchestrator.state import AgentResult, TaskState, ToolCall
from prompts.loader import render_prompt


class PlannerAgent:
    name = "planner"
    max_repair_attempts = 2

    def __init__(self, model: str = "qwen2.5-coder:7b", timeout: int = 60) -> None:
        self.model = model
        self._llm = OllamaClient(model=model, timeout=timeout)

    # ── JSON extraction ───────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = self._llm.strip_thinking(text)

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
            return json.loads(text[start:end + 1])

        raise ValidationError("Planner did not return valid JSON.")

    def _build_json_repair_prompt(self, malformed_output: str) -> str:
        return render_prompt("planner_json_repair.md", malformed_output=malformed_output)

    def _get_plan_data(self, prompt: str) -> Dict[str, Any]:
        raw = self._llm.generate(prompt)

        try:
            return self._extract_json(raw)
        except Exception as first_exc:
            repair_error = first_exc
            repair_raw = raw

            for _ in range(self.max_repair_attempts):
                repair_prompt = self._build_json_repair_prompt(repair_raw)
                repair_raw = self._llm.generate(repair_prompt)
                try:
                    return self._extract_json(repair_raw)
                except Exception as exc:
                    repair_error = exc

            raise ValidationError(
                f"Planner JSON repair failed: {repair_error}"
            ) from repair_error

    # ── Plan validation ───────────────────────────────────────────────────────

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

            # Catch guaranteed-to-fail patch_file shapes before execution.
            if tool == "patch_file":
                old_lines = args.get("old_lines", "")
                if not isinstance(old_lines, str) or not old_lines.strip():
                    raise ValidationError(
                        "patch_file action has empty old_lines — this will always fail at runtime. "
                        "Use code_file to create files and read_file to obtain the exact text "
                        "before attempting a patch."
                    )

            actions.append(ToolCall(agent=agent, tool=tool, args=args))

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
            raise ValidationError("Every success criterion must be a non-empty string.")
        success_criteria = [c.strip() for c in sc_raw]

        return AgentResult(
            agent=self.name,
            reasoning_summary=reasoning_summary or "LLM produced a structured plan.",
            actions=actions,
            status=status,
            success_criteria=success_criteria,
        )

    # ── Prompt builders ───────────────────────────────────────────────────────

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
- All file paths must be relative.
- Prefer python3 over python.
- No shell chaining, pipes, redirects, or multiple commands in one run_shell.
- To create a new code file use code_file (coder) with a natural-language specification. Do NOT write the code yourself.
- To rewrite most of an existing file use modify_file (coder) with a natural-language specification. Do NOT write the code yourself.
- To replace a small section of an existing file use patch_file (coder). old_lines must be the exact text to replace (non-empty, copied verbatim). new_lines is the replacement. Do NOT embed full implementations in new_lines.
- Use write_file (coder) only for non-code content (plain text, configs).
- After every code_file or write_file, add a file_exists action (verifier) to confirm creation.
- If the task requires running a Python file, add run_shell (verifier) after the coder action.
- If the task produces an interactive or GUI app (pygame, tkinter, etc.), do NOT add run_shell. Static verification is automatic.
- success_criteria: non-empty JSON array of concrete testable strings. E.g. "file foo.py exists", "python3 foo.py exits with code 0".
- Keep reasoning_summary to one line. status is usually "ready". Use the smallest viable action list."""

    _PERMISSIONS = """\
- coder: code_file, modify_file, patch_file, write_file, read_file, replace_in_file
- verifier: run_shell, read_file, file_exists
- researcher: read_file, list_files, search_in_files, file_exists, summarize_file"""

    _SCHEMA = """\
{
  "reasoning_summary": "one-line description of the plan",
  "status": "ready",
  "success_criteria": [
    "file <target>.py exists",
    "python3 <target>.py exits with code 0"
  ],
  "actions": [
    {"agent": "coder", "tool": "code_file", "args": {"path": "<target>.py", "specification": "<natural-language description of what the file must do>"}},
    {"agent": "verifier", "tool": "file_exists", "args": {"path": "<target>.py"}},
    {"agent": "verifier", "tool": "run_shell", "args": {"command": "python3 <target>.py"}},
    {"agent": "coder", "tool": "patch_file", "args": {"path": "<target>.py", "old_lines": "<exact existing text to replace>", "new_lines": "<replacement text>"}}
  ]
}"""

    def _build_prompt(self, state: TaskState) -> str:
        return render_prompt(
            "planner_plan.md",
            permissions=self._PERMISSIONS,
            rules=self._RULES,
            memory_section=self._build_memory_text(),
            goal=state.goal,
            schema=self._SCHEMA,
        )

    def _build_repair_prompt(self, state: TaskState, diagnostic: dict) -> str:
        attempt = diagnostic["attempt"]
        sc_raw = diagnostic.get("success_criteria", [])
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

        return render_prompt(
            "planner_repair.md",
            prior_section=prior_section,
            goal=state.goal,
            criteria_text=criteria_text,
            failure_section=failure_section,
            permissions=self._PERMISSIONS,
            rules=self._RULES,
            memory_section=self._build_memory_text(),
            schema=self._SCHEMA,
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    def act(self, state: TaskState, tool_registry: Dict[str, object]) -> AgentResult:
        repair_events = [e for e in state.history if e["event_type"] == "repair_diagnostic"]
        prompt = (
            self._build_repair_prompt(state, repair_events[-1]["payload"])
            if repair_events
            else self._build_prompt(state)
        )

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
