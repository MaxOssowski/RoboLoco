
from __future__ import annotations

import json
import subprocess

from agents.interactive import (
    find_target_python_file,
    is_interactive_python_task,
    run_py_compile,
)
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

    def _get_success_criteria(self, state: TaskState) -> list[str]:
        """Extract structured success_criteria from the most recent planner result in history."""
        for event in reversed(state.history):
            if (
                event["event_type"] == "agent_result"
                and event["payload"].get("agent") == "planner"
            ):
                val = event["payload"].get("success_criteria", [])
                if isinstance(val, list):
                    return val
                # backward-compat: plain string stored by older code
                if isinstance(val, str) and val:
                    return [val]
        return []

    def _build_source_verification_prompt(
        self,
        state: TaskState,
        target_file: str,
        file_content: str,
        compile_ok: bool,
        compile_output: str,
        success_criteria: list[str],
    ) -> str:
        compile_section = "PASSED (no errors)" if compile_ok else f"FAILED:\n{compile_output}"
        if success_criteria:
            criteria_section = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(success_criteria))
        else:
            criteria_section = "(none specified)"
        return f"""
You are the verifier agent in a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

Assess the Python source code below to determine whether the task goal is satisfied.
Do NOT assume you can run the code. Base your verdict entirely on static source inspection.

Goal:
{state.goal}

Success criteria:
{criteria_section}

File: {target_file}
Compile check: {compile_section}

Source code:
{file_content}

Return JSON in exactly this shape:
{{
  "status": "passed",
  "reasoning_summary": "..."
}}

Allowed statuses:
- passed   (file compiles and source satisfies the goal and criteria)
- failed   (file is missing required elements or does not compile)
- inconclusive
""".strip()

    def _interactive_verify(self, state: TaskState, target_file: str) -> AgentResult:
        """Static verification path for interactive/GUI Python apps."""
        file_path = WORKSPACE / target_file

        if not file_path.is_file():
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"File not found: {target_file}. Coder must write it first.",
                status="failed",
            )

        try:
            file_content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Could not read {target_file}: {exc}",
                status="failed",
            )

        compile_ok, compile_output = run_py_compile(WORKSPACE, target_file)
        success_criteria = self._get_success_criteria(state)

        try:
            prompt = self._build_source_verification_prompt(
                state, target_file, file_content, compile_ok, compile_output, success_criteria
            )
            raw = self._call_ollama(prompt)
            data = self._extract_json(raw)

            status = data.get("status", "inconclusive")
            reasoning = data.get("reasoning_summary", "Static verification completed.")

            if status not in {"passed", "failed", "inconclusive"}:
                status = "inconclusive"
            if not isinstance(reasoning, str):
                reasoning = "Static verification completed."

            if not compile_ok and status != "failed":
                status = "failed"

            if not compile_ok:
                reasoning = f"Compile error in {target_file}:\n{compile_output}\n\n{reasoning}"

            return AgentResult(agent=self.name, reasoning_summary=reasoning, status=status)

        except Exception:
            # LLM unavailable — fall back to deterministic result
            if compile_ok:
                return AgentResult(
                    agent=self.name,
                    reasoning_summary=f"{target_file} compiled successfully.",
                    status="passed",
                )
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Compile error in {target_file}:\n{compile_output}",
                status="failed",
            )

    def _written_files_exist(self, tool_events: list) -> tuple[bool, list[str]]:
        """Check every file reported written by write_file/code_file/modify_file exists on disk."""
        missing = []
        _file_writing_tools = {"write_file", "code_file", "modify_file"}
        for e in tool_events:
            p = e["payload"]
            if p.get("tool") in _file_writing_tools and p.get("ok"):
                output = p.get("output", "")
                if output.startswith("Wrote file:"):
                    rel_path = output[len("Wrote file:"):].strip()
                    if rel_path and not (WORKSPACE / rel_path).is_file():
                        missing.append(rel_path)
        return len(missing) == 0, missing

    def _has_confirmed_file_exists(self, tool_events: list) -> bool:
        """Return True if at least one file_exists call confirmed a file is present."""
        return any(
            e["payload"].get("tool") == "file_exists" and e["payload"].get("ok")
            for e in tool_events
        )

    def _build_verification_prompt(self, state: TaskState) -> str:
        recent_tools = [e["payload"] for e in state.history if e["event_type"] == "tool_result"][-6:]
        tool_text = json.dumps(recent_tools, indent=2)
        success_criteria = self._get_success_criteria(state)
        if success_criteria:
            numbered = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(success_criteria))
            criteria_section = f"\nSuccess criteria (evaluate each):\n{numbered}\n"
        else:
            criteria_section = ""

        return f"""
You are the verifier agent in a local multi-agent coding system.
Return ONLY valid JSON. Do not add commentary before or after the JSON.

Evaluate whether the task goal was actually achieved based on the recent tool results.
Do not assume success just because a command exited successfully.

Goal:
{state.goal}
{criteria_section}
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

        # Surface the actual error so the planner can repair it precisely
        failed = [e for e in last_tool_events if not e["payload"].get("ok")]
        if failed:
            first = failed[0]["payload"]
            tool_name = first.get("tool", "unknown")
            error_msg = first.get("output", "no error details")
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Tool '{tool_name}' failed: {error_msg}",
                status="failed",
            )

        # Deterministic pre-check: every file reported written must exist on disk
        all_exist, missing = self._written_files_exist(last_tool_events)
        if not all_exist:
            return AgentResult(
                agent=self.name,
                reasoning_summary=(
                    f"Written file(s) not found on disk: {', '.join(missing)}. "
                    "The write_file tool reported success but the file is absent."
                ),
                status="failed",
            )

        # Fast path: file_exists confirmation is deterministic proof — skip LLM
        if self._has_confirmed_file_exists(last_tool_events):
            return AgentResult(
                agent=self.name,
                reasoning_summary="All tools succeeded and file existence confirmed on disk.",
                status="passed",
            )

        if is_interactive_python_task(state.goal):
            target_file = find_target_python_file(state.goal)
            if target_file:
                return self._interactive_verify(state, target_file)

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

        # Inconclusive but all tools passed and no files are missing:
        # treat as passed only if there is at least some tool evidence of work done
        has_shell_success = any(
            e["payload"].get("tool") == "run_shell" and e["payload"].get("ok")
            for e in last_tool_events
        )
        has_write_success = any(
            e["payload"].get("tool") == "write_file" and e["payload"].get("ok")
            for e in last_tool_events
        )
        if has_shell_success or has_write_success:
            return AgentResult(
                agent=self.name,
                reasoning_summary=(
                    "All tools succeeded with evidence of output; "
                    "semantic verification was inconclusive."
                ),
                status="passed",
            )

        return AgentResult(
            agent=self.name,
            reasoning_summary=(
                "No clear evidence the goal was achieved. "
                "Semantic verification was inconclusive and no output-producing tools ran."
            ),
            status="inconclusive",
        )
