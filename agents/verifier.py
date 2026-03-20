from __future__ import annotations

import json

from agents.interactive import (
    find_target_python_file,
    is_interactive_python_task,
    run_py_compile,
)
from orchestrator.llm import OllamaClient
from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from orchestrator.state import AgentResult, TaskState, ToolCall, ToolResult
from prompts.loader import render_prompt


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
        self._llm = OllamaClient(model=model, timeout=60)

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in AGENT_PERMISSIONS[self.name]

    def execute(self, call: ToolCall, state: TaskState) -> ToolResult:
        if not self.can_execute(call.tool):
            result = ToolResult(ok=False, tool=call.tool, output=f"Permission denied for verifier agent: {call.tool}")
            state.log("tool_result", result.__dict__)
            return result

        tool_cls = self.tool_registry.get(call.tool)
        if tool_cls is None:
            result = ToolResult(ok=False, tool=call.tool, output=f"Unknown tool: {call.tool}")
            state.log("tool_result", result.__dict__)
            return result

        try:
            result = tool_cls.run(call.args)
        except Exception as exc:
            result = ToolResult(ok=False, tool=call.tool, output=f"{type(exc).__name__}: {exc}")

        state.log("tool_result", result.__dict__)
        return result

    # ── JSON extraction ───────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict:
        text = self._llm.strip_thinking(text)
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_success_criteria(self, state: TaskState) -> list[str]:
        for event in reversed(state.history):
            if (
                event["event_type"] == "agent_result"
                and event["payload"].get("agent") == "planner"
            ):
                val = event["payload"].get("success_criteria", [])
                if isinstance(val, list):
                    return val
                if isinstance(val, str) and val:
                    return [val]
        return []

    def _current_attempt_tool_results(self, state: TaskState) -> list:
        """Tool results logged after the most recent planner agent_result."""
        last_planner_idx = -1
        for i, e in enumerate(state.history):
            if e["event_type"] == "agent_result" and e["payload"].get("agent") == "planner":
                last_planner_idx = i
        return [
            e for e in state.history[last_planner_idx + 1:]
            if e["event_type"] == "tool_result"
        ]

    def _written_files_exist(self, tool_events: list) -> tuple[bool, list[str]]:
        missing = []
        _file_writing_tools = {"write_file", "code_file", "modify_file"}
        for e in tool_events:
            p = e["payload"]
            if p.get("tool") in _file_writing_tools and p.get("ok"):
                data = p.get("data", {})
                if data.get("kind") == "file_write" and "path" in data:
                    rel_path = data["path"]
                else:
                    output = p.get("output", "")
                    if not output.startswith("Wrote file:"):
                        continue
                    rel_path = output[len("Wrote file:"):].strip()
                if rel_path and not (WORKSPACE / rel_path).is_file():
                    missing.append(rel_path)
        return len(missing) == 0, missing

    def _has_confirmed_file_exists(self, tool_events: list) -> bool:
        return any(
            e["payload"].get("tool") == "file_exists" and e["payload"].get("ok")
            for e in tool_events
        )

    def _all_shells_succeeded(self, tool_events: list) -> bool:
        """True if at least one run_shell result exists and all of them succeeded."""
        shells = [e for e in tool_events if e["payload"].get("tool") == "run_shell"]
        return bool(shells) and all(e["payload"].get("ok") for e in shells)

    # ── Verification paths ────────────────────────────────────────────────────

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
        criteria_section = (
            "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(success_criteria))
            if success_criteria else "(none specified)"
        )
        return render_prompt(
            "verifier_source_static.md",
            goal=state.goal,
            criteria_section=criteria_section,
            target_file=target_file,
            compile_section=compile_section,
            file_content=file_content,
        )

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
            return AgentResult(agent=self.name, reasoning_summary=f"Could not read {target_file}: {exc}", status="failed")

        compile_ok, compile_output = run_py_compile(WORKSPACE, target_file)
        success_criteria = self._get_success_criteria(state)

        try:
            prompt = self._build_source_verification_prompt(
                state, target_file, file_content, compile_ok, compile_output, success_criteria
            )
            raw = self._llm.generate(prompt)
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
            if compile_ok:
                return AgentResult(agent=self.name, reasoning_summary=f"{target_file} compiled successfully.", status="passed")
            return AgentResult(agent=self.name, reasoning_summary=f"Compile error in {target_file}:\n{compile_output}", status="failed")

    def _build_verification_prompt(self, state: TaskState) -> str:
        recent_tools = [e["payload"] for e in state.history if e["event_type"] == "tool_result"][-6:]
        tool_text = json.dumps(recent_tools, indent=2)
        success_criteria = self._get_success_criteria(state)
        if success_criteria:
            numbered = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(success_criteria))
            criteria_section = f"\nSuccess criteria (evaluate each):\n{numbered}\n"
        else:
            criteria_section = ""
        return render_prompt("verifier_semantic.md", goal=state.goal, criteria_section=criteria_section, tool_text=tool_text)

    def _semantic_verify(self, state: TaskState) -> AgentResult:
        try:
            raw = self._llm.generate(self._build_verification_prompt(state))
            data = self._extract_json(raw)

            status = data.get("status", "inconclusive")
            reasoning_summary = data.get("reasoning_summary", "Semantic verification completed.")

            if status not in {"passed", "failed", "inconclusive"}:
                status = "inconclusive"
            if not isinstance(reasoning_summary, str):
                reasoning_summary = "Semantic verification completed."

            return AgentResult(agent=self.name, reasoning_summary=reasoning_summary, status=status)
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Semantic verification unavailable: {type(exc).__name__}",
                status="inconclusive",
            )

    # ── Main entry point ──────────────────────────────────────────────────────

    def act(self, state: TaskState) -> AgentResult:
        last_tool_events = self._current_attempt_tool_results(state)

        if not last_tool_events:
            return AgentResult(agent=self.name, reasoning_summary="Nothing to verify yet.", status="idle")

        # Stage 1: surface any tool failure immediately
        failed = [e for e in last_tool_events if not e["payload"].get("ok")]
        if failed:
            first = failed[0]["payload"]
            return AgentResult(
                agent=self.name,
                reasoning_summary=f"Tool '{first.get('tool', 'unknown')}' failed: {first.get('output', 'no error details')}",
                status="failed",
            )

        # Stage 2: every written file must exist on disk
        all_exist, missing = self._written_files_exist(last_tool_events)
        if not all_exist:
            return AgentResult(
                agent=self.name,
                reasoning_summary=(
                    f"Written file(s) not found on disk: {', '.join(missing)}. "
                    "The write tool reported success but the file is absent."
                ),
                status="failed",
            )

        # Stage 3: interactive/GUI tasks — static source verification
        if is_interactive_python_task(state.goal):
            target_file = find_target_python_file(state.goal)
            if target_file:
                return self._interactive_verify(state, target_file)

        # Stage 4: semantic LLM verification
        semantic_result = self._semantic_verify(state)
        if semantic_result.status in {"passed", "failed"}:
            return semantic_result

        if self.strict_verifier:
            return AgentResult(
                agent=self.name,
                reasoning_summary=(
                    "Semantic verification was inconclusive. Strict verifier mode requires "
                    "explicit proof before counting the task as successful."
                ),
                status="failed",
            )

        # Non-strict: gather evidence for inconclusive summary
        observed: list[str] = []
        if self._has_confirmed_file_exists(last_tool_events):
            observed.append("file existence confirmed on disk")
        if any(e["payload"].get("tool") in {"write_file", "code_file", "modify_file"} and e["payload"].get("ok") for e in last_tool_events):
            observed.append("file was written")
        if any(e["payload"].get("tool") == "run_shell" and e["payload"].get("ok") for e in last_tool_events):
            observed.append("shell command exited with code 0")
        evidence = "; ".join(observed) if observed else "no notable tool evidence"
        return AgentResult(
            agent=self.name,
            reasoning_summary=(
                f"Semantic verification was inconclusive — output content and "
                f"behavior were not confirmed ({evidence}). "
                "Cannot count this as success without stronger proof."
            ),
            status="inconclusive",
        )
