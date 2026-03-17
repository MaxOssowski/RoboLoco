
from __future__ import annotations

import time
from datetime import datetime, timezone

from agents.coder import CoderAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.verifier import VerifierAgent
from memory.summary_store import save_task_summary
from orchestrator.repair import (
    FailedActionFingerprint,
    RepairDiagnostic,
    fingerprint_failed_action,
)
from orchestrator.router import Router
from orchestrator.state import AgentResult, ModelConfig, TaskState, ToolResult
from tools.filesystem import (
    FileExistsTool,
    FilesystemTool,
    ListFilesTool,
    ReadFileTool,
    ReplaceInFileTool,
    SummarizeFileTool,
)
from tools.search import SearchInFilesTool
from tools.shell import ShellTool


class Orchestrator:
    def __init__(
        self,
        model: str = "llama3.1:8b",
        models: ModelConfig | None = None,
        strict_verifier: bool = False,
    ) -> None:
        if models is None:
            models = ModelConfig(default=model)

        self.tool_registry = {
            FilesystemTool.name: FilesystemTool,
            ReadFileTool.name: ReadFileTool,
            FileExistsTool.name: FileExistsTool,
            SummarizeFileTool.name: SummarizeFileTool,
            ListFilesTool.name: ListFilesTool,
            ReplaceInFileTool.name: ReplaceInFileTool,
            SearchInFilesTool.name: SearchInFilesTool,
            ShellTool.name: ShellTool,
        }

        self.planner = PlannerAgent(model=models.for_agent("planner"))
        self.coder = CoderAgent(self.tool_registry, model=models.for_agent("coder"))
        self.verifier = VerifierAgent(
            model=models.for_agent("verifier"),
            tool_registry=self.tool_registry,
            strict_verifier=strict_verifier,
        )
        self.researcher = ResearcherAgent(self.tool_registry, model=models.for_agent("researcher"))
        self.router = Router(self.coder, self.verifier, self.researcher)

    def _agent_models(self) -> dict:
        return {
            "planner": self.planner.model,
            "coder": self.coder.model,
            "verifier": self.verifier.model,
            "researcher": self.researcher.model,
        }

    def _build_summary(
        self,
        goal: str,
        result: dict,
        all_tool_results: list[dict],
        plan: AgentResult,
        retry_count: int,
        start_time: float,
    ) -> dict:
        files_touched: list[str] = []
        tools_used: list[str] = []
        key_outputs: list[str] = []
        seen_tools: set[str] = set()

        for r in all_tool_results:
            tool = r.get("tool", "")
            if tool not in seen_tools:
                tools_used.append(tool)
                seen_tools.add(tool)

            if r.get("ok"):
                output = r.get("output", "")
                if tool == "write_file" and output.startswith("Wrote file:"):
                    files_touched.append(output[len("Wrote file:"):].strip())
                elif tool == "replace_in_file" and output.startswith("Replaced text in:"):
                    files_touched.append(output[len("Replaced text in:"):].strip())
                elif tool == "run_shell":
                    key_outputs.append(output)

        status = result.get("verification_status", "unknown")
        verification_summary = result.get("verification_summary", "")
        if status == "passed":
            summary_text = f"Task completed successfully. {verification_summary}".strip()
        else:
            summary_text = f"Task did not complete. {verification_summary}".strip()

        # Deduplicate files_touched while preserving order
        seen: set[str] = set()
        deduped_files: list[str] = []
        for f in files_touched:
            if f not in seen:
                deduped_files.append(f)
                seen.add(f)

        return {
            "goal": goal,
            "status": status,
            "summary": summary_text,
            "success_criteria": plan.success_criteria,
            "files_touched": deduped_files,
            "tools_used": tools_used,
            "key_outputs": key_outputs,
            "retry_count": retry_count,
            "agent_models": self._agent_models(),
            "duration_seconds": round(time.monotonic() - start_time, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run(self, goal: str, max_retries: int = 2, on_event=None) -> dict:
        def emit(kind: str, **kwargs) -> None:
            if on_event:
                on_event(kind, **kwargs)

        start_time = time.monotonic()
        state = TaskState(goal=goal)
        state.log("task_started", {"goal": goal})

        attempt = 0
        all_tool_results: list[dict] = []
        plan: AgentResult | None = None
        prior_fingerprints: list[FailedActionFingerprint] = []

        while attempt <= max_retries:
            emit("planning", attempt=attempt)
            plan = self.planner.act(state, self.tool_registry)
            state.log(
                "agent_result",
                {
                    "agent": plan.agent,
                    "reasoning_summary": plan.reasoning_summary,
                    "status": plan.status,
                    "actions": [call.__dict__ for call in plan.actions],
                    "attempt": attempt,
                },
            )

            if plan.status != "ready":
                verification = self.verifier.act(state)
                result = {
                    "goal": goal,
                    "planner_status": plan.status,
                    "planner_summary": plan.reasoning_summary,
                    "tool_results": all_tool_results,
                    "verification_status": verification.status,
                    "verification_summary": verification.reasoning_summary,
                    "history": state.history,
                }
                save_task_summary(self._build_summary(goal, result, all_tool_results, plan, attempt, start_time))
                return result

            had_failure = False
            failed_fingerprint: FailedActionFingerprint | None = None

            for call in plan.actions:
                emit("executing", agent=call.agent, tool=call.tool)
                executor = self.router.get_executor(call.agent)
                if executor is None:
                    result_obj: ToolResult = self.router.unknown_executor_result(call.agent, call.tool)
                    state.log("tool_result", result_obj.__dict__)
                    all_tool_results.append(result_obj.__dict__)
                    had_failure = True
                    failed_fingerprint = fingerprint_failed_action(
                        call.agent, call.tool, call.args, result_obj.output, attempt
                    )
                    break

                result_obj = executor.execute(call, state)
                all_tool_results.append(result_obj.__dict__)

                if not result_obj.ok:
                    had_failure = True
                    failed_fingerprint = fingerprint_failed_action(
                        call.agent, call.tool, call.args, result_obj.output, attempt
                    )
                    break

            if had_failure:
                diagnostic = RepairDiagnostic(
                    attempt=attempt,
                    success_criteria=plan.success_criteria,
                    verification_reasoning="Tool execution failed before verification.",
                    failed_fingerprint=failed_fingerprint,
                    prior_fingerprints=list(prior_fingerprints),
                )
                if failed_fingerprint is not None:
                    prior_fingerprints.append(failed_fingerprint)
                state.log("repair_diagnostic", diagnostic.as_dict())
                attempt += 1
                continue

            emit("verifying", attempt=attempt)
            verification = self.verifier.act(state)
            state.log(
                "agent_result",
                {
                    "agent": verification.agent,
                    "reasoning_summary": verification.reasoning_summary,
                    "status": verification.status,
                    "actions": [],
                    "attempt": attempt,
                },
            )

            if verification.status == "passed":
                result = {
                    "goal": goal,
                    "planner_status": plan.status,
                    "planner_summary": plan.reasoning_summary,
                    "tool_results": all_tool_results,
                    "verification_status": verification.status,
                    "verification_summary": verification.reasoning_summary,
                    "history": state.history,
                }
                save_task_summary(self._build_summary(goal, result, all_tool_results, plan, attempt, start_time))
                return result

            diagnostic = RepairDiagnostic(
                attempt=attempt,
                success_criteria=plan.success_criteria,
                verification_reasoning=verification.reasoning_summary,
                failed_fingerprint=None,
                prior_fingerprints=list(prior_fingerprints),
            )
            state.log("repair_diagnostic", diagnostic.as_dict())
            attempt += 1

        result = {
            "goal": goal,
            "planner_status": "failed",
            "planner_summary": "Max retries reached.",
            "tool_results": all_tool_results,
            "verification_status": "failed",
            "verification_summary": "Agent could not repair the task within retry limit.",
            "history": state.history,
        }
        save_task_summary(
            self._build_summary(goal, result, all_tool_results, plan or AgentResult(agent="planner", reasoning_summary=""), attempt, start_time)
        )
        return result
