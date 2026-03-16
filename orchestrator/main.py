
from __future__ import annotations

from agents.coder import CoderAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.verifier import VerifierAgent
from orchestrator.router import Router
from orchestrator.state import TaskState, ToolResult
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
    def __init__(self, model: str = "llama3.1:8b", strict_verifier: bool = False) -> None:
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

        self.planner = PlannerAgent(model=model)
        self.coder = CoderAgent(self.tool_registry)
        self.verifier = VerifierAgent(
            model=model,
            tool_registry=self.tool_registry,
            strict_verifier=strict_verifier,
        )
        self.researcher = ResearcherAgent(self.tool_registry)
        self.router = Router(self.coder, self.verifier, self.researcher)

    def run(self, goal: str, max_retries: int = 2) -> dict:
        state = TaskState(goal=goal)
        state.log("task_started", {"goal": goal})

        attempt = 0
        all_tool_results: list[dict] = []

        while attempt <= max_retries:
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
                return {
                    "goal": goal,
                    "planner_status": plan.status,
                    "planner_summary": plan.reasoning_summary,
                    "tool_results": all_tool_results,
                    "verification_status": verification.status,
                    "verification_summary": verification.reasoning_summary,
                    "history": state.history,
                }

            had_failure = False

            for call in plan.actions:
                executor = self.router.get_executor(call.agent)
                if executor is None:
                    result: ToolResult = self.router.unknown_executor_result(call.agent, call.tool)
                    state.log("tool_result", result.__dict__)
                    all_tool_results.append(result.__dict__)
                    had_failure = True
                    break

                result = executor.execute(call, state)
                all_tool_results.append(result.__dict__)

                if not result.ok:
                    had_failure = True
                    state.log(
                        "repair_context",
                        {
                            "failed_tool": call.tool,
                            "failed_args": call.args,
                            "failed_agent": call.agent,
                            "error_output": result.output,
                        },
                    )
                    break

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

            if not had_failure and verification.status == "passed":
                return {
                    "goal": goal,
                    "planner_status": plan.status,
                    "planner_summary": plan.reasoning_summary,
                    "tool_results": all_tool_results,
                    "verification_status": verification.status,
                    "verification_summary": verification.reasoning_summary,
                    "history": state.history,
                }

            state.log(
                "repair_context",
                {
                    "failed_tool": "semantic_verification",
                    "failed_args": {},
                    "failed_agent": "verifier",
                    "error_output": verification.reasoning_summary,
                },
            )
            attempt += 1

        return {
            "goal": goal,
            "planner_status": "failed",
            "planner_summary": "Max retries reached.",
            "tool_results": all_tool_results,
            "verification_status": "failed",
            "verification_summary": "Agent could not repair the task within retry limit.",
            "history": state.history,
        }
