
from __future__ import annotations

from orchestrator.policies import AGENT_PERMISSIONS
from orchestrator.state import TaskState, ToolCall, ToolResult


class CoderAgent:
    name = "coder"

    def __init__(self, tool_registry: dict[str, object], model: str = "llama3.1:8b") -> None:
        self.tool_registry = tool_registry
        self.model = model

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in AGENT_PERMISSIONS[self.name]

    def execute(self, call: ToolCall, state: TaskState) -> ToolResult:
        if not self.can_execute(call.tool):
            result = ToolResult(ok=False, tool=call.tool, output=f"Permission denied for coder agent: {call.tool}")
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
