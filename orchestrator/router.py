
from __future__ import annotations

from orchestrator.state import ToolResult


class Router:
    def __init__(self, coder, verifier, researcher) -> None:
        self.coder = coder
        self.verifier = verifier
        self.researcher = researcher

    def get_executor(self, agent_name: str):
        if agent_name == "coder":
            return self.coder
        if agent_name == "verifier":
            return self.verifier
        if agent_name == "researcher":
            return self.researcher
        return None

    def unknown_executor_result(self, agent_name: str, tool_name: str) -> ToolResult:
        return ToolResult(
            ok=False,
            tool=tool_name,
            output=f"Unknown executor agent: {agent_name}",
        )
