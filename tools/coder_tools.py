
from __future__ import annotations

from orchestrator.state import ToolResult


class CodeFileTool:
    """Sentinel registered in tool_registry so the planner can emit 'code_file' actions.

    CoderAgent intercepts 'code_file' calls *before* the registry lookup and
    handles them with LLM-backed code generation.  This class's run() method
    must never be invoked directly.
    """

    name = "code_file"

    @staticmethod
    def run(args: dict) -> ToolResult:
        raise RuntimeError(
            "CodeFileTool.run() must not be called directly — "
            "CoderAgent intercepts 'code_file' before the tool registry."
        )


class ModifyFileTool:
    """Sentinel registered in tool_registry so the planner can emit 'modify_file' actions.

    CoderAgent intercepts 'modify_file' calls *before* the registry lookup and
    handles them with LLM-backed code generation.  This class's run() method
    must never be invoked directly.
    """

    name = "modify_file"

    @staticmethod
    def run(args: dict) -> ToolResult:
        raise RuntimeError(
            "ModifyFileTool.run() must not be called directly — "
            "CoderAgent intercepts 'modify_file' before the tool registry."
        )
