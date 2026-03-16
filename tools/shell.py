
from __future__ import annotations

import subprocess
from typing import Any, Dict

from orchestrator.policies import WORKSPACE, validate_shell_command
from orchestrator.state import ToolResult


class ShellTool:
    name = "run_shell"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        parts = validate_shell_command(command)

        result = subprocess.run(
            parts,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=20,
        )

        output = (result.stdout or "") + ("" + result.stderr if result.stderr else "")
        output = output.strip() or "<no output>"

        return ToolResult(
            ok=result.returncode == 0,
            tool="run_shell",
            output=output,
        )
