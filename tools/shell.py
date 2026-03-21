
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
        stdin_text: str | None = args.get("stdin")
        parts = validate_shell_command(command)

        try:
            result = subprocess.run(
                parts,
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
                timeout=20,
                input=stdin_text,          # None → DEVNULL-equivalent (no stdin pipe)
                stdin=subprocess.DEVNULL if stdin_text is None else None,
            )
        except subprocess.TimeoutExpired as exc:
            partial = ((exc.stdout or "") + (exc.stderr or "")).strip()
            msg = "Command timed out after 20 s."
            if partial:
                msg += f" Partial output: {partial}"
            return ToolResult(ok=False, tool="run_shell", output=msg)

        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "<no output>"

        return ToolResult(
            ok=result.returncode == 0,
            tool="run_shell",
            output=output,
        )
