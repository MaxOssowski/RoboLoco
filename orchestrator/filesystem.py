from __future__ import annotations

from typing import Any, Dict

from orchestrator.policies import resolve_workspace_path, WORKSPACE
from orchestrator.state import ToolResult


class FilesystemTool:
    name = "write_file"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        content = args.get("content", "")

        file_path = resolve_workspace_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

        return ToolResult(
            ok=True,
            tool="write_file",
            output=f"Wrote file: {file_path}",
        )


class ReadFileTool:
    name = "read_file"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        file_path = resolve_workspace_path(path)

        if not file_path.exists():
            return ToolResult(
                ok=False,
                tool="read_file",
                output=f"File not found: {file_path}",
            )

        content = file_path.read_text(encoding="utf-8")
        return ToolResult(ok=True, tool="read_file", output=content)


class ListFilesTool:
    name = "list_files"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        dir_path = resolve_workspace_path(path)

        if not dir_path.exists():
            return ToolResult(ok=False, tool="list_files", output=f"Path not found: {dir_path}")

        if not dir_path.is_dir():
            return ToolResult(ok=False, tool="list_files", output=f"Path is not a directory: {dir_path}")

        entries = []
        for item in sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{prefix} {item.relative_to(WORKSPACE)}")

        output = "\n".join(entries) if entries else "<empty directory>"
        return ToolResult(ok=True, tool="list_files", output=output)