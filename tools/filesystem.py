
from __future__ import annotations

import json
import re
from typing import Any, Dict

from orchestrator.policies import resolve_workspace_path, WORKSPACE, ValidationError
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


class FileExistsTool:
    name = "file_exists"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        file_path = resolve_workspace_path(path)

        if not file_path.exists():
            return ToolResult(ok=False, tool="file_exists", output=f"File not found: {file_path}")

        if not file_path.is_file():
            return ToolResult(ok=False, tool="file_exists", output=f"Path is not a file: {file_path}")

        return ToolResult(ok=True, tool="file_exists", output=f"File exists: {file_path}")


class SummarizeFileTool:
    name = "summarize_file"

    @staticmethod
    def _truncate_line(text: str, limit: int = 120) -> str:
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[: limit - 3] + "..."

    @staticmethod
    def _summarize_python(text: str) -> Dict[str, Any]:
        imports: list[str] = []
        classes: list[str] = []
        functions: list[str] = []

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
                continue

            class_match = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
            if class_match:
                classes.append(class_match.group(1))
                continue

            function_match = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
            if function_match:
                functions.append(function_match.group(1))

        return {
            "imports": imports[:8],
            "classes": classes[:12],
            "functions": functions[:20],
        }

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        preview_lines = args.get("preview_lines", 5)

        if not isinstance(preview_lines, int) or preview_lines < 0 or preview_lines > 20:
            raise ValidationError("preview_lines must be an integer between 0 and 20.")

        file_path = resolve_workspace_path(path)
        if not file_path.exists():
            return ToolResult(ok=False, tool="summarize_file", output=f"File not found: {file_path}")

        if not file_path.is_file():
            return ToolResult(ok=False, tool="summarize_file", output=f"Path is not a file: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="replace")
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        summary: Dict[str, Any] = {
            "path": str(file_path.relative_to(WORKSPACE)),
            "extension": file_path.suffix or "<none>",
            "line_count": len(text.splitlines()),
            "char_count": len(text),
            "preview": [
                SummarizeFileTool._truncate_line(line)
                for line in non_empty_lines[:preview_lines]
            ],
        }

        if file_path.suffix == ".py":
            summary["python_symbols"] = SummarizeFileTool._summarize_python(text)

        return ToolResult(
            ok=True,
            tool="summarize_file",
            output=json.dumps(summary, indent=2),
        )


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


class ReplaceInFileTool:
    name = "replace_in_file"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        old_text = args.get("old_text")
        new_text = args.get("new_text")

        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ValidationError("old_text and new_text must be strings.")

        file_path = resolve_workspace_path(path)
        if not file_path.exists():
            return ToolResult(ok=False, tool="replace_in_file", output=f"File not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        if old_text not in content:
            return ToolResult(ok=False, tool="replace_in_file", output="old_text not found in file.")

        updated = content.replace(old_text, new_text, 1)
        file_path.write_text(updated, encoding="utf-8")
        return ToolResult(ok=True, tool="replace_in_file", output=f"Replaced text in: {file_path}")


class PatchFileTool:
    """Context-based patch tool for modifying sections of existing files.

    Finds ``old_lines`` verbatim in the file and replaces the first occurrence
    with ``new_lines``.  Fails loudly when the file is missing or when the
    context string is not found — no fuzzy matching, no silent corruption.
    """

    name = "patch_file"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        path = args.get("path")
        old_lines = args.get("old_lines")
        new_lines = args.get("new_lines")

        if not isinstance(old_lines, str):
            raise ValidationError("old_lines must be a string.")
        if not isinstance(new_lines, str):
            raise ValidationError("new_lines must be a string.")
        if not old_lines:
            raise ValidationError("old_lines must not be empty.")

        file_path = resolve_workspace_path(path)
        if not file_path.exists():
            return ToolResult(
                ok=False,
                tool="patch_file",
                output=(
                    f"File not found: {file_path}. "
                    "patch_file can only modify existing files — use write_file or code_file to create a new file."
                ),
            )

        content = file_path.read_text(encoding="utf-8")
        if old_lines not in content:
            return ToolResult(
                ok=False,
                tool="patch_file",
                output=(
                    "patch_file: old_lines not found in file — context mismatch. "
                    "Read the file first and ensure old_lines exactly matches existing content."
                ),
            )

        updated = content.replace(old_lines, new_lines, 1)
        file_path.write_text(updated, encoding="utf-8")
        rel = file_path.relative_to(WORKSPACE)
        return ToolResult(ok=True, tool="patch_file", output=f"Patched file: {rel}")
