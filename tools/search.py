
from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.policies import resolve_workspace_path
from orchestrator.state import ToolResult


class SearchInFilesTool:
    name = "search_in_files"

    @staticmethod
    def run(args: Dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern")
        path = args.get("path", ".")
        file_glob = args.get("file_glob", "*.py")
        max_results = args.get("max_results", 20)

        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(ok=False, tool="search_in_files", output="pattern must be a non-empty string.")

        root = resolve_workspace_path(path)
        if not root.exists():
            return ToolResult(ok=False, tool="search_in_files", output=f"Path not found: {root}")

        if root.is_file():
            files = [root]
        else:
            files = [p for p in root.rglob(file_glob) if p.is_file()]

        matches: List[str] = []
        for file_path in sorted(files):
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    rel_path = file_path.relative_to(resolve_workspace_path("."))
                    matches.append(f"{rel_path}:{line_no}: {line.strip()}")
                    if len(matches) >= max_results:
                        return ToolResult(ok=True, tool="search_in_files", output="".join(matches))

        if not matches:
            return ToolResult(ok=True, tool="search_in_files", output="<no matches>")

        return ToolResult(ok=True, tool="search_in_files", output="".join(matches))
