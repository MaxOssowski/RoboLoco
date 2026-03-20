from __future__ import annotations

# Planner / Coder boundary
# ────────────────────────
# Planner  — decides WHAT to build: which files, what specifications, which
#             agents act, and what success looks like.  It emits high-level
#             coding intentions (code_file, modify_file) with a natural-language
#             specification.  It must NOT write code bodies itself.
#
# CoderAgent — decides HOW to build it: calls the LLM with the specification
#              and relevant file context, produces code, then persists it via
#              the existing filesystem tools.

import ast

from orchestrator.llm import OllamaClient
from orchestrator.policies import AGENT_PERMISSIONS
from orchestrator.state import TaskState, ToolCall, ToolResult
from tools.filesystem import FilesystemTool, ReadFileTool

# Tool names that CoderAgent handles with LLM logic before reaching the registry.
_CODER_NATIVE_TOOLS = frozenset({"code_file", "modify_file"})


class CoderAgent:
    name = "coder"

    def __init__(self, tool_registry: dict[str, object], model: str = "qwen2.5-coder:7b", timeout: int = 120) -> None:
        self.tool_registry = tool_registry
        self.model = model
        self._llm = OllamaClient(model=model, timeout=timeout)

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in AGENT_PERMISSIONS[self.name]

    # ── Code extraction & validation ──────────────────────────────────────────

    def _extract_code(self, text: str) -> str:
        """Strip chain-of-thought blocks and markdown fences; return raw code."""
        text = self._llm.strip_thinking(text)

        for fence in ("```python", "```"):
            if fence in text:
                start = text.index(fence) + len(fence)
                end = text.find("```", start)
                if end != -1:
                    return text[start:end].strip()

        return text

    @staticmethod
    def _check_python_syntax(path: str, content: str) -> str | None:
        """Return a SyntaxError description if *content* is invalid Python, else None."""
        if not path.endswith(".py"):
            return None
        try:
            ast.parse(content)
            return None
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"

    def _generate_code(self, prompt: str, path: str) -> str:
        """Call the LLM, extract code, and retry once on Python syntax errors."""
        feedback = ""
        for _ in range(2):
            effective_prompt = (
                prompt if not feedback
                else (
                    f"{prompt}\n\n"
                    f"The previous response contained a Python syntax error: {feedback}\n"
                    "Fix the indentation and syntax issues. Return ONLY valid Python code."
                )
            )
            raw = self._llm.generate(effective_prompt)
            content = self._extract_code(raw)
            if not content:
                raise ValueError("LLM returned empty code content")
            syntax_err = self._check_python_syntax(path, content)
            if syntax_err is None:
                return content
            feedback = syntax_err
        return content  # best effort; verifier will catch remaining issues

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_create_prompt(self, path: str, specification: str, state: TaskState) -> str:
        return (
            f"You are a coding agent. Generate source code for a new file.\n"
            f"Return ONLY the code — no explanations, no markdown fences.\n\n"
            f"File path: {path}\n"
            f"Task goal: {state.goal}\n"
            f"Specification: {specification}"
        )

    def _build_modify_prompt(self, path: str, existing_content: str, specification: str, state: TaskState) -> str:
        return (
            f"You are a coding agent. Modify an existing source file.\n"
            f"Return ONLY the complete new file content — no explanations, no markdown fences.\n\n"
            f"File path: {path}\n"
            f"Task goal: {state.goal}\n"
            f"Modification required: {specification}\n\n"
            f"Current file content:\n{existing_content}"
        )

    # ── LLM-backed action handlers ────────────────────────────────────────────

    def _execute_code_file(self, call: ToolCall, state: TaskState) -> ToolResult:
        """Generate a new file from a specification using the LLM."""
        path = call.args.get("path", "").strip()
        specification = call.args.get("specification", "").strip()

        if not path:
            return ToolResult(ok=False, tool="code_file", output="Missing required arg: path")
        if not specification:
            return ToolResult(ok=False, tool="code_file", output="Missing required arg: specification")

        try:
            content = self._generate_code(self._build_create_prompt(path, specification, state), path)
        except Exception as exc:
            return ToolResult(ok=False, tool="code_file", output=f"LLM generation failed: {exc}")

        if not content:
            return ToolResult(ok=False, tool="code_file", output="LLM returned empty code content")

        write_result = FilesystemTool.run({"path": path, "content": content})
        if not write_result.ok:
            return ToolResult(ok=False, tool="code_file", output=f"Write failed: {write_result.output}")

        rel = write_result.data.get("path", path)
        return ToolResult(ok=True, tool="code_file", output=f"Wrote file: {rel}", data={"kind": "file_write", "path": rel})

    def _execute_modify_file(self, call: ToolCall, state: TaskState) -> ToolResult:
        """Read an existing file, apply a specification-driven LLM edit, and write it back."""
        path = call.args.get("path", "").strip()
        specification = call.args.get("specification", "").strip()

        if not path:
            return ToolResult(ok=False, tool="modify_file", output="Missing required arg: path")
        if not specification:
            return ToolResult(ok=False, tool="modify_file", output="Missing required arg: specification")

        read_result = ReadFileTool.run({"path": path})
        if not read_result.ok:
            return ToolResult(ok=False, tool="modify_file", output=f"Cannot read source file: {read_result.output}")

        try:
            content = self._generate_code(
                self._build_modify_prompt(path, read_result.output, specification, state), path
            )
        except Exception as exc:
            return ToolResult(ok=False, tool="modify_file", output=f"LLM generation failed: {exc}")

        if not content:
            return ToolResult(ok=False, tool="modify_file", output="LLM returned empty code content")

        write_result = FilesystemTool.run({"path": path, "content": content})
        if not write_result.ok:
            return ToolResult(ok=False, tool="modify_file", output=f"Write failed: {write_result.output}")

        rel = write_result.data.get("path", path)
        return ToolResult(ok=True, tool="modify_file", output=f"Wrote file: {rel}", data={"kind": "file_write", "path": rel})

    # ── Main execution entry point ────────────────────────────────────────────

    def execute(self, call: ToolCall, state: TaskState) -> ToolResult:
        if call.tool in _CODER_NATIVE_TOOLS:
            result = (
                self._execute_code_file(call, state)
                if call.tool == "code_file"
                else self._execute_modify_file(call, state)
            )
        elif not self.can_execute(call.tool):
            result = ToolResult(ok=False, tool=call.tool, output=f"Permission denied for coder agent: {call.tool}")
        else:
            tool_cls = self.tool_registry.get(call.tool)
            if tool_cls is None:
                result = ToolResult(ok=False, tool=call.tool, output=f"Unknown tool: {call.tool}")
            else:
                try:
                    result = tool_cls.run(call.args)
                except Exception as exc:
                    result = ToolResult(ok=False, tool=call.tool, output=f"{type(exc).__name__}: {exc}")

        state.log("tool_result", result.__dict__)
        return result
