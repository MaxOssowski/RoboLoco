
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

import re
import subprocess

from orchestrator.policies import AGENT_PERMISSIONS, WORKSPACE
from orchestrator.state import TaskState, ToolCall, ToolResult
from tools.filesystem import FilesystemTool, ReadFileTool

# Tool names that CoderAgent handles with LLM logic before reaching the registry.
_CODER_NATIVE_TOOLS = frozenset({"code_file", "modify_file"})


class CoderAgent:
    name = "coder"

    def __init__(self, tool_registry: dict[str, object], model: str = "qwen3:8b", timeout: int = 120) -> None:
        self.tool_registry = tool_registry
        self.model = model
        self.timeout = timeout

    def can_execute(self, tool_name: str) -> bool:
        return tool_name in AGENT_PERMISSIONS[self.name]

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model, prompt],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "Ollama call failed").strip())
        return (result.stdout or "").strip()

    # Matches <think>...</think> reasoning blocks emitted by thinking models
    # such as Qwen3 and DeepSeek-R1.  The block may span many lines.
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    def _extract_code(self, text: str) -> str:
        """Strip chain-of-thought blocks and markdown fences; return raw code.

        Handles (in order):
        1. ``<think>…</think>`` blocks produced by thinking models (Qwen3, etc.)
        2. ``/think`` tags used by some model variants
        3. Fenced code blocks (````python` or plain ````` ```)
        4. Plain text after all the above are removed
        """
        # 1. Remove <think>…</think> reasoning blocks.
        text = self._THINK_RE.sub("", text)
        # 2. Remove any bare /think marker that some variants emit without a
        #    matching opening tag.
        text = re.sub(r"</think>", "", text)
        text = text.strip()

        # 3. Extract from the first fenced code block if present.
        for fence in ("```python", "```"):
            if fence in text:
                start = text.index(fence) + len(fence)
                end = text.find("```", start)
                if end != -1:
                    return text[start:end].strip()

        # 4. No fences — return whatever remains after reasoning removal.
        return text

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_create_prompt(self, path: str, specification: str, state: TaskState) -> str:
        return f"""You are a coding agent. Generate source code for a new file.
Return ONLY the code — no explanations, no markdown fences.

File path: {path}
Task goal: {state.goal}
Specification: {specification}""".strip()

    def _build_modify_prompt(
        self, path: str, existing_content: str, specification: str, state: TaskState
    ) -> str:
        return f"""You are a coding agent. Modify an existing source file.
Return ONLY the complete new file content — no explanations, no markdown fences.

File path: {path}
Task goal: {state.goal}
Modification required: {specification}

Current file content:
{existing_content}""".strip()

    # ── LLM-backed action handlers ────────────────────────────────────────────

    def _execute_code_file(self, call: ToolCall, state: TaskState) -> ToolResult:
        """Generate a new file from a specification using the LLM."""
        path = call.args.get("path", "").strip()
        specification = call.args.get("specification", "").strip()

        if not path:
            return ToolResult(ok=False, tool="code_file", output="Missing required arg: path")
        if not specification:
            return ToolResult(ok=False, tool="code_file", output="Missing required arg: specification")

        prompt = self._build_create_prompt(path, specification, state)
        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:
            return ToolResult(ok=False, tool="code_file", output=f"LLM generation failed: {exc}")

        content = self._extract_code(raw)
        if not content:
            return ToolResult(ok=False, tool="code_file", output="LLM returned empty code content")

        write_result = FilesystemTool.run({"path": path, "content": content})
        if not write_result.ok:
            return ToolResult(ok=False, tool="code_file", output=f"Write failed: {write_result.output}")

        rel = write_result.data.get("path", path)
        return ToolResult(
            ok=True,
            tool="code_file",
            output=f"Wrote file: {rel}",
            data={"kind": "file_write", "path": rel},
        )

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
            return ToolResult(
                ok=False,
                tool="modify_file",
                output=f"Cannot read source file: {read_result.output}",
            )

        prompt = self._build_modify_prompt(path, read_result.output, specification, state)
        try:
            raw = self._call_ollama(prompt)
        except Exception as exc:
            return ToolResult(ok=False, tool="modify_file", output=f"LLM generation failed: {exc}")

        content = self._extract_code(raw)
        if not content:
            return ToolResult(ok=False, tool="modify_file", output="LLM returned empty code content")

        write_result = FilesystemTool.run({"path": path, "content": content})
        if not write_result.ok:
            return ToolResult(ok=False, tool="modify_file", output=f"Write failed: {write_result.output}")

        rel = write_result.data.get("path", path)
        return ToolResult(
            ok=True,
            tool="modify_file",
            output=f"Wrote file: {rel}",
            data={"kind": "file_write", "path": rel},
        )

    # ── Main execution entry point ────────────────────────────────────────────

    def execute(self, call: ToolCall, state: TaskState) -> ToolResult:
        # LLM-backed native tools are handled before the registry lookup.
        if call.tool in _CODER_NATIVE_TOOLS:
            if call.tool == "code_file":
                result = self._execute_code_file(call, state)
            else:
                result = self._execute_modify_file(call, state)
        elif not self.can_execute(call.tool):
            result = ToolResult(
                ok=False, tool=call.tool, output=f"Permission denied for coder agent: {call.tool}"
            )
        else:
            tool_cls = self.tool_registry.get(call.tool)
            if tool_cls is None:
                result = ToolResult(ok=False, tool=call.tool, output=f"Unknown tool: {call.tool}")
            else:
                try:
                    result = tool_cls.run(call.args)
                except Exception as exc:
                    result = ToolResult(
                        ok=False, tool=call.tool, output=f"{type(exc).__name__}: {exc}"
                    )

        state.log("tool_result", result.__dict__)
        return result
