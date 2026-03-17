
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Goal keywords that indicate an interactive / GUI Python app
_INTERACTIVE_GOAL_KEYWORDS = {
    "pygame", "game", "gui", "window", "tkinter",
    "arcade", "kivy", "pong", "tetris", "snake", "interactive",
}

# Library names whose presence in import statements confirms interactive app
_INTERACTIVE_IMPORT_LIBS = {
    "pygame", "tkinter", "wx", "kivy", "arcade",
    "PyQt5", "PyQt6", "PySide2", "PySide6",
}


def is_interactive_python_task(goal: str, file_content: str | None = None) -> bool:
    """Return True if the goal or generated source looks like an interactive/GUI Python app."""
    goal_lower = goal.lower()
    if any(kw in goal_lower for kw in _INTERACTIVE_GOAL_KEYWORDS):
        return True
    if file_content:
        for lib in _INTERACTIVE_IMPORT_LIBS:
            if f"import {lib}" in file_content or f"from {lib}" in file_content:
                return True
    return False


def find_target_python_file(goal: str) -> str | None:
    """Return the first *.py filename mentioned in the goal, or None."""
    matches = re.findall(r"\b(\w+\.py)\b", goal)
    return matches[0] if matches else None


def run_py_compile(workspace: Path, relative_path: str) -> tuple[bool, str]:
    """
    Compile a Python file inside *workspace* using `python3 -m py_compile`.

    Returns (success, output) where *output* contains any compile error text.
    """
    abs_path = (workspace / relative_path).resolve()
    try:
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(abs_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (result.stderr or result.stdout or "").strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "py_compile timed out"
    except Exception as exc:
        return False, f"py_compile error: {exc}"
