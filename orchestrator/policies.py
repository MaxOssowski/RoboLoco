
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Set


WORKSPACE = Path("./workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

ALLOWED_SHELL_PREFIXES = [
    "python",
    "python3",
    "pytest",
    "ls",
    "pwd",
    "cat",
    "echo",
]

AGENT_PERMISSIONS: Dict[str, Set[str]] = {
    "planner": set(),
    "coder": {"write_file", "read_file", "replace_in_file", "patch_file", "code_file", "modify_file"},
    "verifier": {"read_file", "run_shell", "file_exists"},
    "researcher": {"read_file", "list_files", "search_in_files", "file_exists", "summarize_file"},
}


class SecurityError(Exception):
    pass


class ValidationError(Exception):
    pass


def resolve_workspace_path(relative_path: str) -> Path:
    if not relative_path or not relative_path.strip():
        raise ValidationError("Path must not be empty.")

    candidate = (WORKSPACE / relative_path).resolve()
    if not str(candidate).startswith(str(WORKSPACE)):
        raise SecurityError(f"Path escapes workspace: {relative_path}")

    return candidate


def validate_shell_command(command: str) -> List[str]:
    if not command.strip():
        raise ValidationError("Shell command must not be empty.")

    parts = shlex.split(command)
    if not parts:
        raise ValidationError("Invalid shell command.")

    program = parts[0]
    if program not in ALLOWED_SHELL_PREFIXES:
        raise SecurityError(f"Command not allowed: {program}")

    banned_tokens = {"&&", "||", ";", "|", ">", ">>", "<"}
    if any(token in command for token in banned_tokens):
        raise SecurityError("Shell chaining, piping, or redirection is not allowed.")

    return parts
