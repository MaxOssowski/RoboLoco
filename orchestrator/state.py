
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCall:
    agent: str
    tool: str
    args: Dict[str, Any]


@dataclass
class AgentResult:
    agent: str
    reasoning_summary: str
    actions: List[ToolCall] = field(default_factory=list)
    status: str = "ready"


@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: str


@dataclass
class TaskState:
    goal: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.history.append({"event_type": event_type, "payload": payload})
