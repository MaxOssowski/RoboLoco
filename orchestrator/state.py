
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    default: str = "llama3.1:8b"
    planner: Optional[str] = None
    verifier: Optional[str] = None
    coder: Optional[str] = "qwen3:8b"
    researcher: Optional[str] = None

    def for_agent(self, agent: str) -> str:
        return getattr(self, agent, None) or self.default


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
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    ok: bool
    tool: str
    output: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    goal: str
    history: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.history.append({"event_type": event_type, "payload": payload})
