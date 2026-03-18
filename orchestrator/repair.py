
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailedActionFingerprint:
    """Identifies a failed tool call by its agent, tool, and argument shape."""

    agent: str
    tool: str
    arg_keys: list[str]           # sorted keys — captures the shape of args, not values
    arg_snapshot: dict[str, Any]  # full args, for display in repair prompts
    error_output: str
    attempt: int

    @property
    def signature(self) -> str:
        """Pattern key: agent.tool(sorted_arg_keys).

        Two calls share a signature when they use the same agent, tool, and
        argument shape — even if the argument values differ.  This lets the
        planner detect when it keeps repeating an approach that has already
        failed.
        """
        return f"{self.agent}.{self.tool}({', '.join(self.arg_keys)})"

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "tool": self.tool,
            "arg_keys": self.arg_keys,
            "arg_snapshot": self.arg_snapshot,
            "error_output": self.error_output,
            "attempt": self.attempt,
            "signature": self.signature,
        }


@dataclass
class RepairDiagnostic:
    """Full structured diagnosis of a failed attempt, stored in state history.

    Consumed by PlannerAgent to build a targeted repair prompt instead of
    re-issuing a generic planning prompt.
    """

    attempt: int
    success_criteria: list[str]       # structured criteria from previous plan
    verification_reasoning: str       # why the verifier was unsatisfied (or "Tool failed before verification.")
    failed_fingerprint: FailedActionFingerprint | None = None   # None for semantic-only failures
    prior_fingerprints: list[FailedActionFingerprint] = field(default_factory=list)

    @property
    def is_repeated_failure(self) -> bool:
        """True when the same agent+tool+arg_shape has failed in a previous attempt."""
        if self.failed_fingerprint is None:
            return False
        sig = self.failed_fingerprint.signature
        return any(fp.signature == sig for fp in self.prior_fingerprints)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "success_criteria": self.success_criteria,
            "verification_reasoning": self.verification_reasoning,
            "failed_fingerprint": (
                self.failed_fingerprint.as_dict() if self.failed_fingerprint else None
            ),
            "is_repeated_failure": self.is_repeated_failure,
            "prior_signatures": [fp.signature for fp in self.prior_fingerprints],
        }


def fingerprint_failed_action(
    agent: str,
    tool: str,
    args: dict[str, Any],
    error_output: str,
    attempt: int,
) -> FailedActionFingerprint:
    """Build a FailedActionFingerprint from raw tool-call data."""
    return FailedActionFingerprint(
        agent=agent,
        tool=tool,
        arg_keys=sorted(args.keys()),
        arg_snapshot=dict(args),
        error_output=error_output,
        attempt=attempt,
    )
