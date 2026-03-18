from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.verifier import VerifierAgent
from orchestrator.state import AgentResult, TaskState


def _make_verifier(strict: bool = False) -> VerifierAgent:
    return VerifierAgent(model="test-model", tool_registry={}, strict_verifier=strict)


def _state_with_tools(*tool_payloads: dict) -> TaskState:
    """Return a TaskState pre-seeded with tool_result events."""
    state = TaskState(goal="create foo.py with hello world output")
    for payload in tool_payloads:
        state.log("tool_result", payload)
    return state


_INCONCLUSIVE = AgentResult(
    agent="verifier",
    reasoning_summary="Inconclusive.",
    status="inconclusive",
)

_PASSED = AgentResult(
    agent="verifier",
    reasoning_summary="Goal achieved.",
    status="passed",
)

_FAILED_SEMANTIC = AgentResult(
    agent="verifier",
    reasoning_summary="Goal not met.",
    status="failed",
)


class FileExistsAloneDoesNotPassTest(unittest.TestCase):
    """file_exists success alone must not grant 'passed'."""

    def test_file_exists_alone_is_not_sufficient(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "file_exists", "output": "foo.py exists"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertNotEqual(result.status, "passed")

    def test_file_exists_inconclusive_semantic_gives_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "file_exists", "output": "foo.py exists"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")

    def test_file_exists_with_semantic_passed_gives_passed(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "file_exists", "output": "foo.py exists"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_PASSED):
            result = verifier.act(state)
        self.assertEqual(result.status, "passed")


class RunShellAloneDoesNotPassTest(unittest.TestCase):
    """A successful run_shell alone must not grant 'passed'."""

    def test_run_shell_alone_inconclusive_stays_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "hello world"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")

    def test_run_shell_alone_semantic_passed_gives_passed(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "hello world"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_PASSED):
            result = verifier.act(state)
        self.assertEqual(result.status, "passed")


class WriteFileAloneDoesNotPassTest(unittest.TestCase):
    """A successful write_file alone must not grant 'passed'."""

    def test_write_file_alone_inconclusive_stays_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "write_file", "output": "Wrote file: foo.py"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_written_files_exist", return_value=(True, [])):
            with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
                result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")


class CombinedEvidenceDoesNotPassTest(unittest.TestCase):
    """Even with multiple tools succeeding, inconclusive semantic must stay inconclusive."""

    def test_file_exists_plus_run_shell_inconclusive_stays_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "file_exists", "output": "foo.py"},
            {"ok": True, "tool": "run_shell", "output": "hello world"},
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")

    def test_write_file_plus_run_shell_inconclusive_stays_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "write_file", "output": "Wrote file: foo.py"},
            {"ok": True, "tool": "run_shell", "output": "hello world"},
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_written_files_exist", return_value=(True, [])):
            with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
                result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")


class SemanticPassedAndFailedRespectedTest(unittest.TestCase):
    """When semantic gives a definitive answer it must be honoured."""

    def test_semantic_passed_propagates(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "ok"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_PASSED):
            result = verifier.act(state)
        self.assertEqual(result.status, "passed")

    def test_semantic_failed_propagates(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "ok"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_FAILED_SEMANTIC):
            result = verifier.act(state)
        self.assertEqual(result.status, "failed")


class StrictModeFailsOnInconclusiveTest(unittest.TestCase):
    """strict_verifier=True converts inconclusive to failed."""

    def test_strict_fails_on_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "done"}
        )
        verifier = _make_verifier(strict=True)
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertEqual(result.status, "failed")

    def test_non_strict_stays_inconclusive(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "done"}
        )
        verifier = _make_verifier(strict=False)
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertEqual(result.status, "inconclusive")

    def test_strict_is_strictly_stricter_than_non_strict(self) -> None:
        """strict and non-strict may not both return 'passed' when semantic is inconclusive."""
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "done"}
        )
        strict_verifier = _make_verifier(strict=True)
        lenient_verifier = _make_verifier(strict=False)
        with patch.object(strict_verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            strict_result = strict_verifier.act(state)
        with patch.object(lenient_verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            lenient_result = lenient_verifier.act(state)
        # strict must not pass
        self.assertNotEqual(strict_result.status, "passed")
        # lenient must not pass either — it stays inconclusive
        self.assertNotEqual(lenient_result.status, "passed")
        # and strict must be "worse" (more conservative) than lenient
        self.assertEqual(strict_result.status, "failed")
        self.assertEqual(lenient_result.status, "inconclusive")


class InconclusiveReasoningDescribesEvidenceTest(unittest.TestCase):
    """The inconclusive reasoning_summary must describe what tool evidence was observed."""

    def test_reasoning_mentions_no_evidence_when_only_run_shell(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "ok"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertIn("shell command exited with code 0", result.reasoning_summary)

    def test_reasoning_mentions_file_written_when_write_file(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "write_file", "output": "Wrote file: foo.py"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_written_files_exist", return_value=(True, [])):
            with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
                result = verifier.act(state)
        self.assertIn("file was written", result.reasoning_summary)

    def test_reasoning_mentions_file_existence_when_file_exists(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "file_exists", "output": "foo.py"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertIn("file existence confirmed on disk", result.reasoning_summary)

    def test_reasoning_mentions_no_notable_tool_evidence_when_no_known_tools(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "summarize_file", "output": "summary text"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertIn("no notable tool evidence", result.reasoning_summary)

    def test_reasoning_always_says_cannot_count_as_success(self) -> None:
        state = _state_with_tools(
            {"ok": True, "tool": "run_shell", "output": "ok"}
        )
        verifier = _make_verifier()
        with patch.object(verifier, "_semantic_verify", return_value=_INCONCLUSIVE):
            result = verifier.act(state)
        self.assertIn("Cannot count this as success", result.reasoning_summary)


if __name__ == "__main__":
    unittest.main()
