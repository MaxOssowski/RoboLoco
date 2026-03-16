from __future__ import annotations

import json
import shutil
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import memory.summary_store as store


class SummaryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # Redirect SUMMARIES_DIR to a temp location for isolation
        self._real_dir = store.SUMMARIES_DIR
        self._tmp_dir = self._real_dir.parent / "_test_summaries"
        store.SUMMARIES_DIR = self._tmp_dir

    def tearDown(self) -> None:
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir)
        store.SUMMARIES_DIR = self._real_dir

    def _make_summary(self, goal: str = "do something", status: str = "passed") -> dict:
        return {
            "goal": goal,
            "status": status,
            "summary": "ok",
            "success_criteria": "",
            "files_touched": [],
            "tools_used": [],
            "key_outputs": [],
            "retry_count": 0,
            "agent_models": {},
            "duration_seconds": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def test_save_creates_timestamped_file_and_latest(self) -> None:
        summary = self._make_summary()
        store.save_task_summary(summary)

        files = list(self._tmp_dir.glob("*.json"))
        self.assertEqual(len(files), 2)  # timestamped + latest.json
        self.assertTrue((self._tmp_dir / "latest.json").exists())

    def test_load_latest_returns_saved_summary(self) -> None:
        summary = self._make_summary(goal="create hello.py")
        store.save_task_summary(summary)

        loaded = store.load_latest_summary()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["goal"], "create hello.py")

    def test_load_latest_returns_none_when_empty(self) -> None:
        self.assertIsNone(store.load_latest_summary())

    def test_load_recent_summaries_returns_in_reverse_order(self) -> None:
        for i in range(3):
            s = self._make_summary(goal=f"task {i}")
            s["timestamp"] = f"2026-03-1{i + 1}T00:00:00+00:00"
            store.save_task_summary(s)

        recent = store.load_recent_summaries(limit=3)
        self.assertEqual(len(recent), 3)
        # Most recent first
        self.assertEqual(recent[0]["goal"], "task 2")
        self.assertEqual(recent[2]["goal"], "task 0")

    def test_load_recent_summaries_respects_limit(self) -> None:
        for i in range(5):
            s = self._make_summary(goal=f"task {i}")
            s["timestamp"] = f"2026-03-{10 + i}T00:00:00+00:00"
            store.save_task_summary(s)

        recent = store.load_recent_summaries(limit=2)
        self.assertEqual(len(recent), 2)

    def test_load_recent_returns_empty_when_no_store(self) -> None:
        result = store.load_recent_summaries()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
