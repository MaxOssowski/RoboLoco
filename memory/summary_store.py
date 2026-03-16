from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

SUMMARIES_DIR = Path(__file__).resolve().parent / "summaries"


def save_task_summary(summary: dict) -> None:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    ts = summary.get("timestamp", datetime.now(timezone.utc).isoformat())
    safe_ts = ts[:19].replace(":", "-").replace("T", "T")
    filename = f"{safe_ts}_{uuid.uuid4().hex[:8]}.json"

    encoded = json.dumps(summary, indent=2)
    (SUMMARIES_DIR / filename).write_text(encoded, encoding="utf-8")
    (SUMMARIES_DIR / "latest.json").write_text(encoded, encoding="utf-8")


def load_latest_summary() -> dict | None:
    latest = SUMMARIES_DIR / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_recent_summaries(limit: int = 3) -> list[dict]:
    if not SUMMARIES_DIR.exists():
        return []
    summaries = []
    for f in SUMMARIES_DIR.glob("*.json"):
        if f.name == "latest.json":
            continue
        try:
            summaries.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    summaries.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return summaries[:limit]
