"""Shared Ollama client used by all agents.

Uses the Ollama REST API (POST /api/generate) instead of spawning
``ollama run`` as a subprocess.  This eliminates fork/exec overhead on
every LLM call and keeps the model hot in VRAM between calls.

All agents that previously had their own ``_call_ollama`` copy now import
``OllamaClient`` from here.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_BASE_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, model: str, timeout: int = 120) -> None:
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """POST /api/generate and return the response text (stripped)."""
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read()).get("response", "").strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code} for model '{self.model}'. "
                f"Is the model installed? Run: ollama pull {self.model}"
                + (f"\nServer message: {body}" if body else "")
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama not reachable at {_BASE_URL} — is it running? ({exc.reason})"
            ) from exc

    def strip_thinking(self, text: str) -> str:
        """Remove <think>…</think> blocks and stray </think> tags."""
        text = _THINK_RE.sub("", text)
        return re.sub(r"</think>", "", text).strip()

    @staticmethod
    def list_models() -> list[str]:
        """Return model names available in the local Ollama instance."""
        try:
            with urllib.request.urlopen(f"{_BASE_URL}/api/tags", timeout=10) as resp:
                return [m["name"] for m in json.loads(resp.read()).get("models", [])]
        except Exception:
            return []
