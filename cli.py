#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

try:
    import readline
    _HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".roboloco_history")
    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False

from orchestrator.main import Orchestrator
from orchestrator.state import ModelConfig

# ── ANSI colours ────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Banner ───────────────────────────────────────────────────────────────────
BANNER = f"""{CYAN}{BOLD}
 ██████╗  ██████╗ ██████╗  ██████╗     ██╗      ██████╗  ██████╗ ██████╗
 ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗    ██║     ██╔═══██╗██╔════╝██╔═══██╗
 ██████╔╝██║   ██║██████╔╝██║   ██║    ██║     ██║   ██║██║     ██║   ██║
 ██╔══██╗██║   ██║██╔══██╗██║   ██║    ██║     ██║   ██║██║     ██║   ██║
 ██║  ██║╚██████╔╝██████╔╝╚██████╔╝    ███████╗╚██████╔╝╚██████╗╚██████╔╝
 ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝
{RESET}"""

TAGLINE = "  Modular Local AI Agent System  ·  Powered by Ollama\n"

DIVIDER = f"  {DIM}{'─' * 62}{RESET}"

HELP_TEXT = f"""
{DIVIDER}
  {BOLD}Commands{RESET}

    {YELLOW}<task>{RESET}           Run agents on the given task
    {YELLOW}run <task>{RESET}       Same as above (explicit form)
    {YELLOW}options{RESET}          View and edit runtime settings
    {YELLOW}help{RESET}             Show this message
    {YELLOW}exit  /  quit{RESET}    Close the session

  {BOLD}Examples{RESET}

    {DIM}> Create a Python script that prints Fibonacci numbers{RESET}
    {DIM}> run Write a function that checks if a string is a palindrome{RESET}
{DIVIDER}
"""


# ── Runtime settings ──────────────────────────────────────────────────────────

@dataclass
class Settings:
    default_model: str = "llama3.1:8b"
    planner_model: str | None = None
    coder_model: str | None = None
    verifier_model: str | None = None
    researcher_model: str | None = None
    planner_timeout: int = 60
    coder_timeout: int = 120
    strict_verifier: bool = False
    verbose: bool = False

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            default=self.default_model,
            planner=self.planner_model,
            coder=self.coder_model,
            verifier=self.verifier_model,
            researcher=self.researcher_model,
        )

    def build_orchestrator(self) -> Orchestrator:
        return Orchestrator(
            models=self.to_model_config(),
            strict_verifier=self.strict_verifier,
            planner_timeout=self.planner_timeout,
            coder_timeout=self.coder_timeout,
        )


def _fetch_ollama_models() -> list[str]:
    """Return model names reported by ``ollama list``."""
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0:
            return []
        models: list[str] = []
        for line in r.stdout.strip().splitlines()[1:]:  # skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


# ── Options menu ──────────────────────────────────────────────────────────────

# (key, display-label) pairs in the order they appear in the menu
_OPT_ROWS: list[tuple[str, str]] = [
    ("default-model",    "Default model for all agents"),
    ("planner-model",    "Planner agent model"),
    ("coder-model",      "Coder agent model"),
    ("verifier-model",   "Verifier agent model"),
    ("researcher-model", "Researcher agent model"),
    ("planner-timeout",  "Planner LLM timeout (seconds)"),
    ("coder-timeout",    "Coder LLM timeout (seconds)"),
    ("strict-verifier",  "Require verifier to pass"),
    ("verbose",          "Print full JSON result"),
]

_MODEL_OPTS = frozenset(
    {"default-model", "planner-model", "coder-model", "verifier-model", "researcher-model"}
)
_BOOL_OPTS = frozenset({"strict-verifier", "verbose"})


def _opt_display(key: str, s: Settings) -> str:
    """Coloured current-value string for one option row."""
    mc = s.to_model_config()
    if key == "default-model":
        return f"{YELLOW}{s.default_model}{RESET}"
    if key in _MODEL_OPTS:
        agent = key.replace("-model", "")
        override = getattr(s, f"{agent}_model", None)
        eff = mc.for_agent(agent)
        return f"{YELLOW}{override}{RESET}" if override else f"{DIM}(default → {eff}){RESET}"
    if key == "planner-timeout":
        return f"{YELLOW}{s.planner_timeout}{RESET} s"
    if key == "coder-timeout":
        return f"{YELLOW}{s.coder_timeout}{RESET} s"
    if key == "strict-verifier":
        return f"{GREEN}on{RESET}" if s.strict_verifier else f"{DIM}off{RESET}"
    if key == "verbose":
        return f"{GREEN}on{RESET}" if s.verbose else f"{DIM}off{RESET}"
    return "?"


def _print_options(s: Settings) -> None:
    key_width = max(len(k) for k, _ in _OPT_ROWS)
    print(f"\n{DIVIDER}")
    print(f"  {BOLD}Options{RESET}\n")
    for i, (key, _desc) in enumerate(_OPT_ROWS, 1):
        print(f"    {DIM}{i:>2}{RESET}  {key.ljust(key_width)}  {_opt_display(key, s)}")
    print()


def options_menu(s: Settings) -> bool:
    """Interactive options editor.  Returns True when a value was changed."""
    _print_options(s)
    try:
        raw = input(
            f"  Enter number to edit, or {DIM}Enter{RESET} to cancel: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print(f"{DIVIDER}\n")
        return False
    print(f"{DIVIDER}\n")

    if not raw:
        return False

    try:
        idx = int(raw) - 1
        if not (0 <= idx < len(_OPT_ROWS)):
            raise ValueError
    except ValueError:
        print(f"  {RED}Invalid choice.{RESET}\n")
        return False

    key, desc = _OPT_ROWS[idx]

    # ── Model option ──────────────────────────────────────────────────────────
    if key in _MODEL_OPTS:
        print(f"  {DIM}Fetching available Ollama models…{RESET}")
        available = _fetch_ollama_models()
        if not available:
            print(f"  {RED}Could not retrieve models from Ollama.{RESET}\n")
            return False

        is_per_agent = key != "default-model"
        agent        = key.replace("-model", "")
        current_val  = (
            s.default_model if agent == "default"
            else getattr(s, f"{agent}_model", None) or ""
        )

        print(f"\n  {BOLD}Available models{RESET}  — {desc}\n")
        for j, m in enumerate(available, 1):
            marker = f"  {GREEN}← current{RESET}" if m == current_val else ""
            print(f"    {DIM}{j:>2}{RESET}  {m}{marker}")
        if is_per_agent:
            print(f"    {DIM} 0{RESET}  {DIM}(use default){RESET}")
        print()

        try:
            pick = input(
                f"  Pick [1–{len(available)}]"
                + (", 0 to use default" if is_per_agent else "")
                + f", or {DIM}Enter{RESET} to cancel: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        print()

        if not pick:
            return False

        if is_per_agent and pick == "0":
            setattr(s, f"{agent}_model", None)
            print(f"  {GREEN}✓{RESET}  {key} reset to default.\n")
            return True

        try:
            pidx = int(pick) - 1
            if not (0 <= pidx < len(available)):
                raise ValueError
        except ValueError:
            print(f"  {RED}Invalid choice.{RESET}\n")
            return False

        chosen = available[pidx]
        attr   = "default_model" if agent == "default" else f"{agent}_model"
        setattr(s, attr, chosen)
        print(f"  {GREEN}✓{RESET}  {key} set to {YELLOW}{chosen}{RESET}.\n")
        return True

    # ── planner-timeout ───────────────────────────────────────────────────────
    if key == "planner-timeout":
        try:
            new_val = input(
                f"  planner-timeout [{DIM}current: {s.planner_timeout} s{RESET}]"
                f"  New value (s): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        print()
        if not new_val:
            return False
        try:
            s.planner_timeout = int(new_val)
        except ValueError:
            print(f"  {RED}Must be an integer.{RESET}\n")
            return False
        print(f"  {GREEN}✓{RESET}  planner-timeout set to {YELLOW}{s.planner_timeout}{RESET} s.\n")
        return True

    # ── coder-timeout ─────────────────────────────────────────────────────────
    if key == "coder-timeout":
        try:
            new_val = input(
                f"  coder-timeout [{DIM}current: {s.coder_timeout} s{RESET}]"
                f"  New value (s): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return False
        print()
        if not new_val:
            return False
        try:
            s.coder_timeout = int(new_val)
        except ValueError:
            print(f"  {RED}Must be an integer.{RESET}\n")
            return False
        print(f"  {GREEN}✓{RESET}  coder-timeout set to {YELLOW}{s.coder_timeout}{RESET} s.\n")
        return True

    # ── Boolean toggle ────────────────────────────────────────────────────────
    if key in _BOOL_OPTS:
        attr    = key.replace("-", "_")
        new_val = not getattr(s, attr)
        setattr(s, attr, new_val)
        state   = f"{GREEN}on{RESET}" if new_val else f"{DIM}off{RESET}"
        print(f"  {GREEN}✓{RESET}  {key} → {state}\n")
        return True

    return False


_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_AGENT_ICONS = {
    "planner":    f"{CYAN}planner{RESET}",
    "coder":      f"{YELLOW}coder{RESET}",
    "verifier":   f"{GREEN}verifier{RESET}",
    "researcher": f"{CYAN}researcher{RESET}",
}


class StatusSpinner:
    """Animated status line for the terminal.  No-ops when stdout is not a TTY."""

    def __init__(self) -> None:
        self._msg = ""
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._is_tty = sys.stdout.isatty()

    def start(self, initial: str = "") -> None:
        if not self._is_tty:
            return
        self._msg = initial
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, msg: str) -> None:
        with self._lock:
            self._msg = msg

    def stop(self) -> None:
        if not self._is_tty:
            return
        self._running = False
        if self._thread:
            self._thread.join()
        # clear the spinner line so subsequent print() calls land cleanly
        sys.stdout.write(f"\r\033[K")
        sys.stdout.flush()

    def _loop(self) -> None:
        for frame in itertools.cycle(_SPINNER_FRAMES):
            if not self._running:
                break
            with self._lock:
                msg = self._msg
            sys.stdout.write(f"\r  {CYAN}{frame}{RESET}  {DIM}{msg}{RESET}\033[K")
            sys.stdout.flush()
            time.sleep(0.08)


def _model_line(models: ModelConfig) -> str:
    agents = ("planner", "verifier", "coder", "researcher")
    parts = [f"{a}: {YELLOW}{models.for_agent(a)}{RESET}{DIM}" for a in agents]
    return f"  {DIM}" + "  ·  ".join(parts) + RESET


def print_banner(models: ModelConfig) -> None:
    print(BANNER)
    print(f"{CYAN}{BOLD}{TAGLINE}{RESET}")
    print(_model_line(models))
    print()
    print(f"  {DIM}Type a task to run, {YELLOW}help{RESET}{DIM} for commands, {RED}exit{RESET}{DIM} to quit.{RESET}")
    print(f"\n{DIVIDER}\n")


def run_task(orchestrator: Orchestrator, task: str, verbose: bool) -> None:
    print()
    spinner = StatusSpinner()
    spinner.start("planner · thinking…")

    def on_event(kind: str, **kwargs) -> None:
        if kind == "planning":
            attempt = kwargs.get("attempt", 0)
            if attempt == 0:
                spinner.update("planner · thinking…")
            else:
                spinner.update(f"planner · retrying  (attempt {attempt + 1})…")
        elif kind == "executing":
            agent = kwargs.get("agent", "agent")
            tool  = kwargs.get("tool", "tool")
            icon  = _AGENT_ICONS.get(agent, agent)
            spinner.update(f"{icon} · {tool}")
        elif kind == "verifying":
            spinner.update("verifier · checking result…")

    result = orchestrator.run(task, on_event=on_event)
    spinner.stop()

    planner_status = result.get("planner_status", "")
    status  = result.get("verification_status", "unknown")
    color   = GREEN if status == "passed" else RED
    summary = result.get("verification_summary", "")
    if planner_status not in ("ready", "") and status == "failed" and not summary:
        summary = result.get("planner_summary", "")

    print(DIVIDER)
    print(f"  {color}{BOLD}● {status.upper()}{RESET}")
    if summary:
        print(f"  {DIM}{summary}{RESET}")

    if verbose:
        print()
        print(json.dumps(result, indent=2))

    print(f"{DIVIDER}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboLoco — interactive AI agent session")
    parser.add_argument("--model",            default="llama3.1:8b", help="Default Ollama model for all agents")
    parser.add_argument("--planner-model",    default=None,          help="Ollama model for the planner agent")
    parser.add_argument("--verifier-model",   default=None,          help="Ollama model for the verifier agent")
    parser.add_argument("--coder-model",      default=None,          help="Ollama model for the coder agent")
    parser.add_argument("--researcher-model", default=None,          help="Ollama model for the researcher agent")
    parser.add_argument("--planner-timeout",  type=int, default=60,  help="Seconds before the planner Ollama call times out (default: 60)")
    parser.add_argument("--coder-timeout",    type=int, default=120, help="Seconds before the coder Ollama call times out (default: 120)")
    parser.add_argument("--strict-verifier",  action="store_true",   help="Require semantic verification to pass")
    parser.add_argument("--verbose", "-v",    action="store_true",   help="Print full JSON result after each task")
    args = parser.parse_args()

    settings = Settings(
        default_model=args.model,
        planner_model=args.planner_model,
        coder_model=args.coder_model,
        verifier_model=args.verifier_model,
        researcher_model=args.researcher_model,
        planner_timeout=args.planner_timeout,
        coder_timeout=args.coder_timeout,
        strict_verifier=args.strict_verifier,
        verbose=args.verbose,
    )

    print_banner(settings.to_model_config())

    if _READLINE_AVAILABLE:
        readline.set_history_length(500)
        if os.path.exists(_HISTORY_FILE):
            readline.read_history_file(_HISTORY_FILE)

    orchestrator = settings.build_orchestrator()

    while True:
        try:
            raw = input(f"  {CYAN}❯{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {DIM}Session closed. Goodbye!{RESET}\n")
            if _READLINE_AVAILABLE:
                readline.write_history_file(_HISTORY_FILE)
            sys.exit(0)

        if not raw:
            continue

        if _READLINE_AVAILABLE:
            readline.write_history_file(_HISTORY_FILE)

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):
            print(f"\n  {DIM}Session closed. Goodbye!{RESET}\n")
            sys.exit(0)

        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd == "options":
            if options_menu(settings):
                orchestrator = settings.build_orchestrator()
                print(_model_line(settings.to_model_config()))
                print()
            continue

        task = raw[4:].strip() if cmd.startswith("run ") else raw

        if not task:
            print(f"  {RED}No task provided. Usage: run <task>{RESET}\n")
            continue

        run_task(orchestrator, task, verbose=settings.verbose)


if __name__ == "__main__":
    main()
