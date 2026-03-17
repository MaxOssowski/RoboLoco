#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time

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
    {YELLOW}help{RESET}             Show this message
    {YELLOW}exit  /  quit{RESET}    Close the session

  {BOLD}Examples{RESET}

    {DIM}> Create a Python script that prints Fibonacci numbers{RESET}
    {DIM}> run Write a function that checks if a string is a palindrome{RESET}
{DIVIDER}
"""


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
    parser.add_argument("--model",           default="llama3.1:8b", help="Default Ollama model for all agents")
    parser.add_argument("--planner-model",   default=None,          help="Ollama model for the planner agent")
    parser.add_argument("--verifier-model",  default=None,          help="Ollama model for the verifier agent")
    parser.add_argument("--coder-model",     default=None,          help="Ollama model for the coder agent")
    parser.add_argument("--researcher-model",default=None,          help="Ollama model for the researcher agent")
    parser.add_argument("--planner-timeout",  type=int, default=60,  help="Seconds before the planner Ollama call times out (default: 60)")
    parser.add_argument("--strict-verifier", action="store_true",   help="Require semantic verification to pass")
    parser.add_argument("--verbose", "-v",   action="store_true",   help="Print full JSON result after each task")
    args = parser.parse_args()

    models = ModelConfig(
        default=args.model,
        planner=args.planner_model,
        verifier=args.verifier_model,
        coder=args.coder_model,
        researcher=args.researcher_model,
    )

    print_banner(models)

    orchestrator = Orchestrator(models=models, strict_verifier=args.strict_verifier, planner_timeout=args.planner_timeout)

    while True:
        try:
            raw = input(f"  {CYAN}❯{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {DIM}Session closed. Goodbye!{RESET}\n")
            sys.exit(0)

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):
            print(f"\n  {DIM}Session closed. Goodbye!{RESET}\n")
            sys.exit(0)

        if cmd == "help":
            print(HELP_TEXT)
            continue

        task = raw[4:].strip() if cmd.startswith("run ") else raw

        if not task:
            print(f"  {RED}No task provided. Usage: run <task>{RESET}\n")
            continue

        run_task(orchestrator, task, verbose=args.verbose)


if __name__ == "__main__":
    main()
