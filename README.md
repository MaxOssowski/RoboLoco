# RoboLoco

A local multi-agent coding system that orchestrates specialised AI agents to plan, write, verify, and repair code — all running on local LLMs via [Ollama](https://ollama.ai).

---

## Overview

RoboLoco decomposes software tasks into a pipeline of focused agents:

```
Goal → Planner → Coder / Researcher → Verifier → (repair loop) → Result
```

| Agent | Role | LLM |
|---|---|---|
| **Planner** | Decides *what* to build: produces an action plan with success criteria | Default model |
| **Coder** | Decides *how* to build it: calls an LLM to generate or modify code | `qwen3:8b` by default |
| **Researcher** | Reads and inspects files without modifying them | Default model |
| **Verifier** | Checks whether the goal was actually achieved | Default model |

If verification fails, a **repair loop** re-plans with targeted diagnostics — avoiding previously failed action patterns — for up to a configurable number of retries.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with at least one model pulled (e.g. `ollama pull llama3.1:8b`)

No external Python dependencies beyond the standard library are required for the core system.

---

## Quick Start

**One-shot mode** — run a single task and print the result:

```bash
python3 app.py "Create a Python script that prints the first 10 Fibonacci numbers"
```

**Interactive REPL** — enter tasks one at a time:

```bash
python3 cli.py
```

Type any task at the `>` prompt, or `help`, `exit`, or `quit`.

---

## Command-Line Options

Both `app.py` and `cli.py` accept the same model flags:

```
--model MODEL              Default model for all agents (default: llama3.1:8b)
--planner-model MODEL      Override model for the planner
--coder-model MODEL        Override model for the coder (default: qwen3:8b)
--verifier-model MODEL     Override model for the verifier
--researcher-model MODEL   Override model for the researcher
--strict-verifier          Treat inconclusive verification as failure
--planner-timeout SECS     Planner LLM timeout in seconds (default: 60)
```

**Examples:**

```bash
# Use a single model for everything
python3 app.py "Write a Caesar cipher" --model mistral

# Mix models per role
python3 cli.py --model llama3.1:8b --coder-model qwen3:8b --strict-verifier

# One-shot with a longer timeout for complex tasks
python3 app.py "Refactor utils.py to add type hints" --planner-timeout 120
```

---

## How It Works

### 1. Planning

The **Planner** receives the goal and produces a JSON action plan containing:
- A list of *actions* (agent + tool + args)
- `success_criteria` — a non-empty list of concrete, testable conditions the verifier will evaluate

The planner never writes code itself. For new files it emits `code_file` actions; for surgical edits it emits `patch_file` actions; for larger rewrites it emits `modify_file` actions.

### 2. Execution

Actions are dispatched to the appropriate agent by the Router:

- The **Coder** intercepts `code_file` and `modify_file` actions and calls an LLM to generate actual source code, then writes it to the workspace.
- The **Researcher** handles read-only inspection (listing files, searching, reading, summarising).
- The **Verifier** handles tool calls that probe the result (`run_shell`, `file_exists`, `read_file`).

### 3. Verification

After all actions complete, the **Verifier** evaluates the result in layers:

1. **Deterministic checks** — every file reported as written must exist on disk.
2. **Interactive Python detection** — for GUI apps (pygame, tkinter, etc.), the file is compiled and inspected statically; no shell execution is attempted.
3. **Semantic verification** — the verifier LLM is given the goal, success criteria, and recent tool results and returns `passed`, `failed`, or `inconclusive`.
4. **Strict mode** — if `--strict-verifier` is set, `inconclusive` is treated as `failed`.

The verifier **prefers false negatives over false positives**: a successful `file_exists` or `run_shell` alone is never sufficient to mark the task as passed.

### 4. Repair Loop

On failure the orchestrator:

1. Fingerprints the failed action (`agent.tool(arg_keys...)`).
2. Creates a `RepairDiagnostic` carrying the failure reason, unmet criteria, and all prior failure fingerprints.
3. Re-invokes the planner in *repair mode*, which uses the diagnostic to avoid repeating the same pattern.

Retries default to 2 (configurable via `max_retries`).

---

## Workspace

All file operations are confined to the `workspace/` directory. Paths passed to tools must be **relative**; any attempt to escape the workspace raises a security error.

Allowed shell commands: `python`, `python3`, `pytest`, `ls`, `pwd`, `cat`, `echo`. Shell chaining, piping, and redirection are blocked.

---

## Available Tools

### File creation

| Tool | Agent | Description |
|---|---|---|
| `code_file` | coder | Generate a new code file from a natural-language specification (LLM-backed) |
| `write_file` | coder | Write a new non-code file (plain text, config, etc.) |

### File modification

| Tool | Agent | Description |
|---|---|---|
| `patch_file` | coder | **Primary edit tool.** Replaces an exact block of text (`old_lines`) with new content (`new_lines`). Fails loudly on context mismatch — no fuzzy matching. |
| `modify_file` | coder | Rewrite most of an existing file from a specification (LLM-backed). Use for large-scale refactors. |
| `replace_in_file` | coder | Last-resort literal substring replacement. Prefer `patch_file`. |

### Inspection

| Tool | Agent | Description |
|---|---|---|
| `read_file` | coder, verifier, researcher | Read full file content |
| `file_exists` | verifier, researcher | Check whether a file exists |
| `summarize_file` | researcher | Return size, structure, and Python symbols without reading the full file |
| `list_files` | researcher | List a directory's contents |
| `search_in_files` | researcher | Regex search across files matching a glob pattern |

### Execution

| Tool | Agent | Description |
|---|---|---|
| `run_shell` | verifier | Run an allowed shell command in the workspace |

---

## Memory

After each task RoboLoco saves a JSON summary to `memory/summaries/`. The planner loads the three most recent summaries and uses them as context when planning new tasks — helping it reuse working patterns and avoid repeating past mistakes.

Summaries include: goal, status, files touched, tools used, key outputs, model configuration, retry count, and duration.

---

## Project Structure

```
RoboLoco/
├── app.py                  # One-shot CLI entry point
├── cli.py                  # Interactive REPL entry point
├── agents/
│   ├── planner.py          # PlannerAgent — action planning & repair prompts
│   ├── coder.py            # CoderAgent — LLM-backed code generation
│   ├── verifier.py         # VerifierAgent — multi-layer verification
│   ├── researcher.py       # ResearcherAgent — read-only file inspection
│   └── interactive.py      # GUI/interactive app detection & compile check
├── orchestrator/
│   ├── main.py             # Orchestrator class — main execution loop
│   ├── router.py           # Agent dispatch by name
│   ├── state.py            # TaskState, AgentResult, ToolCall, ToolResult, ModelConfig
│   ├── repair.py           # FailedActionFingerprint & RepairDiagnostic
│   └── policies.py         # Workspace path, agent permissions, shell validation
├── tools/
│   ├── filesystem.py       # write_file, read_file, file_exists, summarize_file,
│   │                       #   list_files, replace_in_file, patch_file
│   ├── shell.py            # run_shell (sandboxed)
│   ├── coder_tools.py      # code_file, modify_file sentinels (LLM-intercepted)
│   └── search.py           # search_in_files
├── memory/
│   ├── summary_store.py    # Save & load task summaries
│   └── summaries/          # JSON logs of completed tasks
├── workspace/              # Sandboxed working directory for all file operations
└── tests/                  # Unit tests (128 tests, no Ollama required)
```

---

## Running Tests

```bash
python3 -m unittest discover tests/
```

Tests are self-contained and do not require Ollama — LLM calls are mocked where needed.

---

## Design Notes

**Planner/Coder boundary** — The planner decides *what* to build and expresses it as a specification. The coder decides *how* to implement it by generating code with an LLM. This keeps the plan human-readable and avoids the planner hallucinating code directly into action args.

**`patch_file` vs `write_file`** — `patch_file` is the primary tool for modifying existing files. It requires an exact context match and fails loudly on mismatch, preventing silent corruption. `write_file` and `code_file` are reserved for creating files that do not yet exist.

**Verifier conservatism** — The verifier never auto-passes based solely on tool success signals. A successful `run_shell` or confirmed `file_exists` is not proof the goal was met; an LLM semantic check is always required for a `passed` verdict.

**Repair fingerprinting** — Failures are identified by `agent.tool(arg_shape)` signatures. The planner is shown all prior failure signatures and warned when it is about to repeat a pattern, encouraging it to try a genuinely different approach rather than retrying the same action.
