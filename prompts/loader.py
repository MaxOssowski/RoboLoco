"""Prompt template loader and renderer.

Templates are stored as Markdown (``.md``) files inside this ``prompts/``
directory.  Each template is a plain-text file that contains ordinary prose
plus ``{{placeholder}}`` markers.

Placeholder syntax
------------------
Use ``{{name}}`` (double curly braces) to mark a substitution site.  Single
curly braces — used throughout the embedded JSON examples in the templates —
are treated as literals and are never touched by the renderer.

Usage
-----
::

    from prompts.loader import render_prompt

    prompt = render_prompt(
        "planner_plan.md",
        goal="Create a hello-world script.",
        rules=PlannerAgent._RULES,
        permissions=PlannerAgent._PERMISSIONS,
        schema=PlannerAgent._SCHEMA,
        memory_section="",
    )

Adding a new template
---------------------
1. Create ``prompts/<descriptive_name>.md`` with ``{{placeholder}}`` markers.
2. Call ``render_prompt("<descriptive_name>.md", placeholder=value, ...)`` from
   the agent code.
3. Add a test in ``tests/test_prompt_templates.py`` asserting that the
   critical content is present in the rendered output.

Error behaviour
---------------
* ``FileNotFoundError`` — template file does not exist.
* ``KeyError``          — a ``{{placeholder}}`` in the template has no
                          corresponding keyword argument.
Extra keyword arguments that do not appear in the template are silently
ignored, making it safe to pass a superset of values.
"""
from __future__ import annotations

import re
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent


def render_prompt(name: str, **kwargs: str) -> str:
    """Load a Markdown prompt template and substitute ``{{placeholder}}`` values.

    Parameters
    ----------
    name:
        Filename of the template (e.g. ``"planner_plan.md"``).  Resolved
        relative to the ``prompts/`` directory.
    **kwargs:
        Values to substitute into the template.  Every ``{{key}}`` present in
        the template must have a matching kwarg; extras are silently ignored.

    Returns
    -------
    str
        The fully rendered prompt string, stripped of leading/trailing
        whitespace.

    Raises
    ------
    FileNotFoundError
        If the template file does not exist.
    KeyError
        If the template references a ``{{placeholder}}`` with no matching kwarg.
    """
    path = _TEMPLATES_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    template = path.read_text(encoding="utf-8").strip()

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise KeyError(f"Missing template placeholder: {{{{{key}}}}}")
        return str(kwargs[key])

    return re.sub(r"\{\{(\w+)\}\}", _replace, template)
