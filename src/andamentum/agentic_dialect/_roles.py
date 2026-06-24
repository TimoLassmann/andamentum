"""Role → law-slice map: the prompt-fragment a given job needs.

Kept tight so a small model does not lose the law in a wall of text. A role is
the kind of step an agent is writing; ``for_role`` returns its job line, the
laws in its slice, and the checklist items for those laws.
"""

from __future__ import annotations

from ._laws import law

ROLES: dict[str, tuple[str, ...]] = {
    "worker": ("L2", "L7", "L8"),
    "orchestrator": ("L2", "L3", "L4", "L6"),
    "state": ("L1", "L3"),
    "agent": ("L6", "L7"),
    "entry": ("L1",),
    "reviewer": ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"),
}

_JOB: dict[str, str] = {
    "worker": "You are writing a worker — one unit of real work, framework-free.",
    "orchestrator": "You are writing an orchestrator step — thin: route and dispatch only.",
    "state": "You are defining State and Deps — the data surfaces.",
    "agent": "You are wiring an agent — a model call as data.",
    "entry": "You are writing the entry point that builds State + Deps and runs the graph.",
    "reviewer": "You are reviewing an agentic system against the whole dialect.",
}


def roles() -> tuple[str, ...]:
    """The briefable role names."""
    return tuple(ROLES)


def for_role(role: str) -> str:
    """The prompt-slice a job needs: its job line, its laws, its checklist.

    Raises ``KeyError`` for an unknown role.
    """
    key = role.lower()
    if key not in ROLES:
        raise KeyError(f"Unknown role: {role!r}. Known: {', '.join(ROLES)}")
    selected = [law(i) for i in ROLES[key]]
    lines = [_JOB[key], "", "Laws that apply:"]
    for lw in selected:
        lines.append(f"- {lw.id} — {lw.name}: {lw.statement}")
    checks = [(item, lw.id) for lw in selected for item in lw.checklist]
    if checks:
        lines.extend(["", "Before you commit:"])
        for item, lid in checks:
            lines.append(f"- {item} ({lid})")
    return "\n".join(lines)
