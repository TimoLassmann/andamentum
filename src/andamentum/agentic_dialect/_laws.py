"""The structured truth of the dialect: the laws, as data.

This is the authoritative spelling of each law's id, name, one-sentence
statement, enforcement tier, and pre-commit checklist items. The prose canon
(rationale, examples) lives in ``DIALECT.md``; ``tests/test_agentic_dialect_drift.py``
binds the two so neither can drift from the other.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Tier = Literal["type-check", "lint", "test", "review-only"]


class Law(BaseModel):
    """One law of the dialect."""

    id: str
    name: str
    statement: str
    tier: Tier
    checklist: tuple[str, ...] = ()


LAWS: tuple[Law, ...] = (
    Law(
        id="L1",
        name="Surface placement",
        statement=(
            "Given → Deps. Produced and read widely → State. Produced for the next "
            "step → Inputs."
        ),
        tier="review-only",
    ),
    Law(
        id="L2",
        name="Thin orchestrator, fat worker",
        statement=(
            "Orchestrators route and dispatch; workers compute. A worker never "
            "imports the graph engine."
        ),
        tier="lint",
        checklist=(
            "Delete every pydantic_graph import from worker/verb/schema files — they still compile.",
            "grep workers for ctx / state / deps parameters → zero hits.",
        ),
    ),
    Law(
        id="L3",
        name="State is written only by orchestrators",
        statement="Workers return values; the orchestrator assigns them.",
        tier="review-only",
    ),
    Law(
        id="L4",
        name="Routing is static, declarative, and deterministic",
        statement=(
            "Every successor is known from the code; you branch among declared "
            "edges, never synthesize one."
        ),
        tier="test",
        checklist=(
            "No datetime / random / unordered iteration in any run().",
            "Every step's successors appear in its return type.",
        ),
    ),
    Law(
        id="L5",
        name="Every loop, recursion, and fan-out is bounded",
        statement=(
            "The bound — iteration count or fan-out width — traces to a Deps value "
            "or a named constant."
        ),
        tier="lint",
        checklist=(
            "Every loop and fan-out bound is a Deps field or SCREAMING_SNAKE constant — no literals.",
        ),
    ),
    Law(
        id="L6",
        name="The model is a component, not the controller",
        statement=(
            "Flow lives in the graph. An agent answers a question; it does not drive "
            "the pipeline."
        ),
        tier="review-only",
    ),
    Law(
        id="L7",
        name="Typed boundaries; fail loud",
        statement=(
            "A structured schema on every edge — no untyped dict[str, Any]. A missing "
            "service raises."
        ),
        tier="type-check",
        checklist=("Result is a Pydantic model returned via End.",),
    ),
    Law(
        id="L8",
        name="Effects are idempotent",
        statement=(
            "A worker that changes the world must be safe to run more than once — "
            "loops, retries, and resume all re-enter it."
        ),
        tier="review-only",
    ),
)

_BY_ID = {law_.id: law_ for law_ in LAWS}


def laws() -> tuple[Law, ...]:
    """All laws, in order."""
    return LAWS


def law(law_id: str) -> Law:
    """One law by id (e.g. ``"L4"``). Raises ``KeyError`` if unknown."""
    key = law_id.upper()
    if key not in _BY_ID:
        raise KeyError(f"Unknown law id: {law_id!r}. Known: {', '.join(_BY_ID)}")
    return _BY_ID[key]


def checklist() -> tuple[tuple[str, str], ...]:
    """The greppable pre-commit checklist as ``(item, law-id)`` pairs."""
    return tuple((item, law_.id) for law_ in LAWS for item in law_.checklist)
