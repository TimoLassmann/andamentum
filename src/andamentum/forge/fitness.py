"""Worker: the front fitness gate — is the brief realisable as a function? (dialect L9)

The design head run right after ``understand`` and before ``frame``. It judges the
brief's SHAPE — does an external driver own the control loop? — and never its
vocabulary. forge builds *functions* (rung 1 stateless, rung 2 stateful-with-memory);
an app / agent / service hands the loop to an outside driver and is refused at the door
with a concrete reshape (fail loud, never fake).

Three pure/engine-free surfaces:

- ``assess_fitness(why, *, sink) -> Fitness`` — the one small LLM call (FITNESS).
- ``is_buildable(fitness) -> bool`` — the deterministic proceed/refuse predicate.
- ``refusal_message(fitness) -> str`` — the fail-loud text (reason + reshape).

Leaf worker (dialect Law 2): pydantic + sibling agents/schemas only; no graph engine.
"""

from __future__ import annotations

from .agents import FITNESS, AgentSink
from .schemas import Fitness, ForgeWhy

# The rungs forge can build: a stateless function (rung 1), and — now that the durable
# store is provisioned and wired (C-STORE-PRD.md §10) — a stateful function (rung 2: the
# run entry loads its record at the start and saves it at the end). Apps / agents /
# services (an external driver owns the loop) stay out of dialect (L9) and are refused.
BUILDABLE_RUNGS: frozenset[str] = frozenset({"function", "stateful_function"})


async def assess_fitness(why: ForgeWhy, *, sink: AgentSink) -> Fitness:
    """One FITNESS call over the restated problem. Returns the typed verdict; the graph
    node decides proceed-or-fail from it (flow control is the graph's job, dialect L6)."""
    out = await sink.run(
        FITNESS,
        purpose=why.purpose,
        boundary_in=why.boundary_in,
        boundary_out=why.boundary_out,
    )
    assert isinstance(out, Fitness)
    return out


def is_buildable(fitness: Fitness) -> bool:
    """Deterministic, operator-trusted predicate: does this rung proceed to design?
    Keys on the single ``rung`` axis (one decision axis; one code path)."""
    return fitness.rung in BUILDABLE_RUNGS


def refusal_message(fitness: Fitness) -> str:
    """The fail-loud message for a non-buildable brief — names the reason and hands back
    the concrete reshape so the user can resubmit (fail loud, never fake). Reached only for
    an app / agent / service (a stateful function is buildable; see BUILDABLE_RUNGS)."""
    return (
        f"forge builds functions (one input, one output, one run — stateless, or stateful "
        f"with a durable store); this brief is a(n) {fitness.rung}, which hands the control "
        "loop to something outside the system and is out of forge's scope.\n"
        f"Why: {fitness.reason}\n"
        f"Reshape and resubmit: {fitness.suggested_reshape}"
    )


__all__ = ["assess_fitness", "is_buildable", "refusal_message", "BUILDABLE_RUNGS"]
