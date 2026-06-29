"""Drive forge over a benchmark case and score the runs.

``run_case`` runs forge ``runs`` times over one brief and aggregates the outcomes into a
:class:`CaseScore`. Each run is **Tier 1** — design-only: ``run_forge(brief, model=...,
dest=None)`` runs understand→frame→decompose→compile→review and returns a ``ForgeResult``
whose ``.spec`` is inspected for control-flow features. A ``ValueError`` is forge refusing
at the fitness gate (or failing the coherence loop) — for the benchmark's purposes a
refusal; any other exception is a genuine design failure.

An optional ``sink`` injects a stub ``AgentSink`` so the whole path runs with no model —
that is how the offline self-tests drive it.

Tier 2 (``full=True``) is a documented hook for a later end-to-end build + sandbox audit;
this pass does not implement it and falls back to Tier-1 behaviour.
"""

from __future__ import annotations

from andamentum.forge import run_forge
from andamentum.forge.agents import AgentSink

from .shape import detect_features, outcome_matches
from .types import Case, CaseScore, RunOutcome


async def run_case(
    case: Case,
    *,
    model: str,
    runs: int = 3,
    full: bool = False,
    sink: AgentSink | None = None,
) -> CaseScore:
    """Run forge ``runs`` times over ``case`` and score the outcomes.

    Each run is Tier-1 design-only (``dest=None``): success → ``built`` with the spec's
    detected features; ``ValueError`` → ``refused``; any other exception → ``design_failed``.
    Passing ``sink`` drives forge with no model (the offline self-test path).

    ``full=True`` is the Tier-2 end-to-end hook — not yet implemented; for now it behaves
    exactly like Tier 1. TODO(tier-2): render + agent-author + sandbox-audit the package and
    score on ``works`` instead of design shape.
    """
    outcomes: list[RunOutcome] = []
    for _ in range(runs):
        outcomes.append(await _run_once(case, model=model, sink=sink, full=full))

    passes = sum(1 for o in outcomes if outcome_matches(case, o))
    total = len(outcomes)
    return CaseScore(
        case=case,
        runs=outcomes,
        passes=passes,
        total=total,
        pass_rate=(passes / total if total else 0.0),
    )


async def _run_once(
    case: Case,
    *,
    model: str,
    sink: AgentSink | None,
    full: bool,
) -> RunOutcome:
    """One forge run over a brief, mapped to a :class:`RunOutcome`."""
    # full=True is the Tier-2 hook; this pass runs Tier 1 regardless (see run_case docstring).
    _ = full
    try:
        result = await run_forge(case.brief, model=model, dest=None, sink=sink)
    except ValueError as exc:
        # forge refusing at the fitness gate, or failing the coherence loop — a refusal.
        return RunOutcome(kind="refused", error=str(exc))
    except Exception as exc:  # genuine design failure (a crash, not a refusal)
        return RunOutcome(kind="design_failed", error=str(exc))
    return RunOutcome(kind="built", features=detect_features(result.spec))


async def run_all(
    cases: list[Case],
    *,
    model: str,
    runs: int,
    full: bool,
    sink: AgentSink | None = None,
) -> list[CaseScore]:
    """Score every case in ``cases`` (sequentially, to keep model load bounded)."""
    scores: list[CaseScore] = []
    for case in cases:
        scores.append(
            await run_case(case, model=model, runs=runs, full=full, sink=sink)
        )
    return scores
