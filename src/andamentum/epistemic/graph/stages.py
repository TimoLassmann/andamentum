"""Stage registry for the epistemic graph.

Each stage is a NAME for a (entry, exit) pair of graph nodes. The
registry below is the single source of truth — `andamentum-epistemic
stage <name>` looks up the entry/exit here and dispatches via
``run_epistemic_graph(start_at=..., stop_after=...)``.

The graph itself is unchanged. Stages are *labels* on subsets of the
graph, not a parallel control flow.

Each StageDef has a single ``exit_invariant`` predicate. When the
stage's exit node returns, the runner checks the invariant against
the live graph state + repository. A failing invariant is a loud
crash — it means the stage boundary isn't quiescent and the next
stage would inherit half-finished work.

See ``docs/superpowers/plans/2026-05-03-stage-runners.md`` (the plan)
and ``docs/superpowers/plans/2026-05-03-stage-runners-phase-0.md``
(the empirical boundary discovery).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .nodes import (
    AbandonOrDemote,
    CheckCompletion,
    ClusterEvidence,
    CombineClaimVerdicts,
    CreateClaims,
    Decompose,
    EnumerateCandidates,
    PlanEvidence,
    PrepareObjective,
    RunVerification,
    Scrutinize,
    Synthesize,
)

InvariantFn = Callable[[Any, Any], Awaitable[bool]]
"""Async signature: ``await fn(state, repo) -> bool``. False crashes
the run with ``StageInvariantError``."""


class StageInvariantError(RuntimeError):
    """Raised when a stage exits with its invariant unsatisfied. The
    failure mode is loud-by-design: a leaky boundary contaminates every
    downstream stage, so we crash here rather than silently passing
    half-finished state forward."""


@dataclass(frozen=True)
class StageDef:
    name: str
    entry: type
    exit_after: type
    exit_invariant: InvariantFn
    description: str


async def _objective(repo: Any, state: Any) -> Any:
    return await repo.get("objective", state.objective_id)


async def _claims(repo: Any) -> list[Any]:
    return await repo.query("claim")


async def _check_preplanning(state: Any, repo: Any) -> bool:
    obj = await _objective(repo, state)
    if obj is None or obj.question_type is None:
        return False
    # Verify mode (claim_to_verify set) skips Decompose entirely; the
    # preplanning exit is reached as soon as the question is classified.
    # Research mode requires a decomposition to have been produced (or
    # to fail loudly so the empty-decomposition fallback in CreateClaims
    # surfaces it) — at preplanning exit the decomposition must be set.
    if obj.claim_to_verify:
        return True
    decomp = obj.decomposition
    return decomp is not None and len(decomp.sub_investigations) >= 1


async def _check_initial_evidence(_state: Any, repo: Any) -> bool:
    claims = await _claims(repo)
    return bool(claims) and all(c.evidence_count > 0 for c in claims)


async def _all_active_claim_evidence_judged(state: Any, repo: Any) -> bool:
    """Every Evidence linked to a non-abandoned, non-cycle-capped claim
    with extracted content must have ``support_judgment`` set before
    the scrutiny-and-investigation stage exits.

    This is the explicit, enforced version of a contract that used to
    be implicit: downstream consumers (``compute_posterior``, the IBE
    chain, ``confidence.py``, the writer) all skip Evidence where
    ``support_judgment`` is None. If a code path adds Evidence without
    judging it, posteriors silently degenerate toward 0.5 and the
    discriminative metrics drop with no exception or test failure.

    The stage-exit invariant raises loudly at the boundary rather than
    letting that class of bug propagate. The fix is always the same:
    route the new Evidence through a judging step before the stage
    exits.
    """
    claims = await _claims(repo)
    active_ids = {c.entity_id for c in claims if not c.abandoned and not c.cycle_capped}
    all_evidence = await repo.query("evidence", objective_id=state.objective_id)
    for ev in all_evidence:
        if (
            ev.depends_on_claim_id in active_ids
            and ev.support_judgment is None
            and not ev.invalidated
            and ev.extracted_content
        ):
            return False
    return True


async def _check_scrutiny(state: Any, repo: Any) -> bool:
    claims = await _claims(repo)
    active = [c for c in claims if not c.abandoned and not c.cycle_capped]
    verdicts_settled = (
        all(c.scrutiny_verdict in {"pass", "fail"} for c in active)
        and len(state.claims_needing_rescrutiny) == 0
    )
    if not verdicts_settled:
        return False
    # Every claim-linked Evidence with content must have a verdict
    # before downstream stages run — see
    # ``_all_active_claim_evidence_judged`` for the contract this
    # enforces and why it's expressed as an invariant.
    return await _all_active_claim_evidence_judged(state, repo)


async def _check_verification(_state: Any, repo: Any) -> bool:
    claims = await _claims(repo)
    return all(c.verification_done or c.cycle_capped or c.abandoned for c in claims)


async def _check_integration(state: Any, repo: Any) -> bool:
    claims = await _claims(repo)
    obj = await _objective(repo, state)
    decomp_ok = (
        obj is None
        or obj.decomposition is None
        or (obj.decomposition.combined_verdict is not None)
    )
    active = [c for c in claims if not c.cycle_capped and not c.abandoned]
    return decomp_ok and all(c.integrated_assessment is not None for c in active)


async def _check_synthesis(state: Any, repo: Any) -> bool:
    obj = await _objective(repo, state)
    return obj is not None and obj.artefact_id is not None


STAGES: dict[str, StageDef] = {
    "preplanning": StageDef(
        name="preplanning",
        entry=PrepareObjective,
        exit_after=Decompose,
        exit_invariant=_check_preplanning,
        description="Clarify, classify, decompose into sub-investigations.",
    ),
    "initial_evidence": StageDef(
        name="initial_evidence",
        entry=PlanEvidence,
        exit_after=CreateClaims,
        exit_invariant=_check_initial_evidence,
        description="Plan first-pass searches; gather initial evidence; create claims.",
    ),
    "scrutiny_and_investigation": StageDef(
        name="scrutiny_and_investigation",
        entry=Scrutinize,
        exit_after=AbandonOrDemote,
        exit_invariant=_check_scrutiny,
        description=(
            "Iterative scrutiny ↔ investigation loop until each claim has a "
            "terminal scrutiny verdict (single stage; the loop's state is "
            "shared so any internal boundary is non-quiescent)."
        ),
    ),
    "verification": StageDef(
        name="verification",
        entry=ClusterEvidence,
        exit_after=RunVerification,
        exit_invariant=_check_verification,
        description="Run verification tracks (deductive, computational, "
        "convergence, ...) per supported claim.",
    ),
    "integration": StageDef(
        name="integration",
        entry=EnumerateCandidates,
        exit_after=CombineClaimVerdicts,
        exit_invariant=_check_integration,
        description="IBE chain per claim; cross-claim combination per the "
        "decomposition rule.",
    ),
    "synthesis": StageDef(
        name="synthesis",
        entry=CheckCompletion,
        exit_after=Synthesize,
        exit_invariant=_check_synthesis,
        description="Synthesis-demand gate, optional loop-back, final report.",
    ),
}


def stage_for_exit_node(node_class: type) -> StageDef | None:
    """Look up the StageDef whose ``exit_after`` matches ``node_class``.

    Used by the runner: when a stop_after kwarg matches a known stage
    boundary, the runner enforces that stage's exit invariant. When
    stop_after is a non-stage node (e.g., debugging mid-pipeline), no
    invariant fires.
    """
    for stage in STAGES.values():
        if stage.exit_after is node_class:
            return stage
    return None


def stage_names() -> list[str]:
    return list(STAGES.keys())


def get_stage(name: str) -> StageDef:
    if name not in STAGES:
        raise ValueError(f"Unknown stage {name!r}. Known stages: {sorted(STAGES)}.")
    return STAGES[name]
