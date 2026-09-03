"""Epistemic pipeline graph nodes.

Each node wraps one or more existing operations and returns the next
node to run.  The graph replaces the pattern-based scheduler with
explicit, typed transitions.

Architecture: Layer 2 (pydantic-graph, depends on operations + entities)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Union

from pydantic_graph import Decision, End, Graph, GraphBuilder, GraphRunContext

from .base import Node
from .invariants import capped_claims_carry_a_cap_evaluation, no_stranded_claims
from .state import EpistemicGraphState
from ..entities.verdict import EpistemicVerdict
from .deps import EpistemicDeps
from .result import EpistemicResult
from ..thresholds import (
    ADVERSARIAL_SURVIVED_THRESHOLD,
    JUDGMENT_DOUBT_ESCALATION_THRESHOLD,
    PEIRCE_CYCLE_CAP,
    POSTERIOR_DECISIVE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


# Retrieval health is evaluated at end-of-gathering, not per-extraction
# (no transient counter, no per-extraction threshold). See
# ``_evaluate_retrieval_health`` below.

# Cap on Scrutinize ↔ ResolveUncertainties oscillation per claim.
# Each pass of Scrutinize on a marked-for-rescrutiny claim counts as
# one cycle. Once a claim hits this many cycles, Scrutinize stops
# rescrutinizing and ResolveUncertainties stops requesting rescrutiny.
# The claim retains its current verdict.
#
# Aliases ``PEIRCE_CYCLE_CAP`` for backward compatibility — every
# Peirce-grounded loop in the graph (investigation attempts,
# scrutiny↔resolve, uncertainty resolution depth) shares the same
# canonical value defined in ``epistemic.thresholds``. See that
# module for the philosophical basis.
SCRUTINY_RESOLVE_CYCLE_CAP = PEIRCE_CYCLE_CAP


def _loopback_eligible(claim: Any, state: EpistemicGraphState) -> bool:
    """Could another round of inquiry on this claim plausibly change anything?

    True when the claim is neither cycle-capped nor already at
    ``SCRUTINY_RESOLVE_CYCLE_CAP``. (Abandonment is the caller's filter —
    both call sites work from an already-active claim list.)

    ONE predicate, two readers: ``CheckSynthesisDemand._maybe_loop_back``
    (which claims to add to the rescrutiny set) and
    ``CheckSynthesisDemand._judgment_doubt_demand`` (whether high judgment
    doubt is worth escalating on at all). They MUST agree: the doubt gate
    routes through ``_maybe_loop_back``, and a doubt gate that fired on a
    claim the loop-back considers ineligible would hand ``_maybe_loop_back``
    an empty eligible set — the exact shape that routes to
    ``SynthesizeInsufficient``. Sharing the predicate makes that
    structurally impossible rather than conventionally avoided.
    """
    if getattr(claim, "cycle_capped", False):
        return False
    cycles = state.scrutiny_resolve_cycles.get(claim.entity_id, 0)
    return cycles < SCRUTINY_RESOLVE_CYCLE_CAP


async def _mark_cycle_capped(
    deps: "EpistemicDeps",
    state: EpistemicGraphState,
    *,
    claim_id: str,
    source: str,
) -> None:
    """Mark a claim as cycle-capped, snapshot its blocking uncertainties,
    and record the terminal verdict its evidence actually supports.

    The mark-and-snapshot half is idempotent: ``Claim.cycle_capped`` is
    set True (no-op if already True) and ``persistent_concerns`` is
    captured *only on first cap firing* — the first-firing snapshot is
    the diagnostic input for deciding follow-up architecture
    (cluster-dedup vs reformulation). Subsequent cap firings on the same
    claim leave the snapshot alone.

    The conclude half runs on the **first** firing, and on a later
    firing only when the claim's evidence fingerprint
    (``gates.claim_evidence_fingerprint``) OR its resolution-bearing
    fingerprint (``gates.claim_resolution_bearing_fingerprint``) differs
    from the one recorded at the last conclusion (``Claim.cap_fingerprint``
    / ``Claim.cap_bearing_fingerprint``): ``ConcludeAtCycleCapOperation``
    re-reads the claim's eligible evidence deterministically (no LLM) and
    revises its own record. That is what stops a capped claim from
    reporting a false "insufficient" when the evidence it already
    gathered was decisive — a cap stops further LLM work; it must not
    zero out signal already acquired. A repeat firing on unchanged
    evidence AND unchanged bearings would just re-derive the same
    reading, so it is skipped entirely (certify-before-contract D7): the
    cap is terminal, and a terminal claim consumes no further work of
    any kind, mechanical or LLM.

    The bearing half of that comparison is self-feeding-loop plan D3: a
    RESOLVED, BLOCKING uncertainty's ``resolution_bearing`` can arrive
    (or change) after the evidence set has stopped moving — a sibling
    claim sharing the uncertainty can keep it live well past this
    claim's first cap firing — so gating the skip on evidence identity
    alone would leave a cap-owned verdict standing against a resolution
    that has since spoken against it. Re-running
    ``ConcludeAtCycleCapOperation`` on a bearing change costs no LLM
    call (D6): the operation's own veto (see its docstring) is what
    actually clears the stale verdict.

    The cap fires from two sites — ResolveUncertainties (primary
    loop-breaker, blocks the rescrutiny add) and Scrutinize (defense
    in depth, discards a stale rescrutiny flag). Both call this
    helper so the consequence is uniform, which is also what makes the
    ``capped_claims_carry_a_cap_evaluation`` invariant checkable.

    ``claim_id`` and ``source`` are keyword-only: two same-typed strings
    side by side are exactly the transposition the repo's
    confusable-parameter rule exists to catch.
    """
    from ..entities import Claim, Uncertainty
    from ..gates import claim_evidence_fingerprint, claim_resolution_bearing_fingerprint
    from ..operations.stage_management import ConcludeAtCycleCapOperation

    claim = await deps.repo.get("claim", claim_id)
    if not isinstance(claim, Claim):
        return
    first_firing = not claim.cycle_capped
    if first_firing:
        claim.cycle_capped = True
        # Snapshot the currently-blocking uncertainties on this claim, if any.
        # Used post-hoc to decide whether the cycle was driven by persistent
        # concern variants (→ cluster-dedup is the right follow-up) or by
        # genuinely orthogonal concerns (→ claim reformulation is the right
        # follow-up). Empty snapshot is fine — it just means there were no
        # active blocking uncertainties at cap-firing time, which is itself
        # a diagnostic observation.
        snapshot: list[str] = []
        for uid in claim.uncertainty_ids:
            try:
                unc = await deps.repo.get("uncertainty", uid)
            except Exception:
                continue
            if (
                isinstance(unc, Uncertainty)
                and unc.is_blocking
                and unc.resolution is None
            ):
                snapshot.append(unc.entity_id)
        claim.persistent_concerns = snapshot
        await deps.repo.save(claim)
        logger.info(
            "Claim %s cycle-capped via %s (snapshot=%d concerns)",
            claim_id[:12],
            source,
            len(snapshot),
        )

    fingerprint = await claim_evidence_fingerprint(claim, deps.repo)
    bearing_fingerprint = await claim_resolution_bearing_fingerprint(claim, deps.repo)
    if (
        not first_firing
        and claim.cap_fingerprint == fingerprint
        and claim.cap_bearing_fingerprint == bearing_fingerprint
    ):
        logger.info(
            "Claim %s cycle cap re-fired on unchanged evidence and "
            "unchanged resolution bearings (both fingerprints match the "
            "last conclusion); skipping the mechanical re-conclusion",
            claim_id[:12],
        )
        return

    await _run_op(
        ConcludeAtCycleCapOperation,
        deps,
        state,
        claim_id,
        "claim",
        "conclude_at_cycle_cap",
    )

    # Reload: ConcludeAtCycleCapOperation loaded and saved its own copy of
    # the claim, so the local ``claim`` object is stale on the fields it
    # wrote. Stamp both fingerprints onto the current record rather than
    # the stale one, or this save would clobber the operation's own writes.
    claim = await deps.repo.get("claim", claim_id)
    if isinstance(claim, Claim):
        claim.cap_fingerprint = fingerprint
        claim.cap_bearing_fingerprint = bearing_fingerprint
        await deps.repo.save(claim)


async def _uncertainty_is_terminal_only(unc: Any, repo: Any) -> bool:
    """True when every claim ``unc`` affects is cycle-capped or abandoned.

    Certify-before-contract D7: a cycle-capped or abandoned claim is
    terminal for this run — it consumes no further LLM work of any
    kind. An uncertainty whose every affected claim is already terminal
    has no live claim left to resolve anything FOR, so resolving it
    would spend an LLM call to inform claims that can no longer act on
    it, and it must not count as still-blocking for routing either —
    otherwise the graph loops forever between ``ResolveUncertainties``
    and ``Scrutinize`` re-dispatching a terminal-only uncertainty with
    no live claim ever advancing. An uncertainty with an empty
    ``affected_claim_ids`` returns False (nothing to check against,
    treated exactly as before D7).
    """
    from ..entities import Claim

    if not unc.affected_claim_ids:
        return False
    affected_claims = [await repo.get("claim", cid) for cid in unc.affected_claim_ids]
    return all(
        isinstance(c, Claim) and (c.cycle_capped or c.abandoned) for c in affected_claims
    )


async def _live_blocking(uncertainties: list[Any], repo: Any) -> list[Any]:
    """Blocking uncertainties with at least one non-terminal affected claim.

    Certify-before-contract D7: an uncertainty whose every affected claim
    is cycle-capped or abandoned is skipped — logged at info, no
    ``resolve_uncertainty`` operation row — rather than routed for
    resolution. This is the single site that both filters and logs that
    skip, so the two call sites in ``ResolveUncertainties.run`` (the
    initial gather and the post-resolution re-check) cannot drift apart.
    """
    live = []
    for u in uncertainties:
        if not u.is_blocking:
            continue
        if await _uncertainty_is_terminal_only(u, repo):
            logger.info(
                "Uncertainty %s skipped: all %d affected claim(s) "
                "are cycle-capped or abandoned",
                u.entity_id[:12],
                len(u.affected_claim_ids),
            )
            continue
        live.append(u)
    return live


async def _evaluate_retrieval_health(
    state: EpistemicGraphState, deps: EpistemicDeps
) -> None:
    """Set ``state.retrieval_failed`` based on actual evidence yield.

    Retrieval has failed when **zero** evidence pieces with content
    exist for the objective at end-of-gathering. Off-topic providers
    returning empty is expected and routine; transient mid-run empty
    clusters are not failures if other providers found content.

    The previous design used a per-extraction "consecutive empties"
    counter that flipped a sticky flag once 3 empties accrued in a
    row. That metric was order-dependent (which providers happened to
    run consecutively), conflated "off-topic provider returned []"
    with "real retrieval failure", and got worse as the system added
    providers (more chances of legitimate empties bunching). The end-
    of-gathering check is invariant under provider count and matches
    the natural-language meaning of "retrieval failed": we attempted
    to gather data and got nothing.

    Non-sticky: if a later extraction sweep produces non-empty
    evidence, the flag flips back to False, because by then the
    objective has data and retrieval has not failed.
    """
    all_evidence = await deps.repo.query("evidence", objective_id=state.objective_id)
    has_any_content = any(
        bool(getattr(ev, "extracted_content", "")) for ev in all_evidence
    )
    state.retrieval_failed = not has_any_content


def _make_op(op_class: type, deps: EpistemicDeps) -> Any:
    """Create an operation instance from graph deps."""
    return op_class(
        repo=deps.repo,
        agent_runner=deps.agent_runner,
        evidence_gatherer=deps.evidence_gatherer,
        quality_scorer=deps.quality_scorer,
        embedding_model=deps.embedding_model,
        judge_model=deps.judge_model,
    )


def _op_input(
    entity_id: str,
    entity_type: str,
    operation: str,
    metadata: dict[str, Any] | None = None,
    *,
    verdict: "EpistemicVerdict | None" = None,
) -> Any:
    """Create an OperationInput for operation execution."""
    from ..operations.base import OperationInput

    return OperationInput(
        entity_id=entity_id,
        entity_type=entity_type,
        operation=operation,
        metadata=metadata or {},
        verdict=verdict,
    )


@dataclass
class _OpOutcome:
    """The result of executing one operation, before any State write.

    Produced by ``_execute_op`` (which touches no State — safe to run
    inside a parallel path) and consumed by ``_commit_op`` (which does
    the State writes — the sole write site, run in the deterministic
    join). ``exception`` is set when the op raised, so ``_commit_op``
    can record the quarantine.
    """

    entity_id: str
    entity_type: str
    operation: str
    result: Any  # OperationResult
    exception: BaseException | None = None


async def _execute_op(
    op_class: type,
    deps: EpistemicDeps,
    entity_id: str,
    entity_type: str,
    operation: str,
    metadata: dict[str, Any] | None = None,
    *,
    verdict: "EpistemicVerdict | None" = None,
) -> _OpOutcome:
    """Execute one operation and return its outcome — no State writes.

    Pure w.r.t. graph State: it runs the op (an LLM/DB effect via Deps
    Ports) and captures the result or the exception into an ``_OpOutcome``.
    Because it never touches State, it is safe to fan out under
    ``asyncio.gather``; the caller commits the outcomes in a deterministic
    join (Law 3 — the join is the sole State-write site during parallelism).
    """
    from ..operations.base import OperationResult

    op = _make_op(op_class, deps)
    work = _op_input(entity_id, entity_type, operation, metadata, verdict=verdict)
    try:
        result = await op.execute(work)
    except Exception as e:
        logger.warning(
            "%s on %s raised %s: %s — quarantining entity",
            operation,
            entity_id[:12],
            type(e).__name__,
            e,
        )
        result = OperationResult(
            success=False,
            entity_id=entity_id,
            message=f"{operation} quarantined: {type(e).__name__}: {e}",
        )
        return _OpOutcome(entity_id, entity_type, operation, result, exception=e)
    return _OpOutcome(entity_id, entity_type, operation, result)


async def _commit_op(
    deps: EpistemicDeps,
    state: EpistemicGraphState,
    outcome: _OpOutcome,
) -> Any:
    """Write one operation outcome into State + the trace — the sole writer.

    Records the quarantine (if the op raised), logs the operation, persists
    the execution-step trace, and fires the progress callback. Sequential by
    construction: called from a node body or from a deterministic reduce over
    a fan-out, never from inside a parallel path.
    """
    entity_id = outcome.entity_id
    entity_type = outcome.entity_type
    operation = outcome.operation
    result = outcome.result
    if outcome.exception is not None:
        state.quarantine(entity_id, entity_type, operation, outcome.exception)
    state.log_operation(operation, entity_id, result.success, result.message)
    # Persist execution trace to the database.
    # Uses DocumentStore.add() (not register_document, which deduplicates
    # by content hash and would collapse same-message steps).
    backend = getattr(deps.repo, "store", None)
    if backend is not None:
        step_number = len(state.operations_log)
        # Disambiguate file_path across concurrent graph runs sharing a
        # DocumentStore: the decomposed orchestrator runs N children
        # against one DB, and re-runs of the same objective also share
        # the DB. ``state.run_id`` is unique per graph run; without it,
        # two children both write "execution_step_1" and crash on the
        # documents.file_path UNIQUE index. step_number stays in
        # metadata so execution-step queries remain orderable.
        await backend.add(
            file_path=(f"execution_step_{state.run_id}_{step_number:04d}"),
            content=result.message or "",
            title=f"{operation} on {entity_id[:12]}",
            metadata={
                "epistemic_type": "execution_step",
                "step_number": step_number,
                "run_id": state.run_id,
                "operation": operation,
                "entity_id": entity_id,
                "entity_type": entity_type,
                "success": result.success,
                "message": result.message or "",
                "created_entities": getattr(result, "created_entities", []),
            },
        )
    if deps.progress_callback:
        extras = {"created_entities": getattr(result, "created_entities", [])}
        if not result.success:
            extras["validation_errors"] = getattr(result, "validation_errors", [])
        deps.progress_callback(
            operation, entity_id, result.success, result.message, extras
        )
    return result


async def _run_op(
    op_class: type,
    deps: EpistemicDeps,
    state: EpistemicGraphState,
    entity_id: str,
    entity_type: str,
    operation: str,
    metadata: dict[str, Any] | None = None,
    *,
    verdict: "EpistemicVerdict | None" = None,
) -> Any:
    """Execute one operation and commit its outcome to State — sequential.

    The default single-entity path: ``_execute_op`` then ``_commit_op`` back
    to back. For a parallel fan-out, call ``_execute_op`` inside the gather
    and ``_commit_op`` in the deterministic reduce so the join stays the
    sole State-write site (Law 3).

    If the operation raises, the quarantine is recorded and a failed
    OperationResult returned — never swallowed silently. Downstream nodes
    call state.is_quarantined(entity_id) before scheduling further work.
    """
    outcome = await _execute_op(
        op_class, deps, entity_id, entity_type, operation, metadata, verdict=verdict
    )
    return await _commit_op(deps, state, outcome)


async def _run_tms_sweep(deps: EpistemicDeps, state: EpistemicGraphState) -> None:
    """Run TMS (Truth Maintenance System) sweep: cascade evidence invalidation.

    This is a reactive correctness check, not a graph node. It runs after
    operations that can invalidate evidence (adversarial search, investigation,
    evidence extraction).

    Certify-before-contract (docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md
    D1): this sweep used to also revalidate — and, on a failed gate,
    contract — every promoted claim (former Steps 2 and 3) immediately
    after any cascade, racing that contraction against IBE for the same
    claim in the same pass. That revalidate/contract check now runs
    exactly once, in ``PromoteSupported``, after ``SelectBestExplanation``
    has already produced a verdict — so a claim is certified before it is
    ever contracted. This sweep is left with only cascade: mark evidence
    invalidated by an upstream operation as such on every claim that
    references it.

    No LLM calls — purely structural graph maintenance.
    """
    from ..entities import Evidence, Claim
    from ..operations.belief_maintenance import InvalidateEvidenceOperation

    # Step 1: Cascade invalidated evidence → affected claims
    #
    # Performance fix (TMS-storm — was 21 ops per ~3-evidence run):
    # ``InvalidateEvidenceOperation`` does real DB work even when an
    # invalidated evidence has no downstream effect (zero claims
    # reference it, no derived evidence depends on it). Most invalidations
    # come from Layer-1 dedup of duplicate URLs that never made it into
    # any claim's evidence_ids — true no-op cascades. Pre-filter those
    # and just flip ``invalidation_cascaded=True`` directly, skipping the
    # full op + execution_step trace per item. This was a 7×-amplification
    # in a 3-evidence run.
    all_evidence = await deps.repo.query("evidence", objective_id=state.objective_id)
    pending: list[Evidence] = []
    for ev in all_evidence:
        if isinstance(ev, Evidence) and ev.invalidated and not ev.invalidation_cascaded:
            pending.append(ev)

    referenced_ids: set[str] = set()
    if pending:
        all_claims_pre = await deps.repo.query("claim", objective_id=state.objective_id)
        # Build a fast lookup: which evidence IDs are referenced by any claim?
        for c in all_claims_pre:
            if isinstance(c, Claim):
                for eid in c.evidence_ids:
                    referenced_ids.add(eid)

    for ev in pending:
        # No-op cascade: nothing references this evidence and no derived
        # evidence depends on it. Mark cascaded directly.
        no_claim_ref = ev.entity_id not in referenced_ids
        no_derived = ev.depends_on_claim_id is None
        if no_claim_ref and no_derived:
            ev.invalidation_cascaded = True
            await deps.repo.save(ev)
            continue
        await _run_op(
            InvalidateEvidenceOperation,
            deps,
            state,
            ev.entity_id,
            "evidence",
            "invalidate_evidence",
        )


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PrepareObjective(Node):
    """Run clarification, classification, and conceptual analysis on the objective.

    Phase 5 of the Move-3 plan. The graph entry point.
    """

    reads = frozenset({"objective_id", "skip_preplanning"})
    writes = frozenset({"question_type"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Decompose":
        from ..operations.preplanning import (
            ClarifyQuestionOperation,
            ClassifyQuestionOperation,
            ConceptualAnalysisOperation,
        )

        state = ctx.state
        deps = ctx.deps
        oid = state.objective_id

        if not state.skip_preplanning:
            # 1. Clarify
            await _run_op(
                ClarifyQuestionOperation,
                deps,
                state,
                oid,
                "objective",
                "clarify_question",
            )
            obj = await deps.repo.get("objective", oid)
            obj.phase = "clarified"
            await deps.repo.save(obj)

            # 2. Classify
            await _run_op(
                ClassifyQuestionOperation,
                deps,
                state,
                oid,
                "objective",
                "classify_question",
            )
            obj = await deps.repo.get("objective", oid)
            if obj.question_type:
                state.question_type = obj.question_type

            # 3. Conceptual analysis
            await _run_op(
                ConceptualAnalysisOperation,
                deps,
                state,
                oid,
                "objective",
                "conceptual_analysis",
            )
            obj = await deps.repo.get("objective", oid)
            obj.phase = "analyzed"
            await deps.repo.save(obj)
        else:
            # Even when skipping, load question_type if already set
            obj = await deps.repo.get("objective", oid)
            if obj.question_type:
                state.question_type = obj.question_type

        return Decompose()


@dataclass
class Decompose(Node):
    """Run DecomposeQuestionOperation before PlanEvidence in research mode.

    Skipped only when ``obj.claim_to_verify`` is set (verify mode — the
    user named the exact claim, no decomposition needed). Otherwise runs
    ``DecomposeQuestionOperation`` to populate ``objective.decomposition``
    so:

    * ``DispatchGatherOperation`` dispatches queries per
      sub-investigation (per-claim evidence pool, Option-2).
    * ``CreateClaims`` routes to ``MultiSeedClaimOperation`` (one Claim
      per sub-investigation, sub_investigation_id-tagged).
    * ``CombineClaimVerdicts`` applies the decomposition's
      ``combination_rule`` (AND / OR / WEIGHTED_AND / UNION) over the
      per-Claim integration verdicts.
    * ``compute_posterior`` honours the rule via the rule-aware
      delegation path (see confidence.py).

    When the decomposer produces no usable sub-investigations, the
    fallback in ``CreateClaims`` (graph/nodes.py:635-649) routes to
    ``ProposeClaims`` — the open-research path emerges naturally without
    a separate flag.

    Idempotent: ``DecomposeQuestionOperation`` short-circuits if
    ``objective.decomposition`` is already set.

    Phase 5 of the Move-3 plan; gate flipped 2026-05-04 for the
    entry-point consolidation (see
    docs/superpowers/plans/2026-05-04-entry-point-consolidation.md).
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PlanEvidence":
        state = ctx.state
        deps = ctx.deps

        obj = await deps.repo.get("objective", state.objective_id)
        if obj.claim_to_verify:
            return PlanEvidence()

        from ..operations.preplanning import DecomposeQuestionOperation

        await _run_op(
            DecomposeQuestionOperation,
            deps,
            state,
            state.objective_id,
            "objective",
            "decompose_question",
        )
        return PlanEvidence()


@dataclass
class PlanEvidence(Node):
    """Plan and execute evidence gathering via description-driven dispatch.

    Runs ``DispatchGatherOperation``: one dispatch-agent call per
    (provider, sub-claim), each combining route + query into a single
    decision. Providers that the agent decides to abstain on never have
    their HTTP-call layer reached. Returned ``GatheredEvidence`` items
    are persisted as Evidence entities with content + quality scoring
    already attached, so the downstream ``ExtractEvidence`` node sees
    no ``extracted=False`` stubs and proceeds straight to the
    retrieval-health check.

    The description-driven dispatch path is the only evidence-gathering
    implementation; the prior multi-agent chain (provider selection,
    ranking, and per-provider query formulation as separate agents) was
    decommissioned in favour of the unified single-agent design.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ExtractEvidence":
        state = ctx.state
        deps = ctx.deps

        await self._run_dispatch(state, deps)

        obj = await deps.repo.get("objective", state.objective_id)
        obj.phase = "planned"
        await deps.repo.save(obj)

        return ExtractEvidence()

    async def _run_dispatch(
        self, state: EpistemicGraphState, deps: EpistemicDeps
    ) -> None:
        """Execute ``DispatchGatherOperation`` with the providers from
        deps. Kept inline (rather than through ``_run_op``) because the
        operation takes a non-standard ``providers`` kwarg that
        ``_make_op`` doesn't know about.

        Because it bypasses ``_make_op``, this construction must pass
        every dependency ``_make_op`` passes — anything omitted here is
        silently absent on the live graph path only.
        ``test_structural_wiring`` pins the two lists equal.
        """
        from ..operations.base import OperationInput, OperationResult
        from ..operations.dispatch_gather import DispatchGatherOperation

        op = DispatchGatherOperation(
            deps.repo,
            deps.agent_runner,
            evidence_gatherer=deps.evidence_gatherer,
            quality_scorer=deps.quality_scorer,
            embedding_model=deps.embedding_model,
            judge_model=deps.judge_model,
            providers=deps.providers,
        )
        work = OperationInput(
            entity_id=state.objective_id,
            entity_type="objective",
            operation="dispatch_gather",
            metadata={},
        )
        try:
            result = await op.execute(work)
        except Exception as e:
            logger.warning(
                "dispatch_gather on %s raised %s: %s — quarantining",
                state.objective_id[:12],
                type(e).__name__,
                e,
            )
            state.quarantine(state.objective_id, "objective", "dispatch_gather", e)
            result = OperationResult(
                success=False,
                entity_id=state.objective_id,
                message=f"dispatch_gather quarantined: {type(e).__name__}: {e}",
            )
        state.log_operation(
            "dispatch_gather", state.objective_id, result.success, result.message
        )


@dataclass
class ExtractEvidence(Node):
    """Extract content from all unextracted evidence stubs.

    Phase 5 of the Move-3 plan. Writes ``retrieval_failed`` once at
    end-of-gathering via ``_evaluate_retrieval_health`` (zero-yield
    semantics).
    """

    reads = frozenset({"objective_id", "claims_created"})
    writes = frozenset({"retrieval_failed"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["CreateClaims", "Scrutinize"]:
        from ..operations.evidence import ExtractEvidenceOperation

        state = ctx.state
        deps = ctx.deps

        unextracted = await deps.repo.query(
            "evidence",
            objective_id=state.objective_id,
            extracted=False,
        )

        for ev in unextracted:
            await _run_op(
                ExtractEvidenceOperation,
                deps,
                state,
                ev.entity_id,
                "evidence",
                "extract_evidence",
            )

        # Cross-provider duplicate sweep: when multiple providers return the
        # same paper, mark all but one as invalidated so judging / scrutiny /
        # gates don't pay LLM cost on redundant copies. Downstream filters
        # already exclude invalidated evidence; this is purely cost reduction.
        from ..dedupe_evidence import dedupe_evidence_by_source_ref

        await dedupe_evidence_by_source_ref(deps.repo, state.objective_id)

        # End-of-gathering retrieval-health check (zero-yield semantics).
        await _evaluate_retrieval_health(state, deps)

        # If claims have not yet been created, go create them.
        # Otherwise we are re-entering after investigation — go back to scrutiny.
        if not state.claims_created:
            return CreateClaims()
        return Scrutinize()


@dataclass
class CreateClaims(Node):
    """Create claims — three branches:

    1. ``claim_to_verify`` set → SeedClaim (one claim from user's text)
    2. ``decomposition`` set → MultiSeedClaim (N claims from
       sub-investigations, per-claim evidence pools)
    3. otherwise → ProposeClaims (claims discovered from evidence)

    Multi-seed-claim is the v0.3 collapse of decomposition spawning into
    the v0.1 multi-claim shape: one Objective hosts N Claims, the graph
    runs once over them, multi-claim convergence dynamics apply.

    Phase 5 of the Move-3 plan.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset({"claim_ids", "claims_created"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Scrutinize":
        from ..operations.seed_claim import SeedClaimOperation
        from ..operations.multi_seed_claim import MultiSeedClaimOperation
        from ..operations.claims import ProposeClaimsOperation

        state = ctx.state
        deps = ctx.deps
        oid = state.objective_id

        obj = await deps.repo.get("objective", oid)
        if obj.claim_to_verify:
            await _run_op(
                SeedClaimOperation, deps, state, oid, "objective", "seed_claim"
            )
        elif obj.decomposition:
            await _run_op(
                MultiSeedClaimOperation,
                deps,
                state,
                oid,
                "objective",
                "multi_seed_claim",
            )
            # Degenerate-decomposition fallback: if MultiSeedClaim minted
            # zero claims (e.g. DecomposeQuestion produced no usable
            # sub_investigations, or every sub had empty seed_claim), the
            # graph would otherwise proceed with no claims at all and
            # produce a silent empty report. Fall back to ProposeClaims —
            # the v0.1 open-research path — so the inquiry still has
            # something to verify.
            claims_after = await deps.repo.query("claim", objective_id=oid)
            if not [c for c in claims_after if not c.abandoned]:
                logger.warning(
                    "MultiSeedClaim minted no claims for objective %s; "
                    "falling back to ProposeClaims to avoid empty inquiry",
                    oid[:12],
                )
                await _run_op(
                    ProposeClaimsOperation,
                    deps,
                    state,
                    oid,
                    "objective",
                    "propose_claims",
                )
        else:
            await _run_op(
                ProposeClaimsOperation, deps, state, oid, "objective", "propose_claims"
            )

        # Populate claim IDs from repo
        claims = await deps.repo.query("claim", objective_id=oid)
        state.claim_ids = [c.entity_id for c in claims if not c.abandoned]
        state.claims_created = True

        obj = await deps.repo.get("objective", oid)
        obj.phase = "claims_proposed"
        obj.claims_proposed = True
        await deps.repo.save(obj)

        return Scrutinize()


@dataclass
class Scrutinize(Node):
    """Run scrutiny on claims that have not yet been scrutinised.

    Phase 3 of the Move-3 plan. Successors live in the run() return
    annotation. post_invariants is empty: this node is mid-flight,
    not terminal.
    """

    reads = frozenset(
        {
            "objective_id",
            "claims_needing_rescrutiny",
            "scrutiny_resolve_cycles",
            "investigation_counts",
        }
    )
    writes = frozenset({"scrutiny_resolve_cycles", "claims_needing_rescrutiny"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["ResolveUncertainties", "Investigate", "AbandonOrDemote"]:
        from ..operations.scrutiny import ScrutiniseClaimOperation

        state = ctx.state
        deps = ctx.deps

        # Refresh claim list
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        active_claims = [c for c in all_claims if not c.abandoned]

        # Scrutinise claims without a verdict or needing re-scrutiny
        for claim in active_claims:
            needs_scrutiny = (
                claim.scrutiny_verdict is None
                or claim.entity_id in state.claims_needing_rescrutiny
            )
            if not needs_scrutiny:
                continue

            is_rescrutiny = claim.entity_id in state.claims_needing_rescrutiny
            if is_rescrutiny:
                cycles = state.scrutiny_resolve_cycles.get(claim.entity_id, 0)
                if cycles >= SCRUTINY_RESOLVE_CYCLE_CAP:
                    # Defense in depth: ResolveUncertainties is the primary
                    # gate, but if anything else added the claim to the set
                    # post-cap, refuse here too. The claim is marked
                    # cycle_capped so downstream (PromoteToSupported,
                    # compute_posterior) can route it to a terminal state
                    # rather than promoting it as if inquiry converged.
                    logger.warning(
                        "Claim %s hit scrutiny-resolve cycle cap (%d); "
                        "skipping rescrutiny, retaining verdict=%s",
                        claim.entity_id[:12],
                        SCRUTINY_RESOLVE_CYCLE_CAP,
                        claim.scrutiny_verdict,
                    )
                    await _mark_cycle_capped(
                        deps,
                        state,
                        claim_id=claim.entity_id,
                        source="Scrutinize-defense-in-depth",
                    )
                    state.claims_needing_rescrutiny.discard(claim.entity_id)
                    continue
                state.scrutiny_resolve_cycles[claim.entity_id] = cycles + 1
                # Reset verdict for re-scrutiny so the operation runs.
                # Clearing scrutiny_fingerprint alongside the verdict tells
                # the operation that this is an intentional re-pass, not a
                # spontaneous re-entry on unchanged inputs.
                claim.scrutiny_verdict = None
                claim.scrutiny_fingerprint = None
                await deps.repo.save(claim)
                state.claims_needing_rescrutiny.discard(claim.entity_id)
            await _run_op(
                ScrutiniseClaimOperation,
                deps,
                state,
                claim.entity_id,
                "claim",
                "scrutinise_claim",
            )

        # Re-read claims after scrutiny
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        active_claims = [c for c in all_claims if not c.abandoned]

        if not active_claims:
            return AbandonOrDemote()

        # Categorise results
        needs_investigation: list[Any] = []
        needs_abandon: list[Any] = []

        from ..entities.claim import ClaimStage

        for claim in active_claims:
            if claim.scrutiny_verdict == "pass":
                continue  # will be promoted
            if claim.scrutiny_verdict in ("needs_resolution", "fail"):
                inv_count = state.investigation_counts.get(claim.entity_id, 0)
                if (
                    claim.stage == ClaimStage.HYPOTHESIS
                    and inv_count >= PEIRCE_CYCLE_CAP
                ):
                    needs_abandon.append(claim)
                elif claim.stage != ClaimStage.HYPOTHESIS:
                    # SUPPORTED+ claim that failed scrutiny -> demote
                    needs_abandon.append(claim)
                else:
                    needs_investigation.append(claim)

        if needs_abandon:
            return AbandonOrDemote()
        if needs_investigation:
            return Investigate()

        # All claims pass or are terminal — resolve uncertainties before promoting
        return ResolveUncertainties(next_on_clear="promote")


@dataclass
class Investigate(Node):
    """Run investigation on claims needing more evidence.

    Phase 3 of the Move-3 plan. The single successor (ExtractNewEvidence)
    lives in the return annotation. post_invariants is empty: this
    node is mid-flight, not terminal.
    """

    reads = frozenset(
        {"objective_id", "investigation_counts", "claims_needing_investigation"}
    )
    writes = frozenset(
        {
            "investigation_counts",
            "claims_needing_rescrutiny",
            "claims_needing_investigation",
        }
    )
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ExtractNewEvidence":
        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            # D4 (self-feeding-loop plan): a claim can ask for
            # investigation two ways — the pre-existing scrutiny-verdict
            # filter (rescrutiny requested a fresh look), or
            # ``claims_needing_investigation`` (PromoteToSupported found
            # the claim refused ONLY for too few supporting sources,
            # scrutiny_verdict left untouched at "pass" — P3/P4: the
            # demand lives on graph state, not fabricated onto the
            # entity). Either way this node re-derives its own cap check
            # from ``investigation_counts`` before doing any work (D5:
            # one backstop, not duplicated at the call site).
            requested = claim.entity_id in state.claims_needing_investigation
            if requested:
                state.claims_needing_investigation.discard(claim.entity_id)
            if requested or claim.scrutiny_verdict in ("needs_resolution", "fail"):
                inv_count = state.investigation_counts.get(claim.entity_id, 0)
                if inv_count < PEIRCE_CYCLE_CAP:
                    result = await self._run_investigate(state, deps, claim.entity_id)
                    state.investigation_counts[claim.entity_id] = inv_count + 1
                    if result.success:
                        state.claims_needing_rescrutiny.add(claim.entity_id)

        return ExtractNewEvidence()

    async def _run_investigate(
        self, state: EpistemicGraphState, deps: EpistemicDeps, claim_id: str
    ) -> Any:
        """Execute ``InvestigateClaimOperation`` with the providers from
        deps. Inline (rather than through ``_run_op``) because the
        operation takes a non-standard ``providers`` kwarg that
        ``_make_op`` doesn't know about — same pattern as PlanEvidence
        uses for DispatchGatherOperation.

        Because it bypasses ``_make_op``, this construction must pass
        every dependency ``_make_op`` passes — anything omitted here is
        silently absent on the live graph path only.
        ``test_structural_wiring`` pins the two lists equal.
        """
        from ..operations.base import OperationInput, OperationResult
        from ..operations.investigation import InvestigateClaimOperation

        op = InvestigateClaimOperation(
            deps.repo,
            deps.agent_runner,
            evidence_gatherer=deps.evidence_gatherer,
            quality_scorer=deps.quality_scorer,
            embedding_model=deps.embedding_model,
            judge_model=deps.judge_model,
            providers=deps.providers,
        )
        work = OperationInput(
            entity_id=claim_id,
            entity_type="claim",
            operation="investigate_claim",
            metadata={},
        )
        try:
            result = await op.execute(work)
        except Exception as e:
            logger.warning(
                "investigate_claim on %s raised %s: %s — quarantining",
                claim_id[:12],
                type(e).__name__,
                e,
            )
            state.quarantine(claim_id, "claim", "investigate_claim", e)
            result = OperationResult(
                success=False,
                entity_id=claim_id,
                message=f"investigate_claim quarantined: {type(e).__name__}: {e}",
            )
        state.log_operation(
            "investigate_claim", claim_id, result.success, result.message
        )
        return result


@dataclass
class ExtractNewEvidence(Node):
    """Extract content from newly created evidence stubs, then re-enter scrutiny.

    Phase 5 of the Move-3 plan. Writes ``retrieval_failed`` once at
    end-of-gathering via ``_evaluate_retrieval_health`` (zero-yield
    semantics).
    """

    reads = frozenset({"objective_id"})
    writes = frozenset({"retrieval_failed"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Scrutinize":
        from ..operations.evidence import ExtractEvidenceOperation
        from ..entities import Claim, Evidence
        from ..judge import apply_judgment, judge_evidence

        state = ctx.state
        deps = ctx.deps

        unextracted = await deps.repo.query(
            "evidence",
            objective_id=state.objective_id,
            extracted=False,
        )

        # Pass 1: extraction. Defensive — the current evidence-gathering
        # path (initial gather via DispatchGatherOperation, follow-up
        # rounds via InvestigateClaimOperation + dispatch_and_persist_
        # for_text) persists Evidence with ``extracted=True`` directly,
        # so this loop is empty on normal runs. It remains in place so
        # any future code path that introduces ``extracted=False`` stubs
        # is still handled.
        for ev in unextracted:
            await _run_op(
                ExtractEvidenceOperation,
                deps,
                state,
                ev.entity_id,
                "evidence",
                "extract_evidence",
            )

        # Cross-provider duplicate sweep before judging fires. Dedupe runs
        # on all evidence for the objective so it catches duplicates within
        # this batch and against earlier-extracted evidence.
        from ..dedupe_evidence import dedupe_evidence_by_source_ref

        await dedupe_evidence_by_source_ref(deps.repo, state.objective_id)

        # End-of-gathering retrieval-health check (zero-yield semantics).
        await _evaluate_retrieval_health(state, deps)

        # Pass 2: judge every Evidence linked to a non-abandoned claim
        # that has extracted content and no support_judgment yet.
        #
        # The judging contract is **source-agnostic**: every Evidence
        # with ``depends_on_claim_id`` set, ``extracted_content``
        # non-empty, ``support_judgment is None``, and ``invalidated=
        # False`` is eligible for judging. The predicate doesn't care
        # HOW the Evidence got into the repo — only that it's a
        # claim-linked piece of content still missing its verdict.
        # The scrutiny_and_investigation stage's exit invariant in
        # ``graph/stages.py`` enforces the same predicate as a safety
        # net, so any future creation path that bypasses judging fails
        # loudly at the stage boundary rather than silently degrading
        # posterior calibration.
        if deps.agent_runner is not None:
            all_evidence = await deps.repo.query(
                "evidence", objective_id=state.objective_id
            )
            unjudged_by_claim: dict[str, list[Evidence]] = {}
            for ev in all_evidence:
                if not isinstance(ev, Evidence):
                    continue
                if not ev.depends_on_claim_id:
                    continue
                if ev.support_judgment is not None:
                    continue
                if ev.invalidated:
                    continue
                if not ev.extracted_content:
                    continue
                unjudged_by_claim.setdefault(ev.depends_on_claim_id, []).append(ev)

            runner = deps.agent_runner

            async def _judge_against_claim(
                ev_to_judge: Evidence, claim_statement: str, claim_scope: str
            ) -> None:
                judgment = await judge_evidence(
                    claim_statement=claim_statement,
                    claim_scope=claim_scope,
                    evidence_content=ev_to_judge.extracted_content,
                    evidence_source=(
                        f"{ev_to_judge.source_type}: {ev_to_judge.source_ref}"
                    ),
                    runner=runner,
                )
                apply_judgment(ev_to_judge, judgment)
                await deps.repo.save(ev_to_judge)

            for claim_id, evs in unjudged_by_claim.items():
                claim = await deps.repo.get("claim", claim_id)
                if not isinstance(claim, Claim) or claim.abandoned:
                    continue
                claim_statement = claim.statement
                claim_scope = claim.scope or ""
                await asyncio.gather(
                    *(
                        _judge_against_claim(ev, claim_statement, claim_scope)
                        for ev in evs
                    )
                )

        # TMS sweep: new evidence may trigger revalidation of claims
        await _run_tms_sweep(deps, state)

        return Scrutinize()


@dataclass
class AbandonOrDemote(Node):
    """Abandon HYPOTHESIS claims that exhausted investigation; demote SUPPORTED+ claims.

    Phase 2 of the Move-3 plan: this is one of the two recurring-bug
    routing hubs. Successors live in the run() return annotation
    (Union["Scrutinize", "PromoteToSupported"]) — pyright + pydantic-graph
    enforce them, and test_topology.py asserts CheckCompletion is
    explicitly NOT in that set.

    post_invariants is empty: this node is mid-flight, not terminal.
    Soft-promoted claims legitimately leave this node in the "stranded"
    state because PromoteToSupported (the next node) routes them to
    ClusterEvidence → IBE. The no_stranded_claims invariant only
    applies at terminal nodes (Synthesize will get it in Phase 1).
    """

    reads = frozenset({"investigation_counts", "objective_id"})
    writes = frozenset({"terminal_claims", "verification_done"})
    operations = frozenset()  # populated below after operations imports resolve
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Scrutinize", "PromoteToSupported"]:
        # Successor set is intentionally narrower than it was before
        # the routing fix in commit d280573. CheckCompletion was
        # removed from the annotation because the body no longer
        # returns it directly — terminal routing now goes through
        # PromoteToSupported (which itself decides whether to route
        # to ClusterEvidence or CheckCompletion). Re-introducing
        # CheckCompletion here would resurrect the recurring routing
        # bug class — the topology test in test_topology.py asserts
        # it stays out.
        from ..operations.cleanup import AbandonStaleClaimOperation
        from ..operations.stage_management import (
            DemoteClaimOperation,
            PromoteAsRefutedOperation,
            SoftPromoteOperation,
        )
        from ..principles.adjudicate import adjudicate
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        demoted_any = False

        for claim in all_claims:
            if claim.abandoned:
                continue

            if claim.scrutiny_verdict in ("needs_resolution", "fail"):
                inv_count = state.investigation_counts.get(claim.entity_id, 0)

                if (
                    claim.stage == ClaimStage.HYPOTHESIS
                    and inv_count >= PEIRCE_CYCLE_CAP
                ):
                    if claim.contraction_count > 0:
                        # This claim was contracted at least once (AGM
                        # revision, ``Claim.contract()``): it already
                        # carries a verdict this run certified, preserved
                        # through the contraction (D2/S8). Re-attempting
                        # refute-promotion or soft-promotion here would
                        # either be refused by
                        # ``gates.promotion_blocked_by_contraction`` on
                        # unchanged evidence, or — worse — abandonment
                        # would erase a verdict that is still correct.
                        # The claim is terminal for this run unless new
                        # evidence arrives (S3/S4); skip the whole
                        # refute/soft/abandon fork.
                        logger.info(
                            "Claim %s was contracted (count=%d); skipping "
                            "refute/soft-promote/abandon fork so its "
                            "preserved verdict is not erased",
                            claim.entity_id[:12],
                            claim.contraction_count,
                        )
                        continue

                    # First try refute-promotion: if evidence overwhelmingly
                    # contradicts, promote to SUPPORTED with
                    # integrated_assessment="contradicts" instead of abandoning.
                    refute_result = await _run_op(
                        PromoteAsRefutedOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "promote_as_refuted",
                    )
                    if refute_result.success:
                        # Claim is now SUPPORTED with "contradicts"
                        # assessment. Mark verification done so downstream
                        # nodes skip re-work and route to completion.
                        state.verification_done.add(claim.entity_id)
                        continue

                    # Refute declined. If the linked evidence still carries
                    # directional judgments, soft-promote: SUPPORTED with
                    # integrated_assessment=None so the IBE chain can
                    # produce a calibrated verdict from the directional
                    # evidence (instead of erasing it via abandonment).
                    #
                    # Critically: do NOT add this claim to
                    # ``verification_done``. SoftPromoteOperation
                    # deliberately leaves ``integrated_assessment=None``
                    # (see its docstring) and the IBE chain is the only
                    # thing that will populate it. Marking verification
                    # done would short-circuit the verification path
                    # whose terminal step is EnumerateCandidates → the
                    # IBE chain, leaving the claim stranded at SUPPORTED
                    # with no integration verdict and ``compute_posterior``
                    # falling back to the no-data 0.5 prior.
                    #
                    # Refute-promotion above DOES set verification_done
                    # because that path pre-sets
                    # ``integrated_assessment="contradicts"`` and IBE
                    # would just overwrite it. The two branches diverge
                    # on this point.
                    soft_result = await _run_op(
                        SoftPromoteOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "soft_promote",
                    )
                    if soft_result.success:
                        continue

                    # Fall through: no directional signal at all, abandon
                    # as before. Posterior stays at 0.5 because there is
                    # genuinely nothing to say.
                    await _run_op(
                        AbandonStaleClaimOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "abandon_stale_claim",
                    )
                    state.terminal_claims.add(claim.entity_id)
                elif claim.stage != ClaimStage.HYPOTHESIS:
                    # Named adjudication (plan §4.2,
                    # docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md) instead of
                    # a buried conditional: ask every principle whether it
                    # wants this claim, in a fixed declared order, and act
                    # on the winner. The declared order (Peirce before AGM
                    # — principles/adjudicate.py) IS the "cap wins over
                    # demotion" policy below, encoded as data rather than
                    # changed:
                    #
                    # NEVER demote a cycle-capped claim. Two independent
                    # reasons, either alone sufficient:
                    #
                    # 1. Demotion IS Peirce cycling — record_demotion
                    #    resets scrutiny_verdict and every track flag so
                    #    the claim re-enters the loop. That is precisely
                    #    what the cap exists to stop.
                    # 2. record_demotion also nulls
                    #    integrated_assessment / _confidence /
                    #    _reasoning. For a capped claim those hold the
                    #    terminal verdict ConcludeAtCycleCapOperation
                    #    derived from the evidence, so demotion restores
                    #    the false "insufficient" that verdict exists to
                    #    remove — and leaves cap_evaluation behind, so
                    #    the record and the verdict then disagree.
                    #
                    # The reachable sequence is: cap fires at HYPOTHESIS
                    # → this node refute- or soft-promotes the claim to
                    # SUPPORTED (neither path filters cycle_capped) →
                    # the next Scrutinize visit sees a SUPPORTED claim
                    # with scrutiny_verdict="fail" and routes back here.
                    # Skipping is safe: the claim is left terminal,
                    # PromoteToSupported already filters capped claims,
                    # and nothing re-scrutinises it, so there is no loop.
                    decision = adjudicate(claim, state)
                    if decision.winner == "peirce":
                        logger.warning(
                            "Claim %s is cycle-capped at %s; skipping demotion "
                            "so its cap-derived verdict is not erased (%s)",
                            claim.entity_id[:12],
                            claim.stage.value,
                            decision.reason,
                        )
                        state.terminal_claims.add(claim.entity_id)
                        continue
                    if decision.winner == "agm":
                        # Demote SUPPORTED+ claims
                        await _run_op(
                            DemoteClaimOperation,
                            deps,
                            state,
                            claim.entity_id,
                            "claim",
                            "demote_claim",
                        )
                        demoted_any = True
                    else:
                        # Unreachable given this branch's own guard (scrutiny
                        # failing + stage != HYPOTHESIS is exactly AGM's own
                        # eligibility condition — see principles/agm.py), but
                        # fail loud rather than silently doing nothing to a
                        # claim scrutiny already flagged, per the repo's
                        # no-silent-failures rule.
                        raise RuntimeError(
                            f"adjudicate() returned no winner for claim "
                            f"{claim.entity_id[:12]} (stage="
                            f"{claim.stage.value}, scrutiny_verdict="
                            f"{claim.scrutiny_verdict!r}): {decision.reason}. "
                            f"considered={decision.considered!r}"
                        )

        if demoted_any:
            # Re-scrutinise demoted claims.
            return Scrutinize()

        # Default: route through PromoteToSupported. This node has loaded
        # only the failed-scrutiny claims into refute / soft-promote /
        # abandon — but a sibling claim may have passed scrutiny in the
        # same Scrutinize round and is sitting at HYPOTHESIS waiting for
        # promote_claim. PromoteToSupported is the dispatcher that
        # handles all such residual work:
        #
        #   * HYPOTHESIS with scrutiny_verdict="pass" → promote_claim
        #   * SUPPORTED needing verification → ClusterEvidence → IBE chain
        #   * Soft-promoted (SUPPORTED, integrated_assessment=None,
        #     not in verification_done) → ClusterEvidence → IBE chain
        #   * Nothing residual → CheckCompletion
        #
        # Returning CheckCompletion here directly was the bug: when
        # Scrutinize found a mixed pass/abandon set it routed to
        # AbandonOrDemote (priority on abandonment), which then
        # short-circuited past PromoteToSupported. The pass-verdict
        # claims were stranded at HYPOTHESIS, never promoted, never
        # entered IBE. Same shape as the soft-promote routing bug
        # found one commit earlier — same root cause (this node
        # over-claimed terminality) and same fix (delegate residual
        # routing to PromoteToSupported, which is idempotent).
        return PromoteToSupported()


@dataclass
class PromoteToSupported(Node):
    """Promote HYPOTHESIS claims and route SUPPORTED claims to verification.

    This is a routing hub that checks actual claim state:
    1. HYPOTHESIS with pass → promote to SUPPORTED, set routing defaults
    1b. HYPOTHESIS refused ONLY for too few supporting sources, still
        under its investigation cap → Investigate (D4, self-feeding-loop
        plan; lazy escalation, P7 — the claim is asking for more
        evidence, not stuck)
    2. SUPPORTED without verification done → route to ClusterEvidence
    3. Everything at PROVISIONAL+ or abandoned → CombineClaimVerdicts

    This correctly handles re-entry after uncertainty resolution:
    a claim at SUPPORTED that was re-scrutinized goes through
    verification again instead of being skipped.

    Phase 2 of the Move-3 plan: this is the second of the two
    recurring-bug routing hubs. Its successor set
    (Union["ClusterEvidence", "CombineClaimVerdicts"]) intentionally
    INCLUDES the terminal-bound branch — that's the legitimate
    idempotent exit when no SUPPORTED claim needs verification.
    PromoteToSupported is the dispatcher that AbandonOrDemote
    delegates residual routing to; this node is allowed to
    terminate where the upstream node is not.

    The no-work exit routes to ``CombineClaimVerdicts``, not straight to
    ``CheckCompletion``. That single retargeted edge makes
    CombineClaimVerdicts the SOLE predecessor of CheckCompletion, so the
    run's verdict object is rebuilt from the live DB immediately before
    every terminal decision — including after a
    ``CheckSynthesisDemand → Scrutinize`` loop-back, where the old edge
    would have left CheckCompletion reading a stale verdict or none at
    all. ``tests/test_verdict_routing.py`` asserts the domination
    property over ``topology()``.

    post_invariants is empty: this node is mid-flight when it
    routes to ClusterEvidence (claims may legitimately be in the
    "stranded" state about to enter IBE); on the other branch
    no_stranded_claims would be the right invariant to enforce — but
    that's the post-condition for CheckCompletion → Synthesize, not for
    this node directly.
    """

    reads = frozenset(
        {
            "objective_id",
            "verification_done",
            "investigation_counts",
            "scrutiny_resolve_cycles",
        }
    )
    writes = frozenset(
        {"terminal_claims", "verification_done", "claims_needing_investigation"}
    )
    operations = frozenset()  # populated below after operations imports resolve
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["ClusterEvidence", "CombineClaimVerdicts", "Investigate"]:
        from ..operations.stage_management import PromoteClaimOperation
        from ..operations.belief_maintenance import SetRoutingDefaultsOperation
        from ..entities.claim import ClaimStage
        from ..gates import claim_evidence_fingerprint, refused_only_for_min_supporting_sources

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # D4 (self-feeding-loop plan): set when a HYPOTHESIS claim was
        # refused promotion for insufficient supporting sources alone and
        # is still under its investigation cap. Checked after the whole
        # loop so every claim in this batch gets its Step-1 pass before we
        # decide routing.
        needs_investigation = False

        # Step 1: Promote any HYPOTHESIS claims with passing scrutiny
        for claim in all_claims:
            if claim.abandoned:
                continue
            if claim.cycle_capped:
                # Inquiry didn't converge — don't promote this claim, and
                # mark it terminal so CheckCompletion sees no pending work.
                # The claim stays at HYPOTHESIS. It is NOT verdict-less:
                # ConcludeAtCycleCapOperation already read its evidence
                # deterministically at the cap and wrote the verdict that
                # reading supports (or recorded, in cap_evaluation, that the
                # reading was not directional). What is skipped here is the
                # LLM-priced IBE chain, not the verdict.
                state.terminal_claims.add(claim.entity_id)
                continue
            if (
                claim.stage == ClaimStage.HYPOTHESIS
                and claim.scrutiny_verdict == "pass"
            ):
                was_contracted = claim.contraction_count > 0
                if was_contracted:
                    # Certify-before-contract D3: this claim was
                    # contracted back to HYPOTHESIS while keeping its
                    # verdict. Re-promoting it is only warranted if the
                    # evidence actually changed since that contraction —
                    # otherwise the verification tracks would just
                    # re-derive the same contraction (S3). On an
                    # unchanged fingerprint, skip the promotion attempt
                    # entirely — the claim stays terminal for this run.
                    current_fp = await claim_evidence_fingerprint(
                        claim, deps.repo
                    )
                    if current_fp == claim.contraction_fingerprint:
                        logger.info(
                            "Claim %s: evidence unchanged since contraction; "
                            "terminal for this run, not re-promoting",
                            claim.entity_id[:12],
                        )
                        state.terminal_claims.add(claim.entity_id)
                        continue
                    # G1: the fingerprint changed, so a re-promotion
                    # attempt is warranted — but `reset_for_reinquiry()`
                    # nulls the verdict that survived the contraction
                    # (D2/S8), so it must NOT run before we know the
                    # attempt succeeds. Attempt promotion first; only on
                    # success does the claim actually start a fresh
                    # inquiry cycle worth resetting for. On refusal (e.g.
                    # a blocking uncertainty) the claim keeps its stage,
                    # its verdict, and its contraction_fingerprint, and
                    # stays terminal for this run — re-promotion can be
                    # attempted again on a later pass if the evidence
                    # moves further.

                result = await _run_op(
                    PromoteClaimOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "promote_claim",
                )
                if result.success:
                    if was_contracted:
                        promoted_claim = await deps.repo.get(
                            "claim", claim.entity_id
                        )
                        promoted_claim.reset_for_reinquiry()
                        await deps.repo.save(promoted_claim)
                        state.verification_done.discard(claim.entity_id)
                    await _run_op(
                        SetRoutingDefaultsOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "set_routing_defaults",
                    )
                elif (
                    refused_only_for_min_supporting_sources(result.validation_errors)
                    and state.investigation_counts.get(claim.entity_id, 0)
                    < PEIRCE_CYCLE_CAP
                    and _loopback_eligible(claim, state)
                ):
                    # D4 (self-feeding-loop plan): the ONLY thing blocking
                    # this HYPOTHESIS claim is too few corroborating
                    # sources — not scrutiny, not a verification track,
                    # nothing else. That is exactly the shape lazy
                    # escalation (P7) exists for: the claim itself is
                    # asking for more evidence, so route it back through
                    # the existing Investigate/rescrutiny loop instead of
                    # leaving it stranded at HYPOTHESIS for this run.
                    #
                    # Both caps have to clear, not just the investigation
                    # one: `Investigate`'s own success path adds the claim
                    # to `claims_needing_rescrutiny`, and the rescrutiny
                    # that satisfies is gated separately by
                    # `scrutiny_resolve_cycles`
                    # (`SCRUTINY_RESOLVE_CYCLE_CAP`, checked in Scrutinize).
                    # A claim that reached here with its resolve-cycle
                    # count already at cap would get investigated, come
                    # back with a rescrutiny request Scrutinize refuses,
                    # and be cap-decided or abandoned on evidence that
                    # never got a chance to be re-read — the outcome §0/S9
                    # exist to reduce. `_loopback_eligible` is the single
                    # predicate `CheckSynthesisDemand` already uses for
                    # this exact "could another round plausibly change
                    # anything" question; reusing it keeps one source of
                    # truth for both caps instead of duplicating the
                    # resolve-cycle comparison here.
                    #
                    # The demand lives on graph state
                    # (`claims_needing_investigation`), not on the entity:
                    # scrutiny never produced "needs_resolution" for this
                    # claim, so fabricating that value onto
                    # `Claim.scrutiny_verdict` would misrepresent what
                    # scrutiny actually found (P3) to every downstream
                    # reader of that field — `ClaimVerdict.scrutiny_verdict`,
                    # the report generator's needs-resolution count, and
                    # `InvestigateClaimOperation`'s own cap branch — and
                    # would strand a claim whose verdict is "pass" at
                    # "needs_resolution" if the investigation that follows
                    # is quarantined or fails. `Investigate` reads this set
                    # (P4: graph state tracks what needs doing) alongside
                    # its existing scrutiny-verdict filter and re-derives
                    # its own cap check against `investigation_counts`
                    # (D5: the cap is the one backstop, not duplicated
                    # here).
                    state.claims_needing_investigation.add(claim.entity_id)
                    needs_investigation = True
                elif was_contracted:
                    state.terminal_claims.add(claim.entity_id)

        if needs_investigation:
            # D4/S5: route straight to Investigate rather than falling
            # through to the verification/completion checks below — those
            # claims aren't stuck, they're asking for more evidence
            # (P7 lazy escalation), and Investigate re-derives its own
            # per-claim cap check from `investigation_counts` before doing
            # any work. Other claims that also need verification this
            # pass simply get their turn on the next arrival here, once
            # this loop's Investigate → ExtractNewEvidence → Scrutinize
            # cycle completes.
            return Investigate()

        # Step 2: Check if any SUPPORTED claims need verification
        # Re-read after promotions.
        #
        # Cycle-capped claims are excluded: cycle-capped means the
        # inquiry loop didn't converge cleanly, so spending verification
        # plus four IBE calls on the claim is not warranted. The reason
        # is LLM COST, not correctness — combine_claim_verdicts has
        # included capped-with-verdict claims (at a confidence penalty)
        # since 2026-05-04, and the claim receives a deterministic
        # terminal verdict at the cap instead
        # (ConcludeAtCycleCapOperation).
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        needs_verification = any(
            not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
            and c.entity_id not in state.verification_done
            for c in all_claims
        )

        if needs_verification:
            return ClusterEvidence()

        # All claims are at PROVISIONAL+ or abandoned or verified.
        # Route through CombineClaimVerdicts so the verdict object is
        # rebuilt before CheckCompletion decides anything.
        return CombineClaimVerdicts()


@dataclass
class ClusterEvidence(Node):
    """Cluster evidence into representatives before verification.

    Deterministic (embedding-based, no LLM). Runs HDBSCAN on evidence
    embeddings and labels each item as representative, corroborative,
    or deferred. Verification operations then only process representatives.

    Separated into its own node to distinguish deterministic steps from
    LLM-calling steps in the graph topology.

    Phase 4 of the Move-3 plan. Empty operations set: this node calls
    ``select_top_k_evidence`` directly (a function, not via ``_run_op``).
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "RunVerification":
        from ..operations.claims import select_top_k_evidence
        from ..entities import Evidence
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        for claim in all_claims:
            if claim.abandoned or claim.stage != ClaimStage.SUPPORTED:
                continue
            all_ev = []
            for eid in claim.evidence_ids:
                ev = await deps.repo.get("evidence", eid)
                if (
                    isinstance(ev, Evidence)
                    and ev.extracted
                    and ev.extracted_content
                    and not ev.invalidated
                ):
                    all_ev.append(ev)
            if len(all_ev) >= 2:
                await select_top_k_evidence(
                    deps.repo, all_ev, embedding_model=deps.embedding_model
                )

        return RunVerification()


@dataclass
class RunVerification(Node):
    """Run verification tracks on SUPPORTED claims based on routing profile.

    LLM-heavy: adversarial search, convergence, deductive validation,
    computational verification, argument analysis, contrastive evaluation.

    Phase 4 of the Move-3 plan. Operations metadata covers the direct
    ``_run_op`` dispatches in the body; the TMS sweep helper
    (``_run_tms_sweep``) dispatches InvalidateEvidence operations
    indirectly — those are side effects via a helper function and would
    require inter-procedural AST analysis to validate. For now the
    validator only sees direct dispatches.

    Certify-before-contract (docs/epistemic/
    CERTIFY_BEFORE_CONTRACT_PLAN.md D1): this node no longer flags a
    claim for TMS revalidation after the adversarial track, and
    ``_run_tms_sweep`` no longer revalidates or demotes/contracts
    claims — that check moved to ``PromoteSupported``, which runs it
    once, after IBE has already produced a verdict, instead of racing
    IBE for the same claim mid-verification. The
    ``claims_needing_rescrutiny`` guard below
    is unreachable from this path now (nothing here populates that set);
    it stays because two other nodes still populate it independently:
    ``Investigate`` (new evidence) and ``CheckSynthesisDemand`` (synthesis
    demand loop-back).
    """

    reads = frozenset({"objective_id", "question_type", "claims_needing_rescrutiny"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["ResolveUncertainties", "EnumerateCandidates"]:
        from ..operations.verification import (
            AdversarialSearchOperation,
            AssessConvergenceOperation,
            ValidateDeductivelyOperation,
            VerifyComputationallyOperation,
        )
        from ..operations.analysis import (
            AnalyzeArgumentOperation,
            ContrastiveEvaluationOperation,
            CrossClaimConsistencyOperation,
        )
        from ..entities.claim import ClaimStage
        from ..entities.objective import Objective
        from ..routing import get_routing_profile, TrackActivation

        state = ctx.state
        deps = ctx.deps

        # Phase 3 multi-seed-claim correction: when the Objective is a
        # verification task (claim_to_verify or decomposition set),
        # route as verificatory regardless of the LLM classifier's
        # output. Each minted claim is binary verification by
        # construction; the parent's classification (e.g. "explanatory"
        # for a declarative SciFact claim) doesn't apply to per-claim
        # routing. Without this override, misclassification cascades
        # into convergence=SECONDARY, A2 doesn't fire, scrutiny ↔
        # resolve loops are bounded only by the runtime cap. With it,
        # verificatory routing ensures convergence is PRIMARY and the
        # convergence-driven termination path is reachable.
        objective = await deps.repo.get("objective", state.objective_id)
        if isinstance(objective, Objective) and objective.is_verification_task():
            question_type = "verificatory"
        else:
            question_type = state.question_type or "verificatory"

        try:
            profile = get_routing_profile(question_type)
        except KeyError:
            profile = get_routing_profile("verificatory")

        track_map: dict[str, tuple[type, str, str]] = {
            "adversarial": (
                AdversarialSearchOperation,
                "adversarial_search",
                "adversarial_checked",
            ),
            "convergence": (
                AssessConvergenceOperation,
                "assess_convergence",
                "convergence_checked",
            ),
            "deductive": (
                ValidateDeductivelyOperation,
                "validate_deductively",
                "deductive_checked",
            ),
            "computational": (
                VerifyComputationallyOperation,
                "verify_computationally",
                "computational_checked",
            ),
            "argument": (
                AnalyzeArgumentOperation,
                "analyze_argument",
                "argument_analyzed",
            ),
            "contrastive": (
                ContrastiveEvaluationOperation,
                "contrastive_evaluation",
                "contrastive_checked",
            ),
            "consistency": (
                CrossClaimConsistencyOperation,
                "cross_claim_consistency",
                "consistency_checked",
            ),
        }

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Phase 2b of the efficiency plan REVERTED: tracks run serially
        # per claim, not concurrently. The parallelization caused a
        # quality regression — running all (claim, track) pairs
        # concurrently lets adversarial counter-evidence accumulate
        # before any single track's TMS sweep can fire, which cascades
        # into more revalidate→demote→re-promote cycles. The serial
        # version naturally paces this: each track runs, its post-state
        # is observed, the next track sees the post-state. Parallel
        # versions all see the pre-state and cumulatively over-invalidate.
        # The fix isn't impossible (e.g. group tracks that don't write
        # claim state vs those that do), but it's a non-trivial design
        # change; reverting to serial keeps the system correct while we
        # think about how to safely re-introduce concurrency here.

        for claim in all_claims:
            if claim.abandoned or claim.stage != ClaimStage.SUPPORTED:
                continue

            for track_name, (op_class, op_name, checked_field) in track_map.items():
                activation = profile.tracks.get(track_name, TrackActivation.SKIP)
                if activation == TrackActivation.SKIP:
                    continue

                if activation == TrackActivation.SECONDARY:
                    # SECONDARY tracks only fire when a condition is met
                    if track_name == "adversarial":
                        # Fire on first pass (balance is None) or when the
                        # claim hasn't yet survived adversarial challenge.
                        # Skip-refire requires *survival* (not just "above
                        # midpoint"); see ADVERSARIAL_SURVIVED_THRESHOLD
                        # in epistemic.thresholds.
                        balance = claim.adversarial_balance
                        if (
                            balance is not None
                            and balance >= ADVERSARIAL_SURVIVED_THRESHOLD
                        ):
                            continue  # Already tested, survived — skip
                    elif track_name == "convergence":
                        # Only fire if claim has 3+ evidence items
                        if claim.evidence_count < 3:
                            continue
                    # Other SECONDARY tracks: fire unconditionally

                # Skip if already done
                if getattr(claim, checked_field, False):
                    continue

                await _run_op(
                    op_class,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    op_name,
                )

        # TMS sweep: adversarial search may have created contradicting
        # evidence that invalidates existing evidence or triggers claim
        # revalidation.
        await _run_tms_sweep(deps, state)

        # Convergence-driven termination: if every active SUPPORTED claim
        # has its convergence track terminal AND at least one is CONVERGENT
        # AND no blocking uncertainties remain, the evidence base has
        # stabilised and there is no resolution work left to do. Skip
        # ResolveUncertainties and go straight to integration.
        #
        # Multi-seed-claim correction (Phase 2): the previous gate fired on
        # *any* claim being CONVERGENT, which under N-claim runs would
        # short-circuit even when sibling SUPPORTED claims still had
        # unfinished verification tracks — dragging them into IBE with
        # half-checked routing. Now requires *all* active SUPPORTED claims
        # to have a terminal convergence verdict (CONVERGENT, or one of
        # the other definitive outcomes).
        #
        # Behaviour under routing profiles where convergence is SKIP or
        # SECONDARY-but-skipped: those claims have ``convergence_verdict ==
        # None``, which is not in the terminal set, so the gate never
        # fires. This is intentional — A2 is convergence-track-driven; if
        # the inquiry doesn't run convergence, there's no convergence
        # signal to short-circuit on, and the run terminates through the
        # regular ResolveUncertainties → integration path. Under multi-
        # seed-claim with Phase-3-forced verificatory routing, convergence
        # is PRIMARY and the gate fires correctly.
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        # Exclude cycle_capped — same rationale as PromoteToSupported's
        # needs_verification filter: cycle-capped means inquiry didn't
        # converge, so don't spend verification or IBE calls on it. The
        # combiner does NOT discard their verdict; the cap writes one
        # deterministically instead.
        active_supported = [
            c
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
        ]
        # Certify-before-contract (D1): nothing in this path populates
        # claims_needing_rescrutiny any more — the set is populated by
        # Investigate (new evidence) and by CheckSynthesisDemand (synthesis
        # demand loop-back). This guard stays so a claim either of those
        # flagged for rescrutiny isn't fast-pathed to IBE before Scrutinize
        # re-establishes its scrutiny verdict.
        if state.claims_needing_rescrutiny:
            return ResolveUncertainties()
        if active_supported:
            # Terminal convergence verdicts that count as "this claim is
            # done with the convergence track". CONVERGENT is the
            # positive case; the others are definitive too (the track
            # ran and reached a stable verdict, not "we never checked").
            terminal_convergence = {
                "CONVERGENT",
                "WEAKLY_CONVERGENT",
                "DIVERGENT",
                "PARTIAL",
                "SINGLE_DOMAIN",
                "NO_EVIDENCE",
            }
            all_terminal = all(
                c.convergence_verdict in terminal_convergence for c in active_supported
            )
            any_positive = any(
                c.convergence_verdict == "CONVERGENT" for c in active_supported
            )
            if all_terminal and any_positive:
                unresolved = await deps.repo.query(
                    "uncertainty",
                    objective_id=state.objective_id,
                    resolution=None,
                )
                blocking_remaining = any(u.is_blocking for u in unresolved)
                if not blocking_remaining:
                    return EnumerateCandidates()

        return ResolveUncertainties()


@dataclass
class ResolveUncertainties(Node):
    """Resolve blocking uncertainties, then deduplicate concerns.

    Appears in TWO places in the graph:
    1. After Scrutinize (next_on_clear="promote") — resolve before first promote
    2. After RunVerification (next_on_clear="integrate") — resolve before integration

    This matches the old pattern scheduler where resolve_uncertainty had
    higher priority than promote and ran first whenever blocking
    uncertainties existed.

    Phase 3 of the Move-3 plan. This node is one of the routing hubs the
    IBE-skip investigation (Phase 2 benchmark) made explicit: a claim
    flagged via ``claims_needing_rescrutiny`` — by ``Investigate`` (new
    evidence, stale scrutiny verdict) or by ``CheckSynthesisDemand``
    (synthesis demand loop-back) — must route back through Scrutinize
    before the subsequent EnumerateCandidates / PromoteToSupported
    filters see it, or it is stranded pre-IBE. Certify-before-contract
    (docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md D1) moved the OTHER
    historical source of this stranding — TMS revalidation demoting a
    verified claim back to HYPOTHESIS before IBE ever ran on it — out of
    this path entirely: that revalidation now runs once, in
    ``PromoteSupported``, after IBE has already produced a verdict.

    post_invariants is empty: this node is mid-flight; the cycle-cap
    invariant lives at terminal nodes.
    """

    depth: int = 0
    next_on_clear: str = "integrate"  # "promote" or "integrate"

    reads = frozenset(
        {
            "objective_id",
            "claims_needing_rescrutiny",
            "scrutiny_resolve_cycles",
        }
    )
    writes = frozenset({"claims_needing_rescrutiny", "scrutiny_resolve_cycles"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union[
        "EnumerateCandidates",
        "PromoteToSupported",
        "Scrutinize",
        "ResolveUncertainties",
    ]:
        from ..operations.uncertainty import ResolveUncertaintyOperation
        from ..operations.concerns import DeduplicateConcernsOperation

        state = ctx.state
        deps = ctx.deps

        # Find unresolved blocking uncertainties for this objective
        all_uncertainties = await deps.repo.query(
            "uncertainty",
            objective_id=state.objective_id,
            resolution=None,
        )
        blocking = await _live_blocking(all_uncertainties, deps.repo)

        if not blocking:
            # Rescrutiny check FIRST — must run before the early return
            # to EnumerateCandidates / PromoteToSupported. Without this
            # ordering, a claim Investigate flagged for rescrutiny (new
            # evidence extracted, scrutiny_verdict stale) is stranded:
            # the subsequent EnumerateCandidates / PromoteToSupported
            # filters require ``stage == SUPPORTED`` with a fresh
            # scrutiny verdict, which the claim no longer satisfies
            # until Scrutinize re-establishes it. This is the IBE-skip
            # bug surfaced in the Phase 2 benchmark of the Move-3 graph
            # contracts work.
            if state.claims_needing_rescrutiny:
                return Scrutinize()
            if self.next_on_clear == "promote":
                return PromoteToSupported()
            return EnumerateCandidates()

        for unc in blocking:
            # Certify-before-contract D7: a cycle-capped or abandoned
            # claim is terminal for this run — it consumes no further
            # LLM work of any kind. An uncertainty whose every affected
            # claim is already terminal has no live claim left to
            # resolve anything FOR, so resolving it would spend an LLM
            # call to inform claims that can no longer act on it. `blocking`
            # was already filtered by `_live_blocking` above, but this
            # re-check is load-bearing, not defensive: an earlier
            # iteration of THIS loop can cap an affected claim (via
            # `_mark_cycle_capped` below) after `blocking` was built, so a
            # later `unc` in the same pass can turn terminal-only mid-loop.
            if await _uncertainty_is_terminal_only(unc, deps.repo):
                logger.info(
                    "Uncertainty %s skipped: all %d affected claim(s) "
                    "are cycle-capped or abandoned",
                    unc.entity_id[:12],
                    len(unc.affected_claim_ids),
                )
                continue
            result = await _run_op(
                ResolveUncertaintyOperation,
                deps,
                state,
                unc.entity_id,
                "uncertainty",
                "resolve_uncertainty",
            )
            # Mark affected claims for re-scrutiny only when this call did
            # real work. Sibling-grouping inside ResolveUncertaintyOperation
            # can resolve N near-duplicate uncertainties in one LLM call; the
            # remaining entries in the `blocking` list then short-circuit
            # with did_work=False. Treating those as fresh state changes
            # would keep bouncing the graph back to Scrutinize on every
            # resolve loop.
            if result.success and result.did_work:
                unc_updated = await deps.repo.get("uncertainty", unc.entity_id)
                if unc_updated.is_blocking and unc_updated.resolution is not None:
                    for cid in unc_updated.affected_claim_ids:
                        # Cycle cap: once a claim has been through
                        # SCRUTINY_RESOLVE_CYCLE_CAP rescrutiny passes,
                        # stop requesting more. The loop driver is
                        # 'genuine resolution → rescrutiny → fresh
                        # uncertainty → genuine resolution …' and the
                        # cap is what breaks it. The claim keeps its
                        # current verdict and the graph proceeds.
                        cycles = state.scrutiny_resolve_cycles.get(cid, 0)
                        if cycles >= SCRUTINY_RESOLVE_CYCLE_CAP:
                            logger.info(
                                "Claim %s hit scrutiny-resolve cycle cap "
                                "(%d); not requesting further rescrutiny",
                                cid[:12],
                                SCRUTINY_RESOLVE_CYCLE_CAP,
                            )
                            await _mark_cycle_capped(
                                deps,
                                state,
                                claim_id=cid,
                                source="ResolveUncertainties-primary",
                            )
                            continue
                        state.claims_needing_rescrutiny.add(cid)

        # Deduplicate concerns on the objective
        await _run_op(
            DeduplicateConcernsOperation,
            deps,
            state,
            state.objective_id,
            "objective",
            "deduplicate_concerns",
        )

        # Check if new blocking uncertainties were created
        remaining = await deps.repo.query(
            "uncertainty",
            objective_id=state.objective_id,
            resolution=None,
        )
        new_blocking = await _live_blocking(remaining, deps.repo)

        if new_blocking and self.depth < PEIRCE_CYCLE_CAP:
            return ResolveUncertainties(
                depth=self.depth + 1,
                next_on_clear=self.next_on_clear,
            )

        if new_blocking:
            # Max depth reached but still have blocking uncertainties.
            # Re-enter scrutiny so claims can be re-evaluated.
            return Scrutinize()

        # If claims were marked for re-scrutiny during resolution,
        # go back to scrutiny before proceeding.
        if state.claims_needing_rescrutiny:
            return Scrutinize()

        if self.next_on_clear == "promote":
            return PromoteToSupported()
        return EnumerateCandidates()


@dataclass
class EnumerateCandidates(Node):
    """Stage 1 of IBE: enumerate candidate verdicts for each SUPPORTED claim.

    Generative role (Peirce). Iterative single-candidate calls with the
    running list as priors so each call diversifies away from prior
    candidates. Idempotent — skips claims that already have candidates.

    Exception, by design (Stage 3, PRINCIPLE_FIDELITY_PLAN.md §5): a
    claim whose enumeration surfaced fewer than 2 genuine candidates
    ends this node with `integration_candidates` still empty (the
    "insufficient explanatory diversity" decline —
    ``EnumerateCandidatesOperation``). Idempotency for THAT claim is not
    "already have candidates" but "already verification_done"
    (``PromoteSupported`` marks it unconditionally once the run reaches
    that far), which prevents re-ENTRY into ``ClusterEvidence`` on a
    later pass. Within a single already-in-progress
    ``ClusterEvidence → ... → EnumerateCandidates`` traversal triggered
    by a SIBLING claim on the same multi-claim objective, this node's
    own per-claim filter would still re-submit a declined claim for
    another enumeration attempt (a retry, not a re-selection of stale
    output — never incorrect, just occasionally redundant LLM cost).
    Not reachable at all in single-claim seed-verify mode, where every
    objective has exactly one claim.

    Phase 4 of the Move-3 plan.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ScoreLoveliness":
        from ..operations.integration import EnumerateCandidatesOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Phase 2c of the efficiency plan: enumerate candidates per
        # claim concurrently. Each claim's enumeration writes only its
        # own ``integration_candidates`` field — no cross-claim
        # interference. The AgentRunner's global semaphore bounds
        # in-flight LLM calls (1 for Ollama, 8 for cloud).
        # Cycle-capped claims excluded on LLM-cost grounds: their
        # inquiry didn't converge, so the four-call IBE chain isn't
        # warranted. Their verdict is NOT discarded — the cap writes a
        # deterministic one (ConcludeAtCycleCapOperation) and the
        # combiner includes it at a confidence penalty.
        eligible_ids = [
            c.entity_id
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
            and c.integrated_assessment is None
            and not c.integration_candidates
        ]
        if eligible_ids:
            # Fan out the LLM work; commit outcomes to State in the
            # deterministic join (Law 3 — join is the sole write site).
            outcomes = await asyncio.gather(
                *(
                    _execute_op(
                        EnumerateCandidatesOperation,
                        deps,
                        cid,
                        "claim",
                        "enumerate_candidates",
                    )
                    for cid in eligible_ids
                )
            )
            for outcome in outcomes:
                await _commit_op(deps, state, outcome)

        return ScoreLoveliness()


@dataclass
class ScoreLoveliness(Node):
    """Stage 2 of IBE: score each candidate's explanatory virtue.

    Evaluative role (Lipton). Per-candidate calls run in parallel inside
    the operation. Each call sees only its candidate (Kahneman
    independence). Idempotent.

    Phase 4 of the Move-3 plan.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ScoreLikeliness":
        from ..operations.integration import ScoreLovelinessOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Phase 2c of the efficiency plan: score loveliness per claim
        # concurrently. Within a claim, candidates are already scored
        # in parallel (asyncio.gather inside ScoreLovelinessOperation);
        # this lifts that to the across-claims layer too.
        # See EnumerateCandidates filter — same cycle-capped exclusion.
        eligible_ids = [
            c.entity_id
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
            and c.integrated_assessment is None
            and c.integration_candidates
            and any(cand.loveliness is None for cand in c.integration_candidates)
        ]
        if eligible_ids:
            # Fan out the LLM work; commit outcomes to State in the
            # deterministic join (Law 3 — join is the sole write site).
            outcomes = await asyncio.gather(
                *(
                    _execute_op(
                        ScoreLovelinessOperation,
                        deps,
                        cid,
                        "claim",
                        "score_loveliness",
                    )
                    for cid in eligible_ids
                )
            )
            for outcome in outcomes:
                await _commit_op(deps, state, outcome)

        return ScoreLikeliness()


@dataclass
class ScoreLikeliness(Node):
    """Stage 3 of IBE: score each candidate's fit-with-evidence.

    Evaluative role (Lipton). Same pattern as ScoreLoveliness but on
    likeliness. Idempotent.

    Phase 4 of the Move-3 plan.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "SelectBestExplanation":
        from ..operations.integration import ScoreLikelinessOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Phase 2c of the efficiency plan: same parallelization
        # pattern as ScoreLoveliness — across-claims gather.
        # See EnumerateCandidates filter — same cycle-capped exclusion.
        eligible_ids = [
            c.entity_id
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
            and c.integrated_assessment is None
            and c.integration_candidates
            and any(cand.likeliness is None for cand in c.integration_candidates)
        ]
        if eligible_ids:
            # Fan out the LLM work; commit outcomes to State in the
            # deterministic join (Law 3 — join is the sole write site).
            outcomes = await asyncio.gather(
                *(
                    _execute_op(
                        ScoreLikelinessOperation,
                        deps,
                        cid,
                        "claim",
                        "score_likeliness",
                    )
                    for cid in eligible_ids
                )
            )
            for outcome in outcomes:
                await _commit_op(deps, state, outcome)

        return SelectBestExplanation()


@dataclass
class SelectBestExplanation(Node):
    """Stage 4 of IBE: pick the best candidate, write the verdict.

    Comparative role (Lipton). Sees all scored candidates; picks the
    best and assigns gap-based confidence. Writes
    integrated_assessment / integrated_confidence / integrated_reasoning
    on the claim — the fields compute_posterior reads. Idempotent.

    Phase 4 of the Move-3 plan. After this node, ``compute_posterior``
    can produce a real posterior for any claim that reaches here. The
    no_stranded_claims invariant becomes statically satisfiable for
    every claim that successfully traverses this stage (it gets a
    verdict).
    """

    reads = frozenset({"objective_id", "ibe_agreement_k"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PromoteSupported":
        from ..operations.integration import SelectBestExplanationOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Phase 2c of the efficiency plan: same across-claims
        # parallelization as the other IBE stages.
        # See EnumerateCandidates filter — same cycle-capped exclusion.
        eligible_ids = [
            c.entity_id
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
            and c.integrated_assessment is None
            and c.integration_candidates
        ]
        if eligible_ids:
            # Pass ibe_agreement_k via metadata so the operation can
            # run the IBE chain K times and require agreement on
            # direction before committing a verdict. Fan out the LLM work;
            # commit to State in the deterministic join (Law 3).
            outcomes = await asyncio.gather(
                *(
                    _execute_op(
                        SelectBestExplanationOperation,
                        deps,
                        cid,
                        "claim",
                        "select_best_explanation",
                        metadata={"ibe_agreement_k": state.ibe_agreement_k},
                    )
                    for cid in eligible_ids
                )
            )
            for outcome in outcomes:
                await _commit_op(deps, state, outcome)

        return PromoteSupported()


@dataclass
class PromoteSupported(Node):
    """Try advancing claims beyond SUPPORTED: S->P, P->R, R->A, then record decisions.

    Phase 4 of the Move-3 plan. Marks each processed claim as
    verification_done so PromoteToSupported's idempotency check
    short-circuits on re-entry.

    Certify-before-contract (docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md
    D1): the contraction check that used to live in the TMS sweep
    (``_run_tms_sweep``, racing IBE for the same claim mid-verification)
    runs here instead, once, AFTER ``SelectBestExplanation`` has already
    produced this run's IBE verdict — so a claim whose evidence
    contradicts its stage is revised, not restarted, and never loses the
    verdict IBE already certified for it (S1, S8). A contracted claim
    does not climb the ladder this pass; its verdict stands as-is.

    D8: the ladder itself follows the verdict — only a claim IBE
    certified as ``integrated_assessment == "supports"`` attempts a
    ``promote_claim``. A claim with no verdict, or one certified
    ``contradicts``/``insufficient``, is marked ``verification_done``
    with no promotion attempt this pass (S6).
    """

    reads = frozenset({"objective_id"})
    writes = frozenset({"verification_done"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "CombineClaimVerdicts":
        from ..operations.stage_management import PromoteClaimOperation
        from ..operations.investigation import (
            GeneratePredictionOperation,
            RecordDecisionOperation,
        )
        from ..operations.belief_maintenance import RevalidateClaimOperation
        from ..entities.claim import ClaimStage
        from ..gates import get_next_stage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Certification-before-contraction: revalidate every eligible
        # claim's current stage gate BEFORE attempting to climb the
        # ladder. A claim whose gate now fails is contracted (verdict
        # preserved) and excluded from this pass's ladder climb — see
        # the class docstring.
        contracted_ids: set[str] = set()
        for claim in all_claims:
            if claim.abandoned or claim.cycle_capped:
                continue
            if claim.stage == ClaimStage.HYPOTHESIS:
                continue

            result = await _run_op(
                RevalidateClaimOperation,
                deps,
                state,
                claim.entity_id,
                "claim",
                "revalidate_claim",
            )
            if result.success and (result.message or "").startswith(
                "AGM contracted from"
            ):
                contracted_ids.add(claim.entity_id)
                state.verification_done.add(claim.entity_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if claim.entity_id in contracted_ids:
                continue

            # Ladder follows the verdict (certify-before-contract, D8):
            # only a claim IBE certified as "supports" climbs. A claim
            # with no verdict, or one IBE certified as "contradicts" /
            # "insufficient", is done for this pass without a
            # promote_claim attempt — climbing the stage ladder on a
            # claim IBE did not certify as supported would let the
            # ladder outrun the verdict it's supposed to follow.
            if claim.integrated_assessment != "supports":
                state.verification_done.add(claim.entity_id)
                continue

            # Try each promotion step once (S->P, P->R, R->A). The bound
            # is structural (L5): a claim can climb at most the height of
            # the finite ClaimStage ladder, so even a promote op that
            # reports success without advancing the stage cannot spin.
            current = claim.stage
            for _ in range(len(ClaimStage)):
                next_stage = get_next_stage(current)
                if next_stage is None:
                    break

                result = await _run_op(
                    PromoteClaimOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "promote_claim",
                )
                if not result.success:
                    break

                # Re-read to get updated stage
                claim = await deps.repo.get("claim", claim.entity_id)
                current = claim.stage

                # At ROBUST: generate predictions
                if current == ClaimStage.ROBUST and not claim.predictions_generated:
                    await _run_op(
                        GeneratePredictionOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "generate_prediction",
                    )

                # At ACTIONABLE: record decision
                if current == ClaimStage.ACTIONABLE and not claim.decision_recorded:
                    await _run_op(
                        RecordDecisionOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "record_decision",
                    )

            state.verification_done.add(claim.entity_id)

        return CombineClaimVerdicts()


@dataclass
class CombineClaimVerdicts(Node):
    """Build the run's single ``EpistemicVerdict`` from the live database.

    THE ONE PLACE the system decides. ``confidence.build_verdict`` calls
    ``compute_posterior`` for the number (so the rule-aware branch, the
    confidence-weighted branch, the counting fallback, the
    no-certified-verdict gate, the concordance tripwire and both
    penalties stay exactly where they are), then assembles the per-claim
    entries and evidence refs from the same helpers that produced it.
    Everything downstream — CheckCompletion's routing and reason text,
    CheckSynthesisDemand's four gates, the synthesis prose, the CLI, the
    offline report, the HTML audit report — RENDERS this object rather
    than re-deciding.

    It runs for EVERY run shape, not just decomposed ones. The previous
    node no-op'd without a decomposition, which meant verify-mode and
    open-research runs — the entire SciFact arm — produced no structured
    verdict at all and the writer agent's prose was the only verdict
    that existed.

    It is also the SOLE predecessor of ``CheckCompletion`` (see
    ``PromoteToSupported``'s retargeted no-work exit), so the object is
    never stale and never absent when a terminal decision is made. Safe
    to re-enter: it rebuilds from the DB rather than accumulating.

    Pure / deterministic — no LLM call. Empty operations: it calls
    ``build_verdict`` directly as a function, not via ``_run_op``.
    """

    reads = frozenset(
        {"objective_id", "retrieval_failed", "judgment_doubt_escalated"}
    )
    writes = frozenset({"verdict"})
    operations = frozenset()
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "CheckCompletion":
        from ..confidence import build_verdict

        state = ctx.state
        deps = ctx.deps

        state.verdict = await build_verdict(
            deps.repo,
            objective_id=state.objective_id,
            retrieval_failed=state.retrieval_failed,
            # Gate 0 runs AFTER this node, but this node re-runs on every
            # completion check, so the final persisted verdict carries the
            # true value.
            judgment_doubt_escalated=state.judgment_doubt_escalated,
        )

        if deps.verbose:
            prov = state.verdict.provenance
            logger.info(
                "Verdict: %s (posterior=%s, rule=%s, adjudicated=%d/%d, orphans=%d)",
                state.verdict.direction,
                state.verdict.posterior,
                prov.combination_rule,
                prov.n_adjudicated,
                prov.n_active_claims,
                prov.n_orphan,
            )

        return CheckCompletion()


@dataclass
class CheckCompletion(Node):
    """Check whether any non-abandoned claims remain and route accordingly.

    Phase 5 of the Move-3 plan. Read-only: this node inspects state
    and dispatches no operations. Many state fields are read because
    it constructs the EpistemicResult terminal value when no claims
    remain.
    """

    reads = frozenset(
        {
            "retrieval_failed",
            "objective_id",
            "verdict",
            "successful",
            "failed",
            "errors",
            "operations_log",
            "quarantined",
        }
    )
    writes = frozenset({"synthesis_insufficient_reason"})
    operations = frozenset()
    post_invariants = (no_stranded_claims, capped_claims_carry_a_cap_evaluation)

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["CheckSynthesisDemand", "SynthesizeInsufficient"]:
        state = ctx.state

        # Single canonical certifier (Phase A — 2026-05-06). The IBE
        # chain (EnumerateCandidates → ScoreLoveliness → ScoreLikeliness
        # → SelectBestExplanation) is the system's certifier of
        # directional verdicts. ``integrated_assessment is not None`` on
        # any active claim is the system's binary answer to "did we
        # adjudicate?" — counted once, on the verdict object, as
        # ``provenance.n_adjudicated``.
        #
        # This node used to query the claims itself and the posterior
        # aggregator applied the same condition independently, the two
        # kept in sync by convention (commit 761a5d9). They are now
        # literally the same object: ``build_verdict`` calls
        # ``compute_posterior``, whose no-certified-verdict gate fires on
        # exactly ``n_adjudicated == 0``. Writer-aggregator coherence is
        # structural rather than maintained.
        #
        # Routing rule:
        #   * n_adjudicated > 0 → CheckSynthesisDemand (the writer
        #     renders the directional verdict; retrieval_failed is
        #     annotated in prose but does not suppress it).
        #   * n_adjudicated == 0 → SynthesizeInsufficient with one of
        #     four reason variants: retrieval_failed > no_claims >
        #     all_abandoned > no_certified_verdict. The reason text
        #     describes WHY the system could not adjudicate; the
        #     terminal artefact is structurally identical regardless.
        if state.verdict is None:
            raise RuntimeError(
                "CheckCompletion reached with no verdict on graph state. "
                "CombineClaimVerdicts is its sole predecessor and must run "
                "first — reaching here without a verdict is a routing bug, "
                "not a state to route around."
            )

        prov = state.verdict.provenance

        if prov.n_adjudicated > 0:
            # IBE (or the deterministic cycle-cap reading) certified at
            # least one active claim. The system has an answer to report.
            return CheckSynthesisDemand()

        # No certified claim. Pick the most informative reason text,
        # every number sourced from the same provenance the aggregator used.
        if prov.retrieval_failed:
            state.synthesis_insufficient_reason = (
                "Retrieval failed: evidence extraction returned empty "
                "content for several consecutive attempts. The system "
                "could not gather fresh evidence to advance any claim, "
                "so synthesis would be invented from absent data. "
                "Likely causes: search providers returned no new results "
                "for the formulated queries; the question's evidence "
                "base is sparser than the providers can surface; or "
                "every newly-found item was a duplicate of evidence "
                "already invalidated upstream."
            )
        elif prov.n_active_claims:
            state.synthesis_insufficient_reason = (
                f"None of the {prov.n_active_claims} non-abandoned claim(s) "
                "reached IBE certification (integrated_assessment is None "
                f"on every active claim; {prov.n_cycle_capped} of them are "
                "cycle-capped, and for those the deterministic cap reading "
                "was not directional either — conflicting evidence, or too "
                "little of it). Without a certified verdict, neither the "
                "writer nor the posterior aggregator should produce a "
                "directional verdict — synthesis suspends judgment."
            )
        elif prov.n_claims_total:
            state.synthesis_insufficient_reason = (
                f"All {prov.n_claims_total} candidate claim(s) were abandoned "
                "during scrutiny. The system did not find any "
                "actionable evidence for any sub-investigation, so no "
                "directional verdict is supportable."
            )
        else:
            state.synthesis_insufficient_reason = (
                "No claims were created from the question. The "
                "decomposition / claim-creation step produced nothing "
                "investigable, so there is nothing to synthesize."
            )
        return SynthesizeInsufficient()


@dataclass
class CheckSynthesisDemand(Node):
    """Phase 4 of the lazy-escalation plan: ASK whether the synthesised
    verdict is satisfying, and if not, LOOP BACK to investigation.

    Phase 1 shipped this node in logging-only mode (always returned
    Synthesize). Phase 4 activates the loop-back: when the demand
    says ``needs_more=True`` AND there are claims that could benefit
    from more investigation, the node adds them to
    ``state.claims_needing_rescrutiny`` and routes back to Scrutinize.
    Otherwise it continues to Synthesize.

    Infinite-loop prevention: only claims that are non-abandoned,
    non-cycle-capped, AND haven't hit the per-claim
    ``SCRUTINY_RESOLVE_CYCLE_CAP`` are added to the rescrutiny set.
    When all eligible claims are terminal, no claims are added; the
    node falls through to Synthesize regardless of the demand. This
    means the loop-back terminates as soon as no claim can make
    progress, even if the satisfaction LLM keeps saying needs_more.

    Deterministic gates run first as cheap pre-filters; the LLM call
    only fires when the deterministic checks don't already determine
    satisfaction.

    Gate 0 (aggregate judgment doubt) runs before all of them. It is the
    only gate that asks about the QUALITY OF THE INPUTS rather than about
    the conclusion: when the evidence judgments behind a claim's verdict
    were internally unsure, it buys the objective one — and only one —
    extra round of inquiry. It fires at most once per run
    (``state.judgment_doubt_escalated``) and only over
    loop-back-eligible claims, so it can never route to the insufficient
    terminal.
    """

    reads = frozenset(
        {
            "objective_id",
            "verdict",
            "claims_needing_rescrutiny",
            "scrutiny_resolve_cycles",
            "judgment_doubt_escalated",
        }
    )
    writes = frozenset({"claims_needing_rescrutiny", "judgment_doubt_escalated"})
    operations = frozenset()  # uses agent_runner directly, not _run_op
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Synthesize", "Scrutinize", "SynthesizeInsufficient"]:
        from ..demand import Demand
        from ..entities.objective import Objective

        state = ctx.state
        deps = ctx.deps

        objective = await deps.repo.get("objective", state.objective_id)
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        active_claims = [c for c in all_claims if not c.abandoned]

        # ── Gate 0: aggregate judgment doubt ─────────────────────────
        # Deterministic, zero-LLM, and FIRST. Placement is load-bearing,
        # not stylistic: Gate 1 below short-circuits to "satisfied" for
        # every run without a combination rule — i.e. every verify-mode
        # and open-research run, exactly the runs this signal exists
        # for. A doubt gate placed after it would be dead code on the
        # whole SciFact arm. Running first also means it pre-empts the
        # expensive reproducibility tripwire in Gate 4, consistent with
        # this node's "deterministic gates first as cheap pre-filters"
        # contract.
        doubt_demand = self._judgment_doubt_demand(active_claims, state)
        if doubt_demand is not None:
            state.judgment_doubt_escalated = True
            self._log_demand(doubt_demand, deps, state.objective_id)
            # Structurally always Scrutinize(): the gate fires only when
            # the loop-back-eligible set is non-empty (both consult
            # ``_loopback_eligible``), so this can never manufacture a
            # false "insufficient".
            return self._maybe_loop_back(doubt_demand, active_claims, state)

        # ── Compute the demand ──────────────────────────────────────
        # Every path below produces a ``demand`` variable. The single
        # loop-back decision at the end of this method consumes it.
        # This keeps Phase 4's "needs_more → loop back" logic in one
        # place rather than scattered across each gate.
        #
        # All four gates read ``state.verdict`` — the same object
        # CheckCompletion routed on and the writer will render. They used
        # to read ``objective.decomposition.combined_verdict``, a
        # decomposed-runs-only field, which is why the gates below could
        # never see a verify-mode run's numbers.
        verdict = state.verdict
        prov = verdict.provenance if verdict is not None else None

        demand: Any = None  # set by exactly one branch below

        # Gate 1: no combination rule (verify mode / open research). The
        # rule-aware satisfaction check doesn't apply; the writer agent
        # frames the answer over the per-claim narrative. Treat as
        # satisfied for this layer.
        #
        # DELIBERATELY PRESERVED, not widened. The old condition was
        # "objective.decomposition is None"; ``combination_rule is None``
        # is its exact analogue on the verdict object (a decomposition
        # without a rule already fell through to ``or "AND"`` downstream).
        # Opening Gates 2-4 to verify mode would activate the decisive
        # gate, the reproducibility tripwire and the demand LLM across the
        # whole SciFact arm — a routing and LLM-cost change that needs a
        # benchmark this task forbids. Flagged, not made.
        if (
            verdict is None
            or prov is None
            or not isinstance(objective, Objective)
            or prov.combination_rule is None
        ):
            demand = Demand.satisfied(
                justification=(
                    "Open-research / verify mode (no combination rule) — "
                    "synthesis demand check does not apply. Synthesis writer "
                    "will frame the answer over per-claim verdicts directly."
                )
            )
            self._log_demand(demand, deps, state.objective_id)
            return Synthesize()

        # Gate 2: nothing was aggregated (every claim abandoned, or none
        # carried an integration verdict), so the headline answer is the
        # no-data fallback.
        if prov.n_adjudicated == 0 or verdict.posterior is None:
            demand = Demand.needs(
                justification=(
                    "No combined verdict produced (every claim was "
                    "abandoned, cycle-capped, or had no integration "
                    "verdict). Without aggregated per-claim posteriors "
                    "the headline answer is the no-data fallback."
                )
            )
        # Gate 3: stranded claims present (the load-bearing invariant
        # caught by no_stranded_claims). Defensive — this should never
        # happen at synthesis if upstream routing is correct.
        #
        # ``prov.n_no_verdict`` is scoped to the COMBINATION SCOPE, not
        # to every active claim, precisely so this gate keeps the
        # pre-verdict-object routing: it counts the decomposition-matched
        # claims the combiner would have dropped, exactly as
        # ``combine_claim_verdicts(ordered, ...).n_no_verdict`` did. See
        # ``confidence.build_verdict`` for why widening it here would be
        # an unbenchmarked routing/LLM-cost change of the same kind Gate 1
        # above declines to make.
        #
        # KNOWN, DEFERRED interaction with Stage 3 (PRINCIPLE_FIDELITY_
        # PLAN.md §5): this node is reached only when ``n_adjudicated >
        # 0`` (CheckCompletion routes n_adjudicated==0 to
        # SynthesizeInsufficient instead), so Gate 3 only fires on a
        # MULTI-claim objective where at least one claim adjudicated and
        # at least one other did not. Since Stage 3, a claim can
        # legitimately be in the latter set via
        # EnumerateCandidatesOperation's "insufficient explanatory
        # diversity" decline — not only via the routing regression this
        # gate's comment describes. The demand below still loops back to
        # Scrutinize for such a claim (bounded by PEIRCE_CYCLE_CAP, so
        # it cannot loop forever — worst case the claim eventually
        # cycle-caps and gets a deterministic terminal verdict from
        # ConcludeAtCycleCapOperation), but the ``target_hint`` below
        # ("likely a routing regression") is misleading for that specific
        # cause. Not reachable at all in single-claim seed-verify mode
        # (SciFact-shaped runs), which is this task's own verification
        # scope; distinguishing the two causes for multi-claim runs is
        # flagged here as a real, open follow-up rather than fixed
        # unilaterally in this pass.
        elif prov.n_no_verdict > 0:
            demand = Demand.needs(
                justification=(
                    f"{prov.n_no_verdict} claim(s) reached synthesis "
                    "without an integration verdict — IBE was bypassed "
                    "for them. The combined posterior excludes their "
                    "evidence, so the headline answer is incomplete."
                ),
                target_hint=(
                    "Investigate why claims were stranded at SUPPORTED "
                    "without integrated_assessment — likely a routing "
                    "regression."
                ),
            )
        # Gate 4: decisive posterior (above POSTERIOR_DECISIVE_THRESHOLD,
        # or symmetrically below ``1 - threshold``). See
        # epistemic.thresholds for the asymmetric-stakes motivation.
        elif (
            verdict.posterior >= POSTERIOR_DECISIVE_THRESHOLD
            or verdict.posterior <= 1.0 - POSTERIOR_DECISIVE_THRESHOLD
        ):
            posterior = verdict.posterior
            direction = (
                "supports"
                if posterior >= POSTERIOR_DECISIVE_THRESHOLD
                else "contradicts"
            )
            # Reproducibility tripwire (P7, demand-gated): a decisive verdict is
            # exactly the load-bearing moment to check that the confident
            # (one-hot) judgments underpinning it are not paraphrase-brittle —
            # the one error class a single verbalized judgment is structurally
            # blind to. Bounded + strictly sequential; only fires here, at the
            # decisive gate. A detected flip withholds the decisive shortcut and
            # loops back for more inquiry (suspend-only — it never asserts a new
            # direction).
            repro = None
            if deps.agent_runner is not None:
                from ..reproducibility import check_claims_reproducibility

                repro = await check_claims_reproducibility(
                    active_claims, deps.repo, deps.agent_runner
                )
            if repro is not None and repro.flip_found:
                demand = Demand.needs(
                    justification=(
                        f"Combined posterior {posterior:.3f} ({direction}) "
                        "looked decisive, but a reproducibility check found a "
                        "confident evidence judgment that flips under paraphrase. "
                        + repro.detail
                    ),
                    target_hint=(
                        "Re-scrutinise the claim whose one-hot judgment was not "
                        "reproducible before finalising a decisive verdict."
                    ),
                )
            else:
                demand = Demand.satisfied(
                    justification=(
                        f"Combined posterior {posterior:.3f} ({direction}) "
                        f"is decisive; the verdict direction is clear from the "
                        f"per-claim verdicts ({prov.combination_rule} over "
                        f"{prov.n_adjudicated} "
                        "aggregated claims). No further investigation likely "
                        "to change the headline."
                        + (
                            f" Reproducibility check: {repro.checks_run} "
                            "one-hot judgment(s) held under paraphrase."
                            if repro is not None and repro.checks_run > 0
                            else ""
                        )
                    )
                )
        # ── LLM judgment ─────────────────────────────────────────────
        # Deterministic gates didn't determine the answer — ask the
        # check_synthesis_demand agent for a judgment.
        elif deps.agent_runner is None:
            # No runner — can't ask the LLM. Default to satisfied so we
            # don't block synthesis. The deterministic gates above
            # already caught the load-bearing cases.
            demand = Demand.satisfied(
                justification="No agent runner available for LLM satisfaction check; "
                "deterministic gates passed."
            )

        else:
            # Build a compact picture for the agent. Per-claim summaries
            # plus the combined verdict. Keep it short — the agent's
            # job is judgment, not re-reading every detail.
            claim_summaries: list[str] = []
            for c in active_claims:
                claim_verdict = c.integrated_assessment or "no-verdict"
                confidence = (
                    f"{c.integrated_confidence:.2f}"
                    if c.integrated_confidence is not None
                    else "—"
                )
                claim_summaries.append(
                    f"  [{c.sub_investigation_id or '-'}] "
                    f"verdict={claim_verdict} (conf={confidence}): {c.statement[:120]}"
                )

            try:
                demand = await deps.agent_runner.run(
                    "epistemic_check_synthesis_demand",
                    research_question=(
                        objective.clarified_question or objective.description
                    ),
                    combined_verdict_label=verdict.direction,
                    combined_posterior=(
                        f"{verdict.posterior:.3f}"
                        if verdict.posterior is not None
                        else "n/a (UNION rule or no aggregated claims)"
                    ),
                    combination_rule=prov.combination_rule,
                    claim_count=len(active_claims),
                    claims_summary="\n".join(claim_summaries) or "(none)",
                    combiner_explanation=prov.combination_explanation,
                )
            except Exception as e:
                # Agent failed — log and treat as satisfied (don't block
                # synthesis on a satisfaction-check failure).
                logger.warning(
                    "CheckSynthesisDemand agent failed (%s: %s); treating "
                    "as satisfied to not block synthesis",
                    type(e).__name__,
                    e,
                )
                demand = Demand.satisfied(
                    justification=f"Satisfaction agent failed: {type(e).__name__}"
                )

        self._log_demand(demand, deps, state.objective_id)

        # Phase 4 of the lazy-escalation plan: ACTIVATE the loop-back.
        # When demand says needs_more, identify claims that could
        # benefit from more investigation and route back to Scrutinize
        # via the rescrutiny set. When no claim can make progress
        # (all eligible claims are terminal or at the per-claim cap),
        # accept the current state and synthesize.
        if not demand.needs_more:
            return Synthesize()

        return self._maybe_loop_back(demand, active_claims, state)

    @staticmethod
    def _judgment_doubt_demand(
        active_claims: list[Any],
        state: EpistemicGraphState,
    ) -> Any:
        """A ``Demand`` when the judgments behind a claim's verdict were
        internally unsure enough to warrant one more round — else ``None``.

        The pipeline makes many sampled LLM judgments per claim and then feeds
        them to threshold-based gates, so small judgment noise becomes
        categorical output flips. This gate turns the measured version of that
        noise into the system's existing "do more work" primitive — a
        ``Demand`` routed through the lazy-escalation loop (P7) — rather than
        into a new mechanism.

        ONE GATE, THREE INPUTS. A claim escalates when ANY of:

        * ``scrutiny_doubt`` — the directional-mass-weighted mean entropy of
          the evidence judgments that produced the verdict, at or above
          ``JUDGMENT_DOUBT_ESCALATION_THRESHOLD``. The judges hedged.
        * ``judgment_dissent_count > 0`` — an INDEPENDENT judge contradicted
          the direction the primary judge committed to on at least one
          load-bearing item. This cannot travel on ``scrutiny_doubt``: a
          confident dissent has entropy near zero, so folding it into an
          entropy-shaped number would report "no doubt" about exactly the item
          most likely to be wrong. It is a separate signal because it measures
          a separate thing.
        * ``scrutiny_opinion`` dissenting or itself doubtful — the
          claim-level independent read on a borderline deterministic weight.

        The three cover disjoint blind spots: the first is silent on the
        53-82% of small-model judgments that come back one-hot, which is
        exactly where the second speaks; the third speaks about the reading
        rather than the judgments.

        Two further conditions, both necessary:

        1. The escalation has not already fired this run
           (``state.judgment_doubt_escalated``). Doubt buys ONE extra round,
           not a budget: it is a suspicion-raiser, and without this bound a
           persistently-unsure claim would spend its whole scrutiny-resolve
           budget on noise. This one flag is also why widening the predicate
           cannot widen the cost: however many signals fire, the whole
           mechanism spends at most one extra scrutiny round per run.
        2. At least one claim is ``_loopback_eligible`` — the same predicate
           ``_maybe_loop_back`` uses, so the escalation can never route to
           ``SynthesizeInsufficient``.

        ``None`` and ``0`` never escalate: an unmeasured claim (no eligible
        item carried a verbalized distribution, no confirmation ran, the
        reading was not borderline) is not a doubtful one, and treating it as
        such would escalate on every pre-Tier-0 and adversarial-only claim.

        Note the deliberate asymmetry against ``quality_doubt``'s two destroy
        gates: there only a CONFIDENT "fails" acts, because the action is
        destruction — a doubtful judge leaves the evidence standing. Here
        both a confident dissent and a doubtful agreement escalate, because
        the action is one bounded extra round of inquiry — cheap, and never
        destructive.

        Pure and LLM-free. It reads persisted state only, which is why it is
        deliberately independent of the run's verdict object — it asks about
        the quality of the inputs, not about the conclusion drawn from them.
        """
        from ..demand import Demand

        if state.judgment_doubt_escalated:
            return None

        def _reasons(c: Any) -> list[str]:
            """Which signal(s) fired on this claim — named, so the demand's
            justification stays actionable rather than saying 'doubt'."""
            found: list[str] = []
            doubt = getattr(c, "scrutiny_doubt", None)
            if doubt is not None and doubt >= JUDGMENT_DOUBT_ESCALATION_THRESHOLD:
                found.append(f"aggregate judgment doubt {doubt:.2f}")
            dissent_count = getattr(c, "judgment_dissent_count", 0) or 0
            if dissent_count > 0:
                found.append(
                    f"{dissent_count} evidence direction(s) contradicted by an "
                    f"independent judge"
                )
            opinion = getattr(c, "scrutiny_opinion", None)
            if opinion is not None and opinion.dissents:
                found.append(
                    f"an independent judge read the evidence as "
                    f"'{opinion.overall}' against a '{opinion.reference}' weight"
                )
            elif opinion is not None and opinion.needs_review:
                found.append(
                    f"the independent read of the evidence was itself unsure "
                    f"(doubt {opinion.doubt:.2f})"
                )
            return found

        doubtful: list[tuple[Any, list[str]]] = []
        for c in active_claims:
            if not _loopback_eligible(c, state):
                continue
            reasons = _reasons(c)
            if reasons:
                doubtful.append((c, reasons))
        if not doubtful:
            return None

        detail = "; ".join(
            f"{c.statement[:80]} ({', '.join(reasons)})" for c, reasons in doubtful[:3]
        )
        more = f" and {len(doubtful) - 3} more" if len(doubtful) > 3 else ""
        return Demand.needs(
            justification=(
                f"{len(doubtful)} claim(s) hold a verdict whose underlying "
                f"evidence judgments did not survive scrutiny of their own "
                f"reliability: {detail}{more}. "
                f"The direction of these verdicts rests on judgments that "
                f"barely committed, or that an independent judge disagreed "
                f"with, so one more round of inquiry may change the headline."
            ),
            target_hint=(
                "Gather evidence that would let a judge commit — clearer, "
                "more directly on-point sources for the claims named above, "
                "rather than more of the same."
            ),
        )

    def _maybe_loop_back(
        self,
        demand: Any,
        active_claims: list[Any],
        state: EpistemicGraphState,
    ) -> Union["Synthesize", "Scrutinize", "SynthesizeInsufficient"]:
        """Decide whether to route back to Scrutinize for another
        round of investigation, or accept the current state.

        We loop back when there's at least one claim that's
        non-abandoned, non-cycle-capped, and hasn't hit the per-claim
        ``SCRUTINY_RESOLVE_CYCLE_CAP`` — the shared module-level
        ``_loopback_eligible`` predicate. Those are the claims for which
        another round of inquiry could plausibly change the outcome.

        When NO claim can make progress (all eligible claims are
        terminal or at the per-claim cap) AND the demand still says
        ``needs_more``, the rational structural response is *not* to
        invent a directional verdict — it's to synthesise an
        insufficient artefact. Route to ``SynthesizeInsufficient``
        rather than ``Synthesize``. The demand justification is
        carried via state so the deterministic body can surface the
        gate's diagnosis. This is the system's fallibilism mode: it
        suspends judgment instead of fabricating one.

        Per-claim cap is the load-bearing safety on the loop-back
        side: claims that are at cap don't get re-added to rescrutiny,
        so we always reach a terminal state.
        """
        eligible_claim_ids = [
            c.entity_id for c in active_claims if _loopback_eligible(c, state)
        ]

        if not eligible_claim_ids:
            # All non-abandoned claims are either cycle-capped or at
            # cap. No re-investigation can make progress.
            #
            # The second false-insufficient door. If any active claim
            # already carries a DIRECTIONAL verdict — which a capped
            # claim now can, via ConcludeAtCycleCapOperation — routing
            # to SynthesizeInsufficient would discard a conclusion the
            # evidence supports and CheckCompletion just certified. The
            # honest terminal in that case is Synthesize: the writer
            # renders the verdict the system holds, however hedged.
            #
            # DIRECTIONAL, not merely non-None: a claim whose verdict is
            # "insufficient" is faithfully rendered by the insufficient
            # terminal, and that is exactly the shape of the existing
            # all-capped fixture in test_phase4_synthesis_loop_back.py.
            #
            # Read off the verdict object rather than the claim entities,
            # so this gate and the headline the writer renders agree by
            # construction.
            directional = [
                c
                for c in (state.verdict.claims if state.verdict else [])
                if not c.abandoned
                and c.integrated_assessment in ("supports", "contradicts")
            ]
            if directional:
                logger.warning(
                    "[synthesis_demand] obj=%s needs_more=True and no claim "
                    "can progress, but %d active claim(s) carry a directional "
                    "verdict; synthesising it rather than suspending. "
                    "justification: %s",
                    state.objective_id[:12] if state.objective_id else "????????????",
                    len(directional),
                    demand.justification,
                )
                return Synthesize()

            # No directional verdict anywhere. Route to the
            # structurally-explicit "insufficient" terminal rather than
            # asking the writer to invent a verdict.
            logger.warning(
                "[synthesis_demand] obj=%s needs_more=True but all "
                "non-abandoned claims have hit per-claim cap; "
                "routing to SynthesizeInsufficient. justification: %s",
                state.objective_id[:12] if state.objective_id else "????????????",
                demand.justification,
            )
            state.synthesis_insufficient_reason = demand.justification
            return SynthesizeInsufficient()

        # Add the eligible claims to the rescrutiny set so Scrutinize
        # picks them up. This composes with the existing scrutiny ↔
        # resolve loop, which already has its own cycle cap (incremented
        # in Scrutinize's predicate when the claim is dequeued).
        for cid in eligible_claim_ids:
            state.claims_needing_rescrutiny.add(cid)

        logger.warning(
            "[synthesis_demand] obj=%s looping back to Scrutinize on %d "
            "eligible claim(s); justification: %s",
            state.objective_id[:12] if state.objective_id else "????????????",
            len(eligible_claim_ids),
            demand.justification,
        )
        return Scrutinize()

    @staticmethod
    def _log_demand(demand: Any, deps: EpistemicDeps, objective_id: str = "") -> None:
        """Emit the synthesis demand at WARNING level so it shows
        through the CLI's verbose-mode log filter (which suppresses
        INFO during runs to keep the rich-progress output clean).
        These logs are also the audit trail for the satisfaction
        check's calibration — they MUST be visible.

        The log level isn't a "this is alarming" signal; it's just
        the only level the CLI keeps during runs. When the CLI's
        logging policy changes, this can drop back to INFO.

        ``objective_id`` is logged as a per-case attribution tag —
        when N cases run in parallel under snakemake their stderr
        interleaves into one log file, and the tag is the only way
        to attribute a demand to a specific case after the fact.

        ``deps`` is unused today but kept on the signature so future
        per-run instrumentation (e.g. write the demand to a trace
        file in deps) doesn't change every call site."""
        del deps  # currently unused; reserved for trace-file emission
        oid_tag = objective_id[:12] if objective_id else "????????????"
        logger.warning(
            "[synthesis_demand] obj=%s needs_more=%s | %s%s",
            oid_tag,
            demand.needs_more,
            demand.justification,
            f" | hint: {demand.target_hint}" if demand.target_hint else "",
        )


@dataclass
class Synthesize(Node):
    """Freeze snapshot and synthesize the final report.

    Phase 5 of the Move-3 plan. The terminal node before End[...].
    Carries the load-bearing ``no_stranded_claims`` invariant: by the
    time the graph reaches Synthesize, no Claim should be at SUPPORTED
    with integrated_assessment=None and not in verification_done.
    Violation means the IBE chain was bypassed for a claim that
    should have entered it — the recurring routing-bug class.
    """

    reads = frozenset(
        {
            "objective_id",
            "verdict",
            "successful",
            "failed",
            "errors",
            "operations_log",
            "quarantined",
            "retrieval_failed",
        }
    )
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = (no_stranded_claims, capped_claims_carry_a_cap_evaluation)

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> End[EpistemicResult]:
        from ..operations.synthesis import (
            FreezeSnapshotOperation,
            SynthesizeReportOperation,
        )

        state = ctx.state
        deps = ctx.deps
        oid = state.objective_id

        # Freeze snapshot — the verdict travels on the typed channel so
        # the frozen Snapshot carries the decision the writer will render.
        await _run_op(
            FreezeSnapshotOperation,
            deps,
            state,
            oid,
            "objective",
            "freeze_snapshot",
            verdict=state.verdict,
        )

        # Get the snapshot ID from the objective
        obj = await deps.repo.get("objective", oid)

        if obj.snapshot_id:
            await _run_op(
                SynthesizeReportOperation,
                deps,
                state,
                obj.snapshot_id,
                "snapshot",
                "synthesize_report",
            )

        obj = await deps.repo.get("objective", oid)
        obj.phase = "complete"
        obj.status = "completed"
        await deps.repo.save(obj)

        return End(
            EpistemicResult(
                objective_id=oid,
                status="complete",
                successful=state.successful,
                failed=state.failed,
                errors=state.errors,
                operations_log=state.operations_log,
                termination_reason="complete",
                quarantined=state.quarantined,
                retrieval_failed=state.retrieval_failed,
            )
        )


@dataclass
class SynthesizeInsufficient(Node):
    """Terminal node for the structurally-insufficient case.

    Companion to ``Synthesize``. Reached when ``CheckSynthesisDemand``
    determines that more work is needed (``demand.needs_more=True``)
    but no claim is eligible for further investigation (per-claim cap
    fired across the board). The system has structural visibility that
    a directional verdict would be invented; routing here makes that
    visibility a topology property rather than relying on the writer
    agent's prompt to produce a coherent "insufficient" answer.

    Encodes Peirce's fallibilism, AGM's contraction operation, and
    Lipton's IBE precondition (no candidates → no inference) as a
    first-class graph state. Downstream consumers distinguish this
    from a directional verdict by reading ``Artefact.artefact_type``
    (``"insufficient"`` vs ``"summary"``) and ``EpistemicResult.status``
    (``"insufficient"`` vs ``"complete"``) — typed signals, not parsed
    prose.

    Carries the same ``no_stranded_claims`` invariant as ``Synthesize``:
    a stranded claim is a routing bug regardless of which terminal the
    graph reaches.
    """

    reads = frozenset(
        {
            "objective_id",
            "successful",
            "failed",
            "errors",
            "operations_log",
            "quarantined",
            "retrieval_failed",
            "synthesis_insufficient_reason",
            "verdict",
        }
    )
    # Stamps provenance.suspended_judgment onto the verdict — the terminal the
    # graph chose, which no renderer can re-derive from the counts (P2/P6).
    writes = frozenset({"verdict"})
    operations = frozenset()  # populated below
    post_invariants = (no_stranded_claims, capped_claims_carry_a_cap_evaluation)

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> End[EpistemicResult]:
        from ..operations.synthesis import (
            FreezeSnapshotOperation,
            SynthesizeInsufficientReportOperation,
        )

        state = ctx.state
        deps = ctx.deps
        oid = state.objective_id

        # Stamp the chosen terminal onto the verdict BEFORE the freeze, so the
        # snapshot and every renderer agree. The terminal is the one piece of
        # provenance a renderer cannot re-derive — a suspended run and a
        # genuinely inconclusive one carry identical counts — and P2 makes this
        # the graph's call, not an operation's. Without it, audit_report and the
        # CLI headlined "Inconclusive — posterior 0.500" above a body reading
        # "No posterior — no IBE-certified verdict."
        if state.verdict is not None:
            state.verdict.provenance.suspended_judgment = True

        await _run_op(
            FreezeSnapshotOperation,
            deps,
            state,
            oid,
            "objective",
            "freeze_snapshot",
            verdict=state.verdict,
        )

        obj = await deps.repo.get("objective", oid)

        if obj.snapshot_id:
            # The deterministic builder reads the structural diagnosis
            # off the operation's metadata bag — the gate wrote it onto
            # state.synthesis_insufficient_reason before routing here.
            # Carrying it through metadata keeps the operation a pure
            # transform: same input shape every time, the body just
            # varies with the supplied reason.
            await _run_op(
                SynthesizeInsufficientReportOperation,
                deps,
                state,
                obj.snapshot_id,
                "snapshot",
                "synthesize_insufficient",
                metadata={
                    "synthesis_insufficient_reason": (
                        state.synthesis_insufficient_reason or ""
                    ),
                },
            )

        obj = await deps.repo.get("objective", oid)
        obj.phase = "complete"
        obj.status = "completed"
        await deps.repo.save(obj)

        return End(
            EpistemicResult(
                objective_id=oid,
                status="insufficient",
                successful=state.successful,
                failed=state.failed,
                errors=state.errors,
                operations_log=state.operations_log,
                termination_reason="insufficient_evidence",
                quarantined=state.quarantined,
                retrieval_failed=state.retrieval_failed,
            )
        )


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

# The node roster. Transitions between these are still declared solely
# by each node's ``run()`` return annotation — ``GraphBuilder.node()``
# reads the same annotation ``topology()`` reflects over, so there is
# still exactly one source of truth for the edges.
NODE_TYPES: tuple[type[Node], ...] = (
    PrepareObjective,
    Decompose,
    PlanEvidence,
    ExtractEvidence,
    CreateClaims,
    Scrutinize,
    Investigate,
    ExtractNewEvidence,
    AbandonOrDemote,
    PromoteToSupported,
    ClusterEvidence,
    RunVerification,
    ResolveUncertainties,
    EnumerateCandidates,
    ScoreLoveliness,
    ScoreLikeliness,
    SelectBestExplanation,
    PromoteSupported,
    CombineClaimVerdicts,
    CheckCompletion,
    CheckSynthesisDemand,
    Synthesize,
    SynthesizeInsufficient,
)


def _build_epistemic_graph() -> Graph[
    EpistemicGraphState, EpistemicDeps, Node, EpistemicResult
]:
    """Assemble the pipeline graph.

    pydantic-graph v2 removed the ``Graph(nodes=[...])`` constructor —
    a ``Graph`` is now a built artefact produced by ``GraphBuilder``.
    The nodes themselves are unchanged ``BaseNode`` subclasses;
    ``builder.node()`` integrates each one by reading its ``run()``
    return annotation, which is where the topology has always lived.

    v2 also fixes the entry point at build time rather than accepting
    it per ``run()`` call. The stage runner needs a *caller-chosen*
    entry (``run_epistemic_graph(start_at=...)`` resumes mid-pipeline
    off the DB checkpoint), so the graph input is the entry node
    instance and the start node routes to it through a decision with
    one branch per node type. That preserves the v1 behaviour of
    ``graph.run(SomeNode(), ...)`` exactly.
    """
    builder = GraphBuilder[EpistemicGraphState, EpistemicDeps, Node, EpistemicResult](
        name="epistemic_pipeline",
        state_type=EpistemicGraphState,
        deps_type=EpistemicDeps,
        input_type=Node,
        output_type=EpistemicResult,
    )

    for node_type in NODE_TYPES:
        builder.add(builder.node(node_type))

    # ``HandledT`` widens one branch at a time, so the accumulator is
    # annotated ``Any``; the branch list below is exhaustive over
    # NODE_TYPES by construction.
    entry: Decision[EpistemicGraphState, EpistemicDeps, Any] = builder.decision(
        note="entry node supplied by the caller (start_at)",
        node_id="entry",
    )
    for node_type in NODE_TYPES:
        entry = entry.branch(builder.match_node(node_type))
    builder.add(builder.edge_from(builder.start_node).to(entry))

    return builder.build()


epistemic_graph: Graph[EpistemicGraphState, EpistemicDeps, Node, EpistemicResult] = (
    _build_epistemic_graph()
)


# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT METADATA — populated after class definitions
# ══════════════════════════════════════════════════════════════════════════════
#
# Phase 2 of the Move-3 plan. Contract metadata for nodes inheriting
# from ``Node`` (in ``base.py``) — populated here, after class
# definitions, so the operations imports can stay lazy at the run()
# call sites and avoid circular imports.
#
# Pattern: each class declared a metadata field with an empty
# frozenset() default; we replace it here with the populated set
# referencing the actual operation classes.
#
# When more nodes migrate in later phases, this block grows.

from ..operations.cleanup import AbandonStaleClaimOperation  # noqa: E402
from ..operations.belief_maintenance import (  # noqa: E402
    RevalidateClaimOperation,
    SetRoutingDefaultsOperation,
)
from ..operations.concerns import (  # noqa: E402
    DeduplicateConcernsOperation,
)
from ..operations.investigation import (  # noqa: E402
    InvestigateClaimOperation,
)
from ..operations.scrutiny import (  # noqa: E402
    ScrutiniseClaimOperation,
)
from ..operations.stage_management import (  # noqa: E402
    ConcludeAtCycleCapOperation,
    DemoteClaimOperation,
    PromoteAsRefutedOperation,
    PromoteClaimOperation,
    SoftPromoteOperation,
)
from ..operations.uncertainty import (  # noqa: E402
    ResolveUncertaintyOperation,
)


AbandonOrDemote.operations = frozenset(
    {
        PromoteAsRefutedOperation,
        SoftPromoteOperation,
        AbandonStaleClaimOperation,
        DemoteClaimOperation,
    }
)


PromoteToSupported.operations = frozenset(
    {
        PromoteClaimOperation,
        SetRoutingDefaultsOperation,
    }
)


# Both cap-firing sites dispatch ConcludeAtCycleCapOperation via the
# shared ``_mark_cycle_capped`` funnel, so both declare it.
Scrutinize.operations = frozenset(
    {ScrutiniseClaimOperation, ConcludeAtCycleCapOperation}
)


Investigate.operations = frozenset({InvestigateClaimOperation})


ResolveUncertainties.operations = frozenset(
    {
        ResolveUncertaintyOperation,
        DeduplicateConcernsOperation,
        ConcludeAtCycleCapOperation,
    }
)


# ── Phase 4: verification + IBE chain ────────────────────────────────

from ..operations.verification import (  # noqa: E402
    AdversarialSearchOperation,
    AssessConvergenceOperation,
    ValidateDeductivelyOperation,
    VerifyComputationallyOperation,
)
from ..operations.analysis import (  # noqa: E402
    AnalyzeArgumentOperation,
    ContrastiveEvaluationOperation,
    CrossClaimConsistencyOperation,
)
from ..operations.integration import (  # noqa: E402
    EnumerateCandidatesOperation,
    ScoreLikelinessOperation,
    ScoreLovelinessOperation,
    SelectBestExplanationOperation,
)
from ..operations.investigation import (  # noqa: E402
    GeneratePredictionOperation,
    RecordDecisionOperation,
)


# ClusterEvidence has no _run_op dispatches (uses select_top_k_evidence
# directly as a function); leaves operations as the empty frozenset.

RunVerification.operations = frozenset(
    {
        AdversarialSearchOperation,
        AssessConvergenceOperation,
        ValidateDeductivelyOperation,
        VerifyComputationallyOperation,
        AnalyzeArgumentOperation,
        ContrastiveEvaluationOperation,
        CrossClaimConsistencyOperation,
    }
)


EnumerateCandidates.operations = frozenset({EnumerateCandidatesOperation})


ScoreLoveliness.operations = frozenset({ScoreLovelinessOperation})


ScoreLikeliness.operations = frozenset({ScoreLikelinessOperation})


SelectBestExplanation.operations = frozenset({SelectBestExplanationOperation})


PromoteSupported.operations = frozenset(
    {
        RevalidateClaimOperation,
        PromoteClaimOperation,
        GeneratePredictionOperation,
        RecordDecisionOperation,
    }
)


# ── Phase 5: pre-claim phases + terminals ────────────────────────────

from ..operations.preplanning import (  # noqa: E402
    ClarifyQuestionOperation,
    ClassifyQuestionOperation,
    ConceptualAnalysisOperation,
    DecomposeQuestionOperation,
)
from ..operations.evidence import (  # noqa: E402
    ExtractEvidenceOperation,
)
from ..operations.dispatch_gather import (  # noqa: E402
    DispatchGatherOperation,
)
from ..operations.seed_claim import SeedClaimOperation  # noqa: E402
from ..operations.multi_seed_claim import (  # noqa: E402
    MultiSeedClaimOperation,
)
from ..operations.claims import ProposeClaimsOperation  # noqa: E402
from ..operations.synthesis import (  # noqa: E402
    FreezeSnapshotOperation,
    SynthesizeInsufficientReportOperation,
    SynthesizeReportOperation,
)


PrepareObjective.operations = frozenset(
    {
        ClarifyQuestionOperation,
        ClassifyQuestionOperation,
        ConceptualAnalysisOperation,
    }
)


Decompose.operations = frozenset({DecomposeQuestionOperation})


PlanEvidence.operations = frozenset({DispatchGatherOperation})


ExtractEvidence.operations = frozenset({ExtractEvidenceOperation})


CreateClaims.operations = frozenset(
    {
        SeedClaimOperation,
        MultiSeedClaimOperation,
        ProposeClaimsOperation,
    }
)


ExtractNewEvidence.operations = frozenset({ExtractEvidenceOperation})


# CombineClaimVerdicts has empty operations — it uses
# combine_claim_verdicts() as a direct function call, not _run_op.

# CheckCompletion has empty operations — it's read-only routing logic.

Synthesize.operations = frozenset({FreezeSnapshotOperation, SynthesizeReportOperation})


SynthesizeInsufficient.operations = frozenset(
    {FreezeSnapshotOperation, SynthesizeInsufficientReportOperation}
)
