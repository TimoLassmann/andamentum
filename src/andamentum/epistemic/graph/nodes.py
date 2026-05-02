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

from pydantic_graph import End, Graph, GraphRunContext

from .base import Node
from .invariants import no_stranded_claims
from .state import EpistemicGraphState
from .deps import EpistemicDeps
from .result import EpistemicResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


_EMPTY_EXTRACTION_THRESHOLD = 3

# Cap on Scrutinize ↔ ResolveUncertainties oscillation per claim.
# Each pass of Scrutinize on a marked-for-rescrutiny claim counts as
# one cycle. Once a claim hits this many cycles, Scrutinize stops
# rescrutinizing and ResolveUncertainties stops requesting rescrutiny.
# The claim retains its current verdict — analogous to the
# investigation cap of 3. The bug this prevents: claims whose seed
# text produces a reliably-novel-but-near-duplicate uncertainty per
# scrutiny pass loop forever, since the genuine resolution per round
# keeps marking the claim for rescrutiny.
SCRUTINY_RESOLVE_CYCLE_CAP = 3


async def _mark_cycle_capped(
    deps: "EpistemicDeps", claim_id: str, source: str
) -> None:
    """Mark a claim as cycle-capped and snapshot its current blocking
    uncertainties as forensic evidence.

    Idempotent: ``Claim.cycle_capped`` is set True (no-op if already
    True) and ``persistent_concerns`` is captured *only on first cap
    firing* — the first-firing snapshot is the diagnostic input for
    deciding follow-up architecture (cluster-dedup vs reformulation).
    Subsequent cap firings on the same claim leave the snapshot alone.

    The cap fires from two sites — ResolveUncertainties (primary
    loop-breaker, blocks the rescrutiny add) and Scrutinize (defense
    in depth, discards a stale rescrutiny flag). Both call this
    helper so the consequence is uniform.
    """
    from ..entities import Claim, Uncertainty

    claim = await deps.repo.get("claim", claim_id)
    if not isinstance(claim, Claim):
        return
    if claim.cycle_capped:
        # Already marked on a prior cap firing — keep first-firing snapshot.
        return
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


def _update_retrieval_health(state: EpistemicGraphState, evidence: Any) -> None:
    """Update retrieval-health counters based on one extraction outcome.

    Empty extraction (``extracted_content`` is falsy) increments the
    consecutive-empty counter; non-empty resets it to zero. When the
    counter crosses ``_EMPTY_EXTRACTION_THRESHOLD``, ``state.retrieval_failed``
    flips True. Once flipped, the flag stays True for the remainder of the
    run — a late successful extraction shouldn't erase the fact that the
    retrieval infrastructure was already flagged as failing.
    """
    content = getattr(evidence, "extracted_content", None)
    if content:
        state.consecutive_empty_extractions = 0
    else:
        state.consecutive_empty_extractions += 1
        if state.consecutive_empty_extractions >= _EMPTY_EXTRACTION_THRESHOLD:
            state.retrieval_failed = True


def _make_op(op_class: type, deps: EpistemicDeps) -> Any:
    """Create an operation instance from graph deps."""
    return op_class(
        repo=deps.repo,
        agent_runner=deps.agent_runner,
        evidence_gatherer=deps.evidence_gatherer,
        quality_scorer=deps.quality_scorer,
        embedding_model=deps.embedding_model,
    )


def _op_input(entity_id: str, entity_type: str, operation: str) -> Any:
    """Create an OperationInput for operation execution."""
    from ..operations.base import OperationInput

    return OperationInput(
        entity_id=entity_id, entity_type=entity_type, operation=operation
    )


async def _run_op(
    op_class: type,
    deps: EpistemicDeps,
    state: EpistemicGraphState,
    entity_id: str,
    entity_type: str,
    operation: str,
) -> Any:
    """Instantiate an operation, execute it, log the result, and return it.

    If the operation raises, record a quarantine on the graph state and
    return a failed OperationResult — never swallow silently. Downstream
    nodes should call state.is_quarantined(entity_id) before scheduling
    further work on the entity; this guard is added per-operation in
    subsequent tasks.
    """
    from ..operations.base import OperationResult

    op = _make_op(op_class, deps)
    work = _op_input(entity_id, entity_type, operation)
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
        state.quarantine(entity_id, entity_type, operation, e)
        result = OperationResult(
            success=False,
            entity_id=entity_id,
            message=f"{operation} quarantined: {type(e).__name__}: {e}",
        )
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
            file_path=(
                f"execution_step_{state.run_id}_{step_number:04d}"
            ),
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


async def _run_tms_sweep(deps: EpistemicDeps, state: EpistemicGraphState) -> None:
    """Run TMS (Truth Maintenance System) sweep: cascade evidence invalidation
    and revalidate claims whose evidence foundation changed.

    This is a reactive correctness check, not a graph node. It runs after
    operations that can invalidate evidence (adversarial search, investigation,
    evidence extraction).

    No LLM calls — purely structural graph maintenance.
    """
    from ..entities import Evidence, Claim
    from ..operations.belief_maintenance import (
        InvalidateEvidenceOperation,
        RevalidateClaimOperation,
    )

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
    had_cascades = False
    pending: list[Evidence] = []
    for ev in all_evidence:
        if isinstance(ev, Evidence) and ev.invalidated and not ev.invalidation_cascaded:
            pending.append(ev)

    referenced_ids: set[str] = set()
    if pending:
        all_claims_pre = await deps.repo.query(
            "claim", objective_id=state.objective_id
        )
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
            had_cascades = True
            continue
        await _run_op(
            InvalidateEvidenceOperation,
            deps,
            state,
            ev.entity_id,
            "evidence",
            "invalidate_evidence",
        )
        had_cascades = True

    # Step 2: Revalidate all non-abandoned claims above HYPOTHESIS after cascade.
    # RevalidateClaimOperation checks the gate and only demotes if it fails,
    # so running it on unaffected claims is a safe no-op.
    if had_cascades:
        from ..entities.claim import ClaimStage

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        for claim in all_claims:
            if (
                isinstance(claim, Claim)
                and not claim.abandoned
                and claim.stage != ClaimStage.HYPOTHESIS
            ):
                result = await _run_op(
                    RevalidateClaimOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "revalidate_claim",
                )
                # If TMS demoted the claim:
                # - Remove from verification_done so PromoteToSupported
                #   re-routes it through verification.
                # - ALSO add to claims_needing_rescrutiny so the demoted
                #   claim is re-scrutinised before the next promotion
                #   attempt. ``record_demotion`` clears
                #   ``scrutiny_verdict`` to None; without rescrutiny,
                #   PromoteClaimOperation rejects the claim with
                #   "Scrutiny not passed (verdict: None)" and the claim
                #   stays stuck at HYPOTHESIS forever (the
                #   demote-can't-repromote trap). ResolveUncertainties
                #   reads claims_needing_rescrutiny and routes back to
                #   Scrutinize when non-empty, restoring the verdict.
                #
                # COUPLING NOTE (defense-in-depth): the TMS-induced
                # demote→re-promote→adversarial→demote loop is bounded
                # only INDIRECTLY by ``SCRUTINY_RESOLVE_CYCLE_CAP``. Each
                # demote routes the claim back through Scrutinize via
                # the rescrutiny set, where ``state.scrutiny_resolve_cycles``
                # increments. After CAP rounds the claim is cycle_capped
                # → terminal. If a future refactor ever bypasses
                # Scrutinize on the demote→re-promote path (e.g. a
                # "fast revalidation" optimization that goes straight
                # back to verification), this loop loses its bound.
                # Either keep all TMS-demote paths going through
                # Scrutinize, or add a per-claim ``tms_demotion_counts``
                # state field with its own cap.
                if result.success and "demoted" in (result.message or "").lower():
                    state.verification_done.discard(claim.entity_id)
                    state.claims_needing_rescrutiny.add(claim.entity_id)

    # Step 3: Process claims flagged for TMS by graph nodes
    for cid in list(state.claims_needing_tms):
        claim = await deps.repo.get("claim", cid)
        if isinstance(claim, Claim) and not claim.abandoned:
            await _run_op(
                RevalidateClaimOperation,
                deps,
                state,
                cid,
                "claim",
                "revalidate_claim",
            )
    state.claims_needing_tms.clear()


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
    """Optionally run DecomposeQuestionOperation before PlanEvidence.

    Gated by ``state.decompose``: when False (the default open-research
    path), this node is a no-op pass-through. When True, runs
    ``DecomposeQuestionOperation`` to populate
    ``objective.decomposition`` so:

    * ``PlanTaskOperation`` formulates queries per sub-investigation
      (per-claim evidence pool, Option-2).
    * ``CreateClaims`` routes to ``MultiSeedClaimOperation`` (one Claim
      per sub-investigation, sub_investigation_id-tagged).
    * ``CombineClaimVerdicts`` applies the decomposition's
      ``combination_rule`` (AND / OR / WEIGHTED_AND / UNION) over the
      per-Claim integration verdicts.
    * ``compute_posterior`` honours the rule via the rule-aware
      delegation path (see confidence.py).

    Idempotent: ``DecomposeQuestionOperation`` short-circuits if
    ``objective.decomposition`` is already set.

    Phase 5 of the Move-3 plan.
    """

    reads = frozenset({"objective_id", "decompose"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PlanEvidence":
        state = ctx.state
        deps = ctx.deps

        if not state.decompose:
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
    """Create evidence stubs via plan_task.

    Phase 5 of the Move-3 plan.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ExtractEvidence":
        from ..operations.preplanning import PlanTaskOperation

        state = ctx.state
        deps = ctx.deps

        await _run_op(
            PlanTaskOperation,
            deps,
            state,
            state.objective_id,
            "objective",
            "plan_task",
        )

        obj = await deps.repo.get("objective", state.objective_id)
        obj.phase = "planned"
        await deps.repo.save(obj)

        return ExtractEvidence()


@dataclass
class ExtractEvidence(Node):
    """Extract content from all unextracted evidence stubs.

    Phase 5 of the Move-3 plan. Writes
    ``consecutive_empty_extractions`` and ``retrieval_failed`` via the
    ``_update_retrieval_health`` helper.
    """

    reads = frozenset({"objective_id", "claims_created"})
    writes = frozenset({"consecutive_empty_extractions", "retrieval_failed"})
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
            updated_ev = await deps.repo.get("evidence", ev.entity_id)
            _update_retrieval_health(state, updated_ev)


        # Cross-provider duplicate sweep: when multiple providers return the
        # same paper, mark all but one as invalidated so judging / scrutiny /
        # gates don't pay LLM cost on redundant copies. Downstream filters
        # already exclude invalidated evidence; this is purely cost reduction.
        from ..dedupe_evidence import dedupe_evidence_by_source_ref

        await dedupe_evidence_by_source_ref(deps.repo, state.objective_id)

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
    writes = frozenset(
        {"scrutiny_resolve_cycles", "claims_needing_rescrutiny"}
    )
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
                        deps, claim.entity_id, "Scrutinize-defense-in-depth"
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
                if claim.stage == ClaimStage.HYPOTHESIS and inv_count >= 3:
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

    reads = frozenset({"objective_id", "investigation_counts"})
    writes = frozenset(
        {
            "investigation_counts",
            "claims_needing_rescrutiny",
            "claims_needing_tms",
        }
    )
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ExtractNewEvidence":
        from ..operations.investigation import InvestigateClaimOperation

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if claim.scrutiny_verdict in ("needs_resolution", "fail"):
                inv_count = state.investigation_counts.get(claim.entity_id, 0)
                if inv_count < 3:
                    result = await _run_op(
                        InvestigateClaimOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "investigate_claim",
                    )
                    state.investigation_counts[claim.entity_id] = inv_count + 1
                    if result.success:
                        state.claims_needing_rescrutiny.add(claim.entity_id)
                        # TMS: if claim is promoted and new evidence was created
                        if result.created_entities:
                            from ..entities.claim import ClaimStage

                            claim_updated = await deps.repo.get(
                                "claim", claim.entity_id
                            )
                            if claim_updated.stage != ClaimStage.HYPOTHESIS:
                                state.claims_needing_tms.add(claim.entity_id)

        return ExtractNewEvidence()


@dataclass
class ExtractNewEvidence(Node):
    """Extract content from newly created evidence stubs, then re-enter scrutiny.

    Phase 5 of the Move-3 plan. Writes
    ``consecutive_empty_extractions`` and ``retrieval_failed`` via the
    ``_update_retrieval_health`` helper.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset({"consecutive_empty_extractions", "retrieval_failed"})
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Scrutinize":
        from ..operations.evidence import ExtractEvidenceOperation
        from ..entities import Claim, Evidence
        from ..judge import judge_evidence

        state = ctx.state
        deps = ctx.deps

        unextracted = await deps.repo.query(
            "evidence",
            objective_id=state.objective_id,
            extracted=False,
        )

        # Pass 1: extraction only. Each extract_evidence call may create
        # extras (gathered[1:]) — we collect them all before judging so
        # the dedupe sweep runs in between and we don't pay LLM cost on
        # cross-provider duplicates.
        per_stub_results: list[tuple[Any, Any]] = []
        for ev in unextracted:
            result = await _run_op(
                ExtractEvidenceOperation,
                deps,
                state,
                ev.entity_id,
                "evidence",
                "extract_evidence",
            )
            updated_ev = await deps.repo.get("evidence", ev.entity_id)
            _update_retrieval_health(state, updated_ev)
            per_stub_results.append((ev, result))

        # Cross-provider duplicate sweep before judging fires. Dedupe runs
        # on all evidence for the objective so it catches duplicates within
        # this batch and against earlier-extracted evidence.
        from ..dedupe_evidence import dedupe_evidence_by_source_ref

        await dedupe_evidence_by_source_ref(deps.repo, state.objective_id)

        # Pass 2: link new evidence to claims and judge. Invalidated items
        # (marked by the dedupe sweep) are skipped — they were duplicates
        # of evidence that's already represented and will be filtered out
        # by all downstream consumers anyway.
        for ev, result in per_stub_results:
            if not result.success:
                continue

            created_ids = getattr(result, "created_entities", []) or []
            if len(created_ids) <= 1:
                continue

            original = await deps.repo.get("evidence", ev.entity_id)
            claim_id = original.depends_on_claim_id
            if not claim_id:
                continue

            claim = await deps.repo.get("claim", claim_id)
            if not isinstance(claim, Claim):
                continue

            extras_linked = 0
            evidence_to_judge: list[Evidence] = []
            for eid in created_ids:
                entity_ev = await deps.repo.get("evidence", eid)
                if not isinstance(entity_ev, Evidence):
                    continue

                # Skip linking + judging for items the dedupe sweep
                # invalidated as cross-provider duplicates.
                if entity_ev.invalidated:
                    continue

                if eid not in claim.evidence_ids:
                    claim.evidence_ids.append(eid)
                    extras_linked += 1

                if (
                    deps.agent_runner
                    and entity_ev.extracted_content
                    and entity_ev.support_judgment is None
                ):
                    evidence_to_judge.append(entity_ev)

            # Phase 2a of the efficiency plan: judge the new evidence
            # extras concurrently. Each judgment writes a different
            # Evidence entity, so the calls are independent. The
            # AgentRunner's global semaphore bounds in-flight calls
            # (defaults: 1 for Ollama, 8 for cloud).
            if evidence_to_judge and deps.agent_runner is not None:
                runner = deps.agent_runner

                async def _judge_extra(entity_ev: Evidence) -> None:
                    judgment = await judge_evidence(
                        claim_statement=claim.statement,
                        claim_scope=claim.scope or "",
                        evidence_content=entity_ev.extracted_content,
                        evidence_source=f"{entity_ev.source_type}: {entity_ev.source_ref}",
                        runner=runner,
                    )
                    verdict = judgment.verdict.lower().strip()
                    if verdict not in ("supports", "contradicts", "no_bearing"):
                        verdict = "no_bearing"
                    entity_ev.support_judgment = verdict
                    entity_ev.judgment_reasoning = judgment.reasoning
                    await deps.repo.save(entity_ev)

                await asyncio.gather(
                    *(_judge_extra(ev) for ev in evidence_to_judge)
                )

            if extras_linked > 0:
                claim.evidence_count = len(claim.evidence_ids)
                await deps.repo.save(claim)

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

                if claim.stage == ClaimStage.HYPOTHESIS and inv_count >= 3:
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
    2. SUPPORTED without verification done → route to ClusterEvidence
    3. Everything at PROVISIONAL+ or abandoned → CheckCompletion

    This correctly handles re-entry after uncertainty resolution:
    a claim at SUPPORTED that was re-scrutinized goes through
    verification again instead of being skipped.

    Phase 2 of the Move-3 plan: this is the second of the two
    recurring-bug routing hubs. Its successor set
    (Union["ClusterEvidence", "CheckCompletion"]) intentionally
    INCLUDES CheckCompletion — that's the legitimate idempotent
    terminal when no SUPPORTED claim needs verification.
    PromoteToSupported is the dispatcher that AbandonOrDemote
    delegates residual routing to; this node is allowed to
    terminate where the upstream node is not.

    post_invariants is empty: this node is mid-flight when it
    routes to ClusterEvidence (claims may legitimately be in the
    "stranded" state about to enter IBE); when it routes to
    CheckCompletion, no_stranded_claims would be the right
    invariant to enforce — but that's the post-condition for
    CheckCompletion → Synthesize, not for this node directly.
    """

    reads = frozenset({"objective_id", "verification_done"})
    writes = frozenset({"terminal_claims"})
    operations = frozenset()  # populated below after operations imports resolve
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["ClusterEvidence", "CheckCompletion"]:
        from ..operations.stage_management import PromoteClaimOperation
        from ..operations.belief_maintenance import SetRoutingDefaultsOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        # Step 1: Promote any HYPOTHESIS claims with passing scrutiny
        for claim in all_claims:
            if claim.abandoned:
                continue
            if claim.cycle_capped:
                # Inquiry didn't converge — don't promote this claim, and
                # mark it terminal so CheckCompletion sees no pending work.
                # The claim stays at HYPOTHESIS; compute_posterior will
                # surface the oscillation via terminal_state, and the
                # IBE chain (which only runs on SUPPORTED claims) won't
                # fabricate a verdict. This is the principled completion
                # of the runtime cycle cap.
                state.terminal_claims.add(claim.entity_id)
                continue
            if (
                claim.stage == ClaimStage.HYPOTHESIS
                and claim.scrutiny_verdict == "pass"
            ):
                result = await _run_op(
                    PromoteClaimOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "promote_claim",
                )
                if result.success:
                    await _run_op(
                        SetRoutingDefaultsOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "set_routing_defaults",
                    )

        # Step 2: Check if any SUPPORTED claims need verification
        # Re-read after promotions.
        #
        # Cycle-capped claims are excluded: cycle-capped means the
        # inquiry loop didn't converge cleanly, so any verification
        # work would feed an IBE chain whose verdict will be discarded
        # by combine_claim_verdicts (which excludes cycle-capped
        # claims from aggregation). Save the LLM cost AND keep
        # cycle-cap's "this claim's inquiry didn't converge"
        # semantics consistent.
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

        # All claims are at PROVISIONAL+ or abandoned or verified
        return CheckCompletion()


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
    (``_run_tms_sweep``) dispatches InvalidateEvidence / RevalidateClaim
    operations indirectly — those are side effects via a helper function
    and would require inter-procedural AST analysis to validate. For
    now the validator only sees direct dispatches.
    """

    reads = frozenset(
        {"objective_id", "question_type", "claims_needing_rescrutiny"}
    )
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
                        # Fire on first pass (balance is None) or if prior balance was poor
                        balance = claim.adversarial_balance
                        if balance is not None and balance >= 0.6:
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
        # converge, so don't pull verification or IBE work that the
        # combiner will discard anyway.
        active_supported = [
            c
            for c in all_claims
            if not c.abandoned
            and not c.cycle_capped
            and c.stage == ClaimStage.SUPPORTED
        ]
        # If TMS just demoted any claims and added them to the rescrutiny
        # set, don't fast-path to IBE — those claims need their scrutiny
        # verdict re-established first. Falling through to
        # ResolveUncertainties routes us back to Scrutinize via the
        # rescrutiny set, restoring the verdict. Without this guard, the
        # demote-can't-repromote trap kicks in (record_demotion cleared
        # scrutiny_verdict; PromoteClaim rejects None-verdict claims).
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
                c.convergence_verdict in terminal_convergence
                for c in active_supported
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

    Phase 3 of the Move-3 plan. This node is one of the routing hubs
    where the IBE-skip path may live (per the Phase 2 benchmark
    investigation): when adversarial evidence challenges a claim,
    revalidation may demote it back to HYPOTHESIS, and the resulting
    re-cycle through Scrutinize → AbandonOrDemote can leave claims
    cycle-capped and terminate via CheckCompletion before the IBE
    chain ever runs. Making the contracts explicit here is the first
    step toward catching that.

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
    writes = frozenset(
        {"claims_needing_rescrutiny", "scrutiny_resolve_cycles"}
    )
    operations = frozenset()  # populated below
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union[
        "EnumerateCandidates", "PromoteToSupported", "Scrutinize", "ResolveUncertainties"
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
        blocking = [u for u in all_uncertainties if u.is_blocking]

        if not blocking:
            # Rescrutiny check FIRST — must run before the early return
            # to EnumerateCandidates / PromoteToSupported. Without this
            # ordering, claims that TMS demoted (e.g. soft-promoted
            # claim → adversarial counter-evidence → revalidate demote
            # → claims_needing_rescrutiny.add()) are stranded: the
            # subsequent EnumerateCandidates / PromoteToSupported
            # filters require ``stage == SUPPORTED``, which the demoted
            # claim no longer satisfies. The claim never re-enters the
            # inquiry cycle, never gets re-promoted, and the IBE chain
            # never runs on it. This is the IBE-skip bug surfaced in
            # the Phase 2 benchmark of the Move-3 graph contracts work.
            if state.claims_needing_rescrutiny:
                return Scrutinize()
            if self.next_on_clear == "promote":
                return PromoteToSupported()
            return EnumerateCandidates()

        for unc in blocking:
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
                                deps, cid, "ResolveUncertainties-primary"
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
        new_blocking = [u for u in remaining if u.is_blocking]

        if new_blocking and self.depth < 3:
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
        # Cycle-capped claims excluded: combine_claim_verdicts will
        # discard their verdict regardless, so don't spend LLM calls
        # on the IBE chain for them. Keeps cycle-cap's "inquiry didn't
        # converge" semantics consistent across the pipeline.
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
            await asyncio.gather(
                *(
                    _run_op(
                        EnumerateCandidatesOperation,
                        deps,
                        state,
                        cid,
                        "claim",
                        "enumerate_candidates",
                    )
                    for cid in eligible_ids
                )
            )

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
            await asyncio.gather(
                *(
                    _run_op(
                        ScoreLovelinessOperation,
                        deps,
                        state,
                        cid,
                        "claim",
                        "score_loveliness",
                    )
                    for cid in eligible_ids
                )
            )

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
            await asyncio.gather(
                *(
                    _run_op(
                        ScoreLikelinessOperation,
                        deps,
                        state,
                        cid,
                        "claim",
                        "score_likeliness",
                    )
                    for cid in eligible_ids
                )
            )

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

    reads = frozenset({"objective_id"})
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
            await asyncio.gather(
                *(
                    _run_op(
                        SelectBestExplanationOperation,
                        deps,
                        state,
                        cid,
                        "claim",
                        "select_best_explanation",
                    )
                    for cid in eligible_ids
                )
            )

        return PromoteSupported()


@dataclass
class PromoteSupported(Node):
    """Try advancing claims beyond SUPPORTED: S->P, P->R, R->A, then record decisions.

    Phase 4 of the Move-3 plan. Marks each processed claim as
    verification_done so PromoteToSupported's idempotency check
    short-circuits on re-entry.
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
        from ..entities.claim import ClaimStage
        from ..gates import get_next_stage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue

            # Try each promotion step once (S->P, P->R, R->A)
            current = claim.stage
            while True:
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
    """Apply the decomposition's combination_rule over per-claim integration
    verdicts and stash the result on the parent Objective for Synthesize.

    Phase 4 of the multi-seed-claim refactor. The combination logic was
    previously in ``decomposed_runner.combine_sub_verdicts`` (operating
    on per-child PipelineResults). Under multi-seed-claim it operates on
    the Objective's claims directly, since "child" sub-investigations
    are now Claim entities on a single Objective.

    Behaviour:

    * If the Objective has no decomposition (open-research run), this
      node is a no-op — nothing to combine. Synthesize handles the
      multi-claim narrative without a rule-aware verdict.
    * If decomposition is set, combine via
      ``combine_claim_verdicts(claims, rule, weights)``. Stash the
      ``CombinedVerdict`` (serialized) on the parent Objective for
      Synthesize to read later via the snapshot.

    The combined verdict is stored on Objective.decomposition (under
    a new "combined_verdict" key) so it travels through FreezeSnapshot
    → SynthesizeReport without needing a separate field.

    Pure / deterministic — no LLM call. Safe to re-enter.

    Phase 5 of the Move-3 plan. Empty operations: this node uses
    ``combine_claim_verdicts`` directly as a function, not via _run_op.
    """

    reads = frozenset({"objective_id"})
    writes = frozenset()
    operations = frozenset()
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "CheckCompletion":
        from .combination import (
            combine_claim_verdicts,
            extract_weights_from_decomposition,
            resolve_combination_rule,
        )
        from ..entities.objective import Objective

        state = ctx.state
        deps = ctx.deps

        objective = await deps.repo.get("objective", state.objective_id)
        if not isinstance(objective, Objective) or not objective.decomposition:
            # Open-research run with no decomposition. Nothing to combine.
            return CheckCompletion()

        # Pull claims in decomposition order so the weight alignment is
        # correct. Multi-seed-claim sets sub_investigation_id on each
        # Claim; we sort by the order in decomposition.sub_investigations.
        # Phase 6 of the Move-3 plan: typed Decomposition access.
        sub_ids_in_order = [
            s.id for s in objective.decomposition.sub_investigations
        ]
        all_claims = await deps.repo.query(
            "claim", objective_id=state.objective_id
        )
        claims_by_sub: dict[str, Any] = {
            c.sub_investigation_id: c
            for c in all_claims
            if c.sub_investigation_id is not None
        }
        ordered_claims = [
            claims_by_sub[sid]
            for sid in sub_ids_in_order
            if sid in claims_by_sub
        ]
        # Orphan diagnostic: claims that exist but have no matching sub-id
        # in the current decomposition (sub-investigation removed by a
        # later reflection round, or claim minted without sub_investigation_id).
        # Without this surfacing, the combiner would silently exclude them.
        n_orphan = sum(
            1
            for c in all_claims
            if c.sub_investigation_id is None
            or c.sub_investigation_id not in sub_ids_in_order
        )
        if not ordered_claims:
            if n_orphan and deps.verbose:
                logger.warning(
                    "CombineClaimVerdicts: no claims matched decomposition "
                    "(%d orphans dropped); skipping",
                    n_orphan,
                )
            return CheckCompletion()

        rule = resolve_combination_rule(objective) or "AND"
        weights = extract_weights_from_decomposition(
            objective.decomposition, ordered_claims
        )
        combined = combine_claim_verdicts(ordered_claims, rule, weights=weights)

        # Append orphan count to the combiner's explanation if any were
        # dropped. The combiner's own diagnostic only sees the claims it
        # was given; orphans are excluded one level above.
        explanation = combined.explanation
        if n_orphan:
            explanation = (
                f"{explanation} ({n_orphan} orphan claim(s) dropped — "
                "sub_investigation_id not in current decomposition)"
            )

        # Stash the CombinedVerdict on the Decomposition so the
        # snapshot freeze step picks it up. Phase 6 of the Move-3 plan:
        # typed CombinedVerdictData replaces the previous dict shape.
        from ..entities.decomposition import CombinedVerdictData

        objective.decomposition.combined_verdict = CombinedVerdictData(
            posterior=combined.posterior,
            verdict=combined.verdict,
            combination_rule=combined.combination_rule,
            claim_posteriors=combined.claim_posteriors,
            n_capped=combined.n_capped,
            n_no_verdict=combined.n_no_verdict,
            n_abandoned=combined.n_abandoned,
            n_orphan=n_orphan,
            explanation=explanation,
        )
        await deps.repo.save(objective)

        if deps.verbose:
            logger.info(
                "Combined verdict: %s (posterior=%s, rule=%s, orphans=%d)",
                combined.verdict,
                combined.posterior,
                combined.combination_rule,
                n_orphan,
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
            "successful",
            "failed",
            "errors",
            "operations_log",
            "quarantined",
        }
    )
    writes = frozenset()
    operations = frozenset()
    post_invariants = (no_stranded_claims,)

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["CheckSynthesisDemand", End[EpistemicResult]]:
        state = ctx.state
        deps = ctx.deps

        # Short-circuit on retrieval failure: evidence extraction kept
        # returning empty content, so there's nothing to synthesize.
        # Terminate with a distinct status so the posterior/report can
        # surface retrieval_failed as the terminal state.
        if state.retrieval_failed:
            return End(
                EpistemicResult(
                    objective_id=state.objective_id,
                    status="retrieval_failed",
                    successful=state.successful,
                    failed=state.failed,
                    errors=state.errors,
                    operations_log=state.operations_log,
                    termination_reason="retrieval_failed",
                    quarantined=state.quarantined,
                    retrieval_failed=True,
                )
            )

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        non_abandoned = [c for c in all_claims if not c.abandoned]

        if non_abandoned:
            return CheckSynthesisDemand()

        # All claims abandoned or no claims exist
        reason = "partial" if all_claims else "no_claims"
        return End(
            EpistemicResult(
                objective_id=state.objective_id,
                status=reason,
                successful=state.successful,
                failed=state.failed,
                errors=state.errors,
                operations_log=state.operations_log,
                termination_reason=reason,
                quarantined=state.quarantined,
                retrieval_failed=state.retrieval_failed,
            )
        )


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
    """

    reads = frozenset(
        {"objective_id", "claims_needing_rescrutiny", "scrutiny_resolve_cycles"}
    )
    writes = frozenset({"claims_needing_rescrutiny"})
    operations = frozenset()  # uses agent_runner directly, not _run_op
    post_invariants = ()

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Synthesize", "Scrutinize"]:
        from ..demand import Demand
        from ..entities.objective import Objective

        state = ctx.state
        deps = ctx.deps

        objective = await deps.repo.get("objective", state.objective_id)
        all_claims = await deps.repo.query(
            "claim", objective_id=state.objective_id
        )
        active_claims = [c for c in all_claims if not c.abandoned]

        # ── Compute the demand ──────────────────────────────────────
        # Every path below produces a ``demand`` variable. The single
        # loop-back decision at the end of this method consumes it.
        # This keeps Phase 4's "needs_more → loop back" logic in one
        # place rather than scattered across each gate.

        demand: Any = None  # set by exactly one branch below

        # Gate 1: no decomposition at all (open-research mode). The
        # combined-verdict satisfaction check doesn't apply; the
        # writer agent will frame the answer over the per-claim
        # narrative without a rule-aware verdict. Treat as satisfied
        # for this layer.
        if not isinstance(objective, Objective) or objective.decomposition is None:
            demand = Demand.satisfied(
                justification=(
                    "Open-research mode (no decomposition) — synthesis "
                    "demand check does not apply. Synthesis writer will "
                    "frame the answer over per-claim verdicts directly."
                )
            )
            self._log_demand(demand, deps)
            return Synthesize()

        combined = objective.decomposition.combined_verdict

        # Gate 2: combine_claim_verdicts didn't produce a verdict (no
        # claims aggregated; usually means orphan claims or a fully-
        # abandoned set).
        if combined is None:
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
        elif combined.n_no_verdict > 0:
            demand = Demand.needs(
                justification=(
                    f"{combined.n_no_verdict} claim(s) reached synthesis "
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
        # Gate 4: decisive posterior (≥0.85 supports OR ≤0.15 contradicts).
        elif combined.posterior is not None and (
            combined.posterior >= 0.85 or combined.posterior <= 0.15
        ):
            direction = "supports" if combined.posterior >= 0.85 else "contradicts"
            demand = Demand.satisfied(
                justification=(
                    f"Combined posterior {combined.posterior:.3f} ({direction}) "
                    f"is decisive; the verdict direction is clear from the "
                    f"per-claim verdicts ({combined.combination_rule} over "
                    f"{len([p for p in combined.claim_posteriors if p is not None])} "
                    "aggregated claims). No further investigation likely "
                    "to change the headline."
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
                verdict = c.integrated_assessment or "no-verdict"
                confidence = (
                    f"{c.integrated_confidence:.2f}"
                    if c.integrated_confidence is not None
                    else "—"
                )
                claim_summaries.append(
                    f"  [{c.sub_investigation_id or '-'}] "
                    f"verdict={verdict} (conf={confidence}): {c.statement[:120]}"
                )

            try:
                demand = await deps.agent_runner.run(
                    "epistemic_check_synthesis_demand",
                    research_question=(
                        objective.clarified_question or objective.description
                    ),
                    combined_verdict_label=combined.verdict,
                    combined_posterior=(
                        f"{combined.posterior:.3f}"
                        if combined.posterior is not None
                        else "n/a (UNION rule or no aggregated claims)"
                    ),
                    combination_rule=combined.combination_rule,
                    claim_count=len(active_claims),
                    claims_summary="\n".join(claim_summaries) or "(none)",
                    combiner_explanation=combined.explanation,
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

        self._log_demand(demand, deps)

        # Phase 4 of the lazy-escalation plan: ACTIVATE the loop-back.
        # When demand says needs_more, identify claims that could
        # benefit from more investigation and route back to Scrutinize
        # via the rescrutiny set. When no claim can make progress
        # (all eligible claims are terminal or at the per-claim cap),
        # accept the current state and synthesize.
        if not demand.needs_more:
            return Synthesize()

        return self._maybe_loop_back(demand, active_claims, state)

    def _maybe_loop_back(
        self,
        demand: Any,
        active_claims: list[Any],
        state: EpistemicGraphState,
    ) -> Union["Synthesize", "Scrutinize"]:
        """Decide whether to route back to Scrutinize for another
        round of investigation, or accept the current state and
        synthesize.

        We loop back when there's at least one claim that's
        non-abandoned, non-cycle-capped, and hasn't hit the per-claim
        ``SCRUTINY_RESOLVE_CYCLE_CAP``. Those are the claims for which
        another round of inquiry could plausibly change the outcome.

        Per-claim cap is the load-bearing safety: even if the
        satisfaction LLM keeps saying needs_more, claims that are at
        cap don't get re-added to rescrutiny, so when ALL eligible
        claims hit cap, this method returns Synthesize and the loop
        terminates. No global "give up after N rounds" cap — the
        per-claim cap composes correctly under the synthesis loop.
        """
        eligible_claim_ids: list[str] = []
        for c in active_claims:
            if getattr(c, "cycle_capped", False):
                continue
            cycles = state.scrutiny_resolve_cycles.get(c.entity_id, 0)
            if cycles >= SCRUTINY_RESOLVE_CYCLE_CAP:
                continue
            eligible_claim_ids.append(c.entity_id)

        if not eligible_claim_ids:
            # All non-abandoned claims are either cycle-capped or at
            # cap. No re-investigation can make progress. Accept the
            # current state.
            logger.warning(
                "[synthesis_demand] needs_more=True but all "
                "non-abandoned claims have hit per-claim cap; "
                "synthesizing anyway. (Existing safety: per-claim "
                "cap is the loop-bound; no global give-up budget.)"
            )
            return Synthesize()

        # Add the eligible claims to the rescrutiny set so Scrutinize
        # picks them up. This composes with the existing scrutiny ↔
        # resolve loop, which already has its own cycle cap (incremented
        # in Scrutinize's predicate when the claim is dequeued).
        for cid in eligible_claim_ids:
            state.claims_needing_rescrutiny.add(cid)

        logger.warning(
            "[synthesis_demand] looping back to Scrutinize on %d "
            "eligible claim(s); justification: %s",
            len(eligible_claim_ids),
            demand.justification,
        )
        return Scrutinize()

    @staticmethod
    def _log_demand(demand: Any, deps: EpistemicDeps) -> None:
        """Emit the synthesis demand at WARNING level so it shows
        through the CLI's verbose-mode log filter (which suppresses
        INFO during runs to keep the rich-progress output clean).
        These logs are also the audit trail for the satisfaction
        check's calibration — they MUST be visible.

        The log level isn't a "this is alarming" signal; it's just
        the only level the CLI keeps during runs. When the CLI's
        logging policy changes, this can drop back to INFO.

        ``deps`` is unused today but kept on the signature so future
        per-run instrumentation (e.g. write the demand to a trace
        file in deps) doesn't change every call site."""
        del deps  # currently unused; reserved for trace-file emission
        logger.warning(
            "[synthesis_demand] needs_more=%s | %s%s",
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
    post_invariants = (no_stranded_claims,)

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

        # Freeze snapshot
        await _run_op(
            FreezeSnapshotOperation,
            deps,
            state,
            oid,
            "objective",
            "freeze_snapshot",
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


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

epistemic_graph: Graph[EpistemicGraphState, EpistemicDeps, EpistemicResult] = Graph(
    nodes=[
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
    ],
    name="epistemic_pipeline",
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


Scrutinize.operations = frozenset({ScrutiniseClaimOperation})


Investigate.operations = frozenset({InvestigateClaimOperation})


ResolveUncertainties.operations = frozenset(
    {ResolveUncertaintyOperation, DeduplicateConcernsOperation}
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
    PlanTaskOperation,
)
from ..operations.evidence import (  # noqa: E402
    ExtractEvidenceOperation,
)
from ..operations.seed_claim import SeedClaimOperation  # noqa: E402
from ..operations.multi_seed_claim import (  # noqa: E402
    MultiSeedClaimOperation,
)
from ..operations.claims import ProposeClaimsOperation  # noqa: E402
from ..operations.synthesis import (  # noqa: E402
    FreezeSnapshotOperation,
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


PlanEvidence.operations = frozenset({PlanTaskOperation})


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

Synthesize.operations = frozenset(
    {FreezeSnapshotOperation, SynthesizeReportOperation}
)
