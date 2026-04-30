"""Epistemic pipeline graph nodes.

Each node wraps one or more existing operations and returns the next
node to run.  The graph replaces the pattern-based scheduler with
explicit, typed transitions.

Architecture: Layer 2 (pydantic-graph, depends on operations + entities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Union

from pydantic_graph import BaseNode, End, Graph, GraphRunContext

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
class PrepareObjective(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run clarification, classification, and conceptual analysis on the objective."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PlanEvidence":
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

        return PlanEvidence()


@dataclass
class PlanEvidence(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Create evidence stubs via plan_task."""

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
class ExtractEvidence(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Extract content from all unextracted evidence stubs."""

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

        state.evidence_extracted = True

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
class CreateClaims(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Create claims — three branches:

    1. ``claim_to_verify`` set → SeedClaim (one claim from user's text)
    2. ``decomposition`` set → MultiSeedClaim (N claims from
       sub-investigations, per-claim evidence pools)
    3. otherwise → ProposeClaims (claims discovered from evidence)

    Multi-seed-claim is the v0.3 collapse of decomposition spawning into
    the v0.1 multi-claim shape: one Objective hosts N Claims, the graph
    runs once over them, multi-claim convergence dynamics apply.
    """

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
class Scrutinize(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run scrutiny on claims that have not yet been scrutinised."""

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
class Investigate(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run investigation on claims needing more evidence."""

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
class ExtractNewEvidence(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Extract content from newly created evidence stubs, then re-enter scrutiny."""

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
                    judgment = await judge_evidence(
                        claim_statement=claim.statement,
                        claim_scope=claim.scope or "",
                        evidence_content=entity_ev.extracted_content,
                        evidence_source=f"{entity_ev.source_type}: {entity_ev.source_ref}",
                        runner=deps.agent_runner,
                    )
                    verdict = judgment.verdict.lower().strip()
                    if verdict not in ("supports", "contradicts", "no_bearing"):
                        verdict = "no_bearing"
                    entity_ev.support_judgment = verdict
                    entity_ev.judgment_reasoning = judgment.reasoning
                    await deps.repo.save(entity_ev)

            if extras_linked > 0:
                claim.evidence_count = len(claim.evidence_ids)
                await deps.repo.save(claim)

        # TMS sweep: new evidence may trigger revalidation of claims
        await _run_tms_sweep(deps, state)

        return Scrutinize()


@dataclass
class AbandonOrDemote(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Abandon HYPOTHESIS claims that exhausted investigation; demote SUPPORTED+ claims."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Scrutinize", "CheckCompletion"]:
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
                    # integrated_assessment="insufficient" preserves the
                    # counts in the posterior instead of erasing them
                    # via abandonment.
                    soft_result = await _run_op(
                        SoftPromoteOperation,
                        deps,
                        state,
                        claim.entity_id,
                        "claim",
                        "soft_promote",
                    )
                    if soft_result.success:
                        state.verification_done.add(claim.entity_id)
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
            # Re-scrutinise demoted claims
            return Scrutinize()

        # All problematic claims abandoned — check if anything is left
        return CheckCompletion()


@dataclass
class PromoteToSupported(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Promote HYPOTHESIS claims and route SUPPORTED claims to verification.

    This is a routing hub that checks actual claim state:
    1. HYPOTHESIS with pass → promote to SUPPORTED, set routing defaults
    2. SUPPORTED without verification done → route to ClusterEvidence
    3. Everything at PROVISIONAL+ or abandoned → CheckCompletion

    This correctly handles re-entry after uncertainty resolution:
    a claim at SUPPORTED that was re-scrutinized goes through
    verification again instead of being skipped.
    """

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
        # Re-read after promotions
        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        needs_verification = any(
            not c.abandoned
            and c.stage == ClaimStage.SUPPORTED
            and c.entity_id not in state.verification_done
            for c in all_claims
        )

        if needs_verification:
            return ClusterEvidence()

        # All claims are at PROVISIONAL+ or abandoned or verified
        return CheckCompletion()


@dataclass
class ClusterEvidence(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Cluster evidence into representatives before verification.

    Deterministic (embedding-based, no LLM). Runs HDBSCAN on evidence
    embeddings and labels each item as representative, corroborative,
    or deferred. Verification operations then only process representatives.

    Separated into its own node to distinguish deterministic steps from
    LLM-calling steps in the graph topology.
    """

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
class RunVerification(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run verification tracks on SUPPORTED claims based on routing profile.

    LLM-heavy: adversarial search, convergence, deductive validation,
    computational verification, argument analysis, contrastive evaluation.
    """

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
        active_supported = [
            c
            for c in all_claims
            if not c.abandoned and c.stage == ClaimStage.SUPPORTED
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
class ResolveUncertainties(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Resolve blocking uncertainties, then deduplicate concerns.

    Appears in TWO places in the graph:
    1. After Scrutinize (next_on_clear="promote") — resolve before first promote
    2. After RunVerification (next_on_clear="integrate") — resolve before integration

    This matches the old pattern scheduler where resolve_uncertainty had
    higher priority than promote and ran first whenever blocking
    uncertainties existed.
    """

    depth: int = 0
    next_on_clear: str = "integrate"  # "promote" or "integrate"

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
class EnumerateCandidates(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Stage 1 of IBE: enumerate candidate verdicts for each SUPPORTED claim.

    Generative role (Peirce). Iterative single-candidate calls with the
    running list as priors so each call diversifies away from prior
    candidates. Idempotent — skips claims that already have candidates.
    """

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ScoreLoveliness":
        from ..operations.integration import EnumerateCandidatesOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if (
                claim.stage == ClaimStage.SUPPORTED
                and claim.integrated_assessment is None
                and not claim.integration_candidates
            ):
                await _run_op(
                    EnumerateCandidatesOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "enumerate_candidates",
                )

        return ScoreLoveliness()


@dataclass
class ScoreLoveliness(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Stage 2 of IBE: score each candidate's explanatory virtue.

    Evaluative role (Lipton). Per-candidate calls run in parallel inside
    the operation. Each call sees only its candidate (Kahneman
    independence). Idempotent.
    """

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ScoreLikeliness":
        from ..operations.integration import ScoreLovelinessOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if (
                claim.stage == ClaimStage.SUPPORTED
                and claim.integrated_assessment is None
                and claim.integration_candidates
                and any(c.loveliness is None for c in claim.integration_candidates)
            ):
                await _run_op(
                    ScoreLovelinessOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "score_loveliness",
                )

        return ScoreLikeliness()


@dataclass
class ScoreLikeliness(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Stage 3 of IBE: score each candidate's fit-with-evidence.

    Evaluative role (Lipton). Same pattern as ScoreLoveliness but on
    likeliness. Idempotent.
    """

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "SelectBestExplanation":
        from ..operations.integration import ScoreLikelinessOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if (
                claim.stage == ClaimStage.SUPPORTED
                and claim.integrated_assessment is None
                and claim.integration_candidates
                and any(c.likeliness is None for c in claim.integration_candidates)
            ):
                await _run_op(
                    ScoreLikelinessOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "score_likeliness",
                )

        return SelectBestExplanation()


@dataclass
class SelectBestExplanation(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Stage 4 of IBE: pick the best candidate, write the verdict.

    Comparative role (Lipton). Sees all scored candidates; picks the
    best and assigns gap-based confidence. Writes
    integrated_assessment / integrated_confidence / integrated_reasoning
    on the claim — the fields compute_posterior reads. Idempotent.
    """

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PromoteSupported":
        from ..operations.integration import SelectBestExplanationOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned:
                continue
            if (
                claim.stage == ClaimStage.SUPPORTED
                and claim.integrated_assessment is None
                and claim.integration_candidates
            ):
                await _run_op(
                    SelectBestExplanationOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "select_best_explanation",
                )

        return PromoteSupported()


@dataclass
class PromoteSupported(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Try advancing claims beyond SUPPORTED: S->P, P->R, R->A, then record decisions."""

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
class CombineClaimVerdicts(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
    """

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "CheckCompletion":
        from .combination import (
            combine_claim_verdicts,
            extract_weights_from_decomposition,
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
        sub_ids_in_order = [
            s.get("id")
            for s in (objective.decomposition.get("sub_investigations") or [])
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

        rule = objective.combination_rule or "AND"
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

        # Stash the serialized CombinedVerdict on the Objective so the
        # snapshot freeze step picks it up. Using a dict keyed under
        # "combined_verdict" inside decomposition keeps the existing
        # decomposition shape (no new top-level Objective field).
        objective.decomposition["combined_verdict"] = {
            "posterior": combined.posterior,
            "verdict": combined.verdict,
            "combination_rule": combined.combination_rule,
            "claim_posteriors": combined.claim_posteriors,
            "n_capped": combined.n_capped,
            "n_no_verdict": combined.n_no_verdict,
            "n_abandoned": combined.n_abandoned,
            "n_orphan": n_orphan,
            "explanation": explanation,
        }
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
class CheckCompletion(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Check whether any non-abandoned claims remain and route accordingly."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Synthesize", End[EpistemicResult]]:
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
            return Synthesize()

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
class Synthesize(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Freeze snapshot and synthesize the final report."""

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
        Synthesize,
    ],
    name="epistemic_pipeline",
)
