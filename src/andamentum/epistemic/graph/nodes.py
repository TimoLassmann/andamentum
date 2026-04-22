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
        await backend.add(
            file_path=f"execution_step_{step_number}",
            content=result.message or "",
            title=f"{operation} on {entity_id[:12]}",
            metadata={
                "epistemic_type": "execution_step",
                "step_number": step_number,
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
    all_evidence = await deps.repo.query("evidence", objective_id=state.objective_id)
    had_cascades = False
    for ev in all_evidence:
        if isinstance(ev, Evidence) and ev.invalidated and not ev.invalidation_cascaded:
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
                # If TMS demoted the claim, remove from verification_done
                # so PromoteToSupported re-routes it through verification
                if result.success and "demoted" in (result.message or "").lower():
                    state.verification_done.discard(claim.entity_id)

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

        # If claims have not yet been created, go create them.
        # Otherwise we are re-entering after investigation — go back to scrutiny.
        if not state.claims_created:
            return CreateClaims()
        return Scrutinize()


@dataclass
class CreateClaims(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Create claims — seed_claim (verification) or propose_claims (research)."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Scrutinize":
        from ..operations.seed_claim import SeedClaimOperation
        from ..operations.claims import ProposeClaimsOperation

        state = ctx.state
        deps = ctx.deps
        oid = state.objective_id

        obj = await deps.repo.get("objective", oid)
        if obj.claim_to_verify:
            await _run_op(
                SeedClaimOperation, deps, state, oid, "objective", "seed_claim"
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
            if needs_scrutiny:
                # Reset verdict for re-scrutiny so the operation runs
                if claim.entity_id in state.claims_needing_rescrutiny:
                    claim.scrutiny_verdict = None
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

            # Link newly created evidence to the same claim as the stub.
            # Extraction from a single query may produce multiple Evidence
            # entities (gathered[1:]). The original stub is linked to a
            # claim via depends_on_claim_id, but the extras are orphans.
            if not result.success:
                continue

            created_ids = getattr(result, "created_entities", []) or []
            if len(created_ids) <= 1:
                continue  # Only the original stub, nothing to link

            # Find the claim this stub was created for
            original = await deps.repo.get("evidence", ev.entity_id)
            claim_id = original.depends_on_claim_id
            if not claim_id:
                continue

            claim = await deps.repo.get("claim", claim_id)
            if not isinstance(claim, Claim):
                continue

            # Link extras to the claim and judge all created entities.
            # When the gatherer returns multiple results, ExtractEvidenceOperation
            # returns early (before the judgment block) so the original stub is
            # also unjudged. Judge every created entity that still lacks a verdict.
            extras_linked = 0
            for eid in created_ids:
                entity_ev = await deps.repo.get("evidence", eid)
                if not isinstance(entity_ev, Evidence):
                    continue

                # Link to claim (extras are not linked; original already is)
                if eid not in claim.evidence_ids:
                    claim.evidence_ids.append(eid)
                    extras_linked += 1

                # Judge if not yet judged
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

                    # Fall through: truly stale, abandon as before.
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
    ) -> "ResolveUncertainties":
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
        from ..routing import get_routing_profile, TrackActivation

        state = ctx.state
        deps = ctx.deps

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
        "IntegrateEvidence", "PromoteToSupported", "Scrutinize", "ResolveUncertainties"
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
            return IntegrateEvidence()

        for unc in blocking:
            result = await _run_op(
                ResolveUncertaintyOperation,
                deps,
                state,
                unc.entity_id,
                "uncertainty",
                "resolve_uncertainty",
            )
            # Graph flow control: mark affected claims for re-scrutiny
            if result.success:
                unc_updated = await deps.repo.get("uncertainty", unc.entity_id)
                if unc_updated.is_blocking and unc_updated.resolution is not None:
                    for cid in unc_updated.affected_claim_ids:
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
        return IntegrateEvidence()


@dataclass
class IntegrateEvidence(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Run abductive integration on each SUPPORTED claim."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "PromoteSupported":
        from ..operations.integration import AbductiveIntegrationOperation
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
            ):
                await _run_op(
                    AbductiveIntegrationOperation,
                    deps,
                    state,
                    claim.entity_id,
                    "claim",
                    "integrate_evidence",
                )

        return PromoteSupported()


@dataclass
class PromoteSupported(BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]):
    """Try advancing claims beyond SUPPORTED: S->P, P->R, R->A, then record decisions."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "CheckCompletion":
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
        IntegrateEvidence,
        PromoteSupported,
        CheckCompletion,
        Synthesize,
    ],
    name="epistemic_pipeline",
)
