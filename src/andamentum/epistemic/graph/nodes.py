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


def _make_op(op_class: type, deps: EpistemicDeps) -> Any:
    """Create an operation instance from graph deps."""
    return op_class(
        repo=deps.repo,
        agent_runner=deps.agent_runner,
        evidence_gatherer=deps.evidence_gatherer,
        quality_scorer=deps.quality_scorer,
        embedding_model=deps.embedding_model,
    )


def _work(entity_id: str, entity_type: str, operation: str) -> Any:
    """Create a WorkItem for operation execution."""
    from ..operations.base import WorkItem

    return WorkItem(entity_id=entity_id, entity_type=entity_type, operation=operation)


async def _run_op(
    op_class: type,
    deps: EpistemicDeps,
    state: EpistemicGraphState,
    entity_id: str,
    entity_type: str,
    operation: str,
) -> Any:
    """Instantiate an operation, execute it, log the result, and return it."""
    op = _make_op(op_class, deps)
    work = _work(entity_id, entity_type, operation)
    try:
        result = await op.execute(work)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "%s on %s failed with exception: %s", operation, entity_id[:12], e
        )
        from ..operations.base import OperationResult

        result = OperationResult(
            success=False, entity_id=entity_id, message=f"{operation} error: {e}"
        )
    state.log_operation(operation, entity_id, result.success, result.message)
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
    all_evidence = await deps.repo.query(
        "evidence", objective_id=state.objective_id
    )
    for ev in all_evidence:
        if (
            isinstance(ev, Evidence)
            and ev.invalidated
            and not ev.invalidation_cascaded
        ):
            await _run_op(
                InvalidateEvidenceOperation, deps, state,
                ev.entity_id, "evidence", "invalidate_evidence",
            )

    # Step 2: Revalidate claims flagged by the cascade
    all_claims = await deps.repo.query(
        "claim", objective_id=state.objective_id
    )
    for claim in all_claims:
        if (
            isinstance(claim, Claim)
            and claim.needs_revalidation
            and not claim.abandoned
        ):
            await _run_op(
                RevalidateClaimOperation, deps, state,
                claim.entity_id, "claim", "revalidate_claim",
            )

    # Step 3: Process claims flagged for TMS by graph nodes
    for cid in list(state.claims_needing_tms):
        try:
            claim = await deps.repo.get("claim", cid)
            if isinstance(claim, Claim) and not claim.abandoned:
                # Set needs_revalidation so RevalidateClaimOperation processes it
                if not claim.needs_revalidation:
                    claim.needs_revalidation = True
                    await deps.repo.save(claim)
        except Exception:
            pass
    state.claims_needing_tms.clear()


# ══════════════════════════════════════════════════════════════════════════════
# NODES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class PrepareObjective(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
                ClarifyQuestionOperation, deps, state, oid, "objective", "clarify_question"
            )
            obj = await deps.repo.get("objective", oid)
            obj.phase = "clarified"
            await deps.repo.save(obj)

            # 2. Classify
            await _run_op(
                ClassifyQuestionOperation, deps, state, oid, "objective", "classify_question"
            )
            obj = await deps.repo.get("objective", oid)
            if obj.question_type:
                state.question_type = obj.question_type

            # 3. Conceptual analysis
            await _run_op(
                ConceptualAnalysisOperation, deps, state, oid, "objective", "conceptual_analysis"
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
class PlanEvidence(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Create evidence stubs via plan_task."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "ExtractEvidence":
        from ..operations.preplanning import PlanTaskOperation

        state = ctx.state
        deps = ctx.deps

        await _run_op(
            PlanTaskOperation, deps, state,
            state.objective_id, "objective", "plan_task",
        )

        obj = await deps.repo.get("objective", state.objective_id)
        obj.phase = "planned"
        await deps.repo.save(obj)

        return ExtractEvidence()


@dataclass
class ExtractEvidence(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
                ExtractEvidenceOperation, deps, state,
                ev.entity_id, "evidence", "extract_evidence",
            )

        state.evidence_extracted = True

        # If claims have not yet been created, go create them.
        # Otherwise we are re-entering after investigation — go back to scrutiny.
        if not state.claims_created:
            return CreateClaims()
        return Scrutinize()


@dataclass
class CreateClaims(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
            await _run_op(SeedClaimOperation, deps, state, oid, "objective", "seed_claim")
        else:
            await _run_op(ProposeClaimsOperation, deps, state, oid, "objective", "propose_claims")

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
class Scrutinize(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Run scrutiny on claims that have not yet been scrutinised."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["PromoteToSupported", "Investigate", "AbandonOrDemote"]:
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
                    ScrutiniseClaimOperation, deps, state,
                    claim.entity_id, "claim", "scrutinise_claim",
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

        # All claims pass or are terminal
        return PromoteToSupported()


@dataclass
class Investigate(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
                        InvestigateClaimOperation, deps, state,
                        claim.entity_id, "claim", "investigate_claim",
                    )
                    state.investigation_counts[claim.entity_id] = inv_count + 1
                    if result.success:
                        state.claims_needing_rescrutiny.add(claim.entity_id)
                        # TMS: if claim is promoted and new evidence was created
                        if result.created_entities:
                            from ..entities.claim import ClaimStage
                            claim_updated = await deps.repo.get("claim", claim.entity_id)
                            if claim_updated.stage != ClaimStage.HYPOTHESIS:
                                state.claims_needing_tms.add(claim.entity_id)

        return ExtractNewEvidence()


@dataclass
class ExtractNewEvidence(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Extract content from newly created evidence stubs, then re-enter scrutiny."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> "Scrutinize":
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
                ExtractEvidenceOperation, deps, state,
                ev.entity_id, "evidence", "extract_evidence",
            )

        # TMS sweep: new evidence may trigger revalidation of claims
        await _run_tms_sweep(deps, state)

        return Scrutinize()


@dataclass
class AbandonOrDemote(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Abandon HYPOTHESIS claims that exhausted investigation; demote SUPPORTED+ claims."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Scrutinize", "CheckCompletion"]:
        from ..operations.cleanup import AbandonStaleClaimOperation
        from ..operations.stage_management import DemoteClaimOperation
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
                    # Abandon
                    await _run_op(
                        AbandonStaleClaimOperation, deps, state,
                        claim.entity_id, "claim", "abandon_stale_claim",
                    )
                    state.terminal_claims.add(claim.entity_id)
                elif claim.stage != ClaimStage.HYPOTHESIS:
                    # Demote SUPPORTED+ claims
                    await _run_op(
                        DemoteClaimOperation, deps, state,
                        claim.entity_id, "claim", "demote_claim",
                    )
                    demoted_any = True

        if demoted_any:
            # Re-scrutinise demoted claims
            return Scrutinize()

        # All problematic claims abandoned — check if anything is left
        return CheckCompletion()


@dataclass
class PromoteToSupported(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Try promoting HYPOTHESIS claims with passing scrutiny to SUPPORTED."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["RunVerification", "CheckCompletion"]:
        from ..operations.stage_management import PromoteClaimOperation
        from ..operations.belief_maintenance import SetRoutingDefaultsOperation
        from ..entities.claim import ClaimStage

        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        promoted_any = False

        for claim in all_claims:
            if claim.abandoned:
                continue
            if claim.stage == ClaimStage.HYPOTHESIS and claim.scrutiny_verdict == "pass":
                result = await _run_op(
                    PromoteClaimOperation, deps, state,
                    claim.entity_id, "claim", "promote_claim",
                )
                if result.success:
                    promoted_any = True
                    # Set routing defaults for the newly promoted claim
                    await _run_op(
                        SetRoutingDefaultsOperation, deps, state,
                        claim.entity_id, "claim", "set_routing_defaults",
                    )

        if promoted_any:
            return RunVerification()

        # No claims could be promoted to SUPPORTED
        return CheckCompletion()


@dataclass
class RunVerification(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Run verification tracks on SUPPORTED claims based on routing profile."""

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
            "adversarial": (AdversarialSearchOperation, "adversarial_search", "adversarial_checked"),
            "convergence": (AssessConvergenceOperation, "assess_convergence", "convergence_checked"),
            "deductive": (ValidateDeductivelyOperation, "validate_deductively", "deductive_checked"),
            "computational": (VerifyComputationallyOperation, "verify_computationally", "computational_checked"),
            "argument": (AnalyzeArgumentOperation, "analyze_argument", "argument_analyzed"),
            "contrastive": (ContrastiveEvaluationOperation, "contrastive_evaluation", "contrastive_checked"),
            "consistency": (CrossClaimConsistencyOperation, "cross_claim_consistency", "consistency_checked"),
        }

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)

        for claim in all_claims:
            if claim.abandoned or claim.stage != ClaimStage.SUPPORTED:
                continue

            for track_name, (op_class, op_name, checked_field) in track_map.items():
                activation = profile.tracks.get(track_name, TrackActivation.SKIP)
                if activation == TrackActivation.SKIP:
                    continue

                # Skip if already done
                if getattr(claim, checked_field, False):
                    continue

                await _run_op(
                    op_class, deps, state,
                    claim.entity_id, "claim", op_name,
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
    """Resolve blocking uncertainties, then deduplicate concerns."""

    depth: int = 0

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["IntegrateEvidence", "Scrutinize", "ResolveUncertainties"]:
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
            return IntegrateEvidence()

        for unc in blocking:
            result = await _run_op(
                ResolveUncertaintyOperation, deps, state,
                unc.entity_id, "uncertainty", "resolve_uncertainty",
            )
            # Graph flow control: mark affected claims for re-scrutiny
            if result.success:
                try:
                    unc_updated = await deps.repo.get("uncertainty", unc.entity_id)
                    if unc_updated.is_blocking and unc_updated.resolution is not None:
                        for cid in unc_updated.affected_claim_ids:
                            state.claims_needing_rescrutiny.add(cid)
                except Exception:
                    pass

        # Deduplicate concerns on the objective
        await _run_op(
            DeduplicateConcernsOperation, deps, state,
            state.objective_id, "objective", "deduplicate_concerns",
        )

        # Check if new blocking uncertainties were created
        remaining = await deps.repo.query(
            "uncertainty",
            objective_id=state.objective_id,
            resolution=None,
        )
        new_blocking = [u for u in remaining if u.is_blocking]

        if new_blocking and self.depth < 3:
            return ResolveUncertainties(depth=self.depth + 1)

        if new_blocking:
            # Max depth reached but still have blocking uncertainties.
            # Re-enter scrutiny so claims can be re-evaluated.
            return Scrutinize()

        return IntegrateEvidence()


@dataclass
class IntegrateEvidence(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
            if claim.stage == ClaimStage.SUPPORTED and claim.integrated_assessment is None:
                await _run_op(
                    AbductiveIntegrationOperation, deps, state,
                    claim.entity_id, "claim", "integrate_evidence",
                )

        return PromoteSupported()


@dataclass
class PromoteSupported(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
                    PromoteClaimOperation, deps, state,
                    claim.entity_id, "claim", "promote_claim",
                )
                if not result.success:
                    break

                # Re-read to get updated stage
                claim = await deps.repo.get("claim", claim.entity_id)
                current = claim.stage

                # At ROBUST: generate predictions
                if current == ClaimStage.ROBUST and not claim.predictions_generated:
                    await _run_op(
                        GeneratePredictionOperation, deps, state,
                        claim.entity_id, "claim", "generate_prediction",
                    )

                # At ACTIONABLE: record decision
                if current == ClaimStage.ACTIONABLE and not claim.decision_recorded:
                    await _run_op(
                        RecordDecisionOperation, deps, state,
                        claim.entity_id, "claim", "record_decision",
                    )

            state.verification_done.add(claim.entity_id)

        return CheckCompletion()


@dataclass
class CheckCompletion(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
    """Check whether any non-abandoned claims remain and route accordingly."""

    async def run(
        self, ctx: GraphRunContext[EpistemicGraphState, EpistemicDeps]
    ) -> Union["Synthesize", End[EpistemicResult]]:
        state = ctx.state
        deps = ctx.deps

        all_claims = await deps.repo.query("claim", objective_id=state.objective_id)
        non_abandoned = [c for c in all_claims if not c.abandoned]

        if non_abandoned:
            return Synthesize()

        # All claims abandoned or no claims exist
        return End(
            EpistemicResult(
                objective_id=state.objective_id,
                status="partial" if all_claims else "no_claims",
                successful=state.successful,
                failed=state.failed,
                errors=state.errors,
                operations_log=state.operations_log,
            )
        )


@dataclass
class Synthesize(
    BaseNode[EpistemicGraphState, EpistemicDeps, EpistemicResult]
):
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
            FreezeSnapshotOperation, deps, state,
            oid, "objective", "freeze_snapshot",
        )

        # Get the snapshot ID from the objective
        obj = await deps.repo.get("objective", oid)

        if obj.snapshot_id:
            await _run_op(
                SynthesizeReportOperation, deps, state,
                obj.snapshot_id, "snapshot", "synthesize_report",
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
        RunVerification,
        ResolveUncertainties,
        IntegrateEvidence,
        PromoteSupported,
        CheckCompletion,
        Synthesize,
    ],
    name="epistemic_pipeline",
)
