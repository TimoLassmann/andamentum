"""Stage management operations.

Gate-based claim promotion and demotion. PromoteClaimOperation validates
gate requirements before allowing a claim to advance to the next stage.
DemoteClaimOperation handles regression when scrutiny fails or new
blocking uncertainties appear.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import Claim
from ..entities.claim import ClaimStage


class PromoteClaimOperation(BaseOperation):
    """Promote claim to next stage with gate validation.

    Validates gate requirements before allowing promotion.
    Records promotion in claim history.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import validate_promotion, get_next_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        # Determine target stage
        target_stage = get_next_stage(claim.stage)
        if not target_stage:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Claim at {claim.stage.value} cannot be promoted further",
            )

        # Get objective's question_type for routing-aware gate thresholds
        objective = await self.repo.get("objective", claim.objective_id)
        question_type = objective.question_type

        # Validate gate requirements
        gate_result = await validate_promotion(
            claim, target_stage, self.repo, question_type=question_type
        )

        if not gate_result.passed:
            await self.log_event(
                "gate_failed",
                claim.entity_id,
                {
                    "target_stage": target_stage.value,
                    "reasons": gate_result.blocking_reasons,
                },
            )
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=gate_result.blocking_reasons[0]
                if gate_result.blocking_reasons
                else "Gate failed",
                validation_errors=gate_result.blocking_reasons,
            )

        # Record promotion (canonical writer: claim.record_promotion).
        old_stage = claim.stage
        claim.record_promotion(
            old_stage,
            target_stage,
            justification=f"Gate requirements met for {target_stage.value}",
        )

        # Compute and set confidence score
        from ..gates import compute_confidence_score, quality_weighted_evidence_sum

        quality_sum = await quality_weighted_evidence_sum(claim, self.repo)
        evidence_count = max(1, claim.evidence_count)
        avg_quality = quality_sum / evidence_count
        claim.confidence_score = compute_confidence_score(
            target_stage, avg_quality, adversarial_balance=claim.adversarial_balance
        )

        await self.repo.save(claim)

        await self.log_event(
            "claim_promoted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] → {target_stage.value}",
        )


class DemoteClaimOperation(BaseOperation):
    """Demote claim to previous stage.

    Used when scrutiny fails or new blocking uncertainties are found.

    NOTE: This does NOT cascade to derived evidence. Normal demotion means
    the claim didn't meet promotion requirements — the evidence is still
    valid and may support other claims. Only TMS-triggered demotion
    (RevalidateClaimOperation) cascades, because there the evidence
    foundation itself has been undermined.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import get_previous_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        target_stage = get_previous_stage(claim.stage)
        if not target_stage:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Claim at HYPOTHESIS cannot be demoted",
            )

        old_stage = claim.stage
        claim.record_demotion(
            target_stage, justification="Scrutiny failure or blocking uncertainty"
        )

        await self.repo.save(claim)

        await self.log_event(
            "claim_demoted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Demoted to {target_stage.value}",
        )


class PromoteAsRefutedOperation(BaseOperation):
    """Promote a HYPOTHESIS claim to SUPPORTED with integrated_assessment="contradicts".

    Bypasses the scrutiny gate's "verdict must be pass" rule. Intended as an
    escape valve when the evidence overwhelmingly contradicts the claim —
    without this path, such claims get abandoned and their contradiction
    signal drops out of the posterior entirely.

    Integration LLM is skipped for these claims because integrated_assessment
    is pre-set; the existing "if integrated_assessment is not None: skip"
    check in IntegrateEvidence handles this correctly.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import count_support_contradict, is_refuted_by_evidence

        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )
        if claim.stage != ClaimStage.HYPOTHESIS:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Refuted-promotion only runs on HYPOTHESIS, not {claim.stage.value}",
            )
        if not await is_refuted_by_evidence(claim, self.repo):
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Evidence balance does not meet refutation threshold",
            )

        n_sup, n_con = await count_support_contradict(claim, self.repo)
        # Confidence from imbalance; cap at 0.9 because this is a mechanical
        # count, not holistic integration.
        confidence = min(0.9, n_con / (n_con + max(1, n_sup)))

        claim.record_promotion(
            claim.stage,
            ClaimStage.SUPPORTED,
            justification=f"Refuted by evidence: {n_con} contradicts vs {n_sup} supports",
        )
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = confidence
        claim.integrated_reasoning = (
            f"Promoted as refuted: evidence balance {n_con} contradicts vs "
            f"{n_sup} supports. Confidence derived mechanically from imbalance; "
            f"integration LLM skipped."
        )
        claim.confidence_score = confidence
        await self.repo.save(claim)

        await self.log_event(
            "claim_promoted_as_refuted",
            claim.entity_id,
            {"n_supports": n_sup, "n_contradicts": n_con, "confidence": confidence},
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] → supported (refuted: {n_con}⊥ / {n_sup}✓)",
        )


class SoftPromoteOperation(BaseOperation):
    """Promote a HYPOTHESIS claim to SUPPORTED with integrated_assessment="insufficient".

    The middle path between PromoteAsRefutedOperation (strong contradicting
    evidence) and AbandonStaleClaimOperation (no signal at all). Used when
    refute-promotion declined but the linked evidence still carries
    directional judgments — promotes the claim to SUPPORTED so it can
    receive the same abductive deliberation (the IBE chain) as any other
    SUPPORTED claim, instead of being erased via abandonment.

    The verdict is NOT pre-set here. SoftPromote is a routing decision
    ("let this claim through the gate"), not an integration verdict.
    Earlier versions of this operation did pre-set
    integrated_assessment="insufficient" — that was a sensible
    optimization when integration was a single rubber-stamp LLM call,
    but it became wrong after the IBE 4-stage refactor (commit
    3affc1f), where the integration step produces qualitatively richer
    output than a hard-coded label. Setting integrated_assessment here
    short-circuits the IBE chain (whose entry condition is
    integrated_assessment is None) and discards real reasoning.

    Returns success=False when n_supports + n_contradicts == 0 so the graph
    falls through to abandon, which is the honest terminal when there is
    genuinely nothing to say.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import count_support_contradict

        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )
        if claim.stage != ClaimStage.HYPOTHESIS:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message=f"Soft-promote only runs on HYPOTHESIS, not {claim.stage.value}",
                did_work=False,
            )

        n_sup, n_con = await count_support_contradict(claim, self.repo)
        if n_sup + n_con == 0:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="No directional evidence; abandonment is appropriate",
                did_work=False,
            )

        claim.record_promotion(
            claim.stage,
            ClaimStage.SUPPORTED,
            justification=(
                f"Soft-promoted: refute threshold not met "
                f"({n_con} contradicts vs {n_sup} supports). "
                f"Promoted to SUPPORTED so the IBE chain can produce a "
                f"calibrated verdict on the directional evidence."
            ),
        )
        # integrated_assessment / integrated_confidence / integrated_reasoning
        # are deliberately left as None. The IBE chain (EnumerateCandidates
        # → ScoreLoveliness → ScoreLikeliness → SelectBestExplanation) will
        # populate them based on the actual evidence pattern.
        await self.repo.save(claim)

        await self.log_event(
            "claim_soft_promoted",
            claim.entity_id,
            {"n_supports": n_sup, "n_contradicts": n_con},
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"[{claim.statement[:60]}] → supported "
                f"(soft-promote, deferring verdict to IBE: {n_sup}✓ / {n_con}⊥)"
            ),
        )
