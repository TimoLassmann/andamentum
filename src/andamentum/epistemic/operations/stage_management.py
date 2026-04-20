"""Stage management operations (Phase 7).

Gate-based claim promotion and demotion. PromoteClaimOperation validates
gate requirements before allowing a claim to advance to the next stage.
DemoteClaimOperation handles regression when scrutiny fails or new
blocking uncertainties appear.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from datetime import datetime

from .base import BaseOperation, OperationResult, WorkItem

from ..entities import Claim


class PromoteClaimOperation(BaseOperation):
    """Promote claim to next stage with gate validation.

    Validates gate requirements before allowing promotion.
    Records promotion in claim history.
    """

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        from ..gates import validate_promotion, get_next_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        # Yield to TMS — if revalidation is pending, don't promote
        if claim.needs_revalidation:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Revalidation pending — TMS must run first",
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
        question_type = None
        try:
            objective = await self.repo.get("objective", claim.objective_id)
            question_type = objective.question_type
        except Exception:
            pass  # Fall back to default thresholds

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

        # Record promotion
        old_stage = claim.stage
        claim.promotion_history.append(
            {
                "from": old_stage.value,
                "to": target_stage.value,
                "timestamp": datetime.now().isoformat(),
                "justification": f"Gate requirements met for {target_stage.value}",
            }
        )
        claim.stage = target_stage

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

    async def execute(self, work: WorkItem) -> OperationResult:
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
