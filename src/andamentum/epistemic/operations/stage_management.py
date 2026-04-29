"""Stage management operations.

Gate-based claim promotion and demotion. PromoteClaimOperation validates
gate requirements before allowing a claim to advance to the next stage.
DemoteClaimOperation handles regression when scrutiny fails or new
blocking uncertainties appear.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from datetime import datetime

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

        claim.promotion_history.append(
            {
                "from": claim.stage.value,
                "to": ClaimStage.SUPPORTED.value,
                "timestamp": datetime.now().isoformat(),
                "justification": f"Refuted by evidence: {n_con} contradicts vs {n_sup} supports",
            }
        )
        claim.stage = ClaimStage.SUPPORTED
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
    directional judgments — preserves those counts in the posterior instead
    of erasing them via abandonment.

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

        # Low confidence by design: the integration LLM did not run, we are
        # only preserving the counting-mode signal. Cap at 0.5 so the value
        # cannot be mistaken for a confident integrated verdict. The +1
        # smoothing keeps tiny-N cases (e.g. 1/0) below the cap.
        majority = max(n_sup, n_con)
        confidence = min(0.5, majority / (n_sup + n_con + 1))

        claim.promotion_history.append(
            {
                "from": claim.stage.value,
                "to": ClaimStage.SUPPORTED.value,
                "timestamp": datetime.now().isoformat(),
                "justification": (
                    f"Soft-promoted: refute threshold not met "
                    f"({n_con} contradicts vs {n_sup} supports). "
                    f"Counting signal preserved."
                ),
            }
        )
        claim.stage = ClaimStage.SUPPORTED
        claim.integrated_assessment = "insufficient"
        claim.integrated_confidence = confidence
        claim.integrated_reasoning = (
            f"Soft-promoted: refute threshold not met "
            f"({n_con} contradicts vs {n_sup} supports). "
            f"Integration LLM skipped; counting signal preserved for the posterior."
        )
        claim.confidence_score = confidence
        await self.repo.save(claim)

        await self.log_event(
            "claim_soft_promoted",
            claim.entity_id,
            {"n_supports": n_sup, "n_contradicts": n_con, "confidence": confidence},
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] → supported (insufficient: {n_sup}✓ / {n_con}⊥)",
        )
