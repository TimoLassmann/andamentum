"""Truth maintenance system (TMS) operations.

Belief maintenance: invalidate evidence, cascade invalidation to dependent
claims, revalidate claims after evidence changes, and set routing defaults
for verification tracks that are skipped by question-type routing.

Depends on: base (BaseOperation, OperationResult)
Operates on: Evidence, Claim entities
"""

from .base import BaseOperation, OperationResult, WorkItem

from ..entities import (
    Claim,
    Evidence,
)


class InvalidateEvidenceOperation(BaseOperation):
    """Cascade evidence invalidation to dependent claims.

    When evidence is marked invalidated=True, this operation:
    1. Finds all claims referencing this evidence
    2. Removes the evidence from their evidence_ids
    3. Marks invalidation_cascaded=True on the evidence

    The graph's TMS sweep handles revalidation of affected claims.
    No LLM calls — purely structural graph maintenance.
    """

    entity_type = "evidence"

    async def execute(self, work: WorkItem) -> OperationResult:
        evidence = await self.repo.get("evidence", work.entity_id)

        if not isinstance(evidence, Evidence):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Evidence",
            )

        if not evidence.invalidated:
            return OperationResult(
                success=True,
                entity_id=evidence.entity_id,
                message="Evidence not invalidated, nothing to cascade",
            )

        if evidence.invalidation_cascaded:
            return OperationResult(
                success=True,
                entity_id=evidence.entity_id,
                message="Cascade already processed",
            )

        # Find all claims in the same objective that reference this evidence
        claims = await self.repo.query(
            "claim",
            objective_id=evidence.objective_id,
        )

        affected_claim_ids: list[str] = []
        for claim in claims:
            if not isinstance(claim, Claim):
                continue
            if evidence.entity_id in claim.evidence_ids:
                claim.evidence_ids.remove(evidence.entity_id)
                claim.evidence_count = len(claim.evidence_ids)
                await self.repo.save(claim)
                affected_claim_ids.append(claim.entity_id)

        # Mark cascade complete
        evidence.invalidation_cascaded = True
        await self.repo.save(evidence)

        await self.log_event(
            "evidence_invalidation_cascaded",
            evidence.entity_id,
            {
                "reason": evidence.invalidation_reason,
                "affected_claims": affected_claim_ids,
            },
        )

        return OperationResult(
            success=True,
            entity_id=evidence.entity_id,
            message=f"Cascaded invalidation to {len(affected_claim_ids)} claims",
        )


class RevalidateClaimOperation(BaseOperation):
    """Re-validate a claim's current stage gate after evidence changes.

    Called by the graph's TMS sweep when a claim's evidence foundation
    has changed. This operation:
    1. Checks if the claim still meets its current stage gate
    2. If yes: no action needed
    3. If no: demotes one stage, resets verification flags (Peirce cycling),
       and invalidates any evidence derived from this claim (cascade)

    No LLM calls — purely structural graph maintenance.
    """

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        from ..gates import validate_current_stage, get_previous_stage

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        gate_result = await validate_current_stage(claim, self.repo)

        if gate_result.passed:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Claim still meets current stage gate",
            )

        # Gate failed — demote one stage
        old_stage = claim.stage
        target_stage = get_previous_stage(claim.stage)

        if not target_stage:
            # At HYPOTHESIS — can't demote further
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="At HYPOTHESIS, cannot demote further",
            )

        # Demote via shared helper (same flags as DemoteClaimOperation)
        claim.record_demotion(
            target_stage,
            justification=f"TMS demotion: {gate_result.reason or 'stage gate failed after evidence invalidation'}",
        )

        await self.repo.save(claim)

        # Cascade: invalidate evidence derived from this claim
        derived_evidence = await self.repo.query(
            "evidence",
            objective_id=claim.objective_id,
        )
        cascaded_evidence_ids: list[str] = []
        for ev in derived_evidence:
            if not isinstance(ev, Evidence):
                continue
            if ev.depends_on_claim_id == claim.entity_id and not ev.invalidated:
                ev.invalidated = True
                ev.invalidation_reason = f"Dependent claim {claim.entity_id} demoted from {old_stage.value} to {target_stage.value}"
                ev.invalidation_cascaded = False
                await self.repo.save(ev)
                cascaded_evidence_ids.append(ev.entity_id)

        await self.log_event(
            "claim_tms_demoted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
                "reason": gate_result.reason,
                "cascaded_evidence": cascaded_evidence_ids,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"TMS demoted from {old_stage.value} to {target_stage.value}, cascaded to {len(cascaded_evidence_ids)} derived evidence",
        )


class SetRoutingDefaultsOperation(BaseOperation):
    """Pre-mark verification tracks as checked when routing says SKIP.

    Runs before verification tracks fire (priority 4). For each verification
    track that the routing config says is SKIP for this objective's question_type,
    sets the corresponding checked field to True so that promotion patterns
    aren't blocked by tracks that will never run.

    Deterministic: no LLM calls. Idempotent: re-running on a claim where
    defaults are already set is a no-op.
    """

    entity_type = "claim"

    # Map track names to claim fields
    TRACK_TO_FIELD: dict[str, str] = {
        "adversarial": "adversarial_checked",
        "convergence": "convergence_checked",
        "deductive": "deductive_checked",
        "computational": "computational_checked",
        "argument": "argument_analyzed",
        "contrastive": "contrastive_checked",
        "consistency": "consistency_checked",
    }

    async def execute(self, work: WorkItem) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        # Get objective's question_type
        objective = await self.repo.get("objective", claim.objective_id)
        question_type = getattr(objective, "question_type", None)

        if not question_type:
            # No routing — all tracks fire (backward compat)
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="No question_type set — all tracks will fire",
            )

        from ..routing import get_active_tracks, TrackActivation

        tracks = get_active_tracks(question_type)
        skipped: list[str] = []

        for track_name, activation in tracks.items():
            if activation == TrackActivation.SKIP:
                field_name = self.TRACK_TO_FIELD.get(track_name)
                if field_name and not getattr(claim, field_name, True):
                    setattr(claim, field_name, True)
                    skipped.append(track_name)

        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=work.entity_id,
            message=f"[{claim.statement[:60]}] active: {[t for t in tracks if t not in skipped]}"
            if skipped
            else f"[{claim.statement[:60]}] all tracks active",
        )
