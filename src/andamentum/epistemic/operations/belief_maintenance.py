"""Truth maintenance system (TMS) operations.

Belief maintenance: invalidate evidence, cascade invalidation to dependent
claims, revalidate claims after evidence changes, and set routing defaults
for verification tracks that are skipped by question-type routing.

Depends on: base (BaseOperation, OperationResult)
Operates on: Evidence, Claim entities
"""

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import (
    Claim,
    Evidence,
)
from ..entities.claim import ClaimStage


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

    async def execute(self, work: OperationInput) -> OperationResult:
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

    Called by ``PromoteSupported`` (before its ladder loop) for every
    non-abandoned, non-cycle-capped claim past HYPOTHESIS. This operation:
    1. Checks if the claim still meets its current stage gate
    2. If yes: no action needed
    3. If no: contracts to the HIGHEST stage whose own gate currently
       passes, in ONE operation (certify-before-contract G2, docs/
       epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md §6) — a REVISION that
       preserves any verdict this run already certified (D1) — and
       invalidates any evidence derived from this claim (cascade)

    Contracting to the highest PASSING stage rather than one stage per
    firing matters for a stage-independent check (adversarial balance,
    contradict-majority): those checks fire identically at every stage
    above HYPOTHESIS, so a claim above SUPPORTED that fails one of them
    would otherwise contract one stage per pass on the SAME evidence
    fingerprint — which ``run_canary.py``'s D9(b) integrity check reads
    as a second, independent contraction with no intervening evidence
    change. Looping down in one operation call makes the whole fall-
    through — however many stages it spans — exactly one
    ``promotion_history`` entry, one ``contraction_count`` increment, and
    one op row.

    No LLM calls — purely structural graph maintenance.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..gates import (
            validate_current_stage,
            get_previous_stage,
            claim_evidence_fingerprint,
        )

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

        old_stage = claim.stage
        first_failing_reason = (
            gate_result.reason or "stage gate failed after evidence invalidation"
        )

        if not get_previous_stage(old_stage):
            # At HYPOTHESIS — can't contract further
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="At HYPOTHESIS, cannot demote further",
            )

        # Walk down the stage ladder, checking each CANDIDATE stage's own
        # gate (not old_stage's), until one passes. HYPOTHESIS has no
        # gate and always passes, so this terminates. claim.stage is
        # mutated only to drive validate_current_stage's gate lookup for
        # each candidate; it is restored below before Claim.contract
        # computes its own old_stage from claim.stage.
        target_stage = ClaimStage.HYPOTHESIS
        candidate = get_previous_stage(old_stage)
        while candidate is not None:
            claim.stage = candidate
            candidate_result = await validate_current_stage(claim, self.repo)
            if candidate_result.passed:
                target_stage = candidate
                break
            candidate = get_previous_stage(candidate)
        claim.stage = old_stage

        fingerprint = await claim_evidence_fingerprint(claim, self.repo)
        claim.contract(
            target_stage,
            justification=f"AGM contraction: {first_failing_reason}",
            fingerprint=fingerprint,
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
                ev.invalidation_reason = f"Dependent claim {claim.entity_id} contracted from {old_stage.value} to {target_stage.value}"
                ev.invalidation_cascaded = False
                await self.repo.save(ev)
                cascaded_evidence_ids.append(ev.entity_id)

        await self.log_event(
            "claim_agm_contracted",
            claim.entity_id,
            {
                "from": old_stage.value,
                "to": target_stage.value,
                "reason": first_failing_reason,
                "cascaded_evidence": cascaded_evidence_ids,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"AGM contracted from {old_stage.value} to {target_stage.value}, "
                f"verdict preserved ({first_failing_reason}), cascaded to "
                f"{len(cascaded_evidence_ids)} derived evidence, "
                f"fingerprint={fingerprint[:12]}"
            ),
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

    async def execute(self, work: OperationInput) -> OperationResult:
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
