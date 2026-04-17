"""Seed Claim Operation — create a claim directly from the objective.

Used in claim-verification mode when ``Objective.claim_to_verify`` is set.
Skips the normal explore→propose path and creates a single Claim entity
whose statement is the verbatim user-provided claim.

No LLM call — just copies a string into an entity.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

from ..entities.claim import Claim, ClaimStage
from ..entities.objective import Objective
from ..patterns import WorkItem
from .base import BaseOperation, OperationResult


class SeedClaimOperation(BaseOperation):
    """Create a Claim entity from the objective's claim_to_verify field.

    Fires instead of ProposeClaimsOperation when claim_to_verify is set.
    Links ALL existing evidence for this objective to the new claim so
    that scrutiny and verification tracks can assess them.
    """

    entity_type = "objective"

    async def execute(self, work: WorkItem) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if not objective.claim_to_verify:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No claim_to_verify set on objective",
            )

        if objective.claims_proposed:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Seed claim already created",
            )

        # Default question_type to "verificatory" when using seed-claim
        # mode. This ensures compute_posterior returns a meaningful
        # value (verificatory is in POSTERIOR_ELIGIBLE) and the correct
        # routing profile is applied for gate checks.
        if not objective.question_type:
            objective.question_type = "verificatory"

        # Collect all evidence IDs for this objective so the claim
        # is linked to every piece of gathered evidence.
        all_evidence = await self.repo.query(
            "evidence", objective_id=objective.entity_id
        )
        evidence_ids = [ev.entity_id for ev in all_evidence if ev.extracted]

        # Create the claim with the verbatim user-provided statement.
        claim = Claim(
            objective_id=objective.entity_id,
            statement=objective.claim_to_verify,
            scope="As stated by the user",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=evidence_ids,
            evidence_count=len(evidence_ids),
        )
        await self.repo.save(claim)

        # Mark objective as having claims proposed (same contract as
        # ProposeClaimsOperation) so the pipeline advances.
        objective.claims_proposed = True
        objective.phase = "claims_proposed"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Seed claim created: {claim.statement[:80]}",
            created_entities=[claim.entity_id],
        )
