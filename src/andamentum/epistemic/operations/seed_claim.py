"""Seed Claim Operation — create a claim directly from the objective.

Used in claim-verification mode when ``Objective.claim_to_verify`` is set.
Skips the normal explore→propose path and creates a single Claim entity
whose statement is the verbatim user-provided claim, then judges each
linked evidence item against the claim (same judge used by ProposeClaimsOperation).

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

from ..entities.claim import Claim, ClaimStage
from ..entities.objective import Objective
from .base import BaseOperation, OperationInput, OperationResult


class SeedClaimOperation(BaseOperation):
    """Create a Claim entity from the objective's claim_to_verify field.

    Fires instead of ProposeClaimsOperation when claim_to_verify is set.
    Links ALL existing evidence for this objective to the new claim so
    that scrutiny and verification tracks can assess them.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
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

        # Judge each evidence item against the seed claim.
        # Same judge call that ProposeClaimsOperation uses (claims.py:369-388).
        # Sets support_judgment = supports / contradicts / no_bearing on each
        # evidence item, which compute_posterior reads for the directional score.
        judged = 0
        if self.agent_runner:
            from ..judge import apply_judgment, judge_evidence as _judge

            for eid in evidence_ids:
                ev = await self.repo.get("evidence", eid)
                if ev.support_judgment is not None:
                    continue
                if not ev.extracted_content:
                    continue
                # No cluster_status filter: clustering hasn't run yet at
                # seed-claim time (all items default to "unclustered"), so
                # this filter would be a no-op — but if it ever did fire
                # it would silently skip judging real evidence.
                judgment = await _judge(
                    claim_statement=claim.statement,
                    claim_scope=claim.scope,
                    evidence_content=ev.extracted_content,
                    evidence_source=f"{ev.source_type}: {ev.source_ref}",
                    runner=self.agent_runner,
                )
                apply_judgment(ev, judgment)
                await self.repo.save(ev)
                judged += 1

        # Mark objective as having claims proposed (same contract as
        # ProposeClaimsOperation) so the pipeline advances.
        objective.claims_proposed = True
        objective.phase = "claims_proposed"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Seed claim created: {claim.statement[:80]} ({judged} evidence items judged)",
            created_entities=[claim.entity_id],
        )
