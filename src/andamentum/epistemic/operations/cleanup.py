"""Cleanup operations for stale claims.

Catches claims stuck in non-terminal states that no other operation
can advance. This is a safety net -- if the system is working correctly,
this operation should rarely fire.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim entities
"""

from .base import BaseOperation, OperationResult
from ..entities import Claim
from ..patterns import WorkItem


class AbandonStaleClaimOperation(BaseOperation):
    """Abandon claims stuck at HYPOTHESIS that cannot make progress.

    No LLM calls -- purely structural cleanup.
    """

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.abandoned:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already abandoned",
            )

        claim.abandoned = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Abandoned stale claim at {claim.stage.value} "
            f"(scrutiny={claim.scrutiny_verdict}, investigations={claim.investigation_count})",
        )
