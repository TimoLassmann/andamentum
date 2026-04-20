"""Tests for investigation cycle limiting.

Investigation cycles are capped by MAX_INVESTIGATION_ATTEMPTS=3 in
InvestigateClaimOperation. After 3 cycles, the claim is abandoned.
No other mechanism (saturation, etc.) limits investigation.
"""

import pytest

from ..entities.claim import Claim
from ..entities.objective import Objective
from ..primitives import ClaimStage
from ..operations.investigation import InvestigateClaimOperation
from ..operations.base import MAX_INVESTIGATION_ATTEMPTS
from ..patterns import OperationInput


class TestInvestigationCap:
    @pytest.fixture
    def backend(self):
        from ..storage import InMemoryStorageBackend
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        from ..repository import EpistemicRepository
        return EpistemicRepository(backend)

    @pytest.mark.asyncio
    async def test_investigation_exhausted_abandons_claim(self, repo):
        """After MAX_INVESTIGATION_ATTEMPTS, claim is abandoned."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=MAX_INVESTIGATION_ATTEMPTS,
        )
        await repo.save(claim)

        op = InvestigateClaimOperation(repo=repo, agent_runner=None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert updated.abandoned is True

    @pytest.mark.asyncio
    async def test_investigation_below_cap_continues(self, repo):
        """Below MAX_INVESTIGATION_ATTEMPTS, investigation proceeds."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=1,
        )
        await repo.save(claim)

        # With no agent_runner, investigation creates no stubs but still
        # increments count. Scrutiny reset moved to graph node.
        op = InvestigateClaimOperation(repo=repo, agent_runner=None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert updated.abandoned is False
        assert updated.investigation_count == 2
        assert updated.scrutiny_verdict == "needs_resolution"  # unchanged by operation
