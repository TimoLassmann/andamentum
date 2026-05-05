"""Tests for investigation cycle limiting.

Investigation cycles are capped by PEIRCE_CYCLE_CAP=3 (the canonical
Peirce-cycling constant in ``andamentum.epistemic.thresholds``) in
InvestigateClaimOperation. After PEIRCE_CYCLE_CAP cycles the claim
is abandoned. No other mechanism (saturation, etc.) limits
investigation.
"""

import pytest

from andamentum.document_store import DocumentStore
from ..entities.claim import Claim
from ..entities.objective import Objective
from ..primitives import ClaimStage
from ..operations.investigation import InvestigateClaimOperation
from ..thresholds import PEIRCE_CYCLE_CAP
from ..operations.base import OperationInput
from ..repository import EpistemicRepository


class TestInvestigationCap:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

    @pytest.mark.asyncio
    async def test_investigation_exhausted_abandons_claim(self, repo):
        """After PEIRCE_CYCLE_CAP, claim is abandoned."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=PEIRCE_CYCLE_CAP,
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
        """Below PEIRCE_CYCLE_CAP, investigation proceeds."""
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
