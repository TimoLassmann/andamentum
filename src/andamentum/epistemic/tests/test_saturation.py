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
from ..patterns import WorkItem, WORK_PATTERNS


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
        work = WorkItem(
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
        # increments count and resets scrutiny
        op = InvestigateClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert updated.abandoned is False
        assert updated.investigation_count == 2
        assert updated.scrutiny_verdict is None


class TestInvestigationPatternFilters:
    def test_investigation_pattern_has_no_saturation_filter(self):
        """Investigation patterns should NOT have a saturated filter."""
        investigate_patterns = [
            p for p in WORK_PATTERNS if p.operation == "investigate_claim"
        ]
        assert len(investigate_patterns) >= 1
        for p in investigate_patterns:
            assert "saturated" not in p.filters, (
                f"Pattern '{p.description}' has saturated filter — should be removed"
            )

    def test_investigation_pattern_has_count_limit(self):
        """Investigation patterns must have investigation_count__lt filter."""
        investigate_patterns = [
            p for p in WORK_PATTERNS if p.operation == "investigate_claim"
        ]
        for p in investigate_patterns:
            assert "investigation_count__lt" in p.filters

    def test_exhausted_claim_does_not_match_investigation(self):
        """A claim with investigation_count >= 3 should not match investigation patterns."""
        investigate_patterns = [
            p for p in WORK_PATTERNS if p.operation == "investigate_claim"
        ]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=3,
        )
        for pattern in investigate_patterns:
            assert not pattern.matches(claim)

    def test_below_cap_matches_investigation(self):
        """A claim with investigation_count < 3 should match investigation patterns."""
        needs_resolution_patterns = [
            p for p in WORK_PATTERNS
            if p.operation == "investigate_claim"
            and p.filters.get("scrutiny_verdict") == "needs_resolution"
        ]
        assert len(needs_resolution_patterns) >= 1
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=1,
        )
        for pattern in needs_resolution_patterns:
            assert pattern.matches(claim)
