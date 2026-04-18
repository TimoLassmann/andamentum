"""Tests that no claim state is a dead end.

Every non-terminal claim state must either match a pattern or be
explicitly abandoned. These tests verify the dead-end states identified
in the scheduling audit are all handled.
"""

import pytest
from ..entities import Claim, ClaimStage
from ..operations.cleanup import AbandonStaleClaimOperation
from ..patterns import WorkItem, WORK_PATTERNS


class TestAbandonStaleClaim:
    @pytest.mark.asyncio
    async def test_fail_exhausted_gets_abandoned(self, repo):
        """HYPOTHESIS + fail + investigation_count=3 -> abandoned."""
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=3,
        )
        await repo.save(claim)

        op = AbandonStaleClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="abandon_stale_claim")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", "cl-1")
        assert updated.abandoned is True

    @pytest.mark.asyncio
    async def test_needs_resolution_exhausted_gets_abandoned(self, repo):
        """HYPOTHESIS + needs_resolution + investigation_count=3 -> abandoned."""
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=3,
        )
        await repo.save(claim)

        op = AbandonStaleClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="abandon_stale_claim")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", "cl-1")
        assert updated.abandoned is True

    @pytest.mark.asyncio
    async def test_already_abandoned_is_noop(self, repo):
        """Already abandoned claim is a no-op."""
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=3,
            abandoned=True,
        )
        await repo.save(claim)

        op = AbandonStaleClaimOperation(repo=repo, agent_runner=None)
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="abandon_stale_claim")
        result = await op.execute(work)

        assert result.success
        assert "Already abandoned" in result.message


class TestAbandonmentPatterns:
    def test_patterns_exist(self):
        """Abandonment patterns must exist in WORK_PATTERNS."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        assert len(abandon_patterns) >= 2

    def test_fail_exhausted_matches(self):
        """HYPOTHESIS + fail + investigation_count=3 matches abandonment pattern."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=3,
        )
        assert any(p.matches(claim) for p in abandon_patterns)

    def test_needs_resolution_exhausted_matches(self):
        """HYPOTHESIS + needs_resolution + investigation_count=3 matches."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=3,
        )
        assert any(p.matches(claim) for p in abandon_patterns)

    def test_pass_does_not_match(self):
        """HYPOTHESIS + pass should NOT match abandonment (should promote instead)."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
            investigation_count=3,
        )
        assert not any(p.matches(claim) for p in abandon_patterns)

    def test_below_cap_does_not_match(self):
        """investigation_count < 3 should NOT match abandonment."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="fail",
            investigation_count=2,
        )
        assert not any(p.matches(claim) for p in abandon_patterns)

    def test_supported_stage_does_not_match(self):
        """Abandonment only fires at HYPOTHESIS stage."""
        abandon_patterns = [p for p in WORK_PATTERNS if p.operation == "abandon_stale_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="fail",
            investigation_count=3,
        )
        assert not any(p.matches(claim) for p in abandon_patterns)
