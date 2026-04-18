"""Tests for the abductive integration operation.

Verifies that AbductiveIntegrationOperation:
- Sets integrated_assessment, integrated_confidence, integrated_reasoning
- Is idempotent (already integrated -> noop)
- Handles no evidence gracefully
- Pattern matches at SUPPORTED + adversarial_checked + integrated_assessment=None
"""

from ..entities import Claim, ClaimStage, Evidence, Objective
from ..operations.integration import AbductiveIntegrationOperation
from ..patterns import WorkItem, WORK_PATTERNS


def _make_objective(obj_id: str = "obj-1") -> Objective:
    return Objective(
        entity_id=obj_id, objective_id=obj_id,
        description="Test objective", phase="claims_done",
    )


class TestAbductiveIntegration:
    async def test_sets_integrated_fields(self, repo, fake_runner):
        """Integration sets all three integrated_* fields."""
        await repo.save(_make_objective())
        ev = Evidence(
            entity_id="ev-1", objective_id="obj-1",
            extracted=True, extracted_content="Test evidence content",
            support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim", evidence_ids=["ev-1"],
            stage=ClaimStage.SUPPORTED, scrutiny_verdict="pass",
            adversarial_checked=True, adversarial_balance=0.8,
        )
        await repo.save(claim)

        op = AbductiveIntegrationOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test",
        )
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="integrate_evidence")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", "cl-1")
        assert updated.integrated_assessment == "supports"
        assert updated.integrated_confidence == 0.75
        assert updated.integrated_reasoning is not None

    async def test_idempotent(self, repo, fake_runner):
        """Already integrated claim is a noop."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
        )
        await repo.save(claim)

        op = AbductiveIntegrationOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test",
        )
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="integrate_evidence")
        result = await op.execute(work)

        assert result.success
        assert "Already integrated" in result.message

    async def test_no_agent_runner_skips(self, repo):
        """No agent runner -> skip gracefully."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = AbductiveIntegrationOperation(
            repo=repo, agent_runner=None, embedding_model="test",
        )
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="integrate_evidence")
        result = await op.execute(work)

        assert result.success
        assert "skipped" in result.message.lower()

    async def test_handles_no_evidence(self, repo, fake_runner):
        """Claim with no evidence still runs integration."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim", evidence_ids=[],
            stage=ClaimStage.SUPPORTED, adversarial_checked=True,
        )
        await repo.save(claim)

        op = AbductiveIntegrationOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test",
        )
        work = WorkItem(entity_id="cl-1", entity_type="claim", operation="integrate_evidence")
        result = await op.execute(work)

        assert result.success

    async def test_demotion_resets_integration(self, repo):
        """record_demotion should clear integrated_* fields."""
        claim = Claim(
            entity_id="cl-1", objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
            integrated_reasoning="Test reasoning",
        )
        claim.record_demotion(ClaimStage.HYPOTHESIS, "Test demotion")

        assert claim.integrated_assessment is None
        assert claim.integrated_confidence is None
        assert claim.integrated_reasoning is None


class TestIntegrationPattern:
    def test_pattern_exists(self):
        """Integration pattern must be in WORK_PATTERNS."""
        patterns = [p for p in WORK_PATTERNS if p.operation == "integrate_evidence"]
        assert len(patterns) == 1

    def test_matches_supported_with_adversarial(self):
        """Pattern matches SUPPORTED + adversarial_checked + no integration yet."""
        patterns = [p for p in WORK_PATTERNS if p.operation == "integrate_evidence"]
        claim = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            adversarial_checked=True,
            integrated_assessment=None,
        )
        assert patterns[0].matches(claim)

    def test_does_not_match_before_adversarial(self):
        """Pattern should NOT match before adversarial search."""
        patterns = [p for p in WORK_PATTERNS if p.operation == "integrate_evidence"]
        claim = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            adversarial_checked=False,
            integrated_assessment=None,
        )
        assert not patterns[0].matches(claim)

    def test_does_not_match_already_integrated(self):
        """Pattern should NOT match if already integrated."""
        patterns = [p for p in WORK_PATTERNS if p.operation == "integrate_evidence"]
        claim = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            adversarial_checked=True,
            integrated_assessment="supports",
        )
        assert not patterns[0].matches(claim)

    def test_does_not_match_hypothesis(self):
        """Pattern should NOT match at HYPOTHESIS stage."""
        patterns = [p for p in WORK_PATTERNS if p.operation == "integrate_evidence"]
        claim = Claim(
            statement="test", objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            adversarial_checked=True,
            integrated_assessment=None,
        )
        assert not patterns[0].matches(claim)
