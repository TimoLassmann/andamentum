"""Tests for the 4-stage IBE abductive integration pipeline.

Verifies that running EnumerateCandidates → ScoreLoveliness →
ScoreLikeliness → SelectBestExplanation in sequence:
- Populates Claim.integration_candidates with the per-candidate trace
- Sets integrated_assessment / integrated_confidence /
  integrated_reasoning (the fields compute_posterior reads)
- Is idempotent at every stage
- Falls back to default candidates when enumeration produces too few
- Resets the candidate trace on demotion
"""

from ..entities import Claim, ClaimStage, Evidence, Objective
from ..operations.base import OperationInput
from ..operations.integration import (
    EnumerateCandidatesOperation,
    ScoreLikelinessOperation,
    ScoreLovelinessOperation,
    SelectBestExplanationOperation,
)


def _make_objective(obj_id: str = "obj-1") -> Objective:
    return Objective(
        entity_id=obj_id,
        objective_id=obj_id,
        description="Test objective",
        phase="claims_done",
    )


async def _run_full_ibe(repo, fake_runner, claim_id: str = "cl-1") -> Claim:
    """Run all four IBE stages on a single claim and return the updated claim."""
    for op_cls, op_name in [
        (EnumerateCandidatesOperation, "enumerate_candidates"),
        (ScoreLovelinessOperation, "score_loveliness"),
        (ScoreLikelinessOperation, "score_likeliness"),
        (SelectBestExplanationOperation, "select_best_explanation"),
    ]:
        op = op_cls(repo=repo, agent_runner=fake_runner, embedding_model="test")
        work = OperationInput(entity_id=claim_id, entity_type="claim", operation=op_name)
        await op.execute(work)
    return await repo.get("claim", claim_id)


class TestIBEIntegrationPipeline:
    async def test_full_pipeline_sets_integrated_fields(self, repo, fake_runner):
        """End-to-end IBE produces verdict + confidence + reasoning + candidate trace."""
        await repo.save(_make_objective())
        ev = Evidence(
            entity_id="ev-1",
            objective_id="obj-1",
            extracted=True,
            extracted_content="Test evidence content",
            support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=["ev-1"],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.8,
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)

        # Integrated fields are populated by Stage 4
        assert updated.integrated_assessment in ("supports", "contradicts", "insufficient")
        assert updated.integrated_confidence is not None
        assert 0.0 <= updated.integrated_confidence <= 1.0
        assert updated.integrated_reasoning is not None

        # Per-candidate trace is preserved
        assert len(updated.integration_candidates) >= 2
        assert all(c.loveliness is not None for c in updated.integration_candidates)
        assert all(c.likeliness is not None for c in updated.integration_candidates)
        # Exactly one chosen, exactly one runner-up
        chosen = [c for c in updated.integration_candidates if c.chosen]
        runner_up = [c for c in updated.integration_candidates if c.runner_up]
        assert len(chosen) == 1
        assert len(runner_up) == 1
        # Gap fields populated on the chosen candidate
        assert chosen[0].gap_loveliness is not None
        assert chosen[0].gap_likeliness is not None

    async def test_select_is_idempotent(self, repo, fake_runner):
        """SelectBestExplanation skips a claim that already has a verdict."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
        )
        await repo.save(claim)

        op = SelectBestExplanationOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="select_best_explanation",
            )
        )
        assert result.success
        assert result.did_work is False
        assert "already" in result.message.lower()

    async def test_enumerate_seeds_defaults_without_agent_runner(self, repo):
        """No agent runner → enumerate falls back to default 3 candidates."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = EnumerateCandidatesOperation(
            repo=repo, agent_runner=None, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="enumerate_candidates",
            )
        )
        assert result.success

        updated = await repo.get("claim", "cl-1")
        # Default set has 3 verdicts: supports / contradicts / insufficient
        assert len(updated.integration_candidates) == 3
        verdicts = {c.verdict for c in updated.integration_candidates}
        assert verdicts == {"supports", "contradicts", "insufficient"}

    async def test_pipeline_handles_no_evidence(self, repo, fake_runner):
        """A claim with no evidence still completes IBE (defaults + scoring)."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            adversarial_checked=True,
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)
        assert updated.integrated_assessment is not None
        assert len(updated.integration_candidates) >= 2

    async def test_demotion_resets_integration_candidates(self, repo):
        """record_demotion clears integrated_* fields AND integration_candidates."""
        from ..entities.claim import CandidateRecord

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
            integrated_reasoning="Test reasoning",
            integration_candidates=[
                CandidateRecord(
                    candidate_id="A",
                    verdict="supports",
                    description="test",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        claim.record_demotion(ClaimStage.HYPOTHESIS, "Test demotion")

        assert claim.integrated_assessment is None
        assert claim.integrated_confidence is None
        assert claim.integrated_reasoning is None
        assert claim.integration_candidates == []
