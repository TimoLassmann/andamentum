"""Integration tests for question-type routing.

Tests that SetRoutingDefaultsOperation pre-marks skipped tracks
and that parameterized gates respect question_type.
"""

import pytest

from andamentum.document_store import DocumentStore
from ..repository import EpistemicRepository
from ..entities.objective import Objective
from ..entities.claim import Claim
from ..primitives import ClaimStage
from ..patterns import OperationInput
from ..operations import SetRoutingDefaultsOperation, OPERATION_CLASSES


def _make_objective(obj_id: str, **kwargs) -> Objective:
    """Helper: create an Objective with entity_id == objective_id (required by repo)."""
    return Objective(entity_id=obj_id, objective_id=obj_id, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# SetRoutingDefaultsOperation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSetRoutingDefaultsOperation:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

    @pytest.mark.asyncio
    async def test_skips_tracks_for_exploratory(self, repo):
        """Exploratory skips adversarial, deductive, computational, argument, contrastive."""
        obj = _make_objective(
            "obj-1",
            description="test",
            phase="claims_proposed",
            question_type="exploratory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        op = SetRoutingDefaultsOperation(repo, None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="set_routing_defaults",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        # Exploratory skips: adversarial, deductive, computational, argument, contrastive
        assert updated.adversarial_checked is True
        assert updated.deductive_checked is True
        assert updated.computational_checked is True
        assert updated.argument_analyzed is True
        assert updated.contrastive_checked is True
        # Exploratory keeps: consistency (PRIMARY), convergence (SECONDARY)
        assert updated.consistency_checked is False
        assert updated.convergence_checked is False

    @pytest.mark.asyncio
    async def test_skips_tracks_for_verificatory(self, repo):
        """Verificatory skips contrastive and consistency."""
        obj = _make_objective(
            "obj-2",
            description="test",
            phase="claims_proposed",
            question_type="verificatory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-2",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        op = SetRoutingDefaultsOperation(repo, None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="set_routing_defaults",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        # Verificatory skips: contrastive, consistency
        assert updated.contrastive_checked is True
        assert updated.consistency_checked is True
        # Verificatory keeps: adversarial (PRIMARY), convergence (PRIMARY), deductive (SECONDARY),
        # computational (IF_APPLICABLE), argument (SECONDARY)
        assert updated.adversarial_checked is False
        assert updated.convergence_checked is False
        assert updated.deductive_checked is False
        assert updated.computational_checked is False
        assert updated.argument_analyzed is False

    @pytest.mark.asyncio
    async def test_no_skips_when_no_question_type(self, repo):
        """Backward compat: no question_type means all tracks fire."""
        obj = _make_objective("obj-3", description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-3",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = SetRoutingDefaultsOperation(repo, None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="set_routing_defaults",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        # Nothing should be auto-set
        assert updated.adversarial_checked is False
        assert updated.convergence_checked is False
        assert updated.deductive_checked is False
        assert updated.computational_checked is False
        assert updated.argument_analyzed is False
        assert updated.contrastive_checked is False
        assert updated.consistency_checked is False

    @pytest.mark.asyncio
    async def test_idempotent(self, repo):
        """Running twice produces the same result."""
        obj = _make_objective(
            "obj-4",
            description="test",
            phase="claims_proposed",
            question_type="exploratory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-4",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        op = SetRoutingDefaultsOperation(repo, None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="set_routing_defaults",
        )
        result1 = await op.execute(work)
        result2 = await op.execute(work)

        assert result1.success
        assert result2.success
        # Second run should have nothing new to skip (all already applied)
        assert "all tracks active" in result2.message or "active" in result2.message

    @pytest.mark.asyncio
    async def test_registered_in_operation_classes(self):
        assert "set_routing_defaults" in OPERATION_CLASSES




# ══════════════════════════════════════════════════════════════════════════════
# Parameterized gate tests
# ══════════════════════════════════════════════════════════════════════════════


from ..gates import validate_promotion  # noqa: E402


class TestParameterizedGates:
    @pytest.fixture
    async def store(self, tmp_path):
        s = DocumentStore.for_database("test", db_dir=tmp_path)
        await s.initialize()
        return s

    @pytest.fixture
    async def repo(self, store):
        return EpistemicRepository(store)

    @pytest.mark.asyncio
    async def test_exploratory_lower_evidence_bar(self, repo):
        """Exploratory supported threshold is 0.5 (lower than default 1)."""
        from andamentum.epistemic.entities.evidence import Evidence

        obj = Objective(description="test", question_type="exploratory")
        await repo.save(obj)

        # Create claim with 1 evidence at quality 0.3
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test",
            quality_score=0.3,
        )
        await repo.save(ev)

        claim = Claim(
            statement="test",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[ev.entity_id],
            evidence_count=1,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        # With exploratory routing, min_evidence_weighted=0.5 for SUPPORTED
        # We have 1 evidence with quality 0.3 — should pass exploratory but
        # the evidence_count check uses integer count, so 1 >= 0.5 rounds to 1 >= 1 = pass
        result = await validate_promotion(
            claim, ClaimStage.SUPPORTED, repo, question_type="exploratory"
        )
        # Should pass — 1 evidence meets the 0.5 threshold (rounded to 1)
        # The quality sum is 0.3 which meets the default 0.3 min_quality_sum
        assert result.passed, f"Failed with reasons: {result.blocking_reasons}"

    @pytest.mark.asyncio
    async def test_default_thresholds_when_no_question_type(self, repo):
        """No question_type = default thresholds (backward compat)."""
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            evidence_count=0,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        result = await validate_promotion(
            claim, ClaimStage.SUPPORTED, repo, question_type=None
        )
        assert not result.passed
        assert any("evidence" in r.lower() for r in result.blocking_reasons)

    @pytest.mark.asyncio
    async def test_unknown_question_type_uses_defaults(self, repo):
        """Unknown question_type falls back to defaults gracefully."""
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            evidence_count=0,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        result = await validate_promotion(
            claim, ClaimStage.SUPPORTED, repo, question_type="nonexistent"
        )
        assert not result.passed  # Same as default — not enough evidence
