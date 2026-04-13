"""Integration tests for question-type routing in the pattern scheduler.

Tests that verification tracks fire conditionally based on the objective's
question_type, and that SetRoutingDefaultsOperation pre-marks skipped tracks.
"""

import pytest

from ..storage import InMemoryStorageBackend
from ..repository import EpistemicRepository
from ..entities.objective import Objective
from ..entities.claim import Claim
from ..primitives import ClaimStage
from ..patterns import PatternScheduler, OPERATION_TO_TRACK, WorkItem
from ..operations import SetRoutingDefaultsOperation, OPERATION_CLASSES


def _make_objective(obj_id: str, **kwargs) -> Objective:
    """Helper: create an Objective with entity_id == objective_id (required by repo)."""
    return Objective(entity_id=obj_id, objective_id=obj_id, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# OPERATION_TO_TRACK mapping tests
# ══════════════════════════════════════════════════════════════════════════════


class TestOperationToTrackMapping:
    def test_all_verification_operations_mapped(self):
        expected = {
            "adversarial_search",
            "assess_convergence",
            "validate_deductively",
            "verify_computationally",
            "analyze_argument",
            "contrastive_evaluation",
            "cross_claim_consistency",
        }
        assert set(OPERATION_TO_TRACK.keys()) == expected

    def test_mapping_values_are_track_names(self):
        expected_tracks = {
            "adversarial",
            "convergence",
            "deductive",
            "computational",
            "argument",
            "contrastive",
            "consistency",
        }
        assert set(OPERATION_TO_TRACK.values()) == expected_tracks


# ══════════════════════════════════════════════════════════════════════════════
# SetRoutingDefaultsOperation tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSetRoutingDefaultsOperation:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

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
        work = WorkItem(
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
        work = WorkItem(
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
        work = WorkItem(
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
        work = WorkItem(
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
# Routing filter integration tests (PatternScheduler)
# ══════════════════════════════════════════════════════════════════════════════


class TestRoutingFilter:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    @pytest.mark.asyncio
    async def test_exploratory_filters_adversarial(self, repo):
        """For exploratory question, adversarial_search should not appear in pending work."""
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
            adversarial_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-1")
        ops = {w.operation for w in work_items}

        # Adversarial should be filtered out for exploratory
        assert "adversarial_search" not in ops

    @pytest.mark.asyncio
    async def test_verificatory_includes_adversarial(self, repo):
        """For verificatory question, adversarial_search should be in pending work."""
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
            adversarial_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-2")
        ops = {w.operation for w in work_items}

        assert "adversarial_search" in ops

    @pytest.mark.asyncio
    async def test_no_question_type_includes_all(self, repo):
        """Backward compat: no question_type means all verification tracks fire."""
        obj = _make_objective("obj-3", description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-3",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-3")
        ops = {w.operation for w in work_items}

        # All verification tracks should be present
        assert "adversarial_search" in ops
        assert "assess_convergence" in ops
        assert "validate_deductively" in ops
        assert "verify_computationally" in ops
        assert "contrastive_evaluation" in ops
        assert "cross_claim_consistency" in ops

    @pytest.mark.asyncio
    async def test_explanatory_includes_contrastive(self, repo):
        """Explanatory includes contrastive evaluation (PRIMARY)."""
        obj = _make_objective(
            "obj-4",
            description="test",
            phase="claims_proposed",
            question_type="explanatory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-4",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            contrastive_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-4")
        ops = {w.operation for w in work_items}

        assert "contrastive_evaluation" in ops

    @pytest.mark.asyncio
    async def test_verificatory_skips_contrastive(self, repo):
        """Verificatory skips contrastive evaluation."""
        obj = _make_objective(
            "obj-5",
            description="test",
            phase="claims_proposed",
            question_type="verificatory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-5",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            contrastive_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-5")
        ops = {w.operation for w in work_items}

        assert "contrastive_evaluation" not in ops

    @pytest.mark.asyncio
    async def test_non_routable_operations_always_included(self, repo):
        """Operations not in OPERATION_TO_TRACK should always appear."""
        obj = _make_objective(
            "obj-6",
            description="test",
            phase="claims_proposed",
            question_type="exploratory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-6",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-6")
        ops = {w.operation for w in work_items}

        # set_routing_defaults is not in OPERATION_TO_TRACK, should always appear
        assert "set_routing_defaults" in ops

    @pytest.mark.asyncio
    async def test_exploratory_includes_consistency(self, repo):
        """Exploratory includes cross_claim_consistency (PRIMARY)."""
        obj = _make_objective(
            "obj-7",
            description="test",
            phase="claims_proposed",
            question_type="exploratory",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-7",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            consistency_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-7")
        ops = {w.operation for w in work_items}

        assert "cross_claim_consistency" in ops

    @pytest.mark.asyncio
    async def test_predictive_skips_consistency(self, repo):
        """Predictive skips cross_claim_consistency."""
        obj = _make_objective(
            "obj-8",
            description="test",
            phase="claims_proposed",
            question_type="predictive",
        )
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-8",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            consistency_checked=False,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-8")
        ops = {w.operation for w in work_items}

        assert "cross_claim_consistency" not in ops


# ══════════════════════════════════════════════════════════════════════════════
# Promotion pattern tests (new fields required)
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionWithNewFields:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    @pytest.mark.asyncio
    async def test_promotion_blocked_without_contrastive(self, repo):
        """Promotion from SUPPORTED to PROVISIONAL requires contrastive_checked=True."""
        obj = _make_objective("obj-1", description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            contrastive_checked=False,  # Missing!
            consistency_checked=True,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-1")

        # Should not have a SUPPORTED->PROVISIONAL promotion (contrastive_checked is False)
        assert not any(
            w.operation == "promote_claim" and w.entity_id == claim.entity_id
            for w in work_items
        )

    @pytest.mark.asyncio
    async def test_promotion_succeeds_with_all_tracks(self, repo):
        """Promotion from SUPPORTED works when all checked fields are True."""
        obj = _make_objective("obj-2", description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(
            statement="test",
            objective_id="obj-2",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            contrastive_checked=True,
            consistency_checked=True,
        )
        await repo.save(claim)

        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-2")
        promote_ops = [
            w
            for w in work_items
            if w.operation == "promote_claim" and w.entity_id == claim.entity_id
        ]

        assert len(promote_ops) == 1

    @pytest.mark.asyncio
    async def test_routing_defaults_enable_promotion(self, repo):
        """SetRoutingDefaultsOperation pre-marks skipped tracks, enabling promotion."""
        obj = _make_objective(
            "obj-3",
            description="test",
            phase="claims_proposed",
            question_type="verificatory",
        )
        await repo.save(obj)
        # Start with all verification tracks done except contrastive/consistency (skipped by verificatory)
        claim = Claim(
            statement="test",
            objective_id="obj-3",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            computational_checked=True,
            argument_analyzed=True,
            contrastive_checked=False,  # Will be set by routing defaults
            consistency_checked=False,  # Will be set by routing defaults
        )
        await repo.save(claim)

        # Apply routing defaults
        op = SetRoutingDefaultsOperation(repo, None)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="set_routing_defaults",
        )
        result = await op.execute(work)
        assert result.success

        # Now promotion should be possible
        scheduler = PatternScheduler(repo)
        work_items = await scheduler.get_pending_work("obj-3")
        promote_ops = [
            w
            for w in work_items
            if w.operation == "promote_claim" and w.entity_id == claim.entity_id
        ]

        assert len(promote_ops) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Parameterized gate tests
# ══════════════════════════════════════════════════════════════════════════════


from ..gates import validate_promotion  # noqa: E402


class TestParameterizedGates:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

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
