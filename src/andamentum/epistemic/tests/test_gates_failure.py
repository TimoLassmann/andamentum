"""Tests for gate validation under failure conditions.

Verifies that gates behave correctly when:
- Repository operations throw exceptions
- Custom check functions throw
- Evidence has edge-case quality scores
- Blocking uncertainties can't be checked

These tests would catch the dangerous pattern where gates silently pass
when they should block, allowing claims to be promoted without validation.
"""

import pytest

from ..entities import Claim, ClaimStage, Evidence, Uncertainty, UncertaintyType
from ..gates import (
    validate_promotion,
    validate_current_stage,
    quality_weighted_evidence_sum,
    GateResult,
    STAGE_GATES,
)
from ..storage import InMemoryStorageBackend
from ..repository import EpistemicRepository


# ── Helpers ──────────────────────────────────────────────────────────────────


class FailingRepo(EpistemicRepository):
    """Repository that fails on specific operations."""

    def __init__(
        self,
        backend,
        fail_on_get: set[str] | None = None,
        fail_on_query: set[str] | None = None,
    ):
        super().__init__(backend)
        self.fail_on_get = fail_on_get or set()
        self.fail_on_query = fail_on_query or set()

    async def get(self, entity_type: str, entity_id: str):
        if entity_id in self.fail_on_get:
            raise RuntimeError(f"Simulated get failure for {entity_id}")
        return await super().get(entity_type, entity_id)

    async def query(self, entity_type: str, **filters):
        if entity_type in self.fail_on_query:
            raise RuntimeError(f"Simulated query failure for {entity_type}")
        return await super().query(entity_type, **filters)


def _make_claim(**overrides) -> Claim:
    """Create a claim with sensible defaults for gate testing."""
    defaults = dict(
        entity_id="c-test",
        objective_id="obj-test",
        statement="Test claim",
        stage=ClaimStage.HYPOTHESIS,
        evidence_ids=["e-1", "e-2"],
        scrutiny_verdict="pass",
    )
    defaults.update(overrides)
    return Claim(**defaults)


# ── quality_weighted_evidence_sum failure tests ──────────────────────────────


class TestQualityWeightedEvidenceSumFailure:
    """Test quality_weighted_evidence_sum when repo throws."""

    async def test_single_evidence_get_fails(self):
        """If one evidence get fails, it should be skipped (not crash)."""
        backend = InMemoryStorageBackend()
        repo = FailingRepo(backend, fail_on_get={"e-2"})

        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)

        claim = _make_claim(evidence_ids=["e-1", "e-2"])
        total = await quality_weighted_evidence_sum(claim, repo)
        assert total == pytest.approx(0.7)

    async def test_all_evidence_gets_fail(self):
        """If all evidence gets fail, sum should be 0."""
        backend = InMemoryStorageBackend()
        repo = FailingRepo(backend, fail_on_get={"e-1", "e-2"})

        claim = _make_claim(evidence_ids=["e-1", "e-2"])
        total = await quality_weighted_evidence_sum(claim, repo)
        assert total == 0.0

    async def test_evidence_with_none_quality_score(self):
        """Evidence with None quality_score should be skipped."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=None)
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.5)
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim()
        total = await quality_weighted_evidence_sum(claim, repo)
        assert total == pytest.approx(0.5)

    async def test_invalidated_evidence_excluded(self):
        """Invalidated evidence should not count toward quality sum."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.8, invalidated=True)
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.6)
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim()
        total = await quality_weighted_evidence_sum(claim, repo)
        assert total == pytest.approx(0.6)


# ── validate_promotion failure tests ─────────────────────────────────────────


class TestValidatePromotionFailure:
    """Test validate_promotion gate under various failure conditions."""

    async def test_quality_sum_below_threshold_blocks(self):
        """When quality sum is below the gate threshold, promotion should be blocked."""
        backend = InMemoryStorageBackend()
        # Fail all evidence gets so quality_sum = 0.0
        repo = FailingRepo(backend, fail_on_get={"e-1", "e-2"})

        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1", "e-2"],
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # quality_weighted_evidence_sum returns 0.0 (all gets fail silently),
        # SUPPORTED requires min_quality_sum=0.3, so this should block.
        assert not result.passed
        assert any("quality sum" in r.lower() for r in result.blocking_reasons)

    async def test_blocking_uncertainty_lookup_failure_blocks_promotion(self):
        """When uncertainty query throws, the gate MUST block promotion.

        Previously (gates.py:484-486) this only produced a warning, allowing
        claims to be promoted without uncertainty validation. Now fixed:
        if we can't verify no blocking uncertainties exist, we deny promotion.
        """
        backend = InMemoryStorageBackend()
        repo = FailingRepo(backend, fail_on_query={"uncertainty"})

        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        # Save evidence so it passes other checks
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # Must block — we can't verify no blocking uncertainties exist
        assert not result.passed
        assert any("uncertaint" in r.lower() for r in result.blocking_reasons)

    async def test_custom_check_failure_blocks_promotion(self):
        """When custom_check throws, the gate MUST block promotion.

        Previously (gates.py:500-501) this only produced a warning, allowing
        claims through without custom validation. Now fixed: a crashing
        safety check denies promotion.
        """
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        # Create a claim that meets all SUPPORTED requirements
        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        # SUPPORTED has no custom_check, so temporarily set a failing one
        gate = STAGE_GATES[ClaimStage.SUPPORTED]

        async def _failing_check(c, r):
            raise RuntimeError("Custom check exploded")

        original = gate.custom_check
        gate.custom_check = _failing_check
        try:
            result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
            # Must block — custom safety check failed
            assert not result.passed
            assert any("custom gate check" in r.lower() for r in result.blocking_reasons)
        finally:
            gate.custom_check = original

    async def test_unknown_target_stage_returns_failure(self):
        """Requesting promotion to HYPOTHESIS (no gate defined) should fail explicitly."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim()
        # HYPOTHESIS has no entry in STAGE_GATES, so validate_promotion returns failure
        result = await validate_promotion(claim, ClaimStage.HYPOTHESIS, repo)
        assert not result.passed
        assert any("unknown" in r.lower() for r in result.blocking_reasons)

    async def test_gate_blocks_with_blocking_uncertainties(self):
        """Gate must block when claim has unresolved blocking uncertainties."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        # Create a blocking uncertainty
        u = Uncertainty(
            entity_id="u-1",
            objective_id="obj-test",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Critical unknown",
            affected_claim_ids=["c-test"],
        )
        await repo.save(u)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("blocking" in r.lower() for r in result.blocking_reasons)

    async def test_gate_passes_with_resolved_uncertainties(self):
        """Gate should pass when all uncertainties are resolved."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        # Create a resolved uncertainty -- resolution is set, so the query
        # with resolution=None should filter it out
        u = Uncertainty(
            entity_id="u-1",
            objective_id="obj-test",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Was unknown, now resolved",
            affected_claim_ids=["c-test"],
            resolution="Resolved via investigation",
        )
        await repo.save(u)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # No blocking reasons from uncertainties since the only one is resolved
        assert not any("blocking" in r.lower() for r in result.blocking_reasons)

    async def test_adversarial_balance_below_threshold(self):
        """Gate should block when adversarial balance is below threshold."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        # PROVISIONAL stage requires adversarial_balance_threshold >= 0.4
        claim = _make_claim(
            stage=ClaimStage.SUPPORTED,
            evidence_ids=["e-1", "e-2"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            computational_checked=True,
            adversarial_balance=0.2,  # Below 0.4 threshold
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.3)
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.3)
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.PROVISIONAL, repo)
        assert not result.passed
        assert any("adversarial balance" in r.lower() for r in result.blocking_reasons)

    async def test_zero_quality_evidence_below_threshold(self):
        """Evidence with quality 0.0 should count toward sum as zero."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            stage=ClaimStage.SUPPORTED,
            evidence_ids=["e-1", "e-2"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            computational_checked=True,
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.0)
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.0)
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.PROVISIONAL, repo)
        # Quality sum = 0.0, PROVISIONAL requires min_quality_sum >= 0.5
        assert not result.passed
        assert any("quality sum" in r.lower() for r in result.blocking_reasons)


# ── validate_current_stage failure tests ─────────────────────────────────────


class TestValidateCurrentStageFailure:
    """Test validate_current_stage (TMS) under failure conditions."""

    async def test_evidence_get_fails_reduces_count(self):
        """CRITICAL: When evidence get fails in validate_current_stage,
        the evidence is silently not counted (gates.py:552-553 -- except Exception: pass).

        With all gets failing, valid_evidence_count = 0, and SUPPORTED requires >= 1.
        """
        backend = InMemoryStorageBackend()
        repo = FailingRepo(backend, fail_on_get={"e-1", "e-2"})

        claim = _make_claim(
            stage=ClaimStage.SUPPORTED,
            evidence_ids=["e-1", "e-2"],
        )

        result = await validate_current_stage(claim, repo)
        # With all evidence gets failing, valid_evidence_count = 0
        # SUPPORTED requires min_evidence >= 1, so this should fail
        assert not result.passed
        assert any("evidence" in r.lower() for r in result.blocking_reasons)

    async def test_quality_sum_partial_failure(self):
        """When one evidence get fails and the remaining quality is below threshold."""
        backend = InMemoryStorageBackend()
        repo = FailingRepo(backend, fail_on_get={"e-1"})

        # Save one evidence, fail on the other
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.3)
        await repo.save(e2)

        claim = _make_claim(
            stage=ClaimStage.PROVISIONAL,  # Has min_quality_sum = 0.5
            evidence_ids=["e-1", "e-2"],
        )

        result = await validate_current_stage(claim, repo)
        # One evidence get fails (not counted), the other has quality 0.3
        # PROVISIONAL needs min_quality_sum >= 0.5, so 0.3 < 0.5 blocks
        assert not result.passed

    async def test_hypothesis_always_passes(self):
        """HYPOTHESIS has no gate -- validate_current_stage should always pass."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(stage=ClaimStage.HYPOTHESIS, evidence_ids=[])
        result = await validate_current_stage(claim, repo)
        assert result.passed

    async def test_all_evidence_invalidated(self):
        """If all evidence is invalidated, claim should fail current stage validation."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.8, invalidated=True)
        e2 = Evidence(entity_id="e-2", objective_id="obj-test", quality_score=0.7, invalidated=True)
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(stage=ClaimStage.SUPPORTED, evidence_ids=["e-1", "e-2"])
        result = await validate_current_stage(claim, repo)
        assert not result.passed


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestGateEdgeCases:
    """Edge cases in gate validation."""

    async def test_empty_evidence_ids(self):
        """Claim with no evidence_ids should fail any stage requiring evidence."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(evidence_ids=[], scrutiny_verdict="pass")
        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("evidence" in r.lower() for r in result.blocking_reasons)

    async def test_negative_quality_score(self):
        """Negative quality scores should still work arithmetically."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=-0.5)
        await repo.save(e1)

        claim = _make_claim(evidence_ids=["e-1"])
        total = await quality_weighted_evidence_sum(claim, repo)
        assert total == pytest.approx(-0.5)

    async def test_gate_result_bool_conversion(self):
        """GateResult should be truthy when passed, falsy when not."""
        assert GateResult(passed=True)
        assert not GateResult(passed=False, blocking_reasons=["something"])

    async def test_non_blocking_uncertainty_does_not_block(self):
        """Non-blocking uncertainty types should not prevent promotion."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        # Create a non-blocking uncertainty (EVIDENCE_GAP is non-blocking)
        u = Uncertainty(
            entity_id="u-1",
            objective_id="obj-test",
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="Could use more evidence",
            affected_claim_ids=["c-test"],
        )
        await repo.save(u)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # EVIDENCE_GAP is non-blocking, so is_blocking=False after model_post_init
        # The gate filters for is_blocking=True, so this should not block
        assert not any("blocking" in r.lower() for r in result.blocking_reasons)

    async def test_missing_scrutiny_verdict_blocks(self):
        """Claims without scrutiny verdict should fail promotion to SUPPORTED."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            evidence_ids=["e-1"],
            scrutiny_verdict=None,  # No scrutiny yet
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("scrutiny" in r.lower() for r in result.blocking_reasons)

    async def test_degeneracy_blocks_promotion(self):
        """Claims with excessive modifications should be blocked by degeneracy detection."""
        backend = InMemoryStorageBackend()
        repo = EpistemicRepository(backend)

        claim = _make_claim(
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
            modification_count=5,  # > 3 triggers DEGEN_001
        )
        e1 = Evidence(entity_id="e-1", objective_id="obj-test", quality_score=0.7)
        await repo.save(e1)
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("degen_001" in r.lower() for r in result.blocking_reasons)
