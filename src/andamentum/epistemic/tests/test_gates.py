"""Tests for stage gates, validation, and degeneracy detection."""

import pytest
from datetime import datetime, timedelta

from ..entities import Claim, ClaimStage, Evidence, Uncertainty, UncertaintyType
from ..gates import (
    STAGE_GATES,
    STAGE_HIERARCHY,
    GateResult,
    validate_promotion,
    get_next_stage,
    get_previous_stage,
    can_demote,
    check_degeneracy,
    quality_weighted_evidence_sum,
    compute_confidence_score,
    count_modifications_in_window,
)


class TestStageHierarchy:
    def test_hierarchy_order(self):
        assert STAGE_HIERARCHY[ClaimStage.HYPOTHESIS] == 0
        assert STAGE_HIERARCHY[ClaimStage.SUPPORTED] == 1
        assert STAGE_HIERARCHY[ClaimStage.PROVISIONAL] == 2
        assert STAGE_HIERARCHY[ClaimStage.ROBUST] == 3
        assert STAGE_HIERARCHY[ClaimStage.ACTIONABLE] == 4

    def test_all_stages_present(self):
        for stage in ClaimStage:
            assert stage in STAGE_HIERARCHY


class TestStageNavigation:
    def test_get_next_stage(self):
        assert get_next_stage(ClaimStage.HYPOTHESIS) == ClaimStage.SUPPORTED
        assert get_next_stage(ClaimStage.SUPPORTED) == ClaimStage.PROVISIONAL
        assert get_next_stage(ClaimStage.PROVISIONAL) == ClaimStage.ROBUST
        assert get_next_stage(ClaimStage.ROBUST) == ClaimStage.ACTIONABLE
        assert get_next_stage(ClaimStage.ACTIONABLE) is None

    def test_get_previous_stage(self):
        assert get_previous_stage(ClaimStage.HYPOTHESIS) is None
        assert get_previous_stage(ClaimStage.SUPPORTED) == ClaimStage.HYPOTHESIS
        assert get_previous_stage(ClaimStage.PROVISIONAL) == ClaimStage.SUPPORTED
        assert get_previous_stage(ClaimStage.ROBUST) == ClaimStage.PROVISIONAL
        assert get_previous_stage(ClaimStage.ACTIONABLE) == ClaimStage.ROBUST

    def test_can_demote(self):
        assert not can_demote(ClaimStage.HYPOTHESIS)
        assert can_demote(ClaimStage.SUPPORTED)
        assert can_demote(ClaimStage.PROVISIONAL)
        assert can_demote(ClaimStage.ROBUST)
        assert can_demote(ClaimStage.ACTIONABLE)


class TestStageGateDefinitions:
    def test_gates_defined_for_all_non_hypothesis_stages(self):
        assert ClaimStage.SUPPORTED in STAGE_GATES
        assert ClaimStage.PROVISIONAL in STAGE_GATES
        assert ClaimStage.ROBUST in STAGE_GATES
        assert ClaimStage.ACTIONABLE in STAGE_GATES
        assert ClaimStage.HYPOTHESIS not in STAGE_GATES

    def test_supported_gate_requirements(self):
        gate = STAGE_GATES[ClaimStage.SUPPORTED]
        assert gate.min_evidence == 1
        assert gate.requires_scrutiny is True
        assert gate.requires_adversarial is False
        assert gate.requires_convergence is False

    def test_provisional_gate_requirements(self):
        gate = STAGE_GATES[ClaimStage.PROVISIONAL]
        assert gate.min_evidence == 2
        assert gate.requires_scrutiny is True
        assert gate.requires_adversarial is True
        assert gate.requires_convergence is True
        assert gate.requires_deductive is True

    def test_robust_gate_has_custom_check(self):
        gate = STAGE_GATES[ClaimStage.ROBUST]
        assert gate.custom_check is not None
        assert gate.min_evidence == 3

    def test_actionable_gate_has_custom_check(self):
        gate = STAGE_GATES[ClaimStage.ACTIONABLE]
        assert gate.custom_check is not None

    def test_gate_describe(self):
        gate = STAGE_GATES[ClaimStage.SUPPORTED]
        desc = gate.describe()
        assert "supported" in desc.lower()
        assert "evidence" in desc.lower()


class TestGateResult:
    def test_passing_result_is_truthy(self):
        r = GateResult(passed=True)
        assert bool(r) is True

    def test_failing_result_is_falsy(self):
        r = GateResult(passed=False, blocking_reasons=["Not enough evidence"])
        assert bool(r) is False


class TestValidatePromotion:
    async def test_hypothesis_to_supported_passes(self, repo):
        e = Evidence(entity_id="e-1", objective_id="o", quality_score=0.5)
        await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.SUPPORTED, repo)
        assert result.passed

    async def test_hypothesis_to_supported_fails_no_scrutiny(self, repo):
        e = Evidence(entity_id="e-1", objective_id="o", quality_score=0.5)
        await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-1"],
            scrutiny_verdict=None,
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("scrutiny" in r.lower() for r in result.blocking_reasons)

    async def test_hypothesis_to_supported_fails_no_evidence(self, repo):
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=[],
            scrutiny_verdict="pass",
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("evidence" in r.lower() for r in result.blocking_reasons)

    async def test_supported_to_provisional_needs_verification_tracks(self, repo):
        for i in range(3):
            e = Evidence(entity_id=f"e-{i}", objective_id="o", quality_score=0.5)
            await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-0", "e-1", "e-2"],
            scrutiny_verdict="pass",
            adversarial_checked=False,
            convergence_checked=False,
            deductive_checked=False,
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.PROVISIONAL, repo)
        assert not result.passed
        assert any("adversarial" in r.lower() for r in result.blocking_reasons)

    async def test_blocking_uncertainties_prevent_promotion(self, repo):
        e = Evidence(entity_id="e-1", objective_id="o", quality_score=0.5)
        await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-1"],
            scrutiny_verdict="pass",
        )
        await repo.save(c)
        u = Uncertainty(
            entity_id="u-1",
            objective_id="o",
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="Blocking issue",
            affected_claim_ids=["c-1"],
        )
        await repo.save(u)
        result = await validate_promotion(c, ClaimStage.SUPPORTED, repo)
        assert not result.passed
        assert any("blocking" in r.lower() for r in result.blocking_reasons)

    async def test_unknown_target_stage_fails(self, repo):
        c = Claim(entity_id="c-1", objective_id="o", statement="Test")
        result = await validate_promotion(c, ClaimStage.HYPOTHESIS, repo)
        assert not result.passed


class TestDegeneracy:
    def test_no_degeneracy_for_few_modifications(self):
        c = Claim(statement="X", objective_id="o", modification_count=2)
        warnings = check_degeneracy(c)
        assert len(warnings) == 0

    def test_degen_001_excessive_modifications(self):
        c = Claim(statement="X", objective_id="o", modification_count=5)
        warnings = check_degeneracy(c)
        assert any("DEGEN_001" in w for w in warnings)

    def test_degen_003_modification_burst(self):
        now = datetime.now()
        timestamps = [
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(hours=2)).isoformat(),
            (now - timedelta(hours=3)).isoformat(),
        ]
        c = Claim(
            statement="X",
            objective_id="o",
            modification_timestamps=timestamps,
        )
        warnings = check_degeneracy(c)
        assert any("DEGEN_003" in w for w in warnings)

    def test_no_burst_for_old_modifications(self):
        old = datetime.now() - timedelta(hours=48)
        timestamps = [(old - timedelta(hours=i)).isoformat() for i in range(5)]
        c = Claim(
            statement="X",
            objective_id="o",
            modification_timestamps=timestamps,
        )
        warnings = check_degeneracy(c)
        assert not any("DEGEN_003" in w for w in warnings)


class TestCountModificationsInWindow:
    def test_empty_timestamps(self):
        assert count_modifications_in_window([], hours=24) == 0

    def test_all_within_window(self):
        now = datetime.now()
        ts = [(now - timedelta(hours=i)).isoformat() for i in range(3)]
        assert count_modifications_in_window(ts, hours=24) == 3

    def test_some_outside_window(self):
        now = datetime.now()
        ts = [
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(hours=48)).isoformat(),
        ]
        assert count_modifications_in_window(ts, hours=24) == 1


class TestQualityWeightedEvidenceSum:
    async def test_scored_evidence(self, repo):
        e1 = Evidence(entity_id="e-1", objective_id="o", quality_score=0.8)
        e2 = Evidence(entity_id="e-2", objective_id="o", quality_score=0.6)
        await repo.save(e1)
        await repo.save(e2)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            evidence_ids=["e-1", "e-2"],
        )
        total = await quality_weighted_evidence_sum(c, repo)
        assert abs(total - 1.4) < 0.01

    async def test_unscored_evidence_is_skipped(self, repo):
        """Unscored evidence contributes 0.0 to quality sum (skipped, not defaulted)."""
        e = Evidence(entity_id="e-1", objective_id="o", quality_score=None)
        await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            evidence_ids=["e-1"],
        )
        total = await quality_weighted_evidence_sum(c, repo)
        assert total == 0.0

    async def test_missing_evidence_is_skipped(self, repo):
        """Missing evidence contributes 0.0 to quality sum (skipped, not defaulted)."""
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            evidence_ids=["e-missing"],
        )
        total = await quality_weighted_evidence_sum(c, repo)
        assert total == 0.0


class TestComputeConfidenceScore:
    def test_hypothesis_low_confidence(self):
        score = compute_confidence_score(ClaimStage.HYPOTHESIS, 0.5)
        assert score == pytest.approx(0.1, abs=0.01)

    def test_supported_with_quality(self):
        score = compute_confidence_score(ClaimStage.SUPPORTED, 0.8)
        assert 0.3 < score < 0.5

    def test_robust_with_high_quality(self):
        score = compute_confidence_score(ClaimStage.ROBUST, 1.0)
        assert score >= 0.7

    def test_actionable_with_high_quality(self):
        score = compute_confidence_score(ClaimStage.ACTIONABLE, 1.0)
        assert score >= 0.85

    def test_never_exceeds_one(self):
        score = compute_confidence_score(ClaimStage.ACTIONABLE, 10.0)
        assert score <= 1.0

    def test_adversarial_balance_penalty(self):
        # No adversarial balance: no penalty
        score_no_balance = compute_confidence_score(ClaimStage.SUPPORTED, 0.8)
        # With good balance: no penalty
        score_good = compute_confidence_score(
            ClaimStage.SUPPORTED, 0.8, adversarial_balance=0.8
        )
        assert score_good == score_no_balance
        # With low balance: penalty applied
        score_challenged = compute_confidence_score(
            ClaimStage.SUPPORTED, 0.8, adversarial_balance=0.2
        )
        assert score_challenged < score_no_balance
        # Penalty magnitude: (0.6 - 0.2) * 0.3 = 0.12
        assert score_no_balance - score_challenged == pytest.approx(0.12, abs=0.01)

    def test_adversarial_balance_never_below_zero(self):
        score = compute_confidence_score(
            ClaimStage.HYPOTHESIS, 0.0, adversarial_balance=0.0
        )
        assert score >= 0.0


class TestProvisionalQualitySumThreshold:
    def test_provisional_quality_sum_is_0_5(self):
        gate = STAGE_GATES[ClaimStage.PROVISIONAL]
        assert gate.min_quality_sum == 0.5

    def test_provisional_has_adversarial_balance_threshold(self):
        gate = STAGE_GATES[ClaimStage.PROVISIONAL]
        assert gate.adversarial_balance_threshold == 0.4


class TestAdversarialBalanceGate:
    async def test_low_adversarial_balance_blocks_provisional(self, repo):
        for i in range(2):
            e = Evidence(entity_id=f"e-{i}", objective_id="o", quality_score=0.5)
            await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-0", "e-1"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            adversarial_balance=0.2,  # Below 0.4 threshold
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.PROVISIONAL, repo)
        assert not result.passed
        assert any("adversarial balance" in r.lower() for r in result.blocking_reasons)

    async def test_good_adversarial_balance_passes_gate(self, repo):
        for i in range(2):
            e = Evidence(entity_id=f"e-{i}", objective_id="o", quality_score=0.5)
            await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-0", "e-1"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            adversarial_balance=0.7,  # Above 0.4 threshold
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.PROVISIONAL, repo)
        assert result.passed

    async def test_no_adversarial_balance_does_not_block(self, repo):
        """If adversarial_balance is None (not yet computed), gate doesn't block on it."""
        for i in range(2):
            e = Evidence(entity_id=f"e-{i}", objective_id="o", quality_score=0.5)
            await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["e-0", "e-1"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            deductive_checked=True,
            adversarial_balance=None,  # Not yet computed
        )
        await repo.save(c)
        result = await validate_promotion(c, ClaimStage.PROVISIONAL, repo)
        assert result.passed


class TestAdversarialSurvivalGate:
    """Popper: surviving severe adversarial testing satisfies supporting sources."""

    async def test_high_balance_satisfies_supporting_sources(self, repo):
        """High adversarial balance with 0 direct supports should pass gate."""
        ev = Evidence(entity_id="ev-1", objective_id="o", quality_score=0.5)
        ev.support_judgment = "no_bearing"
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["ev-1"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.8,
        )
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        # Should not fail on supporting sources
        supporting_reasons = [r for r in result.blocking_reasons if "supporting" in r.lower()]
        assert len(supporting_reasons) == 0

    async def test_low_balance_does_not_satisfy(self, repo):
        """Low adversarial balance should NOT substitute for supporting sources."""
        ev = Evidence(entity_id="ev-1", objective_id="o", quality_score=0.5)
        ev.support_judgment = "no_bearing"
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["ev-1"],
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.3,
        )
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        supporting_reasons = [r for r in result.blocking_reasons if "supporting" in r.lower()]
        assert len(supporting_reasons) > 0

    async def test_adversarial_not_run_does_not_satisfy(self, repo):
        """If adversarial search hasn't run, can't claim survival."""
        ev = Evidence(entity_id="ev-1", objective_id="o", quality_score=0.5)
        ev.support_judgment = "no_bearing"
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["ev-1"],
            scrutiny_verdict="pass",
            adversarial_checked=False,
        )
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        supporting_reasons = [r for r in result.blocking_reasons if "supporting" in r.lower()]
        assert len(supporting_reasons) > 0

    async def test_direct_support_still_works(self, repo):
        """Direct supporting evidence still satisfies gate without adversarial."""
        ev = Evidence(entity_id="ev-1", objective_id="o", quality_score=0.5)
        ev.support_judgment = "supports"
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="o",
            statement="Test",
            evidence_ids=["ev-1"],
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        result = await validate_promotion(claim, ClaimStage.SUPPORTED, repo)
        supporting_reasons = [r for r in result.blocking_reasons if "supporting" in r.lower()]
        assert len(supporting_reasons) == 0
