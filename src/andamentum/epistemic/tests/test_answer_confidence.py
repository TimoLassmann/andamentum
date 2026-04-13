"""Tests for answer-level confidence scoring.

Tests compute_answer_confidence() which produces a checklist-style report
(pass/fail per check) rather than the legacy continuous-score model.
"""

import pytest

from ..confidence import (
    CheckResult,
    AnswerConfidenceReport,
    compute_answer_confidence,
)
from ..entities import Claim, Evidence, Uncertainty
from ..entities.claim import ClaimStage
from ..entities.objective import Objective
from ..entities.uncertainty import UncertaintyType
from ..repository import EpistemicRepository
from ..storage import InMemoryStorageBackend

OBJ_ID = "test-obj-answer"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo() -> EpistemicRepository:
    return EpistemicRepository(InMemoryStorageBackend())


async def _seed_objective(
    repo: EpistemicRepository,
    question_type: str | None = "verificatory",
    objective_id: str = OBJ_ID,
) -> Objective:
    obj = Objective(
        entity_id=objective_id,
        objective_id=objective_id,
        description="Test question",
        question_type=question_type,
    )
    await repo.save(obj)
    return obj


def _make_claim(
    objective_id: str = OBJ_ID,
    stage: ClaimStage = ClaimStage.SUPPORTED,
    scrutiny_verdict: str | None = "pass",
    adversarial_checked: bool = False,
    adversarial_balance: float | None = None,
    convergence_checked: bool = False,
    deductive_checked: bool = False,
    computational_checked: bool = False,
    contrastive_checked: bool = False,
    consistency_checked: bool = False,
    needs_revalidation: bool = False,
    abandoned: bool = False,
    evidence_ids: list[str] | None = None,
) -> Claim:
    return Claim(
        objective_id=objective_id,
        statement="Test claim",
        stage=stage,
        scrutiny_verdict=scrutiny_verdict,
        adversarial_checked=adversarial_checked,
        adversarial_balance=adversarial_balance,
        convergence_checked=convergence_checked,
        deductive_checked=deductive_checked,
        computational_checked=computational_checked,
        contrastive_checked=contrastive_checked,
        consistency_checked=consistency_checked,
        needs_revalidation=needs_revalidation,
        abandoned=abandoned,
        evidence_ids=evidence_ids or [],
    )


def _make_evidence(
    objective_id: str = OBJ_ID,
    invalidated: bool = False,
    support_judgment: str | None = "supports",
) -> Evidence:
    return Evidence(
        objective_id=objective_id,
        source_type="web",
        source_ref="https://example.com",
        extracted_content="Some content",
        extracted=True,
        invalidated=invalidated,
        support_judgment=support_judgment,
    )


def _make_uncertainty(
    objective_id: str = OBJ_ID,
    uncertainty_type: UncertaintyType = UncertaintyType.CONTRADICTION,
    resolved: bool = False,
    affected_claim_ids: list[str] | None = None,
) -> Uncertainty:
    u = Uncertainty(
        objective_id=objective_id,
        uncertainty_type=uncertainty_type,
        description="Test uncertainty",
        affected_claim_ids=affected_claim_ids or [],
    )
    if resolved:
        u.resolve("Resolved")
    return u


# =========================================================================
# 1. All checks passing (verificatory, all PRIMARY tracks done) -> HIGH
# =========================================================================


class TestAllChecksPassing:
    async def test_verificatory_all_primary_done_gives_high(self):
        """Verificatory with all PRIMARY tracks done and universal checks passing -> HIGH."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        # Create evidence that is judged and not invalidated
        ev = _make_evidence(support_judgment="supports", invalidated=False)
        await repo.save(ev)

        # Claim with all verificatory PRIMARY tracks done (adversarial, convergence)
        # plus universal checks (scrutiny pass, no revalidation)
        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        assert isinstance(report, AnswerConfidenceReport)
        assert report.level == "high"
        assert report.confidence >= 0.75

        # All checks should pass
        for check in report.checks:
            assert check.passed, f"Check {check.name} should pass but didn't: {check.detail}"

        # Should have universal checks + routing checks
        check_names = {c.name for c in report.checks}
        assert "evidence_basis" in check_names
        assert "scrutiny_complete" in check_names
        assert "uncertainties_resolved" in check_names
        assert "belief_maintenance" in check_names
        # Verificatory PRIMARY tracks: adversarial, convergence
        assert "track:adversarial" in check_names
        assert "track:convergence" in check_names


# =========================================================================
# 2. Empty run (no claims, no evidence) -> INSUFFICIENT
# =========================================================================


class TestEmptyRun:
    async def test_no_claims_no_evidence_gives_insufficient(self):
        """Empty run with no claims or evidence -> INSUFFICIENT."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        report = await compute_answer_confidence(repo, OBJ_ID)

        assert report.level == "insufficient"
        assert report.confidence < 0.25

        # evidence_basis should fail (no evidence)
        evidence_check = next(c for c in report.checks if c.name == "evidence_basis")
        assert not evidence_check.passed

        # scrutiny_complete should fail (no claims to have scrutiny on)
        scrutiny_check = next(c for c in report.checks if c.name == "scrutiny_complete")
        assert not scrutiny_check.passed

        # belief_maintenance should fail (no claims)
        belief_check = next(c for c in report.checks if c.name == "belief_maintenance")
        assert not belief_check.passed

        # Track checks should fail (no active claims)
        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        for tc in track_checks:
            assert not tc.passed


# =========================================================================
# 3. TMS stuck (needs_revalidation=True) -> belief_maintenance fails
# =========================================================================


class TestTmsStuck:
    async def test_needs_revalidation_fails_belief_maintenance(self):
        """Claim with needs_revalidation=True -> belief_maintenance check fails."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            needs_revalidation=True,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        belief_check = next(c for c in report.checks if c.name == "belief_maintenance")
        assert not belief_check.passed
        assert "revalidation" in belief_check.detail.lower()


# =========================================================================
# 4. Unresolved blocking uncertainty -> uncertainties_resolved fails
# =========================================================================


class TestUnresolvedBlockingUncertainty:
    async def test_unresolved_blocking_fails_uncertainties_resolved(self):
        """Unresolved blocking uncertainty -> uncertainties_resolved check fails."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        # Unresolved blocking uncertainty
        u = _make_uncertainty(
            uncertainty_type=UncertaintyType.CONTRADICTION,
            resolved=False,
        )
        await repo.save(u)

        report = await compute_answer_confidence(repo, OBJ_ID)

        unc_check = next(c for c in report.checks if c.name == "uncertainties_resolved")
        assert not unc_check.passed

    async def test_resolved_blocking_passes(self):
        """All blocking uncertainties resolved -> uncertainties_resolved passes."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        # Resolved blocking uncertainty
        u = _make_uncertainty(
            uncertainty_type=UncertaintyType.CONTRADICTION,
            resolved=True,
        )
        await repo.save(u)

        report = await compute_answer_confidence(repo, OBJ_ID)

        unc_check = next(c for c in report.checks if c.name == "uncertainties_resolved")
        assert unc_check.passed

    async def test_nonblocking_uncertainty_doesnt_fail(self):
        """Non-blocking uncertainties should not cause uncertainties_resolved to fail."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        # Non-blocking uncertainty (EVIDENCE_GAP), unresolved
        u = _make_uncertainty(
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            resolved=False,
        )
        await repo.save(u)

        report = await compute_answer_confidence(repo, OBJ_ID)

        unc_check = next(c for c in report.checks if c.name == "uncertainties_resolved")
        assert unc_check.passed


# =========================================================================
# 5. Partial track completion (one PRIMARY track missing) -> mixed
# =========================================================================


class TestPartialTrackCompletion:
    async def test_one_primary_track_missing_gives_mixed(self):
        """Verificatory: adversarial done but convergence missing -> mixed results."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=False,  # missing PRIMARY track
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        # adversarial track should pass
        adv_check = next(c for c in report.checks if c.name == "track:adversarial")
        assert adv_check.passed

        # convergence track should fail
        conv_check = next(c for c in report.checks if c.name == "track:convergence")
        assert not conv_check.passed

        # At least one failure present
        assert report.failures >= 1

    async def test_multi_claim_one_missing_track_fails(self):
        """If one of two active claims lacks a PRIMARY track flag, track check fails."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        c1 = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        c2 = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=False,  # this one is missing
            evidence_ids=[ev.entity_id],
        )
        await repo.save(c1)
        await repo.save(c2)

        report = await compute_answer_confidence(repo, OBJ_ID)

        conv_check = next(c for c in report.checks if c.name == "track:convergence")
        assert not conv_check.passed


# =========================================================================
# 6. Different question types produce different check sets
# =========================================================================


class TestQuestionTypeCheckSets:
    async def test_explanatory_has_deductive_contrastive_not_adversarial(self):
        """Explanatory: deductive + contrastive are PRIMARY (not adversarial)."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="explanatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            deductive_checked=True,
            contrastive_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        check_names = {c.name for c in report.checks}
        # explanatory PRIMARY: deductive, argument, contrastive
        assert "track:deductive" in check_names
        assert "track:contrastive" in check_names
        # argument track has no checked flag, so it should be skipped
        assert "track:argument" not in check_names
        # adversarial is SECONDARY for explanatory, NOT in checks
        assert "track:adversarial" not in check_names

    async def test_exploratory_has_consistency_only(self):
        """Exploratory: only consistency is PRIMARY."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="exploratory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            consistency_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        track_names = {c.name for c in track_checks}
        assert track_names == {"track:consistency"}

    async def test_predictive_has_deductive_and_computational(self):
        """Predictive: deductive + computational are PRIMARY."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="predictive")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            deductive_checked=True,
            computational_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        track_names = {c.name for c in track_checks}
        assert track_names == {"track:deductive", "track:computational"}

    async def test_comparative_has_contrastive_and_consistency(self):
        """Comparative: contrastive + consistency are PRIMARY."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="comparative")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            contrastive_checked=True,
            consistency_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        track_names = {c.name for c in track_checks}
        assert track_names == {"track:contrastive", "track:consistency"}


# =========================================================================
# 7. No question_type -> only universal checks, no track checks
# =========================================================================


class TestNoQuestionType:
    async def test_none_question_type_skips_track_checks(self):
        """question_type=None -> no track: checks, only universal checks."""
        repo = _make_repo()
        await _seed_objective(repo, question_type=None)

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        assert len(track_checks) == 0

        # Universal checks should still be present
        check_names = {c.name for c in report.checks}
        assert "evidence_basis" in check_names
        assert "scrutiny_complete" in check_names
        assert "uncertainties_resolved" in check_names
        assert "belief_maintenance" in check_names

    async def test_unknown_question_type_skips_track_checks(self):
        """Unknown question_type (not in routing table) -> no track checks."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="made_up_type_xyz")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        assert len(track_checks) == 0


# =========================================================================
# 8. Abandoned claims excluded from checks
# =========================================================================


class TestAbandonedClaimsExcluded:
    async def test_abandoned_claims_ignored(self):
        """Abandoned claims should be excluded from all checks."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        # Active claim: everything done
        active = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(active)

        # Abandoned claim: nothing done (should be ignored)
        abandoned = _make_claim(
            scrutiny_verdict=None,
            adversarial_checked=False,
            convergence_checked=False,
            abandoned=True,
        )
        await repo.save(abandoned)

        report = await compute_answer_confidence(repo, OBJ_ID)

        # The abandoned claim should not drag down any checks
        for check in report.checks:
            assert check.passed, f"Check {check.name} failed but shouldn't (abandoned claim should be ignored)"

    async def test_all_claims_abandoned_fails(self):
        """If all claims are abandoned, checks requiring active claims should fail."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        abandoned = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            abandoned=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(abandoned)

        report = await compute_answer_confidence(repo, OBJ_ID)

        # scrutiny_complete requires active claims
        scrutiny_check = next(c for c in report.checks if c.name == "scrutiny_complete")
        assert not scrutiny_check.passed

        # belief_maintenance requires active claims
        belief_check = next(c for c in report.checks if c.name == "belief_maintenance")
        assert not belief_check.passed

        # track checks require active claims
        track_checks = [c for c in report.checks if c.name.startswith("track:")]
        for tc in track_checks:
            assert not tc.passed


# =========================================================================
# Aggregation correctness
# =========================================================================


class TestAggregation:
    async def test_log_odds_and_confidence_consistent(self):
        """log_odds = passes - failures, confidence = sigmoid(log_odds)."""
        import math

        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=True,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        passes = sum(1 for c in report.checks if c.passed)
        failures = sum(1 for c in report.checks if not c.passed)

        assert report.passes == passes
        assert report.failures == failures
        assert report.log_odds == passes - failures

        expected_conf = 1.0 / (1.0 + math.exp(-report.log_odds))
        assert report.confidence == pytest.approx(expected_conf, abs=1e-6)

    async def test_report_has_correct_objective_id(self):
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        report = await compute_answer_confidence(repo, OBJ_ID)
        assert report.objective_id == OBJ_ID


# =========================================================================
# CheckResult model tests
# =========================================================================


class TestCheckResultModel:
    def test_check_result_fields(self):
        cr = CheckResult(name="test", tradition="peirce", passed=True, detail="ok")
        assert cr.name == "test"
        assert cr.tradition == "peirce"
        assert cr.passed is True
        assert cr.detail == "ok"


# =========================================================================
# Evidence basis edge cases
# =========================================================================


class TestEvidenceBasis:
    async def test_only_invalidated_evidence_fails(self):
        """If all evidence is invalidated, evidence_basis should fail."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence(invalidated=True, support_judgment="supports")
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        eb = next(c for c in report.checks if c.name == "evidence_basis")
        assert not eb.passed

    async def test_unjudged_evidence_fails(self):
        """Evidence with no support_judgment should not count toward evidence_basis."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence(support_judgment=None, invalidated=False)
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        eb = next(c for c in report.checks if c.name == "evidence_basis")
        assert not eb.passed


# =========================================================================
# Scrutiny edge cases
# =========================================================================


class TestScrutinyEdgeCases:
    async def test_scrutiny_fail_verdict_still_counts_as_complete(self):
        """scrutiny_verdict='fail' is still 'complete' (verdict exists)."""
        repo = _make_repo()
        await _seed_objective(repo, question_type=None)

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="fail",
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        sc = next(c for c in report.checks if c.name == "scrutiny_complete")
        assert sc.passed

    async def test_scrutiny_none_is_incomplete(self):
        """scrutiny_verdict=None means scrutiny is not complete."""
        repo = _make_repo()
        await _seed_objective(repo, question_type=None)

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict=None,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        sc = next(c for c in report.checks if c.name == "scrutiny_complete")
        assert not sc.passed


class TestDemotedClaimTrackRecognition:
    """TMS demotion resets checked flags but track results persist.

    A claim demoted by adversarial search has adversarial_checked=False
    (Peirce cycling) but adversarial_balance is not None. The track
    check should recognize the investigation happened.
    """

    async def test_adversarial_balance_proves_track_ran(self):
        """adversarial_checked=False but adversarial_balance set → track passes."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        # Simulates a claim demoted by adversarial: flag reset, balance persists
        claim = _make_claim(
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
            adversarial_checked=False,
            adversarial_balance=0.08,  # Devastating refutation
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        adv_check = next(c for c in report.checks if c.name == "track:adversarial")
        assert adv_check.passed, "adversarial_balance proves the track ran"

    async def test_no_balance_no_flag_means_never_ran(self):
        """adversarial_checked=False AND adversarial_balance=None → track fails."""
        repo = _make_repo()
        await _seed_objective(repo, question_type="verificatory")

        ev = _make_evidence()
        await repo.save(ev)

        claim = _make_claim(
            scrutiny_verdict="pass",
            adversarial_checked=False,
            adversarial_balance=None,
            convergence_checked=True,
            evidence_ids=[ev.entity_id],
        )
        await repo.save(claim)

        report = await compute_answer_confidence(repo, OBJ_ID)

        adv_check = next(c for c in report.checks if c.name == "track:adversarial")
        assert not adv_check.passed, "no evidence the track ever ran"
