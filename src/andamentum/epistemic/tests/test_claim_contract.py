"""Tests for Claim.contract() / Claim.reset_for_reinquiry() (D2, D4).

Certify-before-contract plan (docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md),
Phase A. Covers:

S8 — Claim.contract() never nulls integrated_assessment,
     integrated_confidence, integrated_reasoning, integration_candidates,
     or scrutiny_verdict. Claim.reset_for_reinquiry() is the only place
     those track/verdict resets live.
S9 — The Lakatos degeneracy checks (DEGEN_001, DEGEN_003) never fire
     because of in-process AGM contractions. A contraction does not call
     record_modification.
"""

from typing import Any

from ..entities import Claim, ClaimStage
from ..entities.claim import CandidateRecord
from ..gates import check_degeneracy


def _certified_claim(**overrides: Any) -> Claim:
    """A SUPPORTED claim carrying a full IBE-certified verdict."""
    defaults: dict[str, Any] = dict(
        statement="Caffeine improves alertness",
        objective_id="obj-1",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        integrated_assessment="supports",
        integrated_confidence=0.82,
        integrated_reasoning="Two independent RCTs converge on a positive effect.",
        integration_candidates=[
            CandidateRecord(
                candidate_id="A",
                verdict="supports",
                description="Evidence converges on support.",
                loveliness=0.7,
                likeliness=0.8,
                chosen=True,
            )
        ],
    )
    defaults.update(overrides)
    return Claim(**defaults)


class TestContractPreservesVerdict:
    """S8: contract() never touches the verdict fields."""

    def test_contract_preserves_integrated_assessment(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.integrated_assessment == "supports"

    def test_contract_preserves_integrated_confidence(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.integrated_confidence == 0.82

    def test_contract_preserves_integrated_reasoning(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert (
            claim.integrated_reasoning
            == "Two independent RCTs converge on a positive effect."
        )

    def test_contract_preserves_integration_candidates(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert len(claim.integration_candidates) == 1
        assert claim.integration_candidates[0].chosen is True

    def test_contract_preserves_scrutiny_verdict(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.scrutiny_verdict == "pass"

    def test_contract_preserves_adversarial_balance(self):
        claim = _certified_claim(adversarial_checked=True, adversarial_balance=0.9)
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.adversarial_checked is True
        assert claim.adversarial_balance == 0.9

    def test_contract_updates_stage_and_history(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.stage == ClaimStage.HYPOTHESIS
        assert len(claim.promotion_history) == 1
        entry = claim.promotion_history[0]
        assert entry.from_stage == ClaimStage.SUPPORTED
        assert entry.to_stage == ClaimStage.HYPOTHESIS
        assert entry.justification == "Evidence invalidated by TMS cascade"

    def test_contract_records_count_and_fingerprint(self):
        claim = _certified_claim()
        assert claim.contraction_count == 0
        assert claim.contraction_fingerprint is None
        claim.contract(
            ClaimStage.HYPOTHESIS,
            justification="Evidence invalidated by TMS cascade",
            fingerprint="fp-1",
        )
        assert claim.contraction_count == 1
        assert claim.contraction_fingerprint == "fp-1"

    def test_two_contractions_increment_count_and_update_fingerprint(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="first", fingerprint="fp-1"
        )
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="second", fingerprint="fp-2"
        )
        assert claim.contraction_count == 2
        assert claim.contraction_fingerprint == "fp-2"
        # Still preserved after two rounds.
        assert claim.integrated_assessment == "supports"


class TestResetForReinquiryIsTheOnlyResetPath:
    """S8: reset_for_reinquiry() is the sole holder of the reset list."""

    def test_reset_for_reinquiry_nulls_verdict_fields(self):
        claim = _certified_claim(
            adversarial_checked=True,
            convergence_checked=True,
            convergence_verdict="CONVERGENT",
            deductive_checked=True,
            computational_checked=True,
            contrastive_checked=True,
            consistency_checked=True,
        )
        claim.contraction_fingerprint = "fp-1"
        claim.reset_for_reinquiry()

        assert claim.adversarial_checked is False
        assert claim.convergence_checked is False
        assert claim.convergence_verdict is None
        assert claim.deductive_checked is False
        assert claim.computational_checked is False
        assert claim.contrastive_checked is False
        assert claim.consistency_checked is False
        assert claim.integrated_assessment is None
        assert claim.integrated_confidence is None
        assert claim.integrated_reasoning is None
        assert claim.integration_candidates == []
        assert claim.contraction_fingerprint is None

    def test_reset_for_reinquiry_does_not_touch_scrutiny_stage_or_history(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="x", fingerprint="fp-1"
        )
        history_len_before = len(claim.promotion_history)
        stage_before = claim.stage

        claim.reset_for_reinquiry()

        assert claim.scrutiny_verdict == "pass"
        assert claim.stage == stage_before
        assert len(claim.promotion_history) == history_len_before

    def test_record_demotion_uses_reset_for_reinquiry_reset_list(self):
        """record_demotion's resets are the same list reset_for_reinquiry defines.

        Exercised via record_demotion (the scrutiny-driven "start over"
        path) landing at HYPOTHESIS, which is the one branch that resets.
        """
        claim = _certified_claim(
            adversarial_checked=True,
            convergence_checked=True,
            convergence_verdict="CONVERGENT",
            deductive_checked=True,
            computational_checked=True,
            contrastive_checked=True,
            consistency_checked=True,
        )
        claim.record_demotion(ClaimStage.HYPOTHESIS, "scrutiny failed")

        assert claim.scrutiny_verdict is None  # record_demotion's own reset
        assert claim.adversarial_checked is False
        assert claim.convergence_checked is False
        assert claim.convergence_verdict is None
        assert claim.deductive_checked is False
        assert claim.computational_checked is False
        assert claim.contrastive_checked is False
        assert claim.consistency_checked is False
        assert claim.integrated_assessment is None
        assert claim.integrated_confidence is None
        assert claim.integrated_reasoning is None
        assert claim.integration_candidates == []

    def test_record_demotion_still_increments_modification_count(self):
        """Unlike contract(), record_demotion IS a Lakatos-tracked modification."""
        claim = _certified_claim()
        assert claim.modification_count == 0
        claim.record_demotion(ClaimStage.HYPOTHESIS, "scrutiny failed")
        assert claim.modification_count == 1
        assert len(claim.modification_timestamps) == 1


class TestContractionCannotTripDegeneracyBrake:
    """S9: DEGEN_001/DEGEN_003 never fire from in-process contractions alone."""

    def test_contract_does_not_call_record_modification(self):
        claim = _certified_claim()
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="x", fingerprint="fp-1"
        )
        assert claim.modification_count == 0
        assert claim.modification_timestamps == []

    def test_three_contractions_leave_degeneracy_checks_empty(self):
        claim = _certified_claim()
        for i in range(3):
            claim.contract(
                ClaimStage.HYPOTHESIS,
                justification=f"contraction {i}",
                fingerprint=f"fp-{i}",
            )
        assert claim.contraction_count == 3
        assert check_degeneracy(claim) == []

    def test_many_contractions_still_leave_degeneracy_checks_empty(self):
        """DEGEN_001 fires on modification_count > 3; contractions must never feed it."""
        claim = _certified_claim()
        for i in range(10):
            claim.contract(
                ClaimStage.HYPOTHESIS,
                justification=f"contraction {i}",
                fingerprint=f"fp-{i}",
            )
        assert claim.contraction_count == 10
        assert claim.modification_count == 0
        assert check_degeneracy(claim) == []
