"""Tests for EvidenceJudgmentOutput cross-field invariants (K12).

The judge now returns a 3-way belief distribution; the verdict is the argmax
(``judgment_signal.argmax_verdict``). The prompt instructs:
    "If in_scope is False, the majority of points MUST go on no_bearing."
    "If in_scope is True, the majority MUST go on supports or contradicts."

The model_validator on EvidenceJudgmentOutput enforces these over the *derived*
verdict and raises ValueError on the bad combinations; pydantic-ai's
output_retries=3 (configured on the judge agent) catches the ValidationError
and re-prompts the model with the error as feedback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.epistemic.agents.output_models import EvidenceJudgmentOutput


class TestValidCombinations:
    """The three legal (in_scope, argmax-verdict) combinations."""

    def test_in_scope_supports(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            reasoning="Evidence is on-topic and points the same way as the claim.",
            belief_supports=85,
            belief_contradicts=10,
            belief_no_bearing=5,
        )
        assert result.in_scope is True
        assert result.verdict == "supports"
        assert result.confidence == pytest.approx(0.85)
        assert 0.0 < result.entropy < 1.0
        assert result.is_one_hot is False

    def test_in_scope_contradicts(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            reasoning="Evidence is on-topic and counters the claim.",
            belief_supports=10,
            belief_contradicts=88,
            belief_no_bearing=2,
        )
        assert result.verdict == "contradicts"

    def test_out_of_scope_no_bearing(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic B (different)",
            in_scope=False,
            reasoning="Evidence covers a different topic; no bearing on the claim.",
            belief_supports=5,
            belief_contradicts=5,
            belief_no_bearing=90,
        )
        assert result.in_scope is False
        assert result.verdict == "no_bearing"

    def test_in_scope_direction_uncertainty_splits_supports_contradicts(self) -> None:
        """In-scope direction doubt is a near-tie between supports/contradicts,
        with low no_bearing — high entropy, still a directional verdict."""
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            reasoning="On-topic but the evidence is genuinely mixed in direction.",
            belief_supports=50,
            belief_contradicts=45,
            belief_no_bearing=5,
        )
        assert result.verdict == "supports"  # argmax; tie-break favours supports order
        assert result.entropy > 0.7  # near-uniform over two classes → high
        assert result.is_one_hot is False


class TestInvalidCombinations:
    """The illegal combinations the model_validator must reject."""

    def test_in_scope_true_with_no_bearing_majority_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic A specifically",
                in_scope=True,
                reasoning="On-topic but I'm not sure which way it leans.",
                belief_supports=20,
                belief_contradicts=20,
                belief_no_bearing=60,
            )
        assert "in_scope=True is incompatible" in str(exc_info.value)

    def test_in_scope_false_with_supports_majority_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic B (different)",
                in_scope=False,
                reasoning="Different topic but seems supportive.",
                belief_supports=70,
                belief_contradicts=10,
                belief_no_bearing=20,
            )
        assert "in_scope=False requires the belief mass on no_bearing" in str(
            exc_info.value
        )

    def test_in_scope_false_with_contradicts_majority_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic B (different)",
                in_scope=False,
                reasoning="Different topic but seems to counter.",
                belief_supports=10,
                belief_contradicts=70,
                belief_no_bearing=20,
            )
        assert "in_scope=False requires the belief mass on no_bearing" in str(
            exc_info.value
        )

    def test_all_zero_belief_rejected(self) -> None:
        """A degenerate all-zero distribution can't be normalised → reject."""
        with pytest.raises(ValidationError):
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic A specifically",
                in_scope=True,
                reasoning="No points assigned.",
                belief_supports=0,
                belief_contradicts=0,
                belief_no_bearing=0,
            )
