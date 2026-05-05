"""Tests for EvidenceJudgmentOutput cross-field invariants (K12).

The judge prompt instructs:
    "If in_scope is False, verdict MUST be 'no_bearing'."
    "If in_scope is True, set verdict to 'supports' or 'contradicts'."

The schema previously did not enforce these rules — small models
occasionally returned logically inconsistent combinations like
``in_scope=True, verdict='no_bearing'`` (on-topic but no direction).
The model_validator on EvidenceJudgmentOutput now raises ValueError
on the bad combinations; pydantic-ai's output_retries=3 (configured
on the judge agent) catches the ValidationError and re-prompts the
model with the error as feedback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.epistemic.agents.output_models import EvidenceJudgmentOutput


class TestValidCombinations:
    """The four legal (in_scope, verdict) combinations."""

    def test_in_scope_supports(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            verdict="supports",
            reasoning="Evidence is on-topic and points the same way as the claim.",
        )
        assert result.in_scope is True
        assert result.verdict == "supports"

    def test_in_scope_contradicts(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic A specifically",
            in_scope=True,
            verdict="contradicts",
            reasoning="Evidence is on-topic and counters the claim.",
        )
        assert result.verdict == "contradicts"

    def test_out_of_scope_no_bearing(self) -> None:
        result = EvidenceJudgmentOutput(
            claim_scope_summary="topic A",
            evidence_scope_summary="topic B (different)",
            in_scope=False,
            verdict="no_bearing",
            reasoning="Evidence covers a different topic; no bearing on the claim.",
        )
        assert result.in_scope is False
        assert result.verdict == "no_bearing"


class TestInvalidCombinations:
    """The two illegal combinations the model_validator must reject."""

    def test_in_scope_true_with_no_bearing_rejected(self) -> None:
        """In-scope evidence must have a directional verdict.

        Pre-fix: schema accepted this; the judge could shrug and
        say "on-topic but I dunno." Post-fix: ValueError; pydantic-ai
        retries the agent with the error message.
        """
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic A specifically",
                in_scope=True,
                verdict="no_bearing",
                reasoning="On-topic but I'm not sure which way it leans.",
            )
        msg = str(exc_info.value)
        assert "in_scope=True is incompatible with verdict='no_bearing'" in msg

    def test_in_scope_false_with_supports_rejected(self) -> None:
        """Out-of-scope evidence cannot support a claim by definition."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic B (different)",
                in_scope=False,
                verdict="supports",
                reasoning="Different topic but seems supportive.",
            )
        msg = str(exc_info.value)
        assert "in_scope=False requires verdict='no_bearing'" in msg

    def test_in_scope_false_with_contradicts_rejected(self) -> None:
        """Out-of-scope evidence cannot contradict a claim by definition."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceJudgmentOutput(
                claim_scope_summary="topic A",
                evidence_scope_summary="topic B (different)",
                in_scope=False,
                verdict="contradicts",
                reasoning="Different topic but seems to counter.",
            )
        msg = str(exc_info.value)
        assert "in_scope=False requires verdict='no_bearing'" in msg
