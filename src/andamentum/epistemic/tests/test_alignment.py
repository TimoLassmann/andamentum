"""Tests for the alignment validation module."""

from dataclasses import asdict

import pytest

from ..alignment import AlignmentCheck, AlignmentResult, _get_mode_instructions


def test_alignment_result_defaults():
    result = AlignmentResult(aligned=True)
    assert result.aligned is True
    assert result.issue == ""
    assert result.suggestion == ""


def test_alignment_result_with_issue():
    result = AlignmentResult(
        aligned=False,
        issue="Subject changed from X to Y",
        suggestion="Stay focused on X",
    )
    assert result.aligned is False
    assert "Subject changed" in result.issue


def test_alignment_check_fields():
    check = AlignmentCheck(
        check_type="clarification",
        research_question="Does X affect Y?",
        output_to_validate="Does X have measurable effects on Y?",
        context="Moderate ambiguity, clarified wording",
    )
    d = asdict(check)
    assert d["check_type"] == "clarification"
    assert d["research_question"] == "Does X affect Y?"


def test_clarification_instructions():
    instructions = _get_mode_instructions("clarification")
    assert "same subject" in instructions.lower()
    assert "same breadth" in instructions.lower()


def test_assertion_instructions():
    instructions = _get_mode_instructions("assertion")
    assert "finding" in instructions.lower()
    assert "falsif" in instructions.lower() or "counterevidence" in instructions.lower()


def test_claim_instructions():
    instructions = _get_mode_instructions("claim")
    assert "falsifiable" in instructions.lower()
    assert "research question" in instructions.lower()


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown check_type"):
        _get_mode_instructions("unknown")
