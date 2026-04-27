"""Tests for the anchoring helper that verifies verbatim quotes."""

from __future__ import annotations

from andamentum.whetstone.v2.anchoring import anchor_quote


def test_returns_none_for_empty_text() -> None:
    assert anchor_quote("", "section text here", "sec_001") is None


def test_returns_none_for_whitespace_only_text() -> None:
    assert anchor_quote("   ", "section text here", "sec_001") is None


def test_returns_none_for_fabricated_quote() -> None:
    """A quote that doesn't appear anywhere in the section returns None."""
    assert anchor_quote("not in source", "the actual section", "sec_001") is None


def test_succeeds_for_exact_substring_match() -> None:
    section = "The quick brown fox jumps over the lazy dog."
    q = anchor_quote("quick brown fox", section, "sec_001")
    assert q is not None
    assert q.section_id == "sec_001"
    assert q.text == "quick brown fox"
    assert section[q.char_start : q.char_end] == "quick brown fox"


def test_succeeds_for_full_section() -> None:
    section = "Just one sentence."
    q = anchor_quote(section, section, "sec_002")
    assert q is not None
    assert q.char_start == 0
    assert q.char_end == len(section)


def test_returned_text_is_sliced_from_source_not_model_input() -> None:
    """Even when fuzzy / whitespace-tolerant tier matches, the persisted
    quote text comes from the source bytes, not the model's submission."""
    section = "We saw the  cat sleeping."  # two spaces between "the" and "cat"
    q = anchor_quote("the cat", section, "sec_003")
    if q is not None:
        # However the matcher resolved it, q.text is sliced from section.
        assert q.text == section[q.char_start : q.char_end]


def test_distinct_sections_produce_distinct_quotes() -> None:
    text_a = "alpha beta gamma"
    text_b = "delta epsilon zeta"
    qa = anchor_quote("alpha", text_a, "sec_a")
    qb = anchor_quote("delta", text_b, "sec_b")
    assert qa is not None and qb is not None
    assert qa.section_id == "sec_a"
    assert qb.section_id == "sec_b"
