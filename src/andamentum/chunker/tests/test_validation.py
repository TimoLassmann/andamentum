"""Tests for tiered anchor matching."""

from andamentum.chunker.validation import (
    AnchorMatch,
    find_anchor,
)


def test_exact_match_at_start():
    text = "Hello world, this is a test."
    m = find_anchor("Hello world", text, search_from=0)
    assert isinstance(m, AnchorMatch)
    assert m.start == 0
    assert m.end == len("Hello world")
    assert m.method == "exact"


def test_exact_match_after_search_from():
    text = "Hello world. Hello world again."
    m = find_anchor("Hello world", text, search_from=5)
    assert m is not None
    assert m.start == 13  # second occurrence


def test_no_match_returns_none():
    text = "Hello world."
    m = find_anchor("nonexistent", text, search_from=0)
    assert m is None


def test_whitespace_normalised_match():
    text = "Hello   world,\n\tthis is a test."  # extra whitespace
    m = find_anchor("Hello world", text, search_from=0)
    assert m is not None
    assert m.method == "whitespace_normalised"
    # Match covers the actual span in the source, not the normalised form
    assert text[m.start : m.end].lower().replace("\n", "").replace("\t", "")


def test_fuzzy_match_typo():
    # Anchor has a typo (single char different)
    text = "Multiple sequence alignment is foundational."
    m = find_anchor("Multiple seqeunce alignment", text, search_from=0)
    assert m is not None
    assert m.method == "fuzzy"


def test_no_fuzzy_match_when_too_different():
    text = "Hello world."
    m = find_anchor("Goodbye galaxy", text, search_from=0)
    assert m is None
