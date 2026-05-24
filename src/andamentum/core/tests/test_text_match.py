"""Tests for the unified string-matching helper.

Covers the canonical contract (markdown stripping + smart-quote stripping +
case folding + whitespace collapsing) plus each individual toggle, the
fuzzy backends, and the within-window scoping. The five existing matchers
across the codebase (whetstone/docx/anchor.py, whetstone/v3/locate.py,
chunker/validation.py, epistemic/passage_extraction.py, and the dead
whetstone/docx/text_processor.py) all migrate to call this module — so
its behaviour is the ground truth.
"""

from __future__ import annotations

import pytest

from andamentum.core.text_match import (
    Match,
    find_span,
    is_present,
    normalize_for_match,
)


# ── normalize_for_match ────────────────────────────────────────────────


def test_normalize_drops_markdown_markers() -> None:
    norm, idx_map = normalize_for_match("**bold** word")
    assert norm == "bold word"
    # The 'b' in normalized is at index 2 of original (after '**')
    assert idx_map[0] == 2


def test_normalize_folds_case() -> None:
    norm, _ = normalize_for_match("MixedCASE")
    assert norm == "mixedcase"


def test_normalize_collapses_whitespace() -> None:
    norm, _ = normalize_for_match("a   b\n\nc\td")
    assert norm == "a b c d"


def test_normalize_strips_leading_trailing_whitespace() -> None:
    norm, _ = normalize_for_match("  hello  ")
    assert norm == "hello"


def test_normalize_strips_edge_quotes() -> None:
    norm, _ = normalize_for_match('"hello world"')
    assert norm == "hello world"
    norm, _ = normalize_for_match("“hello world”")  # curly double
    assert norm == "hello world"
    norm, _ = normalize_for_match("‘hello’")  # curly single
    assert norm == "hello"


def test_normalize_does_not_strip_internal_quotes() -> None:
    norm, _ = normalize_for_match('he said "yes" loudly')
    # Quotes in the middle remain (strip is edge-only)
    assert '"yes"' in norm


def test_normalize_toggles_can_be_disabled() -> None:
    text = "**Bold** TEXT"
    norm_strict, _ = normalize_for_match(
        text,
        strip_markdown=False,
        fold_case=False,
        collapse_whitespace=False,
        strip_quotes=False,
    )
    assert norm_strict == text  # zero normalisation = identity


def test_normalize_idx_map_round_trips() -> None:
    """The idx_map must let a normalised span map back to original."""
    source = "## Heading\n\n**bold** word"
    norm, idx_map = normalize_for_match(source)
    # Find "bold" in normalised form
    pos = norm.find("bold")
    assert pos >= 0
    # Map back to original
    orig_start = idx_map[pos]
    orig_end = idx_map[pos + len("bold") - 1] + 1
    assert source[orig_start:orig_end] == "bold"


# ── find_span: byte-exact tier ─────────────────────────────────────────


def test_find_span_byte_exact_match() -> None:
    m = find_span("world", "hello world")
    assert m == Match(start=6, end=11, method="exact", score=1.0)


def test_find_span_byte_exact_fast_path_beats_normalized() -> None:
    """When the target appears byte-identical in source, return that
    match with method='exact' — even if normalization would also find
    something."""
    m = find_span("Hello", "Hello World")
    assert m is not None
    assert m.method == "exact"  # not 'normalized'


# ── find_span: normalized tier ─────────────────────────────────────────


def test_find_span_normalized_matches_through_markdown() -> None:
    """The canonical use case: LLM emits a markdown-flavoured quote, we
    locate it in plain text."""
    source = "the robust alignment is fine"
    m = find_span("**robust** alignment", source)
    assert m is not None
    assert m.method == "normalized"
    # Original-source coordinates point at the plain-text span
    assert source[m.start : m.end] == "robust alignment"


def test_find_span_normalized_case_insensitive() -> None:
    m = find_span("HELLO", "Hello World")
    assert m is not None
    assert m.method == "normalized"


def test_find_span_normalized_collapses_whitespace() -> None:
    m = find_span("hello world", "Hello   World")
    assert m is not None
    assert m.method == "normalized"


def test_find_span_normalized_strips_edge_quotes() -> None:
    """A primary smoke-time failure mode: LLMs wrap quotes in curly
    double-quotes that aren't in the source."""
    m = find_span("“hello world”", "say hello world now")
    assert m is not None
    assert m.method == "normalized"
    assert m.start == 4
    assert m.end == 15


def test_find_span_returns_none_for_missing_target() -> None:
    assert find_span("not in source", "completely different text") is None


def test_find_span_returns_none_for_empty_target() -> None:
    assert find_span("", "some text") is None


# ── find_span: within-window scoping ────────────────────────────────────


def test_find_span_within_window_limits_search() -> None:
    """The 'within' scope is critical for short-quote disambiguation —
    the same phrase may appear multiple times in a long document, and the
    caller wants the occurrence in a specific section."""
    source = "the method works the method fails"
    m = find_span("the method", source)
    assert m is not None
    assert m.start == 0  # first occurrence
    # Scope to the tail
    m = find_span("the method", source, within=(17, len(source)))
    assert m is not None
    assert m.start == 17


# ── find_span: try_exact / try_normalized toggles ──────────────────────


def test_find_span_disable_exact_tier() -> None:
    """When the caller wants only normalised matching (the v3 hallucination
    gate's contract — byte-exact would still pass through this, but the
    semantics are "verbatim in normalised space")."""
    m = find_span("hello", "Hello", try_exact=False)
    assert m is not None
    assert m.method == "normalized"


def test_find_span_disable_normalized_tier() -> None:
    """When the caller wants only byte-identical (chunker's load-bearing
    contract for FTS5)."""
    m = find_span("HELLO", "Hello", try_normalized=False)
    assert m is None  # byte-different and normalised tier disabled


# ── find_span: fuzzy tiers ─────────────────────────────────────────────


def test_find_span_fuzzy_off_by_default() -> None:
    """No fuzzy matching unless the caller explicitly opts in — keeps
    the canonical 'no hallucinated quotes' invariant by default."""
    m = find_span("hellow world", "hello world", fuzzy="off")
    assert m is None  # one extra letter, no fuzzy → no match


def test_find_span_fuzzy_sequence_recovers_close_match() -> None:
    m = find_span(
        "hellow world", "say hello world now", fuzzy="sequence", fuzzy_threshold=0.85
    )
    assert m is not None
    assert m.method == "fuzzy"
    assert m.score >= 0.85


def test_find_span_fuzzy_rapidfuzz_recovers_close_match() -> None:
    pytest.importorskip("rapidfuzz")
    m = find_span(
        "the method described",
        "we describe the method here",
        fuzzy="rapidfuzz",
        fuzzy_threshold=0.7,
    )
    assert m is not None
    assert m.method == "fuzzy"


def test_find_span_fuzzy_respects_threshold() -> None:
    """A high threshold should reject a truly-different target."""
    m = find_span(
        "completely unrelated text",
        "this source has nothing in common",
        fuzzy="sequence",
        fuzzy_threshold=0.9,
    )
    assert m is None


def test_find_span_fuzzy_low_threshold_epistemic_style() -> None:
    """The epistemic passage-extractor uses a 0.30 threshold for loose
    pointer-to-chunk attribution. The shared default 0.85 must not silently
    elevate this; the caller passes the threshold explicitly."""
    m = find_span(
        "method described",
        "we describe the method here",
        fuzzy="sequence",
        fuzzy_threshold=0.30,
    )
    assert m is not None  # would fail at 0.85


# ── is_present convenience ────────────────────────────────────────────


def test_is_present_returns_bool() -> None:
    assert is_present("world", "hello world") is True
    assert is_present("absent", "hello world") is False


# ── regression: the inconsistency cases the unification fixes ──────────


def test_smart_quotes_no_longer_break_v3_locate_pattern() -> None:
    """Before unification: B (v3 locate) did NOT strip smart quotes, so
    an LLM returning the curly-quoted form would silently fail the
    hallucination gate. The unified API handles this in the default path."""
    llm_quote = "“Adam combines AdaGrad and RMSProp.”"
    source = "We propose Adam. Adam combines AdaGrad and RMSProp."
    m = find_span(llm_quote, source)
    assert m is not None
    assert source[m.start : m.end] == "Adam combines AdaGrad and RMSProp."


def test_markdown_quote_against_plain_source() -> None:
    """The canonical v3 hallucination-gate contract: markdown-flavoured
    LLM quote against plain-text source."""
    m = find_span("**robust** alignment", "the robust alignment is fine")
    assert m is not None
    assert m.method == "normalized"
