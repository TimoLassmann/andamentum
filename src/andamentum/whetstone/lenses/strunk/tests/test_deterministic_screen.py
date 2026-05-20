"""Tests for DeterministicScreen. R2 only in Phase A.

The R2 regex operates on the whole section's text directly (no
sentence tokenization in the new design). Tests pass strings and
check offsets relative to the input string.
"""

from __future__ import annotations

from andamentum.whetstone.lenses.strunk.nodes.deterministic_screen import (
    _check_series_comma,
    _insert_oxford,
)


def test_r2_fires_on_missing_oxford_three_items():
    findings = _check_series_comma("Red, white and blue.")
    assert len(findings) == 1
    assert findings[0].rule_number == 2
    assert findings[0].rule_name == "series-comma"
    assert findings[0].span_text == "Red, white and blue"
    assert findings[0].suggested_replacement == "Red, white, and blue"


def test_r2_fires_on_missing_oxford_four_items():
    findings = _check_series_comma("Red, white, blue and green.")
    assert len(findings) == 1
    assert findings[0].span_text == "Red, white, blue and green"


def test_r2_does_not_fire_with_oxford_three_items():
    assert _check_series_comma("Red, white, and blue.") == []


def test_r2_does_not_fire_with_oxford_four_items():
    assert _check_series_comma("Red, white, blue, and green.") == []


def test_r2_does_not_fire_on_two_items():
    assert _check_series_comma("Red and blue.") == []


def test_r2_fires_with_or_conjunction():
    findings = _check_series_comma("Eggs, ham or toast?")
    assert len(findings) == 1
    assert findings[0].span_text == "Eggs, ham or toast"


def test_r2_offsets_relative_to_input_text():
    text = "We tried apples, oranges and pears yesterday."
    findings = _check_series_comma(text)
    assert len(findings) == 1
    f = findings[0]
    assert text[f.char_start : f.char_end] == f.span_text
    assert f.span_text == "apples, oranges and pears"


def test_r2_finds_multiple_violations_across_section():
    text = (
        "We had eggs, bacon and toast for breakfast. "
        "Later we ate chips, pickles and crackers."
    )
    findings = _check_series_comma(text)
    assert len(findings) == 2
    assert findings[0].span_text == "eggs, bacon and toast"
    assert findings[1].span_text == "chips, pickles and crackers"
    # offsets non-overlapping and in order
    assert findings[0].char_end <= findings[1].char_start


def test_insert_oxford_basic():
    assert _insert_oxford("Red, white and blue") == "Red, white, and blue"
    assert _insert_oxford("A, B, C and D") == "A, B, C, and D"


def test_r2_case_insensitive_for_conjunction():
    findings = _check_series_comma("Red, white And blue.")
    assert len(findings) == 1
