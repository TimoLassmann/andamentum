"""The shape of a written number: parse, compare within precision, write another like it."""

from __future__ import annotations

import pytest

from andamentum.compose.harness.bindings.written import (
    format_like,
    parse_written,
    scale_for,
)


def like(template: str, shown: float) -> str:
    parsed = parse_written(template)
    assert parsed is not None
    return format_like(parsed, shown)


@pytest.mark.parametrize(
    ("template", "shown", "expected"),
    [
        ("23.4%", 25.15, "25.2%"),  # half rounds up, not to even
        ("23.4%", 25.05, "25.1%"),
        ("48", 52.4, "52"),
        ("1,234", 12345.0, "12,345"),
        ("1234", 12345.0, "12345"),
        (".03", 0.071, ".07"),
        ("0.03", 0.071, "0.07"),
        ("−0.5", -0.25, "−0.3"),  # typographic minus is kept
        ("-0.5", -0.25, "-0.3"),
        ("-0.5", 0.04, "0.0"),  # no negative zero
        ("-0.5", 0.6, "0.6"),
        ("+2.0", 3.0, "+3.0"),
        ("0.20", 0.5, "0.50"),
    ],
)
def test_format_like_keeps_the_shape(template: str, shown: float, expected: str):
    assert like(template, shown) == expected


def test_parse_reads_value_decimals_and_unit():
    parsed = parse_written("−1,234.50%")
    assert parsed is not None
    assert (parsed.value, parsed.decimals, parsed.percent, parsed.thousands) == (
        -1234.5,
        2,
        True,
        True,
    )
    assert parse_written(".05").value == 0.05  # type: ignore[union-attr]
    assert parse_written(".05").leading_zero is False  # type: ignore[union-attr]
    assert parse_written("p") is None
    assert parse_written("12 mg") is None


def test_scale_is_one_or_a_hundred_within_the_written_precision():
    percent = parse_written("23.4%")
    assert percent is not None
    assert scale_for(percent, 23.4) == 1.0
    assert scale_for(percent, 0.234) == 100.0
    assert scale_for(percent, 0.2344) == 100.0  # rounds to 23.4
    assert scale_for(percent, 0.236) is None
    plain = parse_written("0.23")
    assert plain is not None
    assert scale_for(plain, 0.234) == 1.0
    assert scale_for(plain, 23.4) is None  # a plain number is never read as percent


def test_a_written_zero_does_not_match_a_small_non_zero_value():
    """The "0" of "week 0" bound to a p-value of .003 in a live draft."""
    zero = parse_written("0")
    assert zero is not None
    assert scale_for(zero, 0.003) is None
    assert scale_for(zero, 0) == 1.0
