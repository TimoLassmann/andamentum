"""Tests for the ModelRetry-raising output validator."""

import pytest
from pydantic_ai import ModelRetry

from andamentum.chunker.types import NextUnitResult
from andamentum.chunker.validation import make_validator
from andamentum.chunker.windowing import Window


def _w(text: str) -> Window:
    return Window(
        text=text,
        cursor=0,
        window_end_offset=len(text),
        full_end_offset=len(text),
    )


def test_validator_passes_clean_unit():
    text = "Multiple sequence alignment is foundational. We propose a new method."
    v = make_validator(_w(text))
    out = NextUnitResult(
        found=True,
        title="Intro",
        start_anchor="Multiple sequence alignment is",
        end_anchor="propose a new method.",
        kind="prose",
    )
    assert v(out) is out  # validator returns the output unchanged


def test_validator_passes_not_found():
    text = "<nav>...</nav>"
    v = make_validator(_w(text))
    out = NextUnitResult(found=False, skip_to="</nav>")
    assert v(out) is out


def test_validator_raises_for_missing_start_anchor():
    text = "Hello world. This is a test."
    v = make_validator(_w(text))
    out = NextUnitResult(
        found=True,
        title="t",
        start_anchor="totally not in the text",
        end_anchor="This is a test.",
        kind="prose",
    )
    with pytest.raises(ModelRetry, match="start_anchor"):
        v(out)


def test_validator_raises_for_missing_end_anchor():
    text = "Hello world. This is a test."
    v = make_validator(_w(text))
    out = NextUnitResult(
        found=True,
        title="t",
        start_anchor="Hello world",
        end_anchor="bogus phrase not here",
        kind="prose",
    )
    with pytest.raises(ModelRetry, match="end_anchor"):
        v(out)


def test_validator_raises_when_end_before_start():
    text = "AAA BBB CCC DDD"
    v = make_validator(_w(text))
    out = NextUnitResult(
        found=True,
        title="t",
        start_anchor="DDD",
        end_anchor="AAA",
        kind="prose",
    )
    with pytest.raises(ModelRetry, match="after"):
        v(out)


def test_validator_raises_when_found_true_but_anchors_empty():
    text = "Hello world."
    v = make_validator(_w(text))
    out = NextUnitResult(
        found=True, title="t", start_anchor="", end_anchor="", kind="prose"
    )
    with pytest.raises(ModelRetry, match="anchor"):
        v(out)


def test_validator_raises_when_not_found_but_skip_to_empty():
    text = "junk junk junk"
    v = make_validator(_w(text))
    out = NextUnitResult(found=False, skip_to="")
    with pytest.raises(ModelRetry, match="skip_to"):
        v(out)
