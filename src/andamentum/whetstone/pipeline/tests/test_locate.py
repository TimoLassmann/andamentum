"""Tests for the locate primitive (pure, deterministic)."""

from __future__ import annotations

from andamentum.whetstone.v3.locate import is_present, locate


def test_locates_exact_substring() -> None:
    src = "The methods were significantly robust and clearly described."
    span = locate("significantly robust", src)
    assert span is not None
    assert src[span[0] : span[1]] == "significantly robust"


def test_normalised_match_ignores_markdown_and_whitespace() -> None:
    src = "We propose a **novel** approach to the problem."
    # The quote arrives markdown-flavoured / re-spaced; should still match.
    span = locate("a *novel*  approach", src)
    assert span is not None
    assert "novel" in src[span[0] : span[1]]


def test_absent_quote_returns_none() -> None:
    assert locate("completely different text", "the methods were robust") is None
    assert is_present("nope", "the methods were robust") is False


def test_within_scopes_the_search() -> None:
    # "the model" appears twice; scoping to the second section picks the 2nd.
    src = "Section A: the model is fast. Section B: the model is accurate."
    second = src.index("Section B")
    span = locate("the model", src, within=(second, len(src)))
    assert span is not None
    assert span[0] >= second  # matched the occurrence inside section B


def test_within_excludes_outside_matches() -> None:
    src = "intro mentions baselines. body has no such word here."
    body = src.index("body")
    # "baselines" only exists in the intro, outside the scoped range → not found.
    assert locate("baselines", src, within=(body, len(src))) is None


def test_offsets_are_original_coordinates() -> None:
    src = "## Heading\n\nThe quick brown fox."
    span = locate("quick brown", src)
    assert span is not None
    assert src[span[0] : span[1]] == "quick brown"


def test_empty_quote_returns_none() -> None:
    assert locate("", "anything") is None
    assert locate("   ", "anything") is None
