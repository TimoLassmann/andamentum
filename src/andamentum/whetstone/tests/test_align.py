"""Mapping converted text back to the document it came from.

The cases that matter are the ones that cost 15 rewrites on a real manuscript:
a converter that puts spaces around a superscript citation, wraps italics in
asterisks, and invents table pipes. None of that text exists in the Word file,
which is why searching for it failed and why mapping it works.
"""

from __future__ import annotations

import pytest

from andamentum.whetstone.docx.align import (
    OffsetMap,
    Seed,
    colinear_chain,
    find_seeds,
)

# What Word holds, and what a converter makes of it.
DOCUMENT = (
    "A large body of work exists to categorise agentic systems according to "
    "their aim3,4. For the purposes of this article we refer to two types. "
    "The main challenge is their lack of transparency10. A black-box model "
    "provides access to text-only output."
)
MARKDOWN = (
    "A large body of work exists to categorise agentic systems according to "
    "their aim 3,4 . For the purposes of this article we refer to two types.\n\n"
    "The main challenge is their lack of transparency 10 . A *black-box* model "
    "provides access to text-only output."
)


def _mapped(markdown: str, document: str, quote: str) -> str | None:
    offsets = OffsetMap(markdown, document)
    at = markdown.find(quote)
    assert at >= 0, "fixture error: the quote is not in the markdown"
    span = offsets.map_span(at, at + len(quote))
    return None if span is None else document[span[0] : span[1]]


# ── the failures this module exists to remove ───────────────────────────


def test_a_quote_with_converter_spaces_maps_to_the_document_text() -> None:
    """`aim 3,4 .` is not in the .docx; `aim3,4.` is."""
    assert (
        _mapped(MARKDOWN, DOCUMENT, "according to their aim 3,4 .")
        == "according to their aim3,4."
    )


def test_a_quote_with_emphasis_markers_maps_to_the_plain_text() -> None:
    assert (
        _mapped(MARKDOWN, DOCUMENT, "A *black-box* model")
        == "A black-box model"
    )


def test_a_quote_spanning_a_converter_paragraph_break() -> None:
    landed = _mapped(MARKDOWN, DOCUMENT, "two types.\n\nThe main challenge")
    assert landed == "two types. The main challenge"


# ── position settles what a score cannot ────────────────────────────────


def test_repeated_sentences_map_to_their_own_occurrence() -> None:
    """A search must refuse a quote appearing twice. Position need not."""
    document = "Alpha. The same claim appears here. Beta. The same claim appears here. Gamma."
    markdown = document
    offsets = OffsetMap(markdown, document)

    first = markdown.find("The same claim appears here.")
    second = markdown.find("The same claim appears here.", first + 1)
    first_span = offsets.map_span(first, first + 28)
    second_span = offsets.map_span(second, second + 28)

    assert first_span is not None and second_span is not None
    assert first_span[0] != second_span[0]
    assert document[first_span[0] : first_span[1]].startswith("The same claim")
    assert document[second_span[0] : second_span[1]].startswith("The same claim")


# ── invented characters map nowhere, which is correct ───────────────────


def test_characters_the_converter_invented_map_nowhere() -> None:
    offsets = OffsetMap("## Heading\n\nBody text here.", "Heading Body text here.")
    # The '#' and the space after it exist in no Word run.
    assert offsets.map_span(0, 2) is None


def test_coverage_reports_how_much_of_the_source_is_real() -> None:
    offsets = OffsetMap(MARKDOWN, DOCUMENT)
    assert 0.85 < offsets.coverage < 1.0


# ── the alignment is independent of a whitespace tie ────────────────────


def test_a_span_does_not_depend_on_which_newline_the_aligner_picked() -> None:
    """Two markdown newlines collapse into one document break: two optimal
    alignments exist. Trimming the edges makes the span the same either way."""
    document = "First paragraph ends. Second paragraph starts."
    markdown = "First paragraph ends.\n\nSecond paragraph starts."
    offsets = OffsetMap(markdown, document)

    quote = "ends.\n\nSecond"
    at = markdown.find(quote)
    span = offsets.map_span(at, at + len(quote))
    assert span is not None
    assert document[span[0] : span[1]] == "ends. Second"


# ── seeds and chaining ──────────────────────────────────────────────────


def test_a_phrase_repeated_anywhere_is_not_a_seed() -> None:
    """Uniqueness on BOTH sides is what makes a seed unambiguous."""
    source = "the same words here and the same words here again"
    seeds = find_seeds(source, source, words_per_seed=3)
    assert all("the same words" not in source[s.source_start :][:14] for s in seeds)


def test_the_chain_drops_seeds_that_go_backwards() -> None:
    """A seed inside a moved block would pull the alignment sideways."""
    seeds = [
        Seed(source_start=0, target_start=0, length=5),
        Seed(source_start=10, target_start=200, length=5),  # the moved block
        Seed(source_start=20, target_start=20, length=5),
        Seed(source_start=30, target_start=30, length=5),
    ]
    chain = colinear_chain(seeds)
    starts = [(s.source_start, s.target_start) for s in chain]
    assert starts == [(0, 0), (20, 20), (30, 30)]


def test_chaining_an_empty_seed_list_is_safe() -> None:
    assert colinear_chain([]) == []


# ── degenerate inputs ───────────────────────────────────────────────────


def test_no_seeds_still_produces_a_correct_map() -> None:
    """Too short for a three-word phrase, so it falls through to plain
    alignment over the whole string."""
    offsets = OffsetMap("hi  there", "hi there")
    span = offsets.map_span(0, 9)
    assert span is not None
    assert "hi there"[span[0] : span[1]] == "hi there"


def test_empty_inputs() -> None:
    assert OffsetMap("", "document").map_span(0, 0) is None
    assert OffsetMap("markdown", "").map_span(0, 8) is None
    assert OffsetMap("", "").coverage == 0.0


def test_identical_strings_map_one_to_one() -> None:
    text = "Every character of this sentence maps straight through unchanged."
    offsets = OffsetMap(text, text)
    assert offsets.coverage == 1.0
    span = offsets.map_span(6, 15)
    assert span == (6, 15)


# ── verification: never trust the map, check it ─────────────────────────

VERIFY_DOCUMENT = (
    "A large body of work exists to categorise agentic systems according to "
    "their aim3,4. For the purposes of this article we refer to two types. "
    "The main challenge is their lack of transparency10. A black-box model "
    "provides access to text-only output. "
    "Reproducibility depends on recording the version of every tool that ran."
)
CONVERTED = (
    "A large body of work exists to categorise agentic systems according to "
    "their aim 3,4 . For the purposes of this article we refer to two types.\n\n"
    "The main challenge is their lack of transparency 10 . A *black-box* model "
    "provides access to text-only output.\n\n"
    "- Reproducibility depends on recording the version of every tool that ran."
)


@pytest.mark.parametrize(
    "quote",
    [
        "according to their aim 3,4 .",
        "The main challenge is their lack of transparency 10 .",
        "A *black-box* model provides access to text-only output.",
        "- Reproducibility depends on recording the version of every tool that ran.",
    ],
)
def test_converter_notation_costs_no_drift(quote: str) -> None:
    """Spacing, markers and pipes are not letters, so they measure zero —
    however long the quote is."""
    offsets = OffsetMap(CONVERTED, VERIFY_DOCUMENT)
    at = CONVERTED.find(quote)
    span = offsets.map_span(at, at + len(quote))
    assert span is not None
    assert offsets.drift(span, quote) == 0
    assert offsets.verifies(span, quote)


@pytest.mark.parametrize("span", [(120, 175), (155, 208), (270, 330), (0, 55)])
def test_a_span_that_landed_elsewhere_is_refused(span: tuple[int, int]) -> None:
    """Whether a converter reorders content is not knowable in advance, so the
    check has to catch a wrong span however it arose."""
    offsets = OffsetMap(CONVERTED, VERIFY_DOCUMENT)
    quote = "The main challenge is their lack of transparency 10 ."
    assert not offsets.verifies(span, quote)


def test_reordered_input_never_yields_wrong_text() -> None:
    """A converter that shuffles blocks must produce refusals, not silent
    misplacement."""
    scrambled = "\n\n".join(reversed(CONVERTED.split("\n\n")))
    offsets = OffsetMap(scrambled, VERIFY_DOCUMENT)
    for quote in (
        "according to their aim 3,4 .",
        "The main challenge is their lack of transparency 10 .",
        "A *black-box* model provides access to text-only output.",
    ):
        at = scrambled.find(quote)
        span = offsets.map_span(at, at + len(quote))
        if span is None or not offsets.verifies(span, quote):
            continue  # refused, which is a safe outcome
        landed = VERIFY_DOCUMENT[span[0] : span[1]].replace(" ", "")
        assert landed in quote.replace(" ", "").replace("*", "")


def test_a_near_duplicate_differing_by_one_digit_is_refused() -> None:
    """The case a ratio gets wrong: 412 against 413 scores 0.98."""
    document = (
        "Validation reached 412 of the sampled records. "
        "Validation reached 413 of the sampled records."
    )
    offsets = OffsetMap(document, document)
    quote = "Validation reached 412 of the sampled records."
    second = document.find("Validation reached 413")

    own = offsets.map_span(0, len(quote))
    assert own is not None and offsets.verifies(own, quote)
    assert not offsets.verifies((second, second + len(quote)), quote)


def test_drift_does_not_shrink_as_the_quote_grows() -> None:
    """A ratio hides one changed word inside a long quote. A count does not."""
    filler = "The calibration figures below are computed from every arm. " * 6
    document = filler + "The cohort of 412 patients was recruited in 2019."
    other = filler + "The cohort of 907 patients was recruited in 2019."
    offsets = OffsetMap(document, other)
    assert offsets.drift((0, len(other)), document) >= 3


def test_an_empty_quote_never_verifies() -> None:
    offsets = OffsetMap("some text here", "some text here")
    assert not offsets.verifies((0, 14), "")
