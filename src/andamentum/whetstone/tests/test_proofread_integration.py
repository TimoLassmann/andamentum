"""Tests for the proofread → whetstone integration.

Two layers:
  1. Adapter unit tests against synthetic ProofreadResults.
  2. End-to-end test through ``review_document`` confirming proofread
     findings appear in ``ReviewResult.deterministic_findings`` when
     ``proofread=True`` (default) and don't when ``proofread=False``.
"""

from __future__ import annotations

from andamentum.proofread import (
    AdverbStats,
    DuplicateWordFinding,
    PassiveFinding,
    ProofreadResult,
    ReadabilityScores,
    Span,
    WeakOpenerFinding,
    WeaselFinding,
)
from andamentum.whetstone import review_document
from andamentum.whetstone.structural.proofread_adapter import (
    _expand_to_word_boundaries,
    _locate_in_section,
    proofread_to_findings,
)
from andamentum.whetstone.structural.types import SectionRef


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


def _empty_proofread_result() -> ProofreadResult:
    """A ProofreadResult with no flags. Building block for selective tests."""
    return ProofreadResult(
        readability=ReadabilityScores(
            smog_index=10.0,
            flesch_kincaid_grade=10.0,
            flesch_reading_ease=60.0,
            gunning_fog=10.0,
            coleman_liau_index=10.0,
            automated_readability_index=10.0,
            word_count=100,
            sentence_count=5,
            avg_sentence_length=20.0,
            avg_syllables_per_word=1.5,
        ),
        weasel_words=[],
        passive_voice=[],
        duplicate_words=[],
        weak_openers=[],
        adverbs=AdverbStats(adverb_count=0, adverb_density=0.0),
        summary="",
    )


def test_expand_to_word_boundaries_snaps_to_whitespace() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    # Expand a span over "brown" by 5 chars each side
    start, end = _expand_to_word_boundaries(text, 10, 15, padding=5)
    # Should snap to whitespace boundaries
    assert text[start:end].strip() == text[start:end]
    assert "brown" in text[start:end]


def test_locate_in_section_finds_correct_section() -> None:
    sections = [
        SectionRef(id="sec_001", title="Intro", text="x", char_start=0, char_end=100),
        SectionRef(
            id="sec_002", title="Methods", text="x", char_start=100, char_end=300
        ),
    ]
    first = _locate_in_section(sections, 50)
    assert first is not None and first.id == "sec_001"
    second = _locate_in_section(sections, 150)
    assert second is not None and second.id == "sec_002"
    assert _locate_in_section(sections, 500) is None


def test_weasel_finding_becomes_anchored_finding() -> None:
    markdown = (
        "The methods were significantly more robust than prior work. "
        "We measured everything in triplicate."
    )
    sections = [
        SectionRef(
            id="sec_001",
            title="Methods",
            text=markdown,
            char_start=0,
            char_end=len(markdown),
        )
    ]
    pr = _empty_proofread_result()
    pr.weasel_words = [
        WeaselFinding(
            word="significantly",
            span=Span(start=17, end=30),
            sentence_index=0,
        )
    ]

    findings = proofread_to_findings(
        result=pr, markdown=markdown, sections=sections
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == "minor"
    assert finding.confidence == "high"
    assert finding.priority == "consider"
    assert finding.source == "deterministic"
    assert finding.category == "style:weasel"
    assert "significantly" in finding.title
    assert len(finding.quotes) == 1
    q = finding.quotes[0]
    assert q.section_id == "sec_001"
    # The anchor should be wider than just the flagged word
    assert len(q.text) > len("significantly")
    assert "significantly" in q.text
    # In-section offsets — char_start/char_end refer to the section's text
    assert q.text == markdown[q.char_start : q.char_end]


def test_all_categories_produce_unique_categories() -> None:
    """Each proofread category maps to a distinct whetstone category tag."""
    markdown = (
        "There are very many things that were measured measured. "
        "It is clear that significantly more work was done."
    )
    sections = [
        SectionRef(
            id="sec_001",
            title="Section",
            text=markdown,
            char_start=0,
            char_end=len(markdown),
        )
    ]
    # Build one of each finding type at known spans
    pr = _empty_proofread_result()
    pr.weasel_words = [
        WeaselFinding(word="very", span=Span(start=10, end=14), sentence_index=0)
    ]
    pr.passive_voice = [
        PassiveFinding(
            matched_text="were measured",
            span=Span(start=30, end=43),
            sentence_index=0,
        )
    ]
    pr.duplicate_words = [
        DuplicateWordFinding(
            word="measured", span=Span(start=35, end=52), sentence_index=0
        )
    ]
    pr.weak_openers = [
        WeakOpenerFinding(
            matched_text="There are",
            span=Span(start=0, end=9),
            sentence_index=0,
        )
    ]

    findings = proofread_to_findings(
        result=pr, markdown=markdown, sections=sections
    )
    categories = {f.category for f in findings}
    assert categories == {
        "style:weasel",
        "style:passive",
        "style:duplicate_word",
        "style:weak_opener",
    }


def test_flag_outside_any_section_is_skipped() -> None:
    """Flags whose char span doesn't fall in any section are silently dropped."""
    markdown = "The brown fox jumps over."
    sections = [
        SectionRef(
            id="sec_001",
            title="Section",
            text=markdown,
            char_start=0,
            char_end=10,  # only covers "The brown "
        )
    ]
    pr = _empty_proofread_result()
    # Span at chars 18-22 ("over") — past the section's char_end of 10
    pr.weasel_words = [
        WeaselFinding(word="over", span=Span(start=18, end=22), sentence_index=0)
    ]

    findings = proofread_to_findings(
        result=pr, markdown=markdown, sections=sections
    )
    assert findings == []


# ---------------------------------------------------------------------------
# End-to-end through review_document
# ---------------------------------------------------------------------------


PAPER_WITH_STYLE_ISSUES = """
# A Study of Some Things

## Introduction

There are many factors that very significantly influence outcomes.
The study was conducted by us in 2024, and we measured everything
that could be measured.

## Methods

We obtained data from various sources. The data was analyzed using
standard methods. The the results were then computed.

## Results

It is clear that our findings are very interesting.

## Conflicts of interest

The author declares no conflicts of interest.

## Data availability

Data are available on request from the corresponding author.

## Ethics

This study was approved by the IRB.
"""


async def test_review_document_includes_proofread_findings_by_default() -> None:
    """proofread=True is the default; surface-style flags appear in
    deterministic_findings. The exact mix depends on proofread's internal
    matchers; here we assert the integration flows at least two categories."""
    result = await review_document(PAPER_WITH_STYLE_ISSUES)

    categories = [f.category for f in result.deterministic_findings]
    style_categories = {c for c in categories if c.startswith("style:")}
    # At minimum: weasel + passive should fire on this fixture
    assert "style:weasel" in style_categories, (
        f"No weasel finding in: {sorted(style_categories)}"
    )
    assert "style:passive" in style_categories, (
        f"No passive finding in: {sorted(style_categories)}"
    )
    # And "the the" is the canonical duplicate pattern
    assert "style:duplicate_word" in style_categories, (
        f"No duplicate-word finding in: {sorted(style_categories)}"
    )


async def test_review_document_proofread_false_skips_style_findings() -> None:
    """proofread=False disables the proofread pass."""
    result = await review_document(PAPER_WITH_STYLE_ISSUES, proofread=False)

    categories = [f.category for f in result.deterministic_findings]
    for c in categories:
        assert not c.startswith("style:"), (
            f"Got style finding with proofread=False: {c}"
        )


async def test_proofread_findings_have_anchored_quotes() -> None:
    """Every proofread-emitted Finding must carry at least one quote so the
    docx renderer can convert it to a Word comment."""
    result = await review_document(PAPER_WITH_STYLE_ISSUES)

    style_findings = [
        f for f in result.deterministic_findings if f.category.startswith("style:")
    ]
    assert style_findings, "Expected at least one style finding on the fixture"
    for f in style_findings:
        assert len(f.quotes) >= 1, f"Style finding {f.title!r} has no quote anchor"
        # The anchor text should be substantive — wider than the flagged token
        for q in f.quotes:
            assert len(q.text.strip()) >= 3
