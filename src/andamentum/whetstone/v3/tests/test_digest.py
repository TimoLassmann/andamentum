"""Tests for the deterministic digest helpers + document-model assembly."""

from __future__ import annotations

from andamentum.references import Hint
from andamentum.whetstone.v3.digest import (
    build_document_model,
    citations_line,
    gist_for,
    read_citations,
)
from andamentum.whetstone.v3.model import CitationSummary, Claim, Section, Span


def _section(text: str, *, id="s1", title="S", start=0) -> Section:
    return Section(id=id, title=title, text=text, start=start, end=start + len(text))


def _two_sections(text: str) -> list[Section]:
    """The body, and the reference list under its heading — as a reader of
    the document's headings would give them."""
    cut = text.index("\nReferences\n") + 1
    return [
        Section(id="s1", title="Body", text=text[:cut], start=0, end=cut),
        Section(id="s2", title="References", text=text[cut:], start=cut, end=len(text)),
    ]


_LIST = (
    "\nReferences\n"
    "1. Smith J (2020) A study of rare disease costs. J Test 1:1\n"
    "2. Lee K (2021) Delays in rare disease care. J Test 2:2\n"
)


async def test_citations_are_read_by_the_reference_checker() -> None:
    text = "Costs are high [1]. Care is slow [2]." + _LIST
    summary = await read_citations(text, _two_sections(text))
    assert summary == CitationSummary(
        found_list=True, entries=2, style="numeric_brackets", citations=2
    )
    assert citations_line(summary) == (
        "CITATIONS: 2 in the text, as numbers in square brackets, like [3]; "
        "a reference list of 2 entries."
    )


async def test_superscript_citations_count_when_the_file_marks_them() -> None:
    """The case the old marker regexes could not see: a Nature-style draft
    read as "CITATIONS PRESENT: (none)"."""
    text = "Costs are high1. Care is slow2." + _LIST
    hints = [
        Hint(kind="superscript", start=text.index("1."), end=text.index("1.") + 1),
        Hint(kind="superscript", start=text.index("2."), end=text.index("2.") + 1),
    ]
    summary = await read_citations(text, _two_sections(text), hints=hints)
    assert (summary.style, summary.citations) == ("numeric_superscript", 2)
    assert "superscript numbers" in citations_line(summary)


async def test_no_list_and_no_notes_says_so() -> None:
    text = "The system is robust to noise."
    summary = await read_citations(text, [_section(text)])
    assert summary.style is None
    assert citations_line(summary) == (
        "CITATIONS: none found, and the document has no reference list."
    )


def test_gist_is_first_sentence_without_heading() -> None:
    s = _section("## Methods\n\nWe trained the model on data. Then we evaluated it.")
    g = gist_for(s)
    assert g == "We trained the model on data."


def test_build_document_model_assembles_the_facets() -> None:
    src = "We recover 4x more [12]. The system is robust."
    sec = _section(src)
    claims = [
        Claim(
            id="c1",
            quote="We recover 4x more [12].",
            span=Span(section_id="s1", start=0, end=24),
        ),
    ]
    summary = CitationSummary(found_list=False)
    model = build_document_model(src, [sec], claims, citations=summary)
    assert [c.id for c in model.claims] == ["c1"]
    assert len(model.gists) == 1 and model.gists[0].section_id == "s1"
    assert model.citations is summary
    assert model.source == src
    # Checklist mode does not read citations; the model then has none.
    assert build_document_model(src, [sec], claims).citations is None


def test_only_a_citations_facet_criterion_sees_the_citation_line() -> None:
    from andamentum.whetstone.v3.criteria import Criterion
    from andamentum.whetstone.v3.review import _project

    src = "We recover 4x more [1]."
    summary = CitationSummary(
        found_list=True, entries=1, style="numeric_brackets", citations=1
    )
    model = build_document_model(src, [_section(src)], [], citations=summary)
    evidence = Criterion(name="Evidence", facets=["claims", "citations"])
    clarity = Criterion(name="Clarity", facets=["claims"])
    assert citations_line(summary) in _project(evidence, model)
    assert "CITATIONS:" not in _project(clarity, model)
