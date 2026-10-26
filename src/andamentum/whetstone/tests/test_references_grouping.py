"""Reference-lookup problems in a review: capped severity, one comment per kind,
and no rewrite by the editor. All text is invented."""

from __future__ import annotations

import pytest

from andamentum.references.models import Problem, ReferenceEntry, ReferenceReport
from andamentum.whetstone.pipeline.editor import to_editor_finding
from andamentum.whetstone.references_check import to_review_result
from andamentum.whetstone.reviewed_document import ReviewedDocument
from andamentum.whetstone.schemas import Finding, Quote

TEXT = (
    "# SaltLog\n\nSaltLog logs tidal sensors [1] [2] [3].\n\n"
    "# References\n\n"
    "1. Doe J (2020) First paper.\n\n"
    "2. Roe A (2019) Second paper.\n\n"
    "3. Doe J (2018) Third paper.\n"
)


@pytest.fixture
def document():
    doc = ReviewedDocument.from_markdown(TEXT)
    yield doc
    doc.cleanup()


def _span(doc, needle: str) -> tuple[int, int]:
    start = doc.text.index(needle)
    return start, start + len(needle)


def _problem(doc, needle: str, number: int | None, kind: str, **kw) -> Problem:
    start, end = _span(doc, needle)
    return Problem(
        kind=kind,  # type: ignore[arg-type]
        severity=kw.pop("severity", "major"),
        headline=kw.pop("headline", "Entry not found"),
        issue=kw.pop("issue", f"{needle} was not found."),
        fix="Check the entry.",
        start=start,
        end=end,
        entry_number=number,
        **kw,
    )


def _report(doc, problems, *, lookups=()) -> ReferenceReport:
    return ReferenceReport(
        found_list=True,
        list_start=doc.text.index("1. Doe"),
        list_end=len(doc.text),
        problems=list(problems),
        lookups=list(lookups),
        looked_up=True,
    )


def test_a_single_lookup_problem_stays_on_its_entry_at_minor(document) -> None:
    p = _problem(document, "Doe J (2020) First paper.", 1, "not_found")
    result = to_review_result(_report(document, [p]), document=document)
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.severity == "minor" and f.priority == "consider"
    assert f.quotes[0].char_start == p.start
    assert f.source == "lookup"


def test_offline_problems_keep_their_severity_and_are_never_grouped(document) -> None:
    ps = [
        _problem(
            document,
            n,
            i,
            "cited_not_listed",
            severity="major",
            headline="Cited, not listed",
        )
        for i, n in ((1, "Doe J (2020)"), (2, "Roe A (2019)"))
    ]
    result = to_review_result(_report(document, ps), document=document)
    assert len(result.findings) == 2
    assert {f.severity for f in result.findings} == {"major"}
    assert all(not f.title.startswith("2 entries") for f in result.findings)


def test_two_problems_of_one_kind_become_one_finding(document) -> None:
    ps = [
        _problem(document, "Doe J (2020) First paper.", 1, "doi_not_found"),
        _problem(document, "Doe J (2018) Third paper.", 3, "doi_not_found"),
    ]
    result = to_review_result(_report(document, ps), document=document)
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.title.startswith("2 entries:")
    assert "Entry 1" in f.rationale and "Entry 3" in f.rationale
    assert "Entry 2" not in f.rationale
    assert f.severity == "minor" and f.priority == "consider"
    assert f.source == "lookup"
    assert len(f.quotes) == 1
    q = f.quotes[0]
    assert q.text == "References"
    assert document.text[q.char_start : q.char_end] == "References"


def test_kinds_are_grouped_apart_and_singles_stay_single(document) -> None:
    ps = [
        _problem(document, "Doe J (2020) First paper.", 1, "not_found"),
        _problem(document, "Roe A (2019) Second paper.", 2, "not_found"),
        _problem(document, "Doe J (2018) Third paper.", 3, "details_differ"),
    ]
    result = to_review_result(_report(document, ps), document=document)
    assert len(result.findings) == 2
    grouped = [f for f in result.findings if f.title.startswith("2 entries:")]
    single = [f for f in result.findings if not f.title.startswith("2 entries:")]
    assert len(grouped) == 1 and len(single) == 1
    assert single[0].quotes[0].char_start == ps[2].start


def test_unnumbered_entries_are_named_by_their_opening_words(document) -> None:
    ps = [
        _problem(document, "Doe J (2020) First paper.", None, "not_found"),
        _problem(document, "Roe A (2019) Second paper.", None, "not_found"),
    ]
    f = to_review_result(_report(document, ps), document=document).findings[0]
    assert "Doe J (2020) First paper." in f.rationale
    assert "Roe A (2019) Second" in f.rationale


def test_the_group_is_anchored_on_the_first_entry_without_a_heading_title() -> None:
    doc = ReviewedDocument.from_markdown(
        "Intro text [1] [2].\n\nSources\n\n1. Doe J (2020) A.\n\n2. Roe A (2019) B.\n"
    )
    try:
        ps = [
            _problem(doc, "Doe J (2020) A.", 1, "not_found"),
            _problem(doc, "Roe A (2019) B.", 2, "not_found"),
        ]
        report = ReferenceReport(
            found_list=True,
            list_start=doc.text.index("1. Doe"),
            list_end=len(doc.text),
            problems=ps,
            looked_up=True,
        )
        f = to_review_result(report, document=doc).findings[0]
        assert len(f.quotes) == 1
        assert f.quotes[0].text != "Sources"
        assert f.quotes[0].char_start == ps[0].start
    finally:
        doc.cleanup()


def test_the_unchecked_wording_no_longer_blames_crossref(document) -> None:
    from andamentum.references.models import LookupResult

    lookups = [
        LookupResult(entry_number=1, status="service_error", detail="no answer"),
        LookupResult(entry_number=2, status="found"),
    ]
    report = _report(document, [], lookups=lookups)
    report.entries = [
        ReferenceEntry(number=n, text="x", start=0, end=1) for n in (1, 2)
    ]
    result = to_review_result(report, document=document)
    assert "Crossref did not answer" not in result.degraded_reason
    assert "Crossref did not answer" not in result.summary
    assert "unchecked, not wrong" in result.degraded_reason
    assert "not checked" in result.summary


# ── the editor ──────────────────────────────────────────────────────────────


def _public(source: str) -> Finding:
    return Finding(
        title="Entry not found",
        severity="minor",
        confidence="medium",
        rationale="Not found in the lookup.",
        fix="Check the entry.",
        quotes=[Quote(section_id="s1", char_start=0, char_end=5, text="SaltL")],
        source=source,  # type: ignore[arg-type]
    )


def test_the_editor_gets_no_lookup_finding() -> None:
    assert to_editor_finding(_public("lookup")) is None


def test_the_editor_still_gets_a_deterministic_reference_finding() -> None:
    assert to_editor_finding(_public("deterministic")) is not None
