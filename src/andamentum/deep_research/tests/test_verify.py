"""VERIFY: report sentences, and every check run on them (code only, no model)."""

from __future__ import annotations

import httpx
import pytest

from andamentum.references import Work

from .. import read
from ..types import Finding, Passage
from ..verify import (
    UNVERIFIED_MARK,
    check_links,
    check_quotes,
    links_in,
    report_units,
    verify_report,
)
from ..write import NO_EVIDENCE_LINE
from .llm_stub import http


@pytest.fixture(autouse=True)
def _simple_locate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading is another package: a plain substring search stands in for it."""

    def locate(quote: str, text: str) -> tuple[int, int] | None:
        start = text.find(quote)
        return None if start < 0 else (start, start + len(quote))

    monkeypatch.setattr(read, "locate_quote", locate)
    monkeypatch.setattr("andamentum.deep_research.verify.locate_quote", locate)


# -- sentences ---------------------------------------------------------------


def test_sentences_end_at_their_citation() -> None:
    report = (
        "# Title\n\n## Does Apollo help?\n\n"
        "Apollo cut latency [F1]. SaltLog adds overhead [F2, F3]. "
        "Both were tested [F1][F4].\n\n"
        "## Sources\n\n- [F1] Doe (2024). Apollo. https://example.org/a\n"
    )
    units = report_units(report)
    assert [(u.text, u.finding_ids) for u in units] == [
        ("Apollo cut latency", ("F1",)),
        ("SaltLog adds overhead", ("F2", "F3")),
        ("Both were tested", ("F1", "F4")),
    ]
    assert report[units[0].end - 5 : units[0].end] == "[F1]."


def test_list_items_headings_and_the_sources_section() -> None:
    report = (
        "## Disagreements\n\n"
        "- Sources disagree. One reports: A [F1]; another reports: B [F2].\n"
        "1. Numbered item [F3].\n"
        "\n## Sources\n\n- [F1] A source line with no sentence\n"
    )
    assert [(u.text, u.finding_ids) for u in report_units(report)] == [
        ("Sources disagree. One reports: A", ("F1",)),
        ("another reports: B", ("F2",)),
        ("Numbered item", ("F3",)),
    ]


def test_text_after_the_last_citation_is_a_sentence_with_no_ids() -> None:
    units = report_units("Cited claim [F1]. And a claim nobody cited.\n")
    assert [(u.text, u.finding_ids) for u in units] == [
        ("Cited claim", ("F1",)),
        ("And a claim nobody cited.", ()),
    ]
    assert report_units("Only punctuation after [F1].\n")[0].finding_ids == ("F1",)
    assert len(report_units("Only punctuation after [F1]. \n")) == 1


# -- fixtures ----------------------------------------------------------------

P1 = Passage.make(
    work_id="W1",
    start=0,
    text="In our benchmark the Apollo cache reduced median latency from 120 ms to 45 ms.",
)
P2 = Passage.make(
    work_id="W2",
    start=0,
    text="SaltLog verbose mode adds roughly 3 ms of overhead per request.",
)
PASSAGES = {p.passage_id: p for p in (P1, P2)}


def _finding(
    number: int,
    passage: Passage,
    quote: str,
    *,
    work_id: str | None = None,
    verified: bool = True,
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="Q1",
        claim=f"claim {number}",
        quote=quote,
        quote_verified=verified,
        passage_id=passage.passage_id,
        work_id=work_id or passage.work_id,
    )


F1 = _finding(1, P1, "reduced median latency from 120 ms to 45 ms")
F2 = _finding(2, P2, "adds roughly 3 ms of overhead per request")
F3 = _finding(3, P1, "reduced median latency from 120 ms to 45 ms", work_id="W3")
U4 = _finding(4, P2, "a quote the page does not hold", verified=False)
U5 = _finding(5, P2, "another lost quote", verified=False)
WORKS = {
    "W1": Work(work_id="W1", title="Apollo caching", authors=["Jane Doe"], year=2024),
    "W2": Work(work_id="W2", title="SaltLog costs", authors=["Max Muster"], year=2023),
    "W3": Work(
        work_id="W3",
        title="A retracted study",
        authors=["Ann Roe"],
        year=2020,
        retracted="retracted",
    ),
}


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"message": {}})


# -- quotes ------------------------------------------------------------------


def test_a_verified_quote_that_is_not_in_its_passage_is_flagged() -> None:
    forged = _finding(9, P1, "reduced median latency from 120 ms to 4 ms")
    orphan = Finding(
        finding_id="F10",
        subquestion_id="Q1",
        claim="c",
        quote="a quote with no passage on record",
        passage_id="nope",
        work_id="W1",
    )
    flags = check_quotes([F1, forged, orphan, U4], PASSAGES)
    # The unverified finding is skipped here.
    assert [(f.kind, f.ref) for f in flags] == [
        ("quote_missing", "F9"),
        ("quote_missing", "F10"),
    ]


# -- the whole report ----------------------------------------------------------


async def test_a_clean_report_passes_unchanged() -> None:
    report = (
        "# Q\n\n## Latency\n\nApollo cut latency [F1]. SaltLog adds overhead [F2].\n\n"
        f"## Empty\n\n{NO_EVIDENCE_LINE}\n"
    )
    outcome = await verify_report(
        report,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        http=http(_ok),
    )
    assert outcome.flags == [] and outcome.unchecked == []
    assert outcome.report_markdown == report
    assert outcome.degraded == []


async def test_unknown_and_retracted_citations() -> None:
    report = "## Q\n\nA claim [F99]. A retracted claim [F3]. Again [F3].\n"
    outcome = await verify_report(
        report,
        findings=[F1, F3],
        passages=PASSAGES,
        works=WORKS,
        http=http(_ok),
    )
    kinds = [(f.kind, f.ref) for f in outcome.flags]
    assert kinds == [("unknown_finding", "F99"), ("retracted", "W3")]
    assert outcome.report_markdown == report  # retraction is flagged, not marked


async def test_an_unverified_quote_is_flagged_once_per_finding() -> None:
    report = "## Q\n\nOne [F4]. Two [F4, F5]. Three [F4].\n"
    outcome = await verify_report(
        report,
        findings=[F4_ := U4, U5],
        passages=PASSAGES,
        works=WORKS,
        http=http(_ok),
    )
    assert F4_.finding_id == "F4"
    assert [(f.kind, f.ref) for f in outcome.flags] == [
        ("unverified_quote", "F4"),
        ("unverified_quote", "F5"),
    ]


async def test_the_mark_goes_on_sentences_whose_findings_are_all_unverified() -> None:
    report = (
        "## Q\n\nBoth lost [F4, F5]. One lost, one fine [F1, F4]. Fine [F1].\n\n"
        "## Sources\n\n- [F4] A source line.\n"
    )
    outcome = await verify_report(
        report,
        findings=[F1, U4, U5],
        passages=PASSAGES,
        works=WORKS,
        http=http(_ok),
    )
    assert f"Both lost [F4, F5].{UNVERIFIED_MARK} One lost" in outcome.report_markdown
    assert outcome.report_markdown.count(UNVERIFIED_MARK) == 1
    assert "Fine [F1]. " not in outcome.report_markdown.replace("Fine [F1].\n", "")
    assert outcome.report_markdown.endswith("- [F4] A source line.\n")
    # Marking is idempotent: verifying the marked report adds nothing.
    again = await verify_report(
        outcome.report_markdown,
        findings=[F1, U4, U5],
        passages=PASSAGES,
        works=WORKS,
        http=http(_ok),
    )
    assert again.report_markdown == outcome.report_markdown


async def test_dead_links_dois_and_unchecked_in_the_report() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.url.host == "api.crossref.org":
            if "10.5555/gone" in url:
                return httpx.Response(404)
            return httpx.Response(200, json={"message": {}})
        if request.url.host == "doi.org":
            return httpx.Response(404)
        if "dead" in url:
            return httpx.Response(404)
        if "slow" in url:
            return httpx.Response(503)
        return httpx.Response(200)

    report = (
        "## Q\n\nFine [F1].\n\n## Sources\n\n"
        "- [F1] https://example.org/ok https://example.org/dead "
        "https://example.org/slow 10.5555/gone\n"
    )
    outcome = await verify_report(
        report,
        findings=[F1],
        passages=PASSAGES,
        works=WORKS,
        http=http(handler),
    )
    assert sorted((f.kind, f.ref) for f in outcome.flags) == [
        ("dead_link", "https://example.org/dead"),
        ("doi_unresolved", "10.5555/gone"),
    ]
    assert len(outcome.unchecked) == 1
    assert outcome.unchecked[0].startswith("https://example.org/slow")


async def test_a_retracted_doi_in_the_report_is_flagged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.crossref.org":
            return httpx.Response(
                200,
                json={
                    "message": {"updated-by": [{"type": "retraction", "DOI": "10.1/n"}]}
                },
            )
        return httpx.Response(200)

    outcome = await verify_report(
        "## Q\n\nFine [F1].\n\n## Sources\n\n- [F1] https://doi.org/10.5555/old\n",
        findings=[F1],
        passages=PASSAGES,
        works=WORKS,
        http=http(handler),
    )
    assert [(f.kind, f.ref) for f in outcome.flags] == [("retracted", "10.5555/old")]


# -- links and DOIs ----------------------------------------------------------


def test_links_in_a_report() -> None:
    report = (
        "See https://example.org/page, (https://example.org/page) and "
        "https://doi.org/10.5555/Abc.1. Also doi 10.5555/xyz-2; "
        "[link](https://example.net/a_b) https://example.org/page."
    )
    urls, dois = links_in(report)
    assert urls == ["https://example.org/page", "https://example.net/a_b"]
    assert dois == ["10.5555/abc.1", "10.5555/xyz-2"]


async def test_check_links_separates_flags_from_unchecked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        host = request.url.host
        if host == "api.crossref.org":
            if "10.5555/gone" in url or "10.48550" in url:
                return httpx.Response(404)
            if "10.5555/retracted" in url:
                return httpx.Response(
                    200,
                    json={
                        "message": {
                            "updated-by": [{"type": "retraction", "DOI": "10.1/n"}]
                        }
                    },
                )
            return httpx.Response(200, json={"message": {}})
        if host == "doi.org":
            return httpx.Response(200 if "10.48550" in url else 404)
        if "dead" in url:
            return httpx.Response(404)
        if "teapot" in url:
            return httpx.Response(418)
        return httpx.Response(200)

    report = (
        "https://example.org/fine https://example.org/dead https://example.org/teapot "
        "10.5555/gone 10.5555/retracted 10.5555/fine 10.48550/arXiv.2101.00001"
    )
    flags, unchecked = await check_links(report, http=http(handler))
    assert sorted((f.kind, f.ref) for f in flags) == [
        ("dead_link", "https://example.org/dead"),
        ("doi_unresolved", "10.5555/gone"),
        ("retracted", "10.5555/retracted"),
    ]
    assert len(unchecked) == 1 and unchecked[0].startswith("https://example.org/teapot")
