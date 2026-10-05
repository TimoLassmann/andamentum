"""VERIFY: report sentences, and every check run on them."""

from __future__ import annotations

import httpx

from andamentum.references import Work

from ..types import Budget, Finding, Passage
from ..verify import (
    UNSUPPORTED_MARK,
    check_links,
    check_quotes,
    links_in,
    report_units,
    verify_report,
)
from ..write import LIMITED_EVIDENCE_PREFIX, NO_EVIDENCE_LINE
from .llm_stub import ScriptedLLM, decisions, http, make_asker

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
    # Each unit ends after its citation and the full stop that follows it.
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
    number: int, passage: Passage, quote: str, *, work_id: str | None = None
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="Q1",
        claim=f"claim {number}",
        quote=quote,
        passage_id=passage.passage_id,
        work_id=work_id or passage.work_id,
    )


F1 = _finding(1, P1, "reduced median latency from 120 ms to 45 ms")
F2 = _finding(2, P2, "adds roughly 3 ms of overhead per request")
F3 = _finding(3, P1, "reduced median latency from 120 ms to 45 ms", work_id="W3")
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


def _judge_by_overlap(state: object, instruction: str) -> float:
    """A stand-in for the decision model: yes when every number of the statement
    appears in the quotes."""
    assert isinstance(state, dict)
    import re

    numbers = set(re.findall(r"\d+", str(state["statement"])))
    quotes = " ".join(state["quotes"])  # type: ignore[arg-type]
    return 0.9 if all(n in quotes for n in numbers) else 0.1


# -- quotes ------------------------------------------------------------------


def test_a_quote_that_is_not_in_its_passage_is_flagged() -> None:
    forged = _finding(9, P1, "reduced median latency from 120 ms to 4 ms")
    orphan = Finding(
        finding_id="F10",
        subquestion_id="Q1",
        claim="c",
        quote="a quote with no passage on record",
        passage_id="nope",
        work_id="W1",
    )
    flags = check_quotes([F1, forged, orphan], PASSAGES)
    assert [(f.kind, f.ref) for f in flags] == [
        ("quote_missing", "F9"),
        ("quote_missing", "F10"),
    ]


# -- the whole report ----------------------------------------------------------


REPORT = (
    "# Does Apollo help?\n\n"
    "## Latency\n\n"
    "Apollo cut median latency from 120 ms to 45 ms [F1]. "
    "SaltLog adds 30 ms of overhead [F2]. "
    "Latency fell for two groups [F1, F2].\n\n"
    f"{NO_EVIDENCE_LINE}\n\n"
    f"{LIMITED_EVIDENCE_PREFIX} 1 independent source found for this question.\n\n"
    "## Sources\n\n- [F1] Doe (2024). Apollo caching.\n"
)


async def test_unsupported_sentences_are_flagged_and_marked() -> None:
    decision_model, _ = decisions(_judge_by_overlap)
    asker = make_asker(decision_model=decision_model)
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
    )

    unsupported = [f for f in outcome.flags if f.kind == "unsupported"]
    assert [f.ref for f in unsupported] == ["SaltLog adds 30 ms of overhead"]
    assert (
        "SaltLog adds 30 ms of overhead [F2]." + UNSUPPORTED_MARK
        in outcome.report_markdown
    )
    assert outcome.report_markdown.count(UNSUPPORTED_MARK) == 1
    assert outcome.sentences_checked == 3
    # The fixed lines of the writer need no id and raise nothing.
    assert not any(NO_EVIDENCE_LINE in f.ref for f in outcome.flags)
    assert not any(LIMITED_EVIDENCE_PREFIX in f.ref for f in outcome.flags)
    assert asker.stats.decision_calls == 3
    assert len(asker.store.decisions(asker.run_id, "claim_support")) == 3


async def test_single_source_sentences_are_flagged() -> None:
    decision_model, _ = decisions(lambda *_: 0.9)
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=make_asker(decision_model=decision_model),
        http=http(_ok),
    )
    single = [f.ref for f in outcome.flags if f.kind == "single_source"]
    # Only the sentence citing F1 and F2 (two origins) is not single-source.
    assert single == [
        "Apollo cut median latency from 120 ms to 45 ms",
        "SaltLog adds 30 ms of overhead",
    ]


async def test_unknown_uncited_and_retracted_citations() -> None:
    report = (
        "## Q\n\nA claim [F99].\n\nA claim with no source at all.\n\n"
        "A retracted claim [F3].\n"
    )
    decision_model, _ = decisions(lambda *_: 0.9)
    asker = make_asker(decision_model=decision_model)
    outcome = await verify_report(
        report,
        findings=[F1, F3],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
    )
    kinds = [(f.kind, f.ref) for f in outcome.flags]
    assert ("unknown_finding", "F99") in kinds
    assert ("retracted", "W3") in kinds
    unsupported = {f.ref: f.detail for f in outcome.flags if f.kind == "unsupported"}
    assert unsupported == {
        "A claim": "cites no known finding",
        "A claim with no source at all.": "cites no finding",
    }
    assert outcome.report_markdown.count(UNSUPPORTED_MARK) == 2
    # Every flag is written to the run.
    items = asker.store.run_items(asker.run_id, "flag")
    assert len(items) == len(outcome.flags)


async def test_without_a_decision_model_the_language_model_checks_support() -> None:
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        body = prompt.split("ITEMS:")[1].strip()
        items = body.split("\n\n[")
        verdicts = []
        for number, item in enumerate(items, start=1):
            verdicts.append(
                {"item": number, "answer": "no" if "30 ms" in item else "yes"}
            )
        return {"verdicts": verdicts}

    llm = ScriptedLLM(respond)
    asker = make_asker(model=llm)
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
    )
    assert asker.stats.llm_calls == 1  # three sentences, one batch
    assert [f.ref for f in outcome.flags if f.kind == "unsupported"] == [
        "SaltLog adds 30 ms of overhead"
    ]
    assert "statement: SaltLog adds 30 ms of overhead" in llm.calls[0][1]
    assert "quotes:" in llm.calls[0][1]


async def test_a_spent_budget_is_reported_not_treated_as_support() -> None:
    decision_model, _ = decisions(lambda *_: 0.9)
    asker = make_asker(
        decision_model=decision_model, budget=Budget(max_decision_calls=1)
    )
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
    )
    # All three sentences were in one chunk, which the budget could not finish.
    assert outcome.sentences_checked == 0
    assert outcome.degraded == [
        "claim support not checked for 3 sentences: budget:max_decision_calls"
    ]
    assert not [f for f in outcome.flags if f.kind == "unsupported"]


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


async def test_dead_links_and_unresolved_dois_are_flags_and_the_rest_is_unchecked() -> (
    None
):
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
    # A bot block settles nothing about the link: it is listed, not flagged.
    assert len(unchecked) == 1 and unchecked[0].startswith("https://example.org/teapot")
