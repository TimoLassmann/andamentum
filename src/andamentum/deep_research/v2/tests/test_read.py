"""READ: passages, the window, quote location, the soft quote check, and one page
end to end (text order, retraction, notes, events). Synthetic text, no network,
no model beyond ``ScriptedLLM``."""

from __future__ import annotations

from typing import Literal

import httpx
import pytest

from andamentum.references import Work

from .. import runlog
from ..ask import UNTRUSTED, BudgetExhausted
from ..read import (
    MAX_FINDINGS_PER_PAGE,
    PASSAGE_MAX_CHARS,
    SNIPPET_MIN_CHARS,
    ReadDeps,
    build_window,
    check_quote,
    check_retraction,
    load_page,
    locate_quote,
    read_page,
    split_passages,
    unreadable_degraded,
)
from ..runlog import RunLogger
from ..sources.base import FetchedText, SearchFilters, SourceError
from ..sources.browser_fetch import CONTENT_TYPE as CHROME_TYPE
from ..sources.crossref import CrossrefSource
from ..store import ResearchStore
from ..types import Budget, Candidate, LogEvent, Passage, SubQuestion
from .llm_stub import ScriptedLLM, http, make_asker

SUBQ = SubQuestion(id="Q1", text="How does Apollo caching affect latency?")
TEXT = (
    "Apollo is a caching layer for read-heavy services.\n\n"
    "In our benchmark, enabling the Apollo cache reduced median latency from "
    "120 ms to 45 ms across all tested workloads.\n\n"
    "SaltLog, the logging component, adds roughly 3 ms of overhead per request "
    "when verbose mode is on."
)
QUOTE = (
    "enabling the Apollo cache reduced median latency from 120 ms to 45 ms "
    "across all tested workloads"
)
SNIPPET = (
    "Apollo caching cut median latency from 120 ms to 45 ms in the benchmark, "
    "across all of the tested workloads."
)


def _passage(text: str, *, work_id: str = "W1", start: int = 0) -> Passage:
    return Passage.make(work_id=work_id, start=start, text=text)


# -- passages ----------------------------------------------------------------


def test_passages_are_exact_slices_of_the_text() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(8))
    passages = split_passages(text, work_id="W1")
    assert len(passages) > 1
    for p in passages:
        assert text[p.start : p.end] == p.text
        assert len(p.text) <= PASSAGE_MAX_CHARS


def test_small_paragraphs_are_packed_up_to_the_target() -> None:
    text = "\n\n".join(f"Short paragraph number {i} says one thing." for i in range(5))
    passages = split_passages(text, work_id="W1")
    assert len(passages) == 1 and passages[0].text == text


def test_a_sentence_longer_than_the_maximum_is_cut_at_whitespace() -> None:
    passages = split_passages(("word " * 1000).strip(), work_id="W1")
    assert len(passages) > 1
    assert all(len(p.text) <= PASSAGE_MAX_CHARS for p in passages)


def test_repeated_text_is_one_passage_and_empty_text_none() -> None:
    assert split_passages("", work_id="W1") == []
    assert split_passages("  \n\n  ", work_id="W1") == []
    assert (
        len(split_passages("Same boilerplate sentence here.\n\n" * 2, work_id="W1"))
        == 1
    )


# -- window ------------------------------------------------------------------


def _paragraphs(count: int, *, marked: dict[int, str]) -> str:
    """*count* paragraphs of about 1,000 characters, each its own passage; the
    ones in *marked* carry a distinctive sentence."""
    out = []
    for i in range(count):
        base = f"Paragraph {i} filler. " + "plain filler words go here. " * 35
        out.append(base[:900] + (" " + marked[i] if i in marked else ""))
    return "\n\n".join(out)


def test_a_text_within_the_limit_is_one_whole_passage_with_exact_offsets() -> None:
    text = "\n  Short page about Apollo.\n\nSecond paragraph.  \n"
    [window] = build_window(text, work_id="W1", query="apollo", max_chars=5000)
    assert window.text == text.strip()
    assert text[window.start : window.end] == window.text
    assert build_window("  \n ", work_id="W1", query="x", max_chars=5000) == []


def test_the_window_takes_the_best_passages_in_document_order() -> None:
    text = _paragraphs(
        20,
        marked={
            3: "Zephyr turbine output rose strongly.",
            14: "The zephyr turbine was tested.",
        },
    )
    window = build_window(
        text, work_id="W1", query="zephyr turbine output", max_chars=2200
    )
    assert [text[p.start : p.end] for p in window] == [p.text for p in window]
    assert [p.start for p in window] == sorted(p.start for p in window)
    joined = "\n".join(p.text for p in window)
    assert (
        "Zephyr turbine output rose" in joined and "zephyr turbine was tested" in joined
    )
    assert sum(len(p.text) for p in window) <= 2200 + 4
    assert len(window) == 2  # far apart: two separate spans


def test_adjacent_passages_are_merged_into_one_exact_slice() -> None:
    text = _paragraphs(
        12, marked={5: "Quasar lattice width measured.", 6: "Quasar lattice depth too."}
    )
    window = build_window(text, work_id="W1", query="quasar lattice", max_chars=2200)
    assert len(window) == 1
    [merged] = window
    assert "width measured" in merged.text and "depth too" in merged.text
    assert text[merged.start : merged.end] == merged.text
    assert "\n\n" in merged.text  # the whitespace between them is inside the span


def test_the_query_terms_drive_the_choice() -> None:
    text = _paragraphs(
        15, marked={2: "Alpha river flows north.", 11: "Beta mountain stands tall."}
    )
    river = build_window(text, work_id="W1", query="alpha river", max_chars=1100)
    mountain = build_window(text, work_id="W1", query="beta mountain", max_chars=1100)
    assert "Alpha river" in " ".join(p.text for p in river)
    assert "Beta mountain" in " ".join(p.text for p in mountain)
    assert "Beta mountain" not in " ".join(p.text for p in river)


# -- quotes ------------------------------------------------------------------


def test_locate_quote_exact_and_folded() -> None:
    start = TEXT.index("median latency")
    assert locate_quote("median latency", TEXT) == (
        start,
        start + len("median latency"),
    )
    text = "The “Apollo” cache — in short bursts —\n reduced   latency."
    span = locate_quote('the "apollo" cache - in short bursts - reduced latency', text)
    assert span is not None and text[span[0] : span[1]] == text[: len(text) - 1]


def test_locate_quote_not_found_empty_and_no_false_match() -> None:
    assert locate_quote("a claim that is nowhere in the text", TEXT) is None
    assert locate_quote("   ", TEXT) is None
    assert locate_quote("Apollo cache increased median latency", TEXT) is None


def test_literal_backslash_n_and_t_read_as_a_space() -> None:
    text = "The Apollo cache holds hot keys.\nIt evicts the oldest first."
    quote = r"holds hot keys.\nIt evicts\tthe oldest first."
    span = locate_quote(quote, text)
    assert span is not None
    assert text[span[0] : span[1]] == "holds hot keys.\nIt evicts the oldest first."
    assert locate_quote(r"hot keys.\r\nIt evicts", text) is not None


def test_a_literal_backslash_n_in_the_source_still_matches_exactly() -> None:
    text = r"Split the lines on \n before parsing the header."
    assert locate_quote(r"on \n before", text) is not None


def test_a_table_row_is_found_without_its_pipes() -> None:
    text = (
        "| Product | Year | Latency |\n| --- | :---: | ---: |\n"
        "| Apollo | 2019 | 45 ms |\n| SaltLog | 2021 | 3 ms |\n"
    )
    row = locate_quote("| Apollo | 2019 | 45 ms |", text)
    assert row is not None and text[row[0] : row[1]] == "| Apollo | 2019 | 45 ms |"
    cells = locate_quote("Apollo 2019 45 ms", text)
    assert cells is not None and text[cells[0] : cells[1]] == "Apollo | 2019 | 45 ms"
    # header, separator row and a row, as a model copies a table
    block = locate_quote(
        "Year | Latency |\n| --- | :---: | ---: |\n| Apollo | 2019", text
    )
    assert block is not None
    assert text[block[0] : block[1]].startswith("Year") and text[
        block[0] : block[1]
    ].endswith("2019")
    assert locate_quote("|---|:---:|---:|", text) is None  # a separator is no quote
    assert locate_quote("Apollo | 2018", text) is None


def test_a_markdown_link_reads_as_its_text() -> None:
    text = "The service uses [Apollo](https://example.org/apollo)for caching reads."
    span = locate_quote("The service uses Apollo for caching reads.", text)
    assert span is not None and span[0] == 0 and span[1] == len(text)


def test_an_elided_quote_is_not_matched() -> None:
    assert locate_quote("Apollo is a caching layer ... adds roughly 3 ms", TEXT) is None


def test_check_quote_tries_the_named_passage_first_then_the_others() -> None:
    a = _passage("Alpha says the cache is warm every morning.", start=0)
    b = _passage("Beta says the cache is warm every morning.", start=100)
    both = [a, b]
    found = check_quote("says the cache is warm", both, named=2)
    assert found is not None and found[0] is b
    found = check_quote("says the cache is warm", both, named=1)
    assert found is not None and found[0] is a
    found = check_quote(
        "Beta says the cache", both, named=1
    )  # wrong number: found anyway
    assert found is not None and found[0] is b and found[1] == "Beta says the cache"
    assert check_quote("nothing like it", both, named=1) is None
    assert check_quote("says the cache", both, named=99) is not None
    assert check_quote("SAYS THE CACHE", both) is not None


# -- doubles -----------------------------------------------------------------


class FakeProvider:
    """A text source: holds ``text``, holds nothing (None), or fails."""

    def __init__(
        self,
        text: str | None = TEXT,
        *,
        name: str = "fake",
        access: Literal["full_text", "abstract"] = "full_text",
        content_type: str = "",
        error: SourceError | None = None,
    ) -> None:
        self.name = name
        self.text = text
        self.access: Literal["full_text", "abstract"] = access
        self.content_type = content_type
        self.error = error
        self.calls = 0

    async def search(
        self, query: str, *, limit: int = 20, filters: SearchFilters | None = None
    ) -> list[Candidate]:
        return []

    async def fetch_text(self, work: Work) -> FetchedText | None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.text is None:
            return None
        return FetchedText(
            source=self.name,
            url="https://example.org/p",
            text=self.text,
            access=self.access,
            content_type=self.content_type,
        )


def _crossref(*, retracted: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        message: dict[str, object] = {}
        if retracted:
            message = {
                "updated-by": [
                    {
                        "type": "retraction",
                        "DOI": "10.5555/notice",
                        "source": "retraction-watch",
                    }
                ]
            }
        return httpx.Response(200, json={"message": message})

    return handler


def _finding(passage: int, claim: str, quote: str) -> dict[str, object]:
    return {"passage": passage, "claim": claim, "quote": quote}


def _llm(*findings: dict[str, object]) -> ScriptedLLM:
    given = list(findings) or [
        _finding(1, "Apollo caching cut median latency from 120 ms to 45 ms.", QUOTE)
    ]
    return ScriptedLLM(lambda _i, _p: {"findings": given})


def _deps(
    *providers: FakeProvider,
    llm: ScriptedLLM | None = None,
    retracted: bool = False,
    budget: Budget | None = None,
    window_chars: int = 12_000,
    logged: bool = True,
) -> ReadDeps:
    asker = make_asker(model=llm or _llm(), budget=budget)
    if logged:
        asker.log = RunLogger(clock=lambda: 0.0)
    return ReadDeps(
        asker=asker,
        crossref=CrossrefSource(
            store=asker.store, http=http(_crossref(retracted=retracted))
        ),
        providers=list(providers) or [FakeProvider()],
        window_chars=window_chars,
    )


def _candidate(deps: ReadDeps, *, snippet: str = "", **fields: object) -> Candidate:
    base: dict[str, object] = {
        "doi": "10.5555/apollo",
        "title": "Apollo caching",
        "year": 2024,
        "url": "https://example.org/apollo",
    }
    work = deps.asker.store.upsert_work(Work(**{**base, **fields}))  # type: ignore[arg-type]
    return Candidate(work=work, sources=["web"], snippet=snippet)


def _events(deps: ReadDeps, name: str | None = None) -> list[LogEvent]:
    assert deps.asker.log is not None
    return [e for e in deps.asker.log.log.events if name is None or e.event == name]


# -- notes and the soft quote check ---------------------------------------------


async def test_a_page_gives_a_verified_finding_in_the_passages_own_words() -> None:
    llm = _llm(_finding(1, "  Apollo caching cut\nlatency. ", QUOTE.upper()))
    deps = _deps(llm=llm)
    reading = await read_page(
        _candidate(deps), subquestion=SUBQ, queries=["apollo latency"], deps=deps
    )
    [finding] = reading.findings
    assert finding.quote_verified and finding.quote == QUOTE  # not the model's casing
    assert finding.claim == "Apollo caching cut latency."
    passage = deps.asker.store.get_passage(finding.passage_id)
    assert passage is not None and finding.quote in passage.text
    assert finding.work_id == reading.work.work_id and finding.subquestion_id == "Q1"
    assert reading.via == "fetched" and reading.readable
    assert deps.asker.stats.llm_calls == 1
    assert deps.asker.stats.step_llm_calls == {"Notes": 1}
    assert deps.asker.store.get_work(reading.work.work_id).access == "full_text"  # type: ignore[union-attr]


async def test_an_unverified_finding_is_kept_not_dropped() -> None:
    llm = _llm(
        _finding(
            1,
            "Apollo makes everything ten times faster.",
            "Apollo makes\n everything   ten times faster",
        ),
        _finding(
            1,
            "SaltLog adds about 3 ms.",
            "adds roughly 3 ms of overhead per request when verbose mode is on",
        ),
        _finding(1, "Short quote that is located.", "Apollo is a caching layer"),
        _finding(9, "A wrong passage number.", "nowhere to be found in the page"),
    )
    deps = _deps(llm=llm)
    reading = await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    first, second, third, fourth = reading.findings
    assert not first.quote_verified
    assert (
        first.quote == "Apollo makes everything ten times faster"
    )  # whitespace squashed
    assert first.passage_id == reading.passages[0].passage_id
    assert second.quote_verified and third.quote_verified
    assert third.quote == "Apollo is a caching layer"  # a short, located quote counts
    assert (
        not fourth.quote_verified
        and fourth.passage_id == reading.passages[0].passage_id
    )
    noted = _events(deps, "finding_noted")
    assert [e.data["verified"] for e in noted] == [False, True, True, False]
    [summary] = _events(deps, "notes")
    assert summary.data["findings"] == 4 and summary.data["unverified"] == 2


async def test_duplicates_are_dropped_and_findings_are_capped() -> None:
    same = _finding(1, "One.", QUOTE)
    again = _finding(1, "The same quote again.", QUOTE.lower())
    many = [
        _finding(1, f"Claim number {i}.", f"an invented quote number {i}")
        for i in range(12)
    ]
    deps = _deps(llm=_llm(same, again, *many))
    reading = await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert len(reading.findings) == MAX_FINDINGS_PER_PAGE
    assert reading.findings[0].claim == "One."
    assert sum(1 for f in reading.findings if f.quote == QUOTE) == 1
    assert any("beyond" in n for n in reading.notes)


async def test_the_prompt_fences_the_passages_and_warns_about_them() -> None:
    llm = _llm()
    deps = _deps(llm=llm)
    await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    [(instructions, message)] = llm.calls
    assert instructions.endswith(UNTRUSTED)
    assert "character for character" in instructions and "No ellipsis" in instructions
    assert "QUESTION: " + SUBQ.text in message
    assert "[1]\n<<<\n" + TEXT + "\n>>>" in message


async def test_a_model_failure_on_the_notes_is_degraded_and_the_page_has_no_findings() -> (
    None
):
    llm = ScriptedLLM(lambda _i, _p: {"unexpected": 1})
    deps = _deps(llm=llm)
    reading = await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.findings == []
    assert (
        len(reading.degraded) == 1
        and "notes for Apollo caching failed" in reading.degraded[0]
    )
    assert reading.passages  # the window was read and stored
    [summary] = _events(deps, "notes")
    assert summary.data["findings"] == 0 and summary.data["error"]


async def test_a_spent_call_budget_propagates() -> None:
    deps = _deps(budget=Budget(max_llm_calls=1))
    deps.asker.stats.llm_calls = 1
    with pytest.raises(BudgetExhausted) as spent:
        await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert spent.value.name == "max_llm_calls"


async def test_an_unreadable_page_asks_the_model_nothing() -> None:
    deps = _deps(FakeProvider(None))
    reading = await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert not reading.readable and reading.findings == []
    assert deps.asker.stats.llm_calls == 0
    assert _events(deps, "notes_input") == [] and _events(deps, "notes") == []


# -- text order ---------------------------------------------------------------


async def test_the_first_source_holding_text_wins_and_the_rest_are_not_asked() -> None:
    nothing, wins, spare = FakeProvider(None), FakeProvider(), FakeProvider()
    deps = _deps(nothing, wins, spare)
    await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert (nothing.calls, wins.calls, spare.calls) == (1, 1, 0)


async def test_text_beats_the_abstract_and_the_abstract_beats_the_snippet() -> None:
    deps = _deps(FakeProvider(TEXT))
    candidate = _candidate(deps, abstract="An abstract about Apollo.", snippet=SNIPPET)
    reading = await load_page(candidate, subquestion=SUBQ, queries=[], deps=deps)
    assert reading.via == "fetched" and reading.passages[0].text == TEXT

    deps = _deps(FakeProvider(None))
    candidate = _candidate(deps, abstract="An abstract about Apollo.", snippet=SNIPPET)
    reading = await load_page(candidate, subquestion=SUBQ, queries=[], deps=deps)
    assert reading.via == "abstract" and reading.work.access == "abstract"
    assert reading.passages[0].text == "An abstract about Apollo."

    deps = _deps(FakeProvider(None))
    reading = await load_page(
        _candidate(deps, snippet=SNIPPET), subquestion=SUBQ, queries=[], deps=deps
    )
    assert reading.via == "snippet" and reading.passages[0].text == SNIPPET
    assert reading.readable


async def test_a_short_snippet_is_nothing_and_a_failure_makes_the_page_unreadable() -> (
    None
):
    short = "x" * (SNIPPET_MIN_CHARS - 1)
    deps = _deps(FakeProvider(None))
    reading = await load_page(
        _candidate(deps, snippet=short), subquestion=SUBQ, queries=[], deps=deps
    )
    assert reading.via == "none" and not reading.readable
    assert reading.work.access == "metadata"
    assert _events(deps, "unreadable")

    failing = FakeProvider(
        error=SourceError("fake", "HTTP 403", kind="http", status=403)
    )
    deps = _deps(failing)
    reading = await load_page(
        _candidate(deps, snippet=short), subquestion=SUBQ, queries=[], deps=deps
    )
    assert reading.via == "none" and reading.work.access == "unreadable"
    assert reading.status == 403 and reading.error == "HTTP 403"
    assert any("403" in n for n in reading.failed_notes) and reading.degraded == []
    [unreadable] = _events(deps, "unreadable")
    assert any("403" in r for r in unreadable.data["reasons"])  # type: ignore[union-attr]


async def test_a_failed_page_is_still_read_from_its_snippet() -> None:
    failing = FakeProvider(
        error=SourceError("fake", "HTTP 403", kind="http", status=403)
    )
    deps = _deps(failing)
    reading = await read_page(
        _candidate(deps, snippet=SNIPPET), subquestion=SUBQ, queries=[], deps=deps
    )
    assert reading.via == "snippet" and reading.failed_notes
    assert reading.status is None and reading.error == ""
    [finding] = reading.findings
    assert finding.quote_verified is False  # QUOTE is not in the snippet
    [fetched] = _events(deps, "fetched")
    assert fetched.data["via"] == "snippet" and fetched.data["ok"] is True


async def test_refused_and_offline_failures_are_not_failed_notes() -> None:
    refused = FakeProvider(error=SourceError("fake", "paywall", kind="refused"))
    offline = FakeProvider(error=SourceError("fake", "not cached", kind="offline"))
    deps = _deps(refused, offline)
    reading = await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.failed_notes == [] and reading.degraded == []
    assert len(reading.notes) == 2
    assert reading.work.access == "unreadable"  # a refusal is a failure, not degraded
    deps = _deps(offline)
    reading = await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.work.access == "metadata"  # offline only: nothing failed


async def test_a_redistributable_abstract_only_and_the_chrome_marker() -> None:
    deps = _deps(FakeProvider(TEXT, content_type=CHROME_TYPE))
    reading = await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.via == "chrome"
    deps = _deps(FakeProvider("Provider abstract text.", access="abstract"))
    reading = await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.via == "abstract" and reading.work.access == "abstract"


async def test_the_fetched_byte_budget_stops_reading_before_the_fetch() -> None:
    provider = FakeProvider()
    deps = _deps(provider, budget=Budget(max_fetch_bytes=100))
    deps.fetched_bytes = lambda: 250  # what the source client has fetched so far
    with pytest.raises(BudgetExhausted) as spent:
        await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert spent.value.name == "max_fetch_bytes"
    assert provider.calls == 0 and deps.asker.stats.fetched_bytes == 250


async def test_fetched_bytes_are_synced_into_the_run_stats() -> None:
    deps = _deps()
    deps.fetched_bytes = lambda: 4096
    await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert deps.asker.stats.fetched_bytes == 4096


# -- retraction ------------------------------------------------------------------


async def test_a_retracted_work_is_excluded_and_never_read() -> None:
    provider = FakeProvider()
    deps = _deps(provider, retracted=True)
    candidate = _candidate(deps)
    reading = await read_page(candidate, subquestion=SUBQ, queries=[], deps=deps)
    assert reading.excluded.startswith("retracted")
    assert reading.findings == [] and reading.passages == [] and not reading.readable
    assert provider.calls == 0 and deps.asker.stats.llm_calls == 0
    assert deps.asker.store.get_work(candidate.work.work_id).retracted == "retracted"  # type: ignore[union-attr]
    [excluded] = _events(deps, "excluded")
    assert (
        excluded.step == "Read" and excluded.data["work_id"] == candidate.work.work_id
    )
    assert _events(deps, "fetched") == []


async def test_a_failed_retraction_check_is_reported_not_assumed_fine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("andamentum.references.lookup.asyncio.sleep", no_sleep)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    deps = _deps()
    deps.crossref = CrossrefSource(store=deps.asker.store, http=http(down))
    candidate = _candidate(deps)
    reading = await read_page(candidate, subquestion=SUBQ, queries=[], deps=deps)
    assert any("retraction check failed" in d for d in reading.degraded)
    assert reading.findings  # the page is still read: unchecked is not retracted


async def test_no_doi_means_no_retraction_check() -> None:
    deps = _deps()
    work = Work(title="No DOI here")
    assert await check_retraction(work, crossref=deps.crossref) == (None, "")


def test_unreadable_pages_degrade_a_run_only_above_the_share() -> None:
    assert unreadable_degraded(attempted=0, unreadable=0) == []
    assert unreadable_degraded(attempted=10, unreadable=5) == []
    [line] = unreadable_degraded(attempted=10, unreadable=6)
    assert "6 of 10 pages" in line


# -- events ----------------------------------------------------------------------


async def test_every_event_is_emitted_with_its_documented_keys() -> None:
    deps = _deps()
    candidate = _candidate(deps)
    reading = await read_page(
        candidate, subquestion=SUBQ, queries=["apollo latency"], deps=deps
    )
    names = [(e.step, e.event) for e in _events(deps) if e.event != "llm_call"]
    assert names == [
        ("Read", "fetched"),
        ("Notes", "notes_input"),
        ("Notes", "finding_noted"),
        ("Notes", "notes"),
    ]
    [fetched] = _events(deps, "fetched")
    assert set(fetched.data) == {
        "work_id", "url", "title", "via", "ok", "chars", "status", "error",
        "truncated", "text",
    }  # fmt: skip
    assert fetched.data["text"] == TEXT and fetched.data["truncated"] is False
    assert fetched.data["via"] == "fetched" and fetched.data["ok"] is True
    assert fetched.data["chars"] == len(TEXT)
    [notes_input] = _events(deps, "notes_input")
    assert set(notes_input.data) == {
        "work_id", "url", "title", "subquestion", "source_chars", "window_chars",
        "window_spans", "text",
    }  # fmt: skip
    assert notes_input.data["source_chars"] == len(TEXT)
    assert notes_input.data["window_chars"] == len(TEXT)
    assert notes_input.data["window_spans"] == [[0, len(TEXT)]]
    assert TEXT in str(notes_input.data["text"])
    [noted] = _events(deps, "finding_noted")
    assert set(noted.data) == {
        "subquestion",
        "work_id",
        "title",
        "claim",
        "quote",
        "verified",
    }
    assert noted.data["quote"] == reading.findings[0].quote
    [notes] = _events(deps, "notes")
    assert notes.data["findings"] == 1 and notes.data["unverified"] == 0
    assert notes.data["subquestion"] == "Q1"


async def test_the_fetched_text_is_clipped_in_the_log_only() -> None:
    long_text = "Apollo cache sentence number one. " * (runlog.TEXT_CHARS // 30)
    assert len(long_text) > runlog.TEXT_CHARS
    deps = _deps(FakeProvider(long_text))
    reading = await load_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    [fetched] = _events(deps, "fetched")
    assert fetched.data["truncated"] is True
    assert len(str(fetched.data["text"])) == runlog.TEXT_CHARS
    assert fetched.data["chars"] == len(long_text)
    assert sum(len(p.text) for p in reading.passages) <= 12_000 + 4


async def test_the_unreadable_and_window_events_report_what_was_lost() -> None:
    text = _paragraphs(20, marked={9: "Quasar lattice width measured."})
    deps = _deps(FakeProvider(text), window_chars=2500)
    reading = await read_page(
        _candidate(deps), subquestion=SUBQ, queries=["quasar lattice"], deps=deps
    )
    [notes_input] = _events(deps, "notes_input")
    assert notes_input.data["source_chars"] == len(text)
    assert int(str(notes_input.data["window_chars"])) < len(text)
    assert "Quasar lattice width" in str(notes_input.data["text"])
    spans = notes_input.data["window_spans"]
    assert spans == [[p.start, p.end] for p in reading.passages]
    stored = deps.asker.store.passages_for_work(reading.work.work_id)
    assert {p.passage_id for p in reading.passages} <= {p.passage_id for p in stored}


async def test_no_logger_is_fine() -> None:
    deps = _deps(logged=False)
    reading = await read_page(_candidate(deps), subquestion=SUBQ, queries=[], deps=deps)
    assert reading.findings


def test_a_store_in_memory_is_not_shared_between_tests() -> None:
    assert ResearchStore.in_memory() is not ResearchStore.in_memory()
