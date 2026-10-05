"""READ: passages, quote location, finding extraction, and one work end to end."""

from __future__ import annotations

import itertools
import json
from typing import Literal

import httpx
import pytest

from andamentum.references import Work

from ..ask import BudgetExhausted
from ..criteria import Screener
from ..read import (
    PASSAGE_MAX_CHARS,
    QUOTE_MIN_CHARS,
    ReadDeps,
    ReadTarget,
    extract_findings,
    locate_quote,
    prefilter,
    quote_is_long_enough,
    read_work,
    read_works,
    split_passages,
)
from ..sources.base import FetchedText, SearchFilters, SourceError
from ..sources.crossref import CrossrefSource
from ..store import ResearchStore
from ..types import Budget, Candidate, Criterion, Passage, SubQuestion
from .llm_stub import ScriptedLLM, decisions, http, make_asker

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
    assert len(passages) == 1
    assert passages[0].text == text


def test_long_paragraph_is_cut_at_sentence_ends() -> None:
    sentence = "The cache holds the hot keys of the service in memory. "
    text = (sentence * 60).strip()
    passages = split_passages(text, work_id="W1")
    assert len(passages) > 1
    assert all(p.text.endswith(".") for p in passages)
    assert all(len(p.text) <= PASSAGE_MAX_CHARS for p in passages)


def test_a_sentence_longer_than_the_maximum_is_cut_at_whitespace() -> None:
    text = "word " * 1000
    passages = split_passages(text.strip(), work_id="W1")
    assert len(passages) > 1
    assert all(len(p.text) <= PASSAGE_MAX_CHARS for p in passages)
    assert "wor" not in {p.text.split()[-1] for p in passages}


def test_repeated_text_is_one_passage_and_empty_text_none() -> None:
    assert split_passages("", work_id="W1") == []
    assert split_passages("  \n\n  ", work_id="W1") == []
    twice = "Same boilerplate sentence appears here.\n\n" * 1
    assert len(split_passages(twice * 2, work_id="W1")) == 1


def test_passage_ids_depend_on_the_work() -> None:
    a = split_passages(TEXT, work_id="W1")
    b = split_passages(TEXT, work_id="W2")
    assert {p.passage_id for p in a}.isdisjoint({p.passage_id for p in b})


# -- quotes ------------------------------------------------------------------


def test_locate_quote_exact() -> None:
    assert locate_quote("median latency", TEXT) == (
        TEXT.index("median latency"),
        TEXT.index("median latency") + len("median latency"),
    )


def test_locate_quote_ignores_case_whitespace_and_typography() -> None:
    text = "The “Apollo” cache — in short bursts —\n reduced   latency."
    span = locate_quote('the "apollo" cache - in short bursts - reduced latency', text)
    assert span is not None
    assert text[span[0] : span[1]] == text[: len(text) - 1]


def test_locate_quote_not_found_and_empty() -> None:
    assert locate_quote("a claim that is nowhere in the text", TEXT) is None
    assert locate_quote("   ", TEXT) is None


def test_locate_quote_maps_back_through_expanding_characters() -> None:
    text = "Die Straße war lang. Weitere Sätze folgen hier."
    span = locate_quote("die STRASSE war lang", text)
    assert span is not None
    assert text[span[0] : span[1]] == "Die Straße war lang"


def test_short_quotes_are_too_little() -> None:
    assert not quote_is_long_enough("x" * (QUOTE_MIN_CHARS - 1))
    assert quote_is_long_enough("x" * QUOTE_MIN_CHARS)


def test_prefilter_keeps_document_order_of_the_best_overlap() -> None:
    passages = [_passage(f"filler text number {i} about nothing") for i in range(6)]
    passages[4] = _passage("Apollo cache latency numbers", start=500)
    kept = prefilter(passages, query="Apollo cache latency", cap=2)
    assert passages[4] in kept
    assert [p.start for p in kept] == sorted(p.start for p in kept)
    assert prefilter(passages, query="anything", cap=10) == passages


# -- extraction --------------------------------------------------------------


async def test_extraction_keeps_only_findings_whose_quote_is_in_a_passage() -> None:
    passages = split_passages(TEXT, work_id="W1")
    assert len(passages) == 1
    passage = passages[0]
    criteria = [
        Criterion(
            id="Q1-C1",
            text="Does it report a latency?",
            kind="must",
            applies_to="passage",
        )
    ]

    def respond(instructions: str, prompt: str) -> dict[str, object]:
        return {
            "findings": [
                {
                    "passage": 1,
                    "claim": "Apollo caching cut median latency from 120 ms to 45 ms.",
                    "quote": QUOTE.upper(),  # same words, different case
                    "criteria": ["Q1-C1", "not-a-criterion"],
                },
                {
                    "passage": 1,
                    "claim": "Apollo makes everything ten times faster.",
                    "quote": "Apollo makes everything ten times faster in every case",
                    "criteria": [],
                },
                {
                    "passage": 1,
                    "claim": "Too short.",
                    "quote": "Apollo",
                    "criteria": [],
                },
                {
                    "passage": 7,  # a wrong passage number: found anyway
                    "claim": "SaltLog adds about 3 ms per request in verbose mode.",
                    "quote": "adds roughly 3 ms of overhead per request when verbose mode is on",
                    "criteria": [],
                },
            ]
        }

    asker = make_asker(model=ScriptedLLM(respond))
    findings, dropped = await extract_findings(
        passages,
        subquestion=SUBQ,
        criteria=criteria,
        asker=asker,
        ids=itertools.count(12),
    )

    assert [f.finding_id for f in findings] == ["F12", "F13"]
    first = findings[0]
    assert first.quote == QUOTE  # the passage's own words, not the model's casing
    assert passage.text[passage.text.index(first.quote) :].startswith(first.quote)
    assert first.criterion_ids == ["Q1-C1"]
    assert first.passage_id == passage.passage_id and first.work_id == "W1"
    assert findings[1].quote.startswith("adds roughly 3 ms")
    assert [d.kind for d in dropped] == ["quote_missing", "quote_missing"]
    assert asker.stats.llm_calls == 1


async def test_extraction_with_no_passages_asks_nothing() -> None:
    asker = make_asker(model=ScriptedLLM(lambda *_: {"findings": []}))
    assert await extract_findings(
        [], subquestion=SUBQ, criteria=[], asker=asker, ids=itertools.count(1)
    ) == ([], [])
    assert asker.stats.llm_calls == 0


# -- one work, end to end ----------------------------------------------------


class FakeProvider:
    """A text source: holds ``text`` (as ``access``), holds nothing, or fails."""

    name = "fake"

    def __init__(
        self,
        text: str | None = TEXT,
        *,
        access: Literal["full_text", "abstract"] = "full_text",
        error: SourceError | None = None,
    ) -> None:
        self.text = text
        self.access: Literal["full_text", "abstract"] = access
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


def _llm() -> ScriptedLLM:
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        if "choose which passages" in instructions.lower():
            return {"passages": [1]}
        return {
            "findings": [
                {
                    "passage": 1,
                    "claim": "Apollo caching cut median latency from 120 ms to 45 ms.",
                    "quote": QUOTE,
                    "criteria": [],
                }
            ]
        }

    return ScriptedLLM(respond)


def _deps(
    *,
    provider: FakeProvider,
    retracted: bool = False,
    llm: ScriptedLLM | None = None,
    store: ResearchStore | None = None,
    budget: Budget | None = None,
    decision_model=None,
) -> ReadDeps:
    asker = make_asker(
        model=llm or _llm(), store=store, budget=budget, decision_model=decision_model
    )
    return ReadDeps(
        screener=Screener(asker=asker),
        crossref=CrossrefSource(
            store=asker.store, http=http(_crossref(retracted=retracted))
        ),
        providers=[provider],
    )


def _work(deps: ReadDeps, **fields: object) -> Work:
    base: dict[str, object] = {
        "doi": "10.5555/apollo",
        "title": "Apollo caching",
        "year": 2024,
    }
    return deps.store.upsert_work(Work(**{**base, **fields}))  # type: ignore[arg-type]


async def test_a_work_is_read_into_stored_passages_and_a_verified_finding() -> None:
    provider = FakeProvider()
    deps = _deps(provider=provider)
    work = _work(deps)

    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )

    assert reading.excluded == "" and reading.done == ["Q1"]
    assert [f.finding_id for f in reading.findings] == ["F1"]
    finding = reading.findings[0]
    stored = deps.store.get_passage(finding.passage_id)
    assert stored is not None and finding.quote in stored.text
    assert deps.store.get_work(work.work_id).retracted == "no"  # type: ignore[union-attr]
    items = deps.store.run_items(deps.asker.run_id, "finding")
    assert [i.ref_id for i in items] == ["F1"]
    assert reading.work.access == "full_text"
    assert deps.store.get_work(work.work_id).access == "full_text"  # type: ignore[union-attr]
    assert provider.calls == 1


async def test_the_first_source_holding_text_wins_and_the_rest_are_not_asked() -> None:
    nothing = FakeProvider(None)
    wins = FakeProvider()
    spare = FakeProvider()
    deps = _deps(provider=nothing)
    deps.providers = [nothing, wins, spare]
    work = _work(deps)
    await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert (nothing.calls, wins.calls, spare.calls) == (1, 1, 0)


async def test_the_fetched_byte_budget_stops_reading_before_the_fetch() -> None:
    provider = FakeProvider()
    deps = _deps(provider=provider, budget=Budget(max_fetch_bytes=100))
    deps.fetched_bytes = lambda: 250  # what the source client has fetched so far
    work = _work(deps)
    with pytest.raises(BudgetExhausted) as spent:
        await read_work(
            work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
        )
    assert spent.value.name == "max_fetch_bytes"
    assert provider.calls == 0 and deps.asker.stats.fetched_bytes == 250


async def test_a_retracted_work_is_excluded_and_never_read() -> None:
    provider = FakeProvider()
    deps = _deps(provider=provider, retracted=True)
    work = _work(deps)

    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )

    assert reading.excluded.startswith("retracted")
    assert reading.findings == [] and reading.passages == []
    assert provider.calls == 0
    assert deps.asker.stats.llm_calls == 0
    assert deps.store.get_work(work.work_id).retracted == "retracted"  # type: ignore[union-attr]


async def test_a_failed_retraction_check_is_reported_not_assumed_fine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("andamentum.references.lookup.asyncio.sleep", no_sleep)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    deps = _deps(provider=FakeProvider())
    deps.crossref = CrossrefSource(store=deps.store, http=http(down))
    work = _work(deps)
    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert any("retraction check failed" in d for d in reading.degraded)
    assert deps.store.get_work(work.work_id).retracted == "unknown"  # type: ignore[union-attr]
    assert reading.findings  # the work is still read: unchecked is not retracted


def _refused() -> SourceError:
    return SourceError("fake", "fetch refused: paywall", kind="refused")


@pytest.mark.parametrize(
    ("provider", "fields", "access", "note"),
    [
        (
            FakeProvider(None),
            {"abstract": "A short abstract about Apollo."},
            "abstract",
            False,
        ),
        (FakeProvider(None), {}, "metadata", False),
        (FakeProvider(error=_refused()), {}, "unreadable", True),
        (
            FakeProvider(error=_refused()),
            {"abstract": "A short abstract about Apollo."},
            "abstract",
            True,
        ),
    ],
)
async def test_access_records_what_we_could_read(
    provider: FakeProvider, fields: dict[str, object], access: str, note: bool
) -> None:
    deps = _deps(provider=provider)
    work = _work(deps, **fields)
    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert reading.work.access == access
    assert deps.store.get_work(work.work_id).access == access  # type: ignore[union-attr]
    assert bool(reading.notes) is note
    if access != "abstract":
        assert reading.passages == [] and reading.findings == []


async def test_a_routine_refusal_is_noted_but_other_failures_degrade_the_run() -> None:
    refused = _deps(provider=FakeProvider(error=_refused()))
    broken = _deps(
        provider=FakeProvider(error=SourceError("fake", "HTTP 503", kind="http"))
    )
    results = []
    for deps in (refused, broken):
        results.append(
            await read_work(
                _work(deps),
                subquestions=[SUBQ],
                criteria={},
                deps=deps,
                ids=itertools.count(1),
            )
        )
    assert results[0].notes and results[0].degraded == []
    assert results[1].notes and len(results[1].degraded) == 1
    assert "HTTP 503" in results[1].degraded[0]


async def test_an_abstract_from_a_source_is_not_read_under_a_no_redistribute_licence() -> (
    None
):
    deps = _deps(provider=FakeProvider("Licensed abstract text.", access="abstract"))
    work = _work(deps, licence="no_redistribute")
    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert reading.work.access == "metadata" and reading.passages == []


async def test_an_abstract_under_a_no_redistribute_licence_is_never_read() -> None:
    deps = _deps(provider=FakeProvider(None))
    work = _work(
        deps, abstract="Licensed abstract text about Apollo.", licence="no_redistribute"
    )
    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert reading.work.access == "metadata"
    assert reading.passages == [] and deps.asker.stats.llm_calls == 0


async def test_findings_for_each_subquestion_get_consecutive_ids() -> None:
    second = SubQuestion(id="Q2", text="What overhead does SaltLog add?")
    deps = _deps(provider=FakeProvider())
    work = _work(deps)
    reading = await read_work(
        work,
        subquestions=[SUBQ, second],
        criteria={},
        deps=deps,
        ids=itertools.count(5),
    )
    assert [(f.finding_id, f.subquestion_id) for f in reading.findings] == [
        ("F5", "Q1"),
        ("F6", "Q2"),
    ]
    assert reading.done == ["Q1", "Q2"]


async def test_a_dropped_quote_is_recorded_as_a_flag_item() -> None:
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        if "choose which passages" in instructions.lower():
            return {"passages": [1]}
        return {
            "findings": [
                {
                    "passage": 1,
                    "claim": "Invented.",
                    "quote": "a sentence the source never contained at all",
                    "criteria": [],
                }
            ]
        }

    deps = _deps(provider=FakeProvider(), llm=ScriptedLLM(respond))
    work = _work(deps)
    reading = await read_work(
        work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
    )
    assert reading.findings == [] and len(reading.dropped) == 1
    flags = deps.store.run_items(deps.asker.run_id, "flag")
    assert len(flags) == 1 and flags[0].data["kind"] == "quote_missing"


async def test_with_a_decision_model_the_passage_criteria_pick_the_passages() -> None:
    paragraphs = [
        "SaltLog is the logging component and writes one line per request to disk.",
        "Apollo caching reduced median latency from 120 ms to 45 ms across all "
        "tested workloads in the benchmark we ran.",
    ]
    text = "\n\n".join(p + " " + "padding " * 150 for p in paragraphs)
    seen: list[str] = []

    def judge(state: object, instruction: str) -> float:
        assert isinstance(state, dict)
        seen.append(str(state["passage"])[:20])
        return 0.9 if "Apollo caching" in str(state["passage"]) else 0.05

    decision_model, stub = decisions(judge)

    def respond(instructions: str, prompt: str) -> dict[str, object]:
        assert "choose which passages" not in instructions.lower()
        return {"findings": []}

    llm = ScriptedLLM(respond)
    deps = _deps(provider=FakeProvider(text), llm=llm, decision_model=decision_model)
    work = _work(deps)
    reading = await read_work(
        work,
        subquestions=[SUBQ],
        criteria={},
        deps=deps,
        ids=itertools.count(1),
        passages_per_work=1,
    )
    assert len(reading.passages) == 1
    assert reading.passages[0].text.startswith("Apollo caching")
    assert deps.asker.stats.decision_calls == len(seen) >= 2
    # The passage the model preferred is what the language model was shown.
    assert "Apollo caching reduced" in llm.calls[-1][1]
    assert "SaltLog is the logging" not in llm.calls[-1][1]
    assert len(stub.calls) == len(seen)


async def test_read_works_skips_pairs_already_read_and_stops_on_a_spent_budget() -> (
    None
):
    deps = _deps(provider=FakeProvider(), budget=Budget(max_llm_calls=2))
    first = _work(deps)
    other = _work(deps, doi="10.5555/other", title="Other Apollo paper", year=2023)
    third = _work(deps, doi="10.5555/third", title="Third Apollo paper", year=2022)
    targets = [ReadTarget(w, ["Q1"]) for w in (first, other, third)]

    outcome = await read_works(
        targets,
        subquestions=[SUBQ],
        criteria={},
        deps=deps,
        ids=itertools.count(1),
        skip={(first.work_id, "Q1")},
    )

    # `first` was skipped; `other` used the two calls (pick, extract); `third` found none left.
    assert [r.work.work_id for r in outcome.readings] == [other.work_id]
    assert outcome.stopped == "max_llm_calls"
    assert outcome.done == {(other.work_id, "Q1")}
    assert [f.finding_id for f in outcome.findings] == ["F1"]


async def test_a_model_failing_on_one_work_is_recorded_and_the_next_is_read() -> None:
    picks = {"n": 0}

    def flaky(instructions: str, prompt: str) -> dict[str, object]:
        if "choose which passages" in instructions.lower():
            picks["n"] += 1
            return {"passages": [1]}
        if picks["n"] == 1:  # every attempt at the first work's extraction fails
            return {"findings": "not a list"}
        return dict(_llm().respond(instructions, prompt))

    deps = _deps(provider=FakeProvider(), llm=ScriptedLLM(flaky))
    first = _work(deps)
    other = _work(deps, doi="10.5555/other", title="Other Apollo paper", year=2023)
    outcome = await read_works(
        [ReadTarget(w, ["Q1"]) for w in (first, other)],
        subquestions=[SUBQ],
        criteria={},
        deps=deps,
        ids=itertools.count(1),
    )
    failed, read = outcome.readings
    assert failed.work.work_id == first.work_id and failed.findings == []
    assert failed.degraded and "failed" in failed.degraded[0]
    assert outcome.done == {(other.work_id, "Q1")}  # the failed work may be retried
    assert [f.work_id for f in read.findings] == [other.work_id]
    assert outcome.stopped == ""


async def test_a_budget_error_from_read_work_itself_is_not_swallowed() -> None:
    deps = _deps(provider=FakeProvider(), budget=Budget(max_llm_calls=1))
    work = _work(deps)
    with pytest.raises(BudgetExhausted):
        await read_work(
            work, subquestions=[SUBQ], criteria={}, deps=deps, ids=itertools.count(1)
        )


def test_the_outcome_is_json_serialisable() -> None:
    from ..read import ReadOutcome, WorkReading

    outcome = ReadOutcome(readings=[WorkReading(work=Work(title="x"))], stopped="")
    assert json.loads(outcome.model_dump_json())["readings"][0]["work"]["title"] == "x"
