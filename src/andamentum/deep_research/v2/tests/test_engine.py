"""The research loop end to end over the invented world.

The reading, writing, verifying and ranking modules are replaced by the world's
stand-ins (``engine_world.install_stand_ins``): these tests are about the loop,
its stops, its budgets and its bookkeeping. No network, no real model.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections import Counter
from collections.abc import Callable

import httpx
import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.core import fetch_gate, url_safety

from .. import engine as engine_module
from .. import fusion, read, verify, write
from ..engine import WRITE_RESERVE, research
from ..sources import SourceConfig, SourceSet
from ..sources.base import SourceError
from ..store import ResearchStore
from ..types import Budget, Candidate, Mode, ResearchResult, RunLog
from . import engine_world
from .engine_world import (
    QUESTION,
    SEARXNG,
    World,
    install_stand_ins,
    respond,
    world_llm,
)
from .llm_stub import ScriptedLLM

SCHOLARLY_ONLY = ("openalex",)

# The real modules, kept here before any test replaces them with the stand-ins.
REAL = {
    (engine_module, "read_page"): read.read_page,
    (engine_module, "write_report"): write.write_report,
    (engine_module, "verify_report"): verify.verify_report,
    (engine_module, "unreadable_degraded"): read.unreadable_degraded,
    (fusion, "fused_scores"): fusion.fused_scores,
    (fusion, "rank_works"): fusion.rank_works,
}

Responder = Callable[[str, str], dict[str, object]]

PLAN = "You turn a question into the sub-questions"
QUERIES = "You are a search query specialist"
CHOOSE = "You choose which search results"
NOTES = "You read passages of ONE document"
REFLECT = "You judge the progress of research"


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The world's hosts are invented; treat every name as a public address."""
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


@pytest.fixture(autouse=True)
def reader(monkeypatch: pytest.MonkeyPatch) -> engine_world.StandInReader:
    return install_stand_ins(monkeypatch)


def scripted(**overrides: Callable[[str], dict[str, object]]) -> Responder:
    """The world's answers, except where *overrides* (keyed by the first words of
    the instructions: PLAN, QUERIES, CHOOSE, NOTES, REFLECT as names) answer."""
    prefixes = {
        "plan": PLAN,
        "queries": QUERIES,
        "choose": CHOOSE,
        "notes": NOTES,
        "reflect": REFLECT,
    }

    def respond_(instructions: str, prompt: str) -> dict[str, object]:
        for name, handler in overrides.items():
            prefix = prefixes.get(name, name)
            if instructions.startswith(prefix):
                return handler(prompt)
        return respond(instructions, prompt)

    return respond_


async def run(
    mode: Mode = "scholarly",
    *,
    llm: Responder | ScriptedLLM | None = None,
    budget: Budget | None = None,
    store: ResearchStore | None = None,
    world: World | None = None,
    clock: Callable[[], float] | None = None,
) -> tuple[ResearchResult, ResearchStore, World, ScriptedLLM]:
    world = world or World()
    store = store or ResearchStore.in_memory()
    model = llm if isinstance(llm, ScriptedLLM) else ScriptedLLM(llm or respond)
    kwargs = {} if clock is None else {"clock": clock}
    result = await research(
        QUESTION,
        mode=mode,
        model=model.model,
        store=store,
        budget=budget or Budget(),
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=SCHOLARLY_ONLY),
        source_client=world.client(store),
        **kwargs,  # type: ignore[arg-type]
    )
    return result, store, world, model


def events(result: ResearchResult, name: str):
    return [e for e in result.log.events if e.event == name]


def llm_steps(result: ResearchResult) -> dict[str, int]:
    return result.stats.step_llm_calls


# -- a full run in each mode ----------------------------------------------------


@pytest.mark.parametrize("mode", ["web", "scholarly", "both"])
async def test_a_run_in_each_mode_yields_findings_a_report_and_ranked_works(
    mode: Mode,
) -> None:
    result, store, _, _ = await run(mode)

    assert [s.id for s in result.subquestions] == ["Q1", "Q2"]
    assert result.degraded == []
    assert result.findings
    assert [f.finding_id for f in result.findings] == [
        f"F{n}" for n in range(1, len(result.findings) + 1)
    ]
    passages = {p.passage_id: p for p in result.passages}
    for finding in result.findings:
        assert finding.quote_verified
        assert finding.quote in passages[finding.passage_id].text
    assert result.report_markdown.startswith("# ")
    assert result.works
    assert result.works[0].work_id in result.cited_work_ids()
    assert len(result.subanswers) == 2
    assert result.queries
    record = store.get_run(result.run_id)
    assert record is not None and record.finished_at is not None
    assert {i.ref_id for i in store.run_items(result.run_id, "finding")} == {
        f.finding_id for f in result.findings
    }


async def test_a_retracted_work_is_excluded_and_never_ranked() -> None:
    result, store, _, _ = await run("scholarly")

    retracted = [w for w in store.iter_works() if w.doi == engine_world.RETRACTED_DOI]
    assert len(retracted) == 1
    assert retracted[0].work_id in result.excluded
    assert retracted[0].work_id not in [w.work_id for w in result.works]
    assert all(f.work_id != retracted[0].work_id for f in result.findings)


async def test_expansion_finds_the_work_known_only_by_a_citation() -> None:
    result, _, _, _ = await run("scholarly")

    # Apollo needs three independent sources; the third is the follow-up that
    # only the citations of the first two lead to, so round 2 expands.
    apollo = next(a for a in result.subanswers if a.subquestion_id == "Q1")
    assert apollo.answered and apollo.rounds == 2
    assert any(w.title.startswith("A follow-up") for w in result.works)
    (expanded,) = [
        e for e in events(result, "expanded") if e.data["subquestion"] == "Q1"
    ]
    assert expanded.step == "Expand"
    assert int(str(expanded.data["count"])) >= 1


class _NoteSource:
    """A search source that answers with nothing but a warning."""

    name = "chatty"

    async def search(
        self, query: str, *, limit: int = 20, filters: object = None
    ) -> list[Candidate]:
        return []

    def take_warnings(self) -> list[str]:
        return ["an engine of the chatty source was down"]


class _OfflineSource:
    name = "offline"

    async def search(
        self, query: str, *, limit: int = 20, filters: object = None
    ) -> list[Candidate]:
        raise SourceError("offline", "not in the snapshot", kind="offline")


async def test_the_log_survives_a_jsonl_round_trip_with_every_event_of_the_catalogue(
    monkeypatch: pytest.MonkeyPatch, reader: engine_world.StandInReader
) -> None:
    from .. import engine as engine_module
    from ..write import Report

    seen: set[str] = set()
    runs: list[ResearchResult] = []
    for mode in ("web", "scholarly", "both"):
        runs.append((await run(mode))[0])
    # a dead source degrades, a spent budget is a result
    runs.append((await run("both", world=World(searxng_down=True)))[0])
    runs.append((await run("scholarly", budget=Budget(max_llm_calls=9)))[0])
    # an unreadable page
    reader.unreadable = ("apollo-docs",)
    runs.append((await run("web"))[0])
    reader.unreadable = ()

    # a verify flag: the writer cites a finding that does not exist
    async def citing_nothing(question: str, **_kw: object) -> Report:
        return Report(markdown="# Report\n\nA sentence. [F99]\n")

    # stub sources: a note (a warning) and a skipped search (an offline miss)
    real_build = engine_module.build_sources

    def with_stubs(*args: object, **kwargs: object) -> SourceSet:
        built = real_build(*args, **kwargs)  # type: ignore[arg-type]
        return dataclasses.replace(
            built, search=(*built.search, _NoteSource(), _OfflineSource())
        )

    monkeypatch.setattr(engine_module, "build_sources", with_stubs)
    install_stand_ins(monkeypatch, reader=reader, writer=citing_nothing)
    runs.append((await run("scholarly"))[0])

    # a failed run keeps its log on the exception
    def broken(instructions: str, prompt: str) -> dict[str, object]:
        if instructions.startswith(QUERIES):
            raise RuntimeError("a bug, not a model failure")
        return respond(instructions, prompt)

    with pytest.raises(RuntimeError) as failure:
        await run("scholarly", llm=broken)
    failed_log: RunLog = failure.value.run_log  # type: ignore[attr-defined]

    for log in [*(r.log for r in runs), failed_log]:
        round_trip = RunLog.from_jsonl(log.to_jsonl())
        assert round_trip == log
        seen |= {e.event for e in round_trip.events}
    catalogue = {
        "config", "models", "sources", "subquestion", "round_start", "queries",
        "query", "search_results", "admitted", "expanded", "choice",
        "chosen_work", "fetched", "unreadable", "excluded", "notes_input",
        "finding_noted", "notes", "reflect", "subquestion_done", "report",
        "verify_flag", "final_ranking", "llm_call", "degraded", "note",
        "skipped", "budget_spent", "finished", "failed",
    }  # fmt: skip
    assert catalogue <= seen, catalogue - seen


async def test_every_event_names_its_step_and_the_steps_run_in_order() -> None:
    result, _, _, _ = await run("both")
    by_event = {e.event: e.step for e in result.log.events}
    assert by_event["config"] == "Setup"
    assert by_event["subquestion"] == "Plan"
    assert by_event["query"] == "Search"
    assert by_event["choice"] == "Choose"
    assert by_event["fetched"] == "Read"
    assert by_event["notes"] == "Notes"
    assert by_event["reflect"] == "Reflect"
    assert by_event["report"] == "Write"
    assert by_event["final_ranking"] == "Finish"
    assert [e.seq for e in result.log.events] == list(range(len(result.log.events)))


# -- stops ------------------------------------------------------------------------


async def test_a_question_answered_in_round_one_runs_one_round() -> None:
    result, _, _, _ = await run(
        "web",
        llm=scripted(
            reflect=lambda p: {
                "answered": True,
                "answer": "Likely 45 ms",
                "confidence": "high",
            }
        ),
    )

    assert result.stop_reason == "answered"
    assert [(a.subquestion_id, a.rounds, a.stop_reason) for a in result.subanswers] == [
        ("Q1", 1, "answered"),
        ("Q2", 1, "answered"),
    ]
    assert all(
        a.answer == "Likely 45 ms" and a.confidence == "high" for a in result.subanswers
    )
    assert len(events(result, "round_start")) == 2
    steps = llm_steps(result)
    assert steps["Plan"] == 1
    assert steps["Search"] == 2 and steps["Choose"] == 2 and steps["Reflect"] == 2
    assert steps["Write"] == 1


async def test_a_question_never_answered_runs_every_round_then_writes() -> None:
    counter = iter(range(1000))
    result, _, _, model = await run(
        "web",
        llm=scripted(
            reflect=lambda p: {
                "answered": False,
                "answer": "Perhaps 45 ms",
                "gaps": ["a second source"],
                "next_queries": [f"extra words {next(counter)}"],
            }
        ),
    )

    assert result.stop_reason == "rounds"
    assert all(a.rounds == 3 and a.stop_reason == "rounds" for a in result.subanswers)
    assert all(
        not a.answered and a.gaps == ["a second source"] for a in result.subanswers
    )
    origins = [(e.data["round"], e.data["origin"]) for e in events(result, "queries")]
    assert sorted(set(origins)) == [(1, "writer"), (2, "reflect"), (3, "reflect")]
    # rounds two and three take reflect's queries: no query-writing call
    assert llm_steps(result)["Search"] == 2
    assert result.report_markdown.startswith("# ")
    assert result.findings
    del model


async def test_a_round_without_proposed_queries_asks_the_writer_with_the_gaps() -> None:
    result, _, _, model = await run(
        "web",
        llm=scripted(
            reflect=lambda p: {"answered": False, "gaps": ["the year asked for"]}
        ),
    )

    assert (
        llm_steps(result)["Search"] == 6
    )  # round one, two and three, both sub-questions
    later = [p for i, p in model.calls if i.startswith(QUERIES) and "GAPS:" in p]
    assert later and all("the year asked for" in p for p in later)
    assert {
        str(e.data["origin"]) for e in events(result, "queries") if e.data["round"] == 2
    } == {"writer"}


async def test_a_spent_call_budget_ends_the_search_and_the_report_is_still_written() -> (
    None
):
    result, store, _, _ = await run("scholarly", budget=Budget(max_llm_calls=9))

    assert result.stop_reason == "budget:max_llm_calls"
    assert result.stats.llm_calls <= 9
    assert llm_steps(result)["Write"] == 1
    assert result.findings
    assert result.report_markdown.startswith("# ")
    assert all(a.stop_reason == "budget:max_llm_calls" for a in result.subanswers)
    record = store.get_run(result.run_id)
    assert record is not None and record.finished_at is not None
    (spent,) = events(result, "budget_spent")
    assert spent.data["bound"] == "max_llm_calls"


async def test_the_search_leaves_the_write_reserve_of_calls() -> None:
    result, _, _, _ = await run("web", budget=Budget(max_llm_calls=12))
    non_write = sum(n for step, n in llm_steps(result).items() if step != "Write")
    assert non_write <= 12 - WRITE_RESERVE
    assert llm_steps(result).get("Write") == 1


async def test_a_spent_wall_clock_ends_the_search_and_the_report_is_still_written() -> (
    None
):
    now = [0.0]

    def ticking(instructions: str, prompt: str) -> dict[str, object]:
        now[0] += 10.0
        return respond(instructions, prompt)

    result, _, _, _ = await run(
        "web", llm=ticking, budget=Budget(max_wall_s=65.0), clock=lambda: now[0]
    )

    assert result.stop_reason == "budget:max_wall_s"
    assert result.report_markdown.startswith("# ")
    assert result.stats.wall_s >= 65.0
    assert any(a.stop_reason == "budget:max_wall_s" for a in result.subanswers)


async def test_a_spent_fetch_budget_ends_the_search() -> None:
    result, _, _, _ = await run("web", budget=Budget(max_fetch_bytes=1))
    # the search itself fetched more than a byte: nothing is read, a report is written
    assert result.stop_reason == "budget:max_fetch_bytes"
    assert result.stats.pages_read == 0
    assert result.report_markdown.startswith("# ")


async def test_no_candidates_still_gives_a_result_without_model_calls_for_nothing() -> (
    None
):
    class Empty(World):
        def handler(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "searx.test":
                return httpx.Response(200, json={"results": []})
            return super().handler(request)

    result, _, _, _ = await run("web", world=Empty())

    assert result.findings == [] and result.works == []
    assert result.stop_reason == "rounds"
    assert all(a.rounds == 3 for a in result.subanswers)
    assert "No evidence could be read for this question." in result.report_markdown
    steps = llm_steps(result)
    assert "Choose" not in steps and "Notes" not in steps and "Reflect" not in steps
    assert "Write" not in steps
    assert events(result, "final_ranking")[0].data["works_ranked"] == 0


# -- reading ----------------------------------------------------------------------


def _narrowed(instructions: str, prompt: str) -> dict[str, object]:
    """The world's choice, but one pick and the rest as reserves."""
    answer = respond(instructions, prompt)
    if instructions.startswith(CHOOSE):
        picks = answer["picks"]
        assert isinstance(picks, list)
        return {"picks": picks[:1], "reserves": picks[1:]}
    return answer


async def test_an_unreadable_pick_is_replaced_by_the_next_reserve(
    reader: engine_world.StandInReader,
) -> None:
    reader.unreadable = ("apollo-docs",)
    result, store, _, _ = await run("web", llm=_narrowed, budget=Budget(max_rounds=1))

    titles = [store.get_work(w).title for w in reader.read if store.get_work(w)]  # type: ignore[union-attr]
    apollo = [t for t in titles if "Apollo" in t]
    assert apollo == ["Apollo cache latency guide", "Ops notes: Apollo in production"]
    assert result.stats.pages_read == 3 and result.stats.pages_unreadable == 1
    q1 = [f for f in result.findings if f.subquestion_id == "Q1"]
    assert q1 and all("ops-notes" in (store.get_work(f.work_id).url or "") for f in q1)  # type: ignore[union-attr]
    (unreadable,) = events(result, "unreadable")
    assert str(unreadable.data["url"]).endswith("apollo-docs.example/latency")


async def test_a_pick_read_only_from_its_snippet_is_also_replaced_by_a_reserve(
    reader: engine_world.StandInReader,
) -> None:
    reader.snippet_only = ("apollo-docs",)
    result, store, _, _ = await run("web", llm=_narrowed, budget=Budget(max_rounds=1))

    titles = [store.get_work(w).title for w in reader.read if store.get_work(w)]  # type: ignore[union-attr]
    apollo = [t for t in titles if "Apollo" in t]
    assert apollo == ["Apollo cache latency guide", "Ops notes: Apollo in production"]
    assert result.stats.pages_unreadable == 0
    assert any(
        e.data.get("via") == "snippet" for e in events(result, "fetched")
    )  # what the snippet gave is kept


async def test_each_reserve_is_used_at_most_once() -> None:
    def three_reserves(instructions: str, prompt: str) -> dict[str, object]:
        if instructions.startswith(CHOOSE):
            return {"picks": [1], "reserves": [2, 3, 4]}
        return respond(instructions, prompt)

    reader = engine_world.StandInReader(unreadable=("example",))
    from pytest import MonkeyPatch

    with MonkeyPatch.context() as patch:
        install_stand_ins(patch, reader=reader)
        result, _, _, _ = await run(
            "web", llm=three_reserves, budget=Budget(max_rounds=1)
        )
    # one pick fails, then one reserve per failure: a sub-question tries each
    # page once at most (both sub-questions see the same four pages here)
    assert max(Counter(reader.read).values()) <= 2
    assert result.stats.pages_read == 8
    assert result.findings == []
    assert any("could not be read" in d for d in result.degraded)


async def test_the_unreadable_page_degrade_rule_is_applied() -> None:
    reader = engine_world.StandInReader(unreadable=("example",))
    from pytest import MonkeyPatch

    with MonkeyPatch.context() as patch:
        install_stand_ins(patch, reader=reader)
        result, _, _, _ = await run("web", budget=Budget(max_rounds=1))
    assert result.stats.pages_unreadable == result.stats.pages_read > 0
    assert result.degraded == ["4 of 4 pages could not be read"]


async def test_an_unverified_quote_keeps_its_finding_and_is_counted() -> None:
    def wrong_quote(prompt: str) -> dict[str, object]:
        return {
            "findings": [
                {
                    "passage": 1,
                    "claim": "Apollo cut latency.",
                    "quote": "words the page never said",
                }
            ]
        }

    result, _, _, _ = await run("web", llm=scripted(notes=wrong_quote))
    assert result.findings
    assert all(not f.quote_verified for f in result.findings)
    assert result.stats.quotes_unverified == len(result.findings)
    assert {bool(e.data["verified"]) for e in events(result, "finding_noted")} == {
        False
    }


async def test_findings_are_numbered_in_subquestion_then_round_then_pick_order() -> (
    None
):
    result, _, _, _ = await run("scholarly")
    subs = [f.subquestion_id for f in result.findings]
    assert subs == sorted(subs)
    assert [f.finding_id for f in result.findings] == [
        f"F{n}" for n in range(1, len(result.findings) + 1)
    ]


# -- failing models ---------------------------------------------------------------

BAD: dict[str, object] = {"unexpected": "shape"}


@pytest.mark.parametrize(
    ("step", "prefix", "line"),
    [
        ("Plan", PLAN, "planning failed"),
        ("Search", QUERIES, "query writing failed"),
        ("Choose", CHOOSE, "choosing what to read failed"),
        ("Notes", NOTES, "notes for"),
        ("Reflect", REFLECT, "reflecting on"),
        ("Write", engine_world.STAND_IN_WRITE[:20], "the report could not be written"),
    ],
)
async def test_a_failing_model_at_each_step_degrades_and_the_run_continues(
    step: str, prefix: str, line: str
) -> None:
    def failing(instructions: str, prompt: str) -> dict[str, object]:
        return BAD if instructions.startswith(prefix) else respond(instructions, prompt)

    result, _, _, _ = await run("web", llm=failing)

    assert any(line in d for d in result.degraded), result.degraded
    assert result.report_markdown.startswith("# ")
    assert result.stop_reason in {"answered", "rounds"}
    if step != "Notes":
        assert result.findings


async def test_a_failing_plan_searches_the_question_as_one_subquestion() -> None:
    def no_plan(instructions: str, prompt: str) -> dict[str, object]:
        return BAD if instructions.startswith(PLAN) else respond(instructions, prompt)

    result, _, _, _ = await run("scholarly", llm=no_plan)
    assert [s.text for s in result.subquestions] == [QUESTION]


async def test_a_dead_source_degrades_the_run_but_does_not_empty_it() -> None:
    result, _, _, _ = await run("both", world=World(searxng_down=True))

    assert any(
        "search failed for" in d and d.startswith("web") for d in result.degraded
    )
    assert result.findings  # the scholarly sources still answered
    assert result.degraded == list(dict.fromkeys(result.degraded))


async def test_a_failed_run_is_recorded_before_it_raises() -> None:
    def broken(instructions: str, prompt: str) -> dict[str, object]:
        if instructions.startswith(QUERIES):
            raise RuntimeError("a bug in the engine, not a model failure")
        return respond(instructions, prompt)

    store = ResearchStore.in_memory()
    with pytest.raises(RuntimeError) as failure:
        await run("scholarly", llm=broken, store=store)
    (record,) = store.list_runs()
    assert record.finished_at is not None
    assert record.degraded and record.degraded[0].startswith("run failed")
    log: RunLog = failure.value.run_log  # type: ignore[attr-defined]
    assert log.events[-1].event == "failed" and log.events[-1].step == "Finish"
    assert [i.kind for i in store.run_items(record.run_id, "log")] == ["log"]


async def test_a_source_client_of_another_store_is_refused() -> None:
    client = World().client(ResearchStore.in_memory())
    with pytest.raises(ValueError, match="must belong"):
        await research(
            QUESTION,
            mode="scholarly",
            model=world_llm().model,
            store=ResearchStore.in_memory(),
            source_client=client,
        )


async def test_a_shared_http_cache_cannot_go_with_an_open_store() -> None:
    with pytest.raises(ValueError, match="http_cache"):
        await research(
            QUESTION,
            mode="scholarly",
            model=world_llm().model,
            store=ResearchStore.in_memory(),
            http_cache=ResearchStore.in_memory(),
        )


async def test_a_second_run_reads_the_first_runs_cache() -> None:
    store = ResearchStore.in_memory()
    first, _, world, _ = await run("scholarly", store=store)
    sent = len(world.requests)
    second, _, _, _ = await run("scholarly", store=store, world=world)
    assert second.stop_reason == first.stop_reason
    assert second.stats.cache_hits > 0
    assert len(world.requests) - sent < sent


# -- parallel sub-questions ---------------------------------------------------------


def _texts(messages: list[ModelMessage]) -> tuple[str, str]:
    instructions, prompt = "", ""
    for message in messages:
        if isinstance(message, ModelRequest):
            instructions = message.instructions or instructions
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not prompt:
                    prompt = str(part.content)
    return instructions, prompt


def _tool_call(info: AgentInfo, fields: dict[str, object]) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name=info.output_tools[0].name, args=json.dumps(fields))
        ]
    )


def _slow_model(gauge: dict[str, int]) -> FunctionModel:
    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        gauge["now"] += 1
        gauge["peak"] = max(gauge["peak"], gauge["now"])
        await asyncio.sleep(0.002)
        gauge["now"] -= 1
        return _tool_call(info, respond(*_texts(messages)))

    return FunctionModel(answer)


async def _parallel_run(concurrency: int) -> tuple[ResearchResult, dict[str, int]]:
    gauge = {"now": 0, "peak": 0}
    store = ResearchStore.in_memory()
    result = await research(
        QUESTION,
        mode="both",
        model=_slow_model(gauge),
        store=store,
        budget=Budget(concurrency=concurrency),
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=SCHOLARLY_ONLY),
        source_client=World().client(store),
    )
    return result, gauge


async def test_two_subquestions_in_parallel_never_exceed_the_concurrency() -> None:
    for width in (1, 2, 4):
        _, gauge = await _parallel_run(width)
        assert gauge["peak"] <= width
    _, gauge = await _parallel_run(4)
    assert gauge["peak"] >= 2  # the sub-questions really did run side by side


async def test_finding_ids_and_the_order_of_noted_findings_repeat_across_runs() -> None:
    """With a model that answers at once (no timers), the schedule of the two
    sub-questions is the same every run; the ids never depend on it anyway."""
    first, _, _, _ = await run("both")
    second, _, _, _ = await run("both")

    def table(result: ResearchResult) -> list[tuple[str, str, str]]:
        return [(f.finding_id, f.subquestion_id, f.claim) for f in result.findings]

    def noted(result: ResearchResult) -> list[tuple[object, object]]:
        return [
            (e.data["subquestion"], e.data["claim"])
            for e in events(result, "finding_noted")
        ]

    assert table(first) == table(second)
    assert noted(first) and noted(first) == noted(second)
    assert first.report_markdown == second.report_markdown
    assert [w.title for w in first.works] == [w.title for w in second.works]


async def test_finding_ids_do_not_depend_on_which_subquestion_finishes_first() -> None:
    """Q2 is slowed down so that it finishes after Q1 in one run and before it in
    the other: the findings are numbered Q1 first both times."""

    async def run_with_delay(slow: str) -> ResearchResult:
        gauge = {"now": 0, "peak": 0}
        store = ResearchStore.in_memory()

        async def answer(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            instructions, prompt = _texts(messages)
            if slow in prompt.lower() and instructions.startswith(NOTES):
                await asyncio.sleep(0.02)
            return _tool_call(info, respond(instructions, prompt))

        del gauge
        return await research(
            QUESTION,
            mode="both",
            model=FunctionModel(answer),
            store=store,
            sources=SourceConfig(searxng_url=SEARXNG, scholarly=SCHOLARLY_ONLY),
            source_client=World().client(store),
        )

    q1_slow = await run_with_delay("apollo")
    q2_slow = await run_with_delay("saltlog")
    for result in (q1_slow, q2_slow):
        assert [f.subquestion_id for f in result.findings] == sorted(
            f.subquestion_id for f in result.findings
        )
    assert [(f.finding_id, f.claim) for f in q1_slow.findings] == [
        (f.finding_id, f.claim) for f in q2_slow.findings
    ]


# -- the log keeps what the loss attribution needs ----------------------------------


async def test_the_run_log_text_fields_are_not_truncated() -> None:
    long_answer = "Likely 45 ms. " * 80
    result, _, _, _ = await run(
        "web",
        llm=scripted(reflect=lambda p: {"answered": True, "answer": long_answer}),
    )
    (reflect, *_) = events(result, "reflect")
    assert reflect.data["answer"] == " ".join(long_answer.split())
    assert len(str(reflect.data["answer"])) > 700
    page = engine_world.PAGES["https://apollo-docs.example/latency"][1]
    fetched = {e.data["url"]: e.data for e in events(result, "fetched")}
    assert fetched["https://apollo-docs.example/latency"]["text"] == page
    given = [e.data["text"] for e in events(result, "notes_input")]
    assert page in given
    snippets = [
        row["snippet"]
        for e in events(result, "search_results")
        for row in e.data["results"]  # type: ignore[union-attr]
    ]
    shown = " ".join(page[:160].split())
    assert any(s.startswith(shown[:100]) and len(s) >= 100 for s in snippets)
    (report,) = events(result, "report")
    assert report.data["text"] == result.report_markdown
    # the llm_call row, on the other hand, keeps 500 characters of output
    assert all(len(str(e.data["output"])) <= 500 for e in events(result, "llm_call"))


async def test_the_stats_count_pages_and_busy_seconds_per_step() -> None:
    result, _, _, _ = await run("both")
    stats = result.stats
    assert stats.pages_read >= len({f.work_id for f in result.findings})
    assert stats.llm_calls == sum(stats.step_llm_calls.values())
    assert {"Plan", "Search", "Choose", "Read", "Reflect", "Write", "Verify"} <= set(
        stats.step_seconds
    )
    assert stats.wall_s >= 0.0
    assert "web" in stats.per_source or "openalex" in stats.per_source


async def test_the_queries_per_round_bound_trims_the_writers_queries() -> None:
    result, _, _, _ = await run("web", budget=Budget(queries_per_round=1, max_rounds=1))
    for event in events(result, "queries"):
        queries = event.data["queries"]
        assert isinstance(queries, list) and len(queries) == 2  # own text + one
        assert queries[0] in {engine_world.SUBQ_APOLLO, engine_world.SUBQ_SALTLOG}


# -- the real reading, writing, verifying and ranking modules ----------------------


@pytest.mark.parametrize("mode", ["web", "scholarly", "both"])
async def test_the_loop_over_the_real_modules_gives_verified_findings_and_a_report(
    mode: Mode, monkeypatch: pytest.MonkeyPatch
) -> None:
    for (module, name), real in REAL.items():
        monkeypatch.setattr(module, name, real)

    result, store, _, _ = await run(mode)

    assert result.findings and result.report_markdown.startswith("# ")
    passages = {p.passage_id: p for p in result.passages}
    for finding in result.findings:
        assert finding.quote_verified
        assert finding.quote in passages[finding.passage_id].text
    known = {f.finding_id for f in result.findings}
    assert {f.kind for f in result.verification}.isdisjoint({"unknown_finding"})
    cited = set(re.findall(r"\bF\d+\b", result.report_markdown))
    assert cited and cited <= known
    assert result.works and result.works[0].work_id in result.cited_work_ids()
    retracted = [w for w in store.iter_works() if w.doi == engine_world.RETRACTED_DOI]
    assert all(w.work_id not in [x.work_id for x in result.works] for w in retracted)
    assert result.stats.pages_read > 0
