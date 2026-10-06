"""Round 2: per-step timing, the lexical pre-filter, code-first verification, the
decision cache, the degraded threshold and the ``fast`` preset, and the proof
that the baseline is what it was (``fixtures/baseline_preset_trace.json``)."""

from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.core.decisions import DecisionModel, Question
from andamentum.core.decisions.testing import StubDecisions, noul_answer
from andamentum.references import Work

from .. import ResearchStore, SourceConfig, research
from ..ask import request_key
from ..config import PRESETS, ResearchConfig, canonical_json, config_hash, preset
from ..criteria import Screener
from ..decision_cache import CHECK_NAME, check_determinism
from ..read import UNREADABLE_DEGRADE_SHARE, unreadable_degraded
from ..strategies import (
    PREFILTER_TOP,
    GenericPrefilterScreen,
    GenericScreen,
    prefilter_order,
    screen_one,
)
from ..types import STEPS, Budget, Candidate, Finding, Mode, Passage, SubQuestion
from ..verify import (
    CODE_SUPPORT_MIN_OVERLAP,
    UNSUPPORTED_MARK,
    content_tokens,
    supported_in_code,
    token_overlap,
    verify_report,
)
from . import engine_world
from .engine_world import (
    PAGES,
    QUESTION,
    SEARXNG,
    World,
    world_decisions,
    world_llm,
)
from .llm_stub import decisions, http, make_asker
from .preset_trace import SCENARIOS, counted_uuids, trace

BASELINE_TRACE = Path(__file__).parent / "fixtures" / "baseline_preset_trace.json"
SUBQ = SubQuestion(id="Q1", text="How much does the Apollo cache reduce latency?")


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


async def _research(
    mode: Mode,
    *,
    world: World | None = None,
    decision: DecisionModel | None = None,
    config: ResearchConfig = PRESETS["baseline"],
    decision_cache: bool = True,
    store: ResearchStore | None = None,
    clock=None,  # type: ignore[no-untyped-def]
):
    world = world or World()
    store = store or ResearchStore.in_memory()
    extra = {} if clock is None else {"clock": clock}
    result = await research(
        QUESTION,
        mode=mode,
        model=world_llm().model,
        decision_model=decision,
        store=store,
        budget=Budget(),
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        config=config,
        source_client=world.client(store),
        decision_cache=decision_cache,
        **extra,
    )
    return result, store


# -- the baseline is what it was ---------------------------------------------------


async def test_the_baseline_reproduces_the_engine_before_round_two() -> None:
    """Every model request, in order, and the evidence table of four runs over
    the invented world are those the engine at commit f339357 produced for the
    baseline preset (recorded with ``preset_trace`` from that commit's code)."""
    recorded = json.loads(BASELINE_TRACE.read_text(encoding="utf-8"))
    assert [(t["mode"], t["decision"]) for t in recorded] == list(SCENARIOS)
    for expected, (mode, decision) in zip(recorded, SCENARIOS, strict=True):
        fetch_gate._robots_cache_clear()
        got = await trace(
            mode=mode,
            decision=decision,
            run=research,
            world=engine_world,
            store_type=ResearchStore,
            source_config=SourceConfig,
            extra={"config": PRESETS["baseline"], "decision_cache": False},
        )
        got = json.loads(json.dumps(got, sort_keys=True))
        for key in expected:
            assert got[key] == expected[key], f"{mode}/{decision}: {key} differs"


def test_the_baseline_hash_is_the_one_recorded_before_the_verify_slot_existed() -> None:
    base = PRESETS["baseline"]
    assert base.verify == "model"
    assert '"verify"' not in canonical_json(base)
    assert config_hash(base) == config_hash(base.model_copy(update={"verify": "model"}))
    # a configuration that does use the slot hashes differently
    assert config_hash(base.model_copy(update={"verify": "code_first"})) != config_hash(
        base
    )


async def test_the_decision_cache_changes_no_answer_of_the_baseline() -> None:
    # work ids are random and break ties between works: count them
    with counted_uuids():
        cached, _ = await _research("scholarly", decision=world_decisions()[0])
    with counted_uuids():
        plain, _ = await _research(
            "scholarly", decision=world_decisions()[0], decision_cache=False
        )
    assert cached.stats.decision_cache == "enabled"
    assert plain.stats.decision_cache == "off"
    assert cached.report_markdown == plain.report_markdown
    assert [(f.claim, f.quote) for f in cached.findings] == [
        (f.claim, f.quote) for f in plain.findings
    ]
    assert [w.doi for w in cached.works] == [w.doi for w in plain.works]
    assert (cached.stop_reason, cached.degraded, cached.notes) == (
        plain.stop_reason,
        plain.degraded,
        plain.notes,
    )
    assert cached.stats.decision_calls <= plain.stats.decision_calls


async def test_a_second_run_over_the_same_store_is_served_from_the_cache() -> None:
    store = ResearchStore.in_memory()
    model, stub = world_decisions()
    with counted_uuids():
        first, _ = await _research(
            "scholarly", decision=world_decisions()[0], store=store
        )
        second, _ = await _research("scholarly", decision=model, store=store)
    assert first.stats.decision_calls > 0 and first.stats.decision_cache_hits == 0
    assert second.stats.decision_cache_hits > 0
    assert second.stats.decision_calls < first.stats.decision_calls
    assert [(f.claim, f.quote) for f in second.findings] == [
        (f.claim, f.quote) for f in first.findings
    ]
    # the determinism verdict was kept in the store: no probe the second time
    assert second.stats.decision_probe_calls == 0
    assert len(stub.calls) == second.stats.decision_calls


# -- the fast preset ---------------------------------------------------------------


def test_fast_is_the_baseline_with_two_strategies_swapped() -> None:
    base, fast = PRESETS["baseline"], PRESETS["fast"]
    assert preset("fast") is fast
    changed = {
        k
        for k in base.model_dump()
        if base.model_dump()[k] != fast.model_dump()[k]
        and k not in ("name", "description")
    }
    assert changed == {"screen", "verify"}
    assert (fast.screen, fast.verify) == ("generic_prefilter", "code_first")
    assert not fast.needs_criteria
    assert config_hash(fast) != config_hash(base)


async def test_the_fast_preset_runs_end_to_end() -> None:
    result, _ = await _research(
        "scholarly", decision=world_decisions()[0], config=PRESETS["fast"]
    )
    assert result.config_name == "fast"
    assert result.stop_reason == "covered"
    assert result.findings
    stats = result.stats
    assert stats.verify_by_code + stats.verify_by_model > 0


# -- per-step timing and calls -----------------------------------------------------


async def test_every_step_is_timed_and_its_calls_counted() -> None:
    ticks = count(0, 2)
    result, store = await _research(
        "scholarly", decision=world_decisions()[0], clock=lambda: float(next(ticks))
    )
    stats = result.stats
    assert set(stats.step_seconds) <= set(STEPS)
    assert {"Plan", "Search", "Screen", "Read", "Coverage", "Write", "Verify"} <= set(
        stats.step_seconds
    )
    assert all(seconds > 0 for seconds in stats.step_seconds.values())
    assert sum(stats.step_seconds.values()) <= stats.wall_s
    assert sum(stats.step_llm_calls.values()) == stats.llm_calls > 0
    assert sum(stats.step_decision_calls.values()) == stats.decision_calls > 0
    assert set(stats.step_llm_calls) <= set(STEPS)
    assert "Plan" in stats.step_llm_calls and "Write" in stats.step_llm_calls
    assert "Screen" in stats.step_decision_calls
    assert "Verify" in stats.step_decision_calls
    record = store.get_run(result.run_id)
    assert record is not None and record.stats is not None
    assert record.stats["step_seconds"] == stats.step_seconds
    assert record.stats["step_llm_calls"] == stats.step_llm_calls


# -- the lexical pre-filter --------------------------------------------------------


def _candidate(n: int, *, relevant: bool) -> Candidate:
    text = (
        f"The Apollo cache reduces latency by a measured amount in study {n}."
        if relevant
        else f"A survey of unrelated gardening practices, volume {n}."
    )
    return Candidate(
        work=Work(
            doi=f"10.5555/p{n}",
            title=f"Apollo cache latency {n}" if relevant else f"Gardening {n}",
            abstract=text,
            year=2024,
            work_id=f"w{n}",
        ),
        sources=["openalex"],
        subquestion_ids=["Q1"],
    )


async def test_the_prefilter_asks_only_the_best_twelve_and_drops_the_rest() -> None:
    relevant = [_candidate(n, relevant=True) for n in range(1, 11)]
    irrelevant = [_candidate(n, relevant=False) for n in range(11, 21)]
    candidates = [c for pair in zip(irrelevant, relevant) for c in pair]  # mixed order
    model, stub = decisions(lambda state, instructions: 0.9)
    asker = make_asker(decision_model=model)
    for c in candidates:
        asker.store.upsert_work(c.work)
    run = screen_one(
        PRESETS["fast"], criteria={}, screener=Screener(asker=asker), question="q"
    )
    standing, scores = await run(SUBQ, candidates)

    assert standing == []
    assert PREFILTER_TOP == 12
    assert len(stub.calls) == PREFILTER_TOP
    asked_titles = {state["title"] for state, _ in stub.calls}  # type: ignore[index]
    assert all(f"Apollo cache latency {n}" in asked_titles for n in range(1, 11))
    assert len(scores) == len(candidates) == 20
    dropped = [s for s in scores if not s.kept]
    assert len(dropped) == 8
    assert all("below lexical pre-filter" in s.reason for s in dropped)
    assert all(s.work_id in {f"w{n}" for n in range(11, 21)} for s in dropped)
    assert len([s for s in scores if s.kept]) == PREFILTER_TOP
    items = asker.store.run_items(asker.run_id, "candidate")
    assert len(items) == 20
    by_screen = {i.data["screen"] for i in items}  # type: ignore[index]
    assert by_screen == {"generic_prefilter"}
    assert sum("pre-filter" in str(i.data["reason"]) for i in items) == 8


async def test_with_twelve_candidates_or_fewer_the_prefilter_is_the_generic_screen() -> (
    None
):
    candidates = [_candidate(n, relevant=n % 2 == 0) for n in range(1, 13)]
    outcomes = []
    for strategy in (GenericScreen(), GenericPrefilterScreen()):
        model, stub = decisions(lambda state, instructions: 0.7)
        asker = make_asker(decision_model=model)
        for c in candidates:
            asker.store.upsert_work(c.work)
        scores = await strategy.screen(
            SUBQ,
            candidates,
            criteria=[],
            screener=Screener(asker=asker),
            config=PRESETS["fast"],
        )
        outcomes.append(([s.model_dump() for s in scores], len(stub.calls)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] == 12


def test_the_prefilter_ranks_by_bm25_and_keeps_arrival_order_on_ties() -> None:
    candidates = [
        _candidate(1, relevant=False),
        _candidate(2, relevant=True),
        _candidate(3, relevant=False),
        _candidate(4, relevant=True),
    ]
    assert prefilter_order(SUBQ, candidates) == [1, 3, 0, 2]


# -- code-first verification -------------------------------------------------------


def test_content_tokens_drop_stop_words_and_keep_numbers_exactly() -> None:
    tokens = content_tokens(
        "The Apollo cache cut latency from 120 ms to 3.5 ms, 1,200 times."
    )
    assert tokens == [
        "apollo",
        "cache",
        "cut",
        "latency",
        "120",
        "3.5",
        "1,200",
        "times",
    ]
    assert content_tokens("is of an a") == []
    assert content_tokens("IL6 and p53") == ["il6", "p53"]


def test_token_overlap_counts_tokens_found_in_the_quotes() -> None:
    quotes = ["reduced median latency from 120 ms to 45 ms"]
    assert token_overlap("median latency 120 45", quotes) == 1.0
    assert token_overlap("median latency 120 46", quotes) == 0.75
    assert token_overlap("of the and", quotes) is None  # nothing to compare


def test_a_sentence_is_supported_in_code_from_the_threshold_up() -> None:
    assert CODE_SUPPORT_MIN_OVERLAP == 0.8
    quotes = ["alpha bravo charlie delta echo"]
    assert supported_in_code("alpha bravo charlie delta zulu", quotes)  # 4 of 5
    assert not supported_in_code("alpha bravo charlie yankee zulu", quotes)  # 3 of 5
    assert not supported_in_code("of the", quotes)  # no content token: the model's


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
F1 = Finding(
    finding_id="F1",
    subquestion_id="Q1",
    claim="claim 1",
    quote="Apollo cache reduced median latency from 120 ms to 45 ms",
    passage_id=P1.passage_id,
    work_id="W1",
)
F2 = Finding(
    finding_id="F2",
    subquestion_id="Q1",
    claim="claim 2",
    quote="adds roughly 3 ms of overhead per request",
    passage_id=P2.passage_id,
    work_id="W2",
)
WORKS = {
    "W1": Work(work_id="W1", title="Apollo caching", authors=["Jane Doe"], year=2024),
    "W2": Work(work_id="W2", title="SaltLog costs", authors=["Max Muster"], year=2023),
}
REPORT = (
    "# Does Apollo help?\n\n## Latency\n\n"
    "The Apollo cache reduced median latency from 120 ms to 45 ms [F1]. "
    "SaltLog was cheaper than Apollo in every benchmark [F2].\n\n"
    "## Sources\n\n- [F1] Doe (2024). Apollo caching.\n"
)


def _ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"message": {}})


async def test_code_first_settles_supported_sentences_in_code_and_asks_the_model_the_rest() -> (
    None
):
    model, stub = decisions(lambda state, instructions: 0.0)  # the model says no
    asker = make_asker(decision_model=model)
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
        code_first=True,
    )
    assert len(stub.calls) == 1  # only the second sentence reached the model
    state = stub.calls[0][0]
    assert isinstance(state, dict) and "SaltLog" in str(state["statement"])
    unsupported = [f.ref for f in outcome.flags if f.kind == "unsupported"]
    assert unsupported == ["SaltLog was cheaper than Apollo in every benchmark"]
    assert outcome.report_markdown.count(UNSUPPORTED_MARK) == 1
    assert (outcome.by_code, outcome.by_model) == (1, 1)
    assert outcome.sentences_checked == 2
    assert asker.stats.verify_by_code == 1 and asker.stats.verify_by_model == 1
    assert asker.stats.decision_calls == 1


async def test_without_code_first_every_sentence_goes_to_the_model() -> None:
    model, stub = decisions(lambda state, instructions: 0.9)
    asker = make_asker(decision_model=model)
    outcome = await verify_report(
        REPORT,
        findings=[F1, F2],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(_ok),
    )
    assert len(stub.calls) == 2
    assert (outcome.by_code, outcome.by_model) == (0, 2)
    assert (asker.stats.verify_by_code, asker.stats.verify_by_model) == (0, 2)


# -- the decision cache ------------------------------------------------------------

STATE: dict[str, object] = {"title": "Apollo", "abstract": "A cache."}
QUESTIONS = {"q": Question(type="noul", instructions="Is it relevant?")}


async def test_an_answered_request_is_served_from_the_store() -> None:
    model, stub = decisions(lambda state, instructions: 0.8)
    asker = make_asker(decision_model=model)
    asker.decision_cache = True
    first = await asker.decide("screen", state=STATE, questions=QUESTIONS)
    again = await asker.decide("screen", state=STATE, questions=QUESTIONS)
    other = await asker.decide(
        "screen", state={**STATE, "title": "SaltLog"}, questions=QUESTIONS
    )
    assert len(stub.calls) == 2
    assert first == again and other["q"].p_true == 0.8
    stats = asker.stats
    assert (stats.decision_calls, stats.decision_cache_hits) == (2, 1)
    # the hit is in the audit trail too, with no time spent
    rows = asker.store.decisions(asker.run_id, "screen")
    assert [r.ms == 0.0 for r in rows] == [False, True, False]


async def test_the_cache_is_off_unless_the_run_switched_it_on() -> None:
    model, stub = decisions(lambda state, instructions: 0.8)
    asker = make_asker(decision_model=model)
    await asker.decide("screen", state=STATE, questions=QUESTIONS)
    await asker.decide("screen", state=STATE, questions=QUESTIONS)
    assert len(stub.calls) == 2 and asker.stats.decision_cache_hits == 0


async def test_a_cached_answer_needs_no_budget_and_is_keyed_by_the_model_digest() -> (
    None
):
    store = ResearchStore.in_memory()
    stub = StubDecisions(lambda state, questions: {"q": noul_answer(0.8)})
    asker = make_asker(
        decision_model=stub.model(),
        store=store,
        budget=Budget(max_decision_calls=1),
    )
    asker.decision_cache = True
    await asker.decide("screen", state=STATE, questions=QUESTIONS)
    await asker.decide("screen", state=STATE, questions=QUESTIONS)  # a hit: no budget
    assert len(stub.calls) == 1
    # the same request to another model (another digest) is a miss
    other = StubDecisions(
        lambda state, questions: {"q": noul_answer(0.1)}, digest="sha256:" + "cd" * 32
    )
    second = make_asker(decision_model=other.model(), store=store)
    second.decision_cache = True
    answers = await second.decide("screen", state=STATE, questions=QUESTIONS)
    assert len(other.calls) == 1 and answers["q"].p_true == 0.1


def test_the_request_key_covers_every_question_and_its_key() -> None:
    a = {"q": Question(type="noul", instructions="one")}
    b = {"q": Question(type="noul", instructions="two")}
    c = {"r": Question(type="noul", instructions="one")}
    assert len({request_key(a), request_key(b), request_key(c)}) == 3
    assert request_key(a) == request_key(dict(a))


async def test_a_deterministic_model_passes_the_check_once_per_store() -> None:
    store = ResearchStore.in_memory()
    stub = StubDecisions(lambda state, questions: {"q": noul_answer(0.6)})
    check = await check_determinism(stub.model(), store=store)
    assert (check.deterministic, check.enabled, check.probe_calls) == (True, True, 2)
    assert len(stub.calls) == 2
    assert stub.calls[0] == stub.calls[1]  # two identical requests
    again = await check_determinism(stub.model(), store=store)
    assert again.enabled and again.reused and again.probe_calls == 0
    assert len(stub.calls) == 2


def _drifting() -> StubDecisions:
    ps = count(1)
    return StubDecisions(
        lambda state, questions: {"q": noul_answer(0.5 + next(ps) / 100)}
    )


def _drifting_world() -> tuple[DecisionModel, StubDecisions]:
    """The world's decision model with every yes/no probability shifted a little
    more on each request (the same side of 0.5): identical requests differ."""
    model, stub = world_decisions()
    answer = stub.respond
    steps = count(1)

    def drift(state: object, questions: dict[str, dict[str, object]]):
        shift = next(steps) / 100_000
        return {
            key: noul_answer(a.p_true + shift) if a.p_true is not None else a
            for key, a in answer(state, questions).items()
        }

    stub.respond = drift
    return model, stub


async def test_a_model_whose_identical_answers_differ_disables_the_cache() -> None:
    store = ResearchStore.in_memory()
    check = await check_determinism(_drifting().model(), store=store)
    assert (check.deterministic, check.enabled) == (False, False)
    assert "different answers" in check.detail
    assert check.state.startswith("disabled: ")
    digest = _drifting().digest
    assert store.determinism_of(digest) == (False, check.detail)


async def test_a_model_without_a_digest_cannot_be_cached() -> None:
    model = StubDecisions(lambda s, q: {"q": noul_answer(0.5)}).model()
    model.digest = None
    check = await check_determinism(model, store=ResearchStore.in_memory())
    assert not check.enabled and "no digest" in check.detail


async def test_a_run_with_a_drifting_model_records_that_the_cache_is_off() -> None:
    handle, _stub = _drifting_world()
    result, store = await _research("scholarly", decision=handle)
    stats = result.stats
    assert stats.decision_cache.startswith("disabled: two identical requests")
    assert stats.decision_cache_hits == 0 and stats.decision_probe_calls == 2
    assert any(n.startswith("decision cache disabled") for n in result.notes)
    items = store.run_items(result.run_id, "check")
    assert [i.ref_id for i in items] == [CHECK_NAME]
    assert items[0].data["deterministic"] is False  # type: ignore[index]
    # the probes are not decision calls of the run
    assert stats.decision_calls == len(
        store.decisions(result.run_id)
    )  # every real request is in the audit trail, the probes are not


async def test_a_run_without_a_decision_model_has_no_cache() -> None:
    result, store = await _research("web")
    assert result.stats.decision_cache == "off"
    assert store.run_items(result.run_id, "check") == []


async def test_the_decision_cache_can_be_turned_off_for_a_run() -> None:
    model, stub = world_decisions()
    result, store = await _research("scholarly", decision=model, decision_cache=False)
    assert result.stats.decision_cache == "off"
    assert result.stats.decision_probe_calls == 0
    assert len(stub.calls) == result.stats.decision_calls
    assert store.run_items(result.run_id, "check") == []


async def test_requests_made_equal_decision_calls_plus_the_probes() -> None:
    model, stub = world_decisions()
    result, _ = await _research("both", decision=model)
    assert result.stats.decision_cache == "enabled"
    assert len(stub.calls) == result.stats.decision_calls + 2


# -- the degraded threshold --------------------------------------------------------


def test_unreadable_works_degrade_a_run_only_above_the_share() -> None:
    assert UNREADABLE_DEGRADE_SHARE == 0.5
    assert unreadable_degraded(attempted=0, unreadable=0) == []
    assert unreadable_degraded(attempted=8, unreadable=1) == []
    assert unreadable_degraded(attempted=4, unreadable=2) == []  # exactly half
    [line] = unreadable_degraded(attempted=4, unreadable=3)
    assert "3 of 4 works" in line


class _FailingPages(World):
    """A world where some of the web pages answer 503 when they are fetched."""

    def __init__(self, failing: set[str]) -> None:
        super().__init__()
        self.failing = failing

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) in self.failing:
            self.requests.append(request)
            return httpx.Response(503, text="try later")
        return super().handler(request)


async def test_a_single_unreadable_page_is_a_note_not_a_degraded_run() -> None:
    urls = list(PAGES)
    result, _ = await _research("web", world=_FailingPages({urls[0]}))
    assert result.degraded == []
    assert any("could not give the text" in n for n in result.notes)
    assert result.findings


async def test_mostly_unreadable_pages_degrade_the_run() -> None:
    urls = list(PAGES)
    result, _ = await _research("web", world=_FailingPages(set(urls[:-1])))
    assert any(
        "works chosen for reading could not be read" in d for d in result.degraded
    )
    assert sum("could not give the text" in n for n in result.notes) == len(urls) - 1
