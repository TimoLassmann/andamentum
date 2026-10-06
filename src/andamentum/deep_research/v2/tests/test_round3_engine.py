"""Round 3, the engine side: merit-ordered screening admission (the ``recall``
preset), the licence of a candidate's snippet, the screening record, the
shared decision cache and the options of the sources.

Everything runs offline over invented works; no test reaches a network or a
model (the decision model is a stub, the language model a scripted one).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.core.decisions import DecisionModel
from andamentum.core.decisions.testing import StubDecisions, noul_answer
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from .. import ResearchStore, SourceConfig, research
from ..config import PRESETS, QUEUES, ResearchConfig, canonical_json, config_hash
from ..decision_cache import check_determinism
from ..frozen import FrozenInputs
from ..identity import merge, merge_pair
from ..sources import ArxivSource, build_sources
from ..sources.arxiv import to_search_query
from ..steps import restore_licensed_text
from ..strategies import (
    ArrivalQueue,
    SourceRankQueue,
    item_state,
    source_rank_score,
)
from ..types import Budget, Candidate, SubQuestion
from . import engine_world
from .engine_world import QUESTION, SEARXNG, World, world_llm
from .llm_stub import decisions, make_asker
from .source_stub import SourceKit, ok

BASELINE_HASH = "9621c6dae218004804ae66dd41983449346745bdf669406159a8a6ff3c90755c"
SECRET = "A restricted abstract sentence about invented kiln results."


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


def candidate(
    number: int, ranks: dict[str, int], *, subquestions: tuple[str, ...] = ("Q1",)
) -> Candidate:
    return Candidate(
        work=Work(
            work_id=f"w{number}",
            title=f"Apollo cache latency study {number}",
            abstract=f"The Apollo cache reduced latency in study {number}.",
            url=f"https://example.org/apollo/{number}",
            year=2020 + number % 5,
        ),
        sources=list(ranks) or ["openalex"],
        subquestion_ids=list(subquestions),
        source_ranks=ranks,
    )


# -- the preset and its hash ------------------------------------------------------


def test_recall_is_the_baseline_with_one_strategy_swapped() -> None:
    base, recall = PRESETS["baseline"], PRESETS["recall"]
    changed = {
        k
        for k in base.model_dump()
        if base.model_dump()[k] != recall.model_dump()[k]
        and k not in ("name", "description")
    }
    assert changed == {"queue"}
    assert (base.queue, recall.queue) == ("arrival", "source_rank")
    assert not recall.needs_criteria
    assert config_hash(recall) != config_hash(base)


def test_the_baseline_hash_does_not_move_and_carries_no_queue() -> None:
    base = PRESETS["baseline"]
    assert config_hash(base) == BASELINE_HASH
    assert '"queue"' not in canonical_json(base)
    assert config_hash(base.model_copy(update={"queue": "arrival"})) == BASELINE_HASH
    assert '"queue":"source_rank"' in canonical_json(PRESETS["recall"])
    for name, preset in PRESETS.items():
        if preset.queue == "arrival":
            assert '"queue"' not in canonical_json(preset), name


def test_an_unknown_queue_is_refused() -> None:
    fields = PRESETS["baseline"].model_dump()
    with pytest.raises(ValueError, match="unknown queue strategy"):
        ResearchConfig(**{**fields, "queue": "random"})
    assert sorted(QUEUES) == ["arrival", "source_rank"]


# -- the queue order ---------------------------------------------------------------


def test_the_source_rank_score_is_one_function() -> None:
    from ..bench import freeze

    assert freeze.source_rank_score is source_rank_score  # no second copy
    both = candidate(1, {"openalex": 5, "europepmc": 5})
    one = candidate(2, {"openalex": 1})
    assert source_rank_score(both) > source_rank_score(one)
    assert source_rank_score(candidate(3, {})) == 0.0
    assert source_rank_score(one) == pytest.approx(1 / 61)


def test_source_rank_orders_by_fusion_ties_in_arrival_order_and_hops_count() -> None:
    pool = {
        c.work.work_id: c
        for c in [
            candidate(1, {"europepmc": 9}),
            candidate(2, {"arxiv": 1}),
            candidate(3, {"arxiv": 1}),  # ties with 2: arrived later
            candidate(4, {"europepmc": 4, "arxiv": 4}),  # two lists beat one good one
            candidate(5, {"openalex:citations": 2}),  # a hop is a list of its own
            candidate(6, {"europepmc": 9, "europepmc:citations": 9}),
        ]
    }
    pairs = [(w, "Q1") for w in pool]
    ordered = [w for w, _ in SourceRankQueue().order(pairs, candidates=pool)]
    # w4 and w6 are on two lists each; w2 and w3 tie (arrival order); the hop
    # list of w5 is a single list; w1 has one poor place
    assert ordered == ["w4", "w6", "w2", "w3", "w5", "w1"]
    # a work on a hop list in addition to a search list gains over the search list alone
    assert source_rank_score(pool["w6"]) > source_rank_score(pool["w1"])


def test_the_arrival_queue_changes_nothing() -> None:
    pool = {"w1": candidate(1, {"arxiv": 9}), "w2": candidate(2, {"arxiv": 1})}
    pairs = [("w1", "Q1"), ("w2", "Q1")]
    assert ArrivalQueue().order(pairs, candidates=pool) == pairs
    assert pairs == [("w1", "Q1"), ("w2", "Q1")]  # the input is not reordered


def test_the_queue_keeps_pairs_of_other_subquestions_in_place_of_arrival() -> None:
    pool = {
        "w1": candidate(1, {"arxiv": 5}),
        "w2": candidate(2, {"arxiv": 1}, subquestions=("Q2",)),
    }
    pairs = [("w1", "Q1"), ("w2", "Q2"), ("w1", "Q2")]
    ordered = SourceRankQueue().order(pairs, candidates=pool)
    assert ordered == [("w2", "Q2"), ("w1", "Q1"), ("w1", "Q2")]


SUBQ = SubQuestion(id="Q1", text="How much does the Apollo cache reduce latency?")


def frozen_with(candidates: list[Candidate]) -> FrozenInputs:
    return FrozenInputs(
        question=QUESTION,
        mode="scholarly",
        captured_at=1.0,
        subquestions=[SUBQ],
        queries={"Q1": ["apollo cache latency"]},
        candidates=candidates,
    )


async def frozen_run(
    candidates: list[Candidate],
    *,
    config: ResearchConfig,
    store: ResearchStore | None = None,
    decision: DecisionModel | None = None,
    shared: ResearchStore | None = None,
    budget: Budget | None = None,
):
    store = store or ResearchStore.in_memory()
    world = World()
    model = decision or decisions(lambda _state, _instructions: 0.9)[0]
    kwargs: dict[str, Any] = {} if shared is None else {"shared_decisions": shared}
    return await research(
        QUESTION,
        mode="scholarly",
        model=world_llm().model,
        decision_model=model,
        store=store,
        sources=SourceConfig(searxng_url=SEARXNG, scholarly=("openalex",)),
        config=config,
        frozen=frozen_with(candidates),
        source_client=world.client(store, offline=True),
        budget=budget or Budget(max_rounds=1),
        **kwargs,
    )


async def test_the_recall_preset_screens_the_best_ranked_first_and_the_baseline_the_first_found() -> (
    None
):
    """Twelve candidates, a cap of 4 a round: arrival order screens the first four,
    the merit order the four with the best source rank (the last four to arrive)."""
    pool = [candidate(n, {"openalex": 13 - n}) for n in range(1, 13)]  # w12 is best
    # no expansion: the round then screens once, and the cap decides what
    base = PRESETS["baseline"].model_copy(
        update={"screen_per_subquestion": 4, "expand": "off"}
    )
    recall = PRESETS["recall"].model_copy(
        update={"screen_per_subquestion": 4, "expand": "off"}
    )

    def screened(result) -> list[str]:  # noqa: ANN001
        return [e.title for e in result.screening if e.p is not None]

    arrival = await frozen_run(pool, config=base, budget=Budget(max_rounds=1))
    merit = await frozen_run(pool, config=recall, budget=Budget(max_rounds=1))
    assert set(screened(arrival)) == {
        f"Apollo cache latency study {n}" for n in (1, 2, 3, 4)
    }
    assert set(screened(merit)) == {
        f"Apollo cache latency study {n}" for n in (9, 10, 11, 12)
    }
    assert len(screened(arrival)) == len(screened(merit)) == 4
    # the unscreened pairs are on record as never screened, in arrival order
    assert [e.arrival for e in merit.screening] == list(range(12))
    assert [e.p for e in merit.screening if e.p is None] and all(
        e.rank is None for e in merit.screening if e.p is None
    )


async def test_the_screening_record_has_the_identity_the_scores_and_the_final_rank() -> (
    None
):
    pool = [candidate(n, {"openalex": n}) for n in range(1, 4)]
    low = decisions(lambda state, _i: 0.1 if "study 3" in str(state) else 0.9)[0]
    result = await frozen_run(pool, config=PRESETS["recall"], decision=low)
    by_title = {e.title: e for e in result.screening}
    assert len(by_title) == 3
    kept = by_title["Apollo cache latency study 1"]
    assert kept.kept and kept.p == pytest.approx(0.9) and kept.rank is not None
    assert [w.work_id for w in result.works][kept.rank - 1] == kept.work_id
    dropped = by_title["Apollo cache latency study 3"]
    assert dropped.p == pytest.approx(0.1) and not dropped.kept and dropped.rank is None
    assert kept.url == "https://example.org/apollo/1"
    assert kept.source_ranks == {"openalex": 1}
    assert kept.identity().url == "https://example.org/apollo/1"
    assert kept.subquestion_id == "Q1" and kept.year == 2021
    # compact: no abstract, no passage
    assert "abstract" not in kept.model_dump()


# -- the licence of a snippet ------------------------------------------------------


def restricted(number: int = 1, *, abstract: str = SECRET) -> Candidate:
    work = Work(
        work_id=f"s2_{number}",
        s2_id=f"{number:040x}",
        title=f"Invented kiln study {number}",
        abstract=abstract,
        year=2022,
        licence=LICENCE_NO_REDISTRIBUTE,
        access="abstract",
    )
    return Candidate(
        work=work,
        sources=["semantic_scholar"],
        snippet=abstract[:500],
        subquestion_ids=["Q1"],
        source_ranks={"semantic_scholar": 1},
    )


def test_a_restricted_candidates_snippet_follows_its_works_licence_on_every_dump() -> (
    None
):
    live = restricted()
    assert live.snippet == SECRET  # in memory it is used
    assert live.model_dump()["snippet"] == ""
    assert SECRET not in live.model_dump_json()
    assert SECRET not in frozen_with([live]).model_dump_json()
    free = candidate(1, {"openalex": 1})
    free = free.model_copy(update={"snippet": "a public snippet"})
    assert free.model_dump()["snippet"] == "a public snippet"


def test_a_merged_free_work_does_not_carry_a_restricted_snippet() -> None:
    free_text = "Short public abstract."
    free = Candidate(
        work=Work(
            work_id="e1",
            s2_id=None,
            doi="10.5555/kiln.1",
            title="Invented kiln study 1",
            abstract=free_text,
            year=2022,
        ),
        sources=["europepmc"],
        snippet=free_text,
    )
    s2 = restricted().model_copy(
        update={"work": restricted().work.model_copy(update={"doi": "10.5555/kiln.1"})}
    )
    merged = merge_pair(free, s2)
    assert merged.work.redistributable and merged.work.abstract == free_text
    assert merged.snippet == free_text  # not the longer, restricted one
    assert SECRET not in merged.model_dump_json()
    [only] = merge([s2, free])
    assert only.snippet == free_text
    # two free records: the longer snippet, as it always was
    longer = free.model_copy(update={"snippet": free_text + " More."})
    assert merge_pair(free, longer).snippet.endswith("More.")


def test_a_frozen_run_reads_a_restricted_text_back_from_the_snapshot_store() -> None:
    live = restricted()
    store = ResearchStore.in_memory()
    store.upsert_work(live.work)  # what the capture's admission did
    snapshot = FrozenInputs.model_validate_json(frozen_with([live]).model_dump_json())
    loaded = snapshot.candidates[0]
    assert loaded.work.abstract == "" and loaded.snippet == ""  # nothing in the file
    [restored] = restore_licensed_text([loaded], store=store)
    assert restored.work.abstract == SECRET and restored.snippet == SECRET[:500]
    # the generic screen sees what the live run saw
    assert item_state(restored) == item_state(live)
    # a free candidate, and one the store does not hold, are returned as they are
    free = candidate(1, {"openalex": 1})
    unknown = restricted(2)
    unknown = unknown.model_copy(
        update={"snippet": "", "work": unknown.work.model_copy(update={"abstract": ""})}
    )
    assert restore_licensed_text([free, unknown], store=store) == [free, unknown]


async def test_the_screen_state_of_a_frozen_s2_only_candidate_equals_a_live_run_and_no_dump_has_the_text() -> (
    None
):
    live = restricted()
    states: list[object] = []

    def record(state: object, _instructions: str) -> float:
        if isinstance(state, dict) and "title" in state:  # not the determinism probe
            states.append(state)
        return 0.9

    # the capture's store holds the text; the snapshot file does not
    store = ResearchStore.in_memory()
    store.upsert_work(live.work)
    snapshot_candidate = FrozenInputs.model_validate_json(
        frozen_with([live]).model_dump_json()
    ).candidates[0]
    result = await frozen_run(
        [snapshot_candidate],
        config=PRESETS["recall"],
        store=store,
        decision=decisions(record)[0],
    )
    assert states and states[0] == item_state(live)
    assert states[0]["abstract"] == SECRET  # type: ignore[index]
    assert result.screening[0].kept
    for dump in (
        result.model_dump_json(),
        frozen_with([live]).model_dump_json(),
        json.dumps([w.model_dump(mode="json") for w in result.works]),
    ):
        assert SECRET not in dump


# -- the shared decision cache ----------------------------------------------------


def counting_decisions(*, flip: bool = False) -> tuple[DecisionModel, StubDecisions]:
    calls = {"n": 0}

    def answer(state: object, questions: dict[str, dict[str, object]]):
        calls["n"] += 1
        p = 0.9 if not flip or calls["n"] % 2 else 0.2
        return {key: noul_answer(p) for key in questions}

    stub = StubDecisions(answer)
    return stub.model(), stub


async def test_a_second_cell_is_served_from_the_shared_decision_cache() -> None:
    pool = [candidate(n, {"openalex": n}) for n in range(1, 6)]
    shared = ResearchStore.in_memory()
    first_model, first_stub = counting_decisions()
    first = await frozen_run(
        pool, config=PRESETS["recall"], decision=first_model, shared=shared
    )
    second_model, second_stub = counting_decisions()
    second = await frozen_run(
        pool, config=PRESETS["recall"], decision=second_model, shared=shared
    )
    assert first.stats.decision_calls > 0
    assert (
        first.stats.decision_cache_hits == 0 and first.stats.decision_shared_hits == 0
    )
    # the second cell asked nothing the first had answered, and says so
    assert second.stats.decision_calls == 0
    assert second.stats.decision_cache_hits == second.stats.decision_shared_hits > 0
    assert len(second_stub.calls) == 0
    assert (
        second.stats.decision_calls + second.stats.decision_cache_hits
        == first.stats.decision_calls
    )
    # the answers are the same: what a request would have got
    assert [(e.title, e.p, e.kept) for e in second.screening] == [
        (e.title, e.p, e.kept) for e in first.screening
    ]
    assert (
        len(first_stub.calls)
        == first.stats.decision_calls + first.stats.decision_probe_calls
    )


async def test_the_determinism_verdict_is_read_from_the_shared_store_once() -> None:
    shared = ResearchStore.in_memory()
    model, stub = counting_decisions()
    own = ResearchStore.in_memory()
    first = await check_determinism(model, store=own, shared=shared)
    assert first.enabled and first.probe_calls == 2 and len(stub.calls) == 2
    other = ResearchStore.in_memory()
    again = await check_determinism(model, store=other, shared=shared)
    assert again.enabled and again.reused and again.probe_calls == 0
    assert len(stub.calls) == 2  # no new probe
    assert other.determinism_of(model.digest or "") is not None  # kept in its own store


async def test_a_model_that_fails_its_determinism_check_gets_no_shared_cache() -> None:
    pool = [candidate(n, {"openalex": n}) for n in range(1, 4)]
    shared = ResearchStore.in_memory()
    flipping, _stub = counting_decisions(flip=True)
    result = await frozen_run(
        pool, config=PRESETS["recall"], decision=flipping, shared=shared
    )
    assert result.stats.decision_cache.startswith("disabled")
    assert result.stats.decision_cache_hits == 0
    # nothing was written to the shared store either
    digest = flipping.digest or ""
    assert (
        shared.cached_answers(digest=digest, question_hash="x", state_hash="y") is None
    )
    again = await frozen_run(
        pool,
        config=PRESETS["recall"],
        decision=counting_decisions(flip=True)[0],
        shared=shared,
    )
    assert again.stats.decision_shared_hits == 0


async def test_without_a_shared_store_the_run_is_what_it_was() -> None:
    pool = [candidate(n, {"openalex": n}) for n in range(1, 4)]
    result = await frozen_run(pool, config=PRESETS["recall"])
    assert result.stats.decision_cache == "enabled"
    assert result.stats.decision_shared_hits == 0


# -- the options of the sources ----------------------------------------------------


def test_the_all_terms_query_is_the_one_it_always_was() -> None:
    text = "Which conformal risk control methods bound the expected loss?"
    assert to_search_query(text) == to_search_query(text, form="all_terms")
    assert to_search_query(text) == (
        "all:conformal AND all:risk AND all:control AND all:methods AND "
        "all:bound AND all:expected AND all:loss"
    )


def test_the_any_term_query_joins_the_same_terms_with_or() -> None:
    text = "Which conformal risk control methods bound the expected loss?"
    every = to_search_query(text)
    any_term = to_search_query(text, form="any_term")
    assert any_term == every.replace(" AND ", " OR ")
    assert " AND " not in any_term
    # the same 8 terms: stop words out, 3+ letters, de-duplicated, the first 8
    long = "alpha beta gamma delta epsilon zeta theta iota kappa lambda alpha the"
    assert to_search_query(long, form="any_term").count("all:") == 8
    assert to_search_query(long, form="any_term").startswith("all:alpha OR all:beta")
    # field syntax is passed through as written, in either form
    assert (
        to_search_query("ti:kiln AND abs:salt", form="any_term")
        == "ti:kiln AND abs:salt"
    )


def arxiv_feed(_request: httpx.Request) -> httpx.Response:
    return ok("arxiv_feed.xml")


async def test_the_two_query_forms_never_share_a_cache_row() -> None:
    kit = SourceKit(arxiv_feed)
    words = "conformal risk control methods"
    all_terms = ArxivSource(client=kit.client)
    any_term = ArxivSource(client=kit.client, query_form="any_term")
    await all_terms.search(words, limit=5)
    await any_term.search(words, limit=5)
    assert len(kit.requests) == 2  # the second was not answered from the first's row
    assert " AND " in kit.param(0, "search_query")  # type: ignore[operator]
    assert " OR " in kit.param(1, "search_query")  # type: ignore[operator]
    assert kit.param(1, "sortBy") == "relevance" and kit.param(1, "max_results") == "5"
    await any_term.search(words, limit=5)
    assert len(kit.requests) == 2  # the same form is cached


async def test_build_sources_passes_the_query_form_and_limits_expansion() -> None:
    store = ResearchStore.in_memory()
    everything = SourceConfig(
        scholarly=("europepmc", "arxiv", "semantic_scholar"), arxiv_query="any_term"
    )
    default = build_sources(everything, store=store, mode="scholarly")
    assert {s.name for s in default.expand} == {"europepmc", "semantic_scholar"}
    limited = build_sources(
        everything.model_copy(update={"expand_with": ("europepmc",)}),
        store=store,
        mode="scholarly",
    )
    assert [s.name for s in limited.expand] == ["europepmc"]
    # searching and reading are not limited by it
    assert [s.name for s in limited.search] == [
        "europepmc",
        "arxiv",
        "semantic_scholar",
    ]
    assert "semantic_scholar" in [s.name for s in limited.text]
    arxiv = next(s for s in limited.search if s.name == "arxiv")
    assert isinstance(arxiv, ArxivSource) and arxiv._query_form == "any_term"  # noqa: SLF001
    none = build_sources(
        everything.model_copy(update={"expand_with": ()}), store=store, mode="scholarly"
    )
    assert none.expand == ()


def test_expand_with_none_gives_todays_expansion() -> None:
    store = ResearchStore.in_memory()
    config = SourceConfig(scholarly=("openalex", "europepmc", "arxiv"))
    assert config.expand_with is None and config.arxiv_query == "all_terms"
    built = build_sources(config, store=store, mode="scholarly")
    assert [s.name for s in built.expand] == ["openalex", "europepmc"]


def test_the_engine_world_is_unchanged_by_the_queue_default() -> None:
    assert engine_world.QUESTION == QUESTION
    assert make_asker().stats.decision_shared_hits == 0
