"""The engine's step functions, one at a time (the graph is tested end to end
in ``test_graph.py``)."""

from __future__ import annotations

import pytest

from andamentum.references import Work

from ..criteria import Screener
from ..sources.base import SearchFilters, SourceError
from ..store import ResearchStore
from ..steps import (
    MAX_EXPAND_SEEDS,
    admit,
    compile_all,
    expand,
    expansion_seeds,
    fallback_criteria,
    filters_for,
    plan,
    read_targets,
    round_robin,
    screen,
    search,
    write_queries,
    write_reserve,
)
from ..types import Budget, Candidate, SubQuestion
from .engine_world import respond
from .llm_stub import ScriptedLLM, make_asker

Q1 = SubQuestion(id="Q1", text="How fast is the Apollo cache?")
Q2 = SubQuestion(id="Q2", text="What does SaltLog cost?")


def _candidate(doi: str, *, subquestion: str = "Q1", title: str = "") -> Candidate:
    return Candidate(
        work=Work(doi=doi, title=title or f"Work {doi}", year=2024),
        sources=["fake"],
        subquestion_ids=[subquestion],
    )


# -- plan -----------------------------------------------------------------------


async def test_a_plan_the_model_cannot_give_falls_back_to_the_question() -> None:
    asker = make_asker(model=ScriptedLLM(lambda i, p: {"subquestions": []}))
    outcome = await plan("Why is the sky blue?", asker=asker)
    assert [s.text for s in outcome.subquestions] == ["Why is the sky blue?"]
    assert outcome.degraded and outcome.degraded[0].startswith("planning failed")


async def test_a_spent_budget_plans_the_question_alone_and_says_so() -> None:
    asker = make_asker(model=ScriptedLLM(respond), budget=Budget(max_llm_calls=1))
    asker.stats.llm_calls = 1
    outcome = await plan("Why?", asker=asker)
    assert outcome.spent == "max_llm_calls" and len(outcome.subquestions) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Which Apollo studies appeared since 2020?", SearchFilters(year_from=2020)),
        ("Apollo papers published after 2019?", SearchFilters(year_from=2020)),
        ("Apollo work before 2015?", SearchFilters(year_to=2014)),
        ("Apollo studies from 2018-2022?", SearchFilters(year_from=2018, year_to=2022)),
        (
            "Apollo papers between 2022 and 2018?",
            SearchFilters(year_from=2018, year_to=2022),
        ),
        ("Apollo work in 2021?", SearchFilters(year_from=2021, year_to=2021)),
        ("Open access Apollo studies?", SearchFilters(open_access_only=True)),
        ("Who wrote about Apollo?", None),
    ],
)
def test_metadata_subquestions_become_source_filters(
    text: str, expected: SearchFilters | None
) -> None:
    assert filters_for(SubQuestion(id="Q1", text=text, kind="metadata")) == expected


def test_only_metadata_subquestions_are_filtered() -> None:
    assert filters_for(SubQuestion(id="Q1", text="Apollo since 2020?")) is None


# -- criteria -------------------------------------------------------------------


async def test_a_failed_compile_falls_back_to_one_relevance_criterion() -> None:
    asker = make_asker(model=ScriptedLLM(lambda i, p: {"criteria": []}))
    outcome = await compile_all([Q1], screener=Screener(asker=asker), question="q")
    assert outcome.criteria["Q1"] == fallback_criteria(Q1)
    assert outcome.degraded and "Q1" in outcome.degraded[0]


# -- queries --------------------------------------------------------------------


async def test_queries_are_capped_new_and_never_repeated() -> None:
    def answer(instructions: str, prompt: str) -> dict[str, object]:
        return {
            "queries": [
                {"subquestion": "Q1", "query": "apollo latency"},
                {"subquestion": "Q1", "query": "Apollo   Latency"},  # a repeat
                {"subquestion": "Q1", "query": "apollo cache benchmark"},
                {"subquestion": "Q1", "query": "apollo third"},
                {"subquestion": "Q2", "query": "saltlog overhead"},
            ]
        }

    outcome = await write_queries(
        "q",
        targets=[Q1, Q2],
        gaps={},
        done=["saltlog overhead"],
        per_subquestion=2,
        asker=make_asker(model=ScriptedLLM(answer)),
    )
    assert outcome.queries == {
        "Q1": ["apollo latency", "apollo cache benchmark"],
        # the model's only query had run; the sub-question's own text stands in
        "Q2": [Q2.text],
    }


async def test_failed_query_writing_searches_the_subquestions_themselves() -> None:
    asker = make_asker(model=ScriptedLLM(lambda i, p: {"queries": "nope"}))
    outcome = await write_queries(
        "q", targets=[Q1], gaps={}, done=[], per_subquestion=3, asker=asker
    )
    assert outcome.queries == {"Q1": [Q1.text]}
    assert outcome.degraded


# -- search and admission -----------------------------------------------------


class FakeSource:
    name = "fake"

    def __init__(self, results: list[Candidate] | SourceError) -> None:
        self.results = results
        self.calls: list[tuple[str, SearchFilters | None]] = []

    async def search(
        self, query: str, *, limit: int = 20, filters: SearchFilters | None = None
    ) -> list[Candidate]:
        self.calls.append((query, filters))
        if isinstance(self.results, SourceError):
            raise self.results
        return self.results[:limit]


async def test_a_failing_source_is_recorded_and_the_others_still_answer() -> None:
    good = FakeSource([_candidate("10.5555/a"), _candidate("10.5555/a")])
    bad = FakeSource(SourceError("broken", "HTTP 500", status=500))
    outcome = await search(
        {"Q1": ["apollo"]},
        sources=[bad, good],
        filters={"Q1": SearchFilters(year_from=2020)},
        limit=5,
    )
    assert [c.work.doi for c in outcome.candidates] == ["10.5555/a"]  # merged
    assert outcome.candidates[0].subquestion_ids == ["Q1"]
    assert outcome.degraded == ["fake search failed for 'apollo': HTTP 500"]
    assert good.calls == [("apollo", SearchFilters(year_from=2020))]


def test_admission_stores_works_and_unions_what_found_them() -> None:
    store = ResearchStore.in_memory()
    first = admit([_candidate("10.5555/a")], store=store, known={})
    (work_id,) = first.candidates
    again = admit(
        [_candidate("10.5555/a", subquestion="Q2")], store=store, known=first.candidates
    )
    assert list(again.candidates) == [work_id]
    assert again.candidates[work_id].subquestion_ids == ["Q1", "Q2"]
    assert again.pairs == [(work_id, "Q2")]


def test_round_robin_takes_a_fair_share_of_each_subquestion() -> None:
    pairs = [("a", "Q1"), ("b", "Q1"), ("c", "Q1"), ("d", "Q2")]
    assert round_robin(pairs, subquestions=[Q1, Q2])[:2] == [("a", "Q1"), ("d", "Q2")]


# -- screen ---------------------------------------------------------------------


async def test_a_spent_screening_budget_leaves_the_rest_pending() -> None:
    store = ResearchStore.in_memory()
    admitted = admit(
        [_candidate("10.5555/a"), _candidate("10.5555/b", subquestion="Q2")],
        store=store,
        known={},
    )
    asker = make_asker(
        model=ScriptedLLM(respond), store=store, budget=Budget(max_llm_calls=1)
    )
    asker.stats.llm_calls = 1
    outcome = await screen(
        admitted.pairs,
        candidates=admitted.candidates,
        subquestions=[Q1, Q2],
        criteria={"Q1": fallback_criteria(Q1), "Q2": fallback_criteria(Q2)},
        screener=Screener(asker=asker),
        question="q",
        limit=10,
    )
    assert outcome.spent == "max_llm_calls"
    assert outcome.screened == [] and outcome.scores == []


# -- expand ---------------------------------------------------------------------


class FakeExpander:
    name = "graph"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def search(self, query: str, *, limit: int = 20, filters=None):
        return []

    async def references(self, work: Work, *, limit: int = 20) -> list[Candidate]:
        self.calls.append(("references", work.doi or ""))
        if work.doi == "10.5555/unplaced":
            raise SourceError("graph", "cannot place the work", kind="config")
        return [_candidate(f"{work.doi}.ref{i}", subquestion="") for i in range(limit)]

    async def citations(self, work: Work, *, limit: int = 20) -> list[Candidate]:
        self.calls.append(("citations", work.doi or ""))
        raise SourceError("graph", "HTTP 503", status=503)


async def test_expansion_is_capped_and_inherits_the_seed_subquestions() -> None:
    seed = Work(doi="10.5555/seed", title="Seed", work_id="w_seed")
    unplaced = Work(doi="10.5555/unplaced", title="Unplaced", work_id="w_u")
    outcome = await expand(
        [(unplaced, ["Q2"]), (seed, ["Q1"])], sources=[FakeExpander()], limit=3
    )
    assert len(outcome.candidates) == 3
    assert all(c.subquestion_ids == ["Q1"] for c in outcome.candidates)
    assert all(c.sources == ["graph:references"] for c in outcome.candidates)
    assert outcome.notes == ["graph references of Unplaced: cannot place the work"]
    assert outcome.degraded == ["graph citations of Unplaced: HTTP 503"]


def test_expansion_seeds_are_scholarly_unexpanded_and_best_first() -> None:
    candidates = {
        f"w{i}": _candidate(f"10.5555/{i}") for i in range(MAX_EXPAND_SEEDS + 2)
    }
    candidates["web"] = Candidate(
        work=Work(url="https://apollo.example/page", title="A page"), sources=["web"]
    )
    kept = {w: (i / 10, ["Q1"]) for i, w in enumerate(candidates)}
    seeds = expansion_seeds(kept, candidates=candidates, expanded={"w6"})
    names = [w.doi for w, _ in seeds]
    assert len(names) == MAX_EXPAND_SEEDS
    assert "10.5555/6" not in names  # already expanded
    assert names[0] == "10.5555/5"  # the best scholarly one


# -- read -----------------------------------------------------------------------


def test_read_targets_prefer_open_subquestions_then_score() -> None:
    kept = {
        "covered-best": (0.9, ["Q1"]),
        "open-low": (0.2, ["Q2"]),
        "done": (1.0, ["Q2"]),
        "retracted": (1.0, ["Q2"]),
    }
    chosen = read_targets(
        kept,
        done={("done", "Q2")},
        excluded={"retracted": "retracted: x"},
        covered={"Q1"},
        limit=5,
    )
    assert chosen == [("open-low", ["Q2"]), ("covered-best", ["Q1"])]


def test_the_write_reserve_grows_with_the_subquestions() -> None:
    assert write_reserve([Q1, Q2]) > write_reserve([Q1])
