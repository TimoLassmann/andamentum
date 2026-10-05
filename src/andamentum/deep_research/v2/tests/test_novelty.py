"""The v2 novelty check, offline: the verdict comes from judged passages, never a guess.

The real engine runs over the invented world of ``engine_world.py``; only the
network, the language model and the decision model are stand-ins.
"""

from __future__ import annotations

import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.core.decisions import Answer, DecisionModel
from andamentum.core.decisions.errors import DecisionError
from andamentum.references import Work

from .. import novelty
from ..ask import Asker
from ..novelty import (
    MIN_WORKS_ASSESSED,
    run_novelty_check_v2,
)
from ..sources import SourceConfig
from ..store import ResearchStore
from ..types import Passage, ResearchResult
from .engine_world import World, holds, world_decisions, world_llm

CLAIM = "The Apollo cache cuts median request latency by half."
SCHOLARLY_ONLY = SourceConfig(scholarly=("openalex",))


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


def _judge(anticipates):
    def judge(question: str, text: str) -> bool:
        if question.startswith("The passage reports the same"):
            return anticipates(text)
        if question.startswith("The passage is about the same topic"):
            return "apollo" in text.lower() or "saltlog" in text.lower()
        return holds(question, text)

    return judge


async def _check(
    *, anticipates, decision: DecisionModel | None = None, **kwargs
) -> tuple[novelty.NoveltyReport, ResearchStore]:
    store = ResearchStore.in_memory()
    handle = decision or world_decisions(judge=_judge(anticipates))[0]
    report = await run_novelty_check_v2(
        CLAIM,
        model=world_llm().model,
        decision_model=handle,
        store=store,
        sources=SCHOLARLY_ONLY,
        source_client=World().client(store),
        **kwargs,
    )
    return report, store


async def test_a_passage_that_anticipates_the_claim_makes_it_not_novel() -> None:
    report, store = await _check(anticipates=lambda text: "120 ms to 45 ms" in text)
    assert report.is_novel is False
    assert report.confidence >= 0.5
    direct = [w for w in report.similar_work if w.relevance.value == "direct"]
    assert direct
    # the evidence is the passage's own words
    passages = [
        p.text
        for run in store.list_runs()
        for w in store.iter_works()
        for p in store.passages_for_work(w.work_id)
    ]
    assert all(any(w.summary in text for text in passages) for w in report.similar_work)
    assert direct[0].url.startswith("https://doi.org/10.5555/apollo-a")
    assert report.search_queries_used  # what was searched is reported


async def test_nothing_anticipating_after_a_healthy_search_is_novel() -> None:
    report, store = await _check(anticipates=lambda text: False)
    assert report.is_novel is True
    assert 0.5 < report.confidence <= 1.0
    run = store.list_runs()[0]
    assert run.degraded == []
    assert "none reports" in report.assessment
    # the decision model's answers are in the audit trail, two per passage
    assert store.decisions(run.run_id, "novelty")


async def test_the_retracted_work_is_never_evidence() -> None:
    report, _ = await _check(
        anticipates=lambda text: "removed all measurable request latency" in text
    )
    assert (
        report.is_novel is True
    )  # the only anticipating passage is a retracted work's


async def test_a_spent_decision_budget_is_undetermined_not_novel() -> None:
    from ..types import Budget

    report, _ = await _check(
        anticipates=lambda text: False,
        budget=Budget(max_rounds=2, max_decision_calls=24),
    )
    assert report.is_novel is None
    assert "decision budget ran out" in report.assessment


async def test_a_decision_model_failure_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = Asker.decide

    async def failing(self: Asker, step: str, **kwargs):  # type: ignore[no-untyped-def]
        if step == novelty.STEP:
            raise DecisionError("the model is down")
        return await real(self, step, **kwargs)

    monkeypatch.setattr(Asker, "decide", failing)
    report, _ = await _check(anticipates=lambda text: True)
    assert report.is_novel is None
    assert "the model is down" in report.assessment


async def test_an_answer_without_a_probability_is_undetermined_not_novel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = Asker.decide

    async def silent(self: Asker, step: str, **kwargs):  # type: ignore[no-untyped-def]
        if step == novelty.STEP:
            return {
                key: Answer(choice="no", probabilities={}, confidence=0.0, p_true=None)
                for key in kwargs["questions"]
            }
        return await real(self, step, **kwargs)

    monkeypatch.setattr(Asker, "decide", silent)
    report, _ = await _check(anticipates=lambda text: False)
    assert report.is_novel is None
    assert "no probability" in report.assessment


async def test_an_empty_claim_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        await run_novelty_check_v2("  ", model="m", decision_model=world_decisions()[0])


# -- the verdict rule, on hand-built results ---------------------------------------


def _result(*, works: int, degraded: list[str] | None = None, stop: str = "covered"):
    items = [
        Work(work_id=f"w{n}", title=f"Work {n}", doi=f"10.5555/t{n}")
        for n in range(works)
    ]
    passages = [
        Passage.make(work_id=w.work_id, start=0, text=f"Text of {w.title}.")
        for w in items
    ]
    return ResearchResult(
        run_id="r",
        question="q",
        mode="scholarly",
        works=items,
        passages=passages,
        degraded=degraded or [],
        stop_reason=stop,
    )


def _scored(result: ResearchResult, p: float) -> list[novelty._Scored]:
    return [
        novelty._Scored(passage=ps, anticipates=p, related=p) for ps in result.passages
    ]


def test_novel_needs_enough_works_read() -> None:
    result = _result(works=MIN_WORKS_ASSESSED - 1)
    report = novelty._verdict(
        "c", result=result, scored=_scored(result, 0.1), cut_short=False, queries=[]
    )
    assert report.is_novel is None
    assert "at least" in report.assessment


def test_novel_needs_a_search_without_failures() -> None:
    result = _result(works=4, degraded=["openalex: http 500"])
    report = novelty._verdict(
        "c", result=result, scored=_scored(result, 0.1), cut_short=False, queries=[]
    )
    assert report.is_novel is None
    assert "degraded" in report.assessment


def test_novel_needs_a_search_that_did_not_hit_its_budget() -> None:
    result = _result(works=4, stop="budget:max_rounds")
    report = novelty._verdict(
        "c", result=result, scored=_scored(result, 0.1), cut_short=False, queries=[]
    )
    assert report.is_novel is None


def test_prior_work_stands_even_from_a_degraded_search() -> None:
    result = _result(works=4, degraded=["openalex: http 500"])
    report = novelty._verdict(
        "c", result=result, scored=_scored(result, 0.9), cut_short=True, queries=[]
    )
    assert report.is_novel is False
    assert report.confidence == pytest.approx(0.9)
    assert report.similar_work[0].relevance.value == "direct"


def test_a_healthy_search_with_nothing_found_is_novel_with_the_complement_confidence() -> (
    None
):
    result = _result(works=4)
    report = novelty._verdict(
        "c", result=result, scored=_scored(result, 0.2), cut_short=False, queries=["q"]
    )
    assert report.is_novel is True
    assert report.confidence == pytest.approx(0.8)
    assert report.search_queries_used == ["q"]
