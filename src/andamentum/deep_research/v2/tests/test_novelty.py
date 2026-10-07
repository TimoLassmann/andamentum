"""The v2 novelty check, offline: the verdict comes from judged findings, never a guess.

``research`` is replaced by a fake returning a hand-built result, so no engine,
network or language model is involved; the decision model is a stub server.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.core.decisions import DecisionModel
from andamentum.core.decisions.errors import DecisionError
from andamentum.references import Work

from .. import novelty
from ..ask import Asker
from ..novelty import (
    ANTICIPATES_AT,
    MIN_WORKS_ASSESSED,
    run_novelty_check_v2,
)
from ..store import ResearchStore
from ..types import Budget, Finding, Passage, ResearchResult
from . import llm_stub

CLAIM = "The Apollo cache cuts median request latency by half."
ANTICIPATING = "halves median request latency"


def _work(n: int, *, retracted: str = "no") -> Work:
    return Work(
        work_id=f"w{n}",
        title=f"Work {n}",
        doi=f"10.5555/t{n}",
        year=2020 + n,
        retracted=retracted,  # type: ignore[arg-type]
    )


def _passage(work: Work) -> Passage:
    return Passage.make(work_id=work.work_id, start=0, text=f"Text of {work.title}.")


def _finding(
    n: int, work: Work, claim: str, *, verified: bool = True, quote: str | None = None
) -> Finding:
    return Finding(
        finding_id=f"F{n}",
        subquestion_id="s1",
        claim=claim,
        quote=quote or f"Quote of {work.title}: {claim}",
        quote_verified=verified,
        passage_id=f"p{n}",
        work_id=work.work_id,
    )


def _result(
    *,
    works: int = 4,
    claims: dict[int, str] | None = None,
    degraded: list[str] | None = None,
    stop: str = "answered",
    retracted: set[int] | None = None,
    unverified: set[int] | None = None,
) -> ResearchResult:
    """*works* works, each with a passage; ``claims[n]`` is work n's finding."""
    retracted = retracted or set()
    unverified = unverified or set()
    items = [
        _work(n, retracted="retracted" if n in retracted else "no")
        for n in range(works)
    ]
    # a retracted work is left out of ``works`` by the engine, listed in ``excluded``
    kept = [w for n, w in enumerate(items) if n not in retracted]
    findings = [
        _finding(n, items[n], claim, verified=n not in unverified)
        for n, claim in (claims or {}).items()
    ]
    return ResearchResult(
        run_id="r",
        question="q",
        mode="scholarly",
        works=kept,
        excluded={items[n].work_id: "retracted" for n in retracted},
        passages=[_passage(w) for w in items],
        findings=findings,
        queries=["apollo cache latency"],
        degraded=degraded or [],
        stop_reason=stop,
    )


def _scored(
    result: ResearchResult, anticipates: float, related: float | None = None
) -> list[novelty._Scored]:
    return [
        novelty._Scored(
            finding=f,
            anticipates=anticipates,
            related=anticipates if related is None else related,
        )
        for f in result.findings
    ]


def _verdict(
    result: ResearchResult, scored: list[novelty._Scored], *, cut_short: bool = False
):
    return novelty._verdict(
        CLAIM, result=result, scored=scored, cut_short=cut_short, queries=["q"]
    )


# -- the verdict rule, on hand-built results ---------------------------------------


def test_an_anticipating_finding_makes_the_claim_not_novel() -> None:
    result = _result(claims={0: "a", 1: "b"})
    scored = [
        novelty._Scored(finding=result.findings[0], anticipates=0.6, related=0.7),
        novelty._Scored(finding=result.findings[1], anticipates=0.2, related=0.3),
    ]
    report = _verdict(result, scored)
    assert report.is_novel is False
    assert report.confidence == pytest.approx(0.6)
    assert [w.title for w in report.similar_work] == ["Work 0"]
    assert report.similar_work[0].relevance.value == "partial"
    assert report.similar_work[0].summary == result.findings[0].quote
    assert report.search_queries_used == ["q"]


def test_a_direct_finding_is_direct_and_one_entry_per_work() -> None:
    result = _result(claims={0: "a", 1: "b"})
    # both findings come from work 0
    second = result.findings[1].model_copy(update={"work_id": "w0"})
    result = result.model_copy(update={"findings": [result.findings[0], second]})
    scored = [
        novelty._Scored(finding=result.findings[0], anticipates=0.85, related=0.9),
        novelty._Scored(finding=second, anticipates=0.95, related=0.9),
    ]
    report = _verdict(result, scored)
    assert report.is_novel is False
    assert report.confidence == pytest.approx(0.95)
    assert len(report.similar_work) == 1
    assert report.similar_work[0].relevance.value == "direct"
    assert report.similar_work[0].summary == second.quote
    assert report.similar_work[0].url == "https://doi.org/10.5555/t0"


def test_prior_work_stands_even_from_a_degraded_search() -> None:
    result = _result(claims={0: "a"}, degraded=["openalex: http 500"])
    report = _verdict(result, _scored(result, 0.9), cut_short=True)
    assert report.is_novel is False
    assert report.similar_work[0].relevance.value == "direct"


def test_a_healthy_search_with_nothing_anticipating_is_novel() -> None:
    result = _result(claims={0: "a", 1: "b"})
    report = _verdict(result, _scored(result, 0.2))
    assert report.is_novel is True
    assert report.confidence == pytest.approx(0.8)
    assert "none reports" in report.assessment


def test_a_healthy_search_with_no_findings_is_novel_and_says_so() -> None:
    result = _result(works=4, claims={})
    report = _verdict(result, [])
    assert report.is_novel is True
    assert report.confidence == pytest.approx(1.0)
    assert "no finding bore on the claim" in report.assessment


def test_novel_needs_enough_works_read() -> None:
    result = _result(works=MIN_WORKS_ASSESSED - 1, claims={0: "a"})
    report = _verdict(result, _scored(result, 0.1))
    assert report.is_novel is None
    assert "at least" in report.assessment
    assert "UNDETERMINED" in report.assessment


def test_novel_needs_a_search_without_failures() -> None:
    result = _result(claims={0: "a"}, degraded=["openalex: http 500"])
    report = _verdict(result, _scored(result, 0.1))
    assert report.is_novel is None
    assert "degraded" in report.assessment


def test_novel_needs_a_search_that_did_not_hit_its_budget() -> None:
    result = _result(claims={0: "a"}, stop="budget:max_rounds")
    report = _verdict(result, _scored(result, 0.1))
    assert report.is_novel is None
    assert "budget" in report.assessment


def test_novel_needs_every_finding_judged() -> None:
    result = _result(claims={0: "a"})
    report = _verdict(result, _scored(result, 0.1), cut_short=True)
    assert report.is_novel is None
    assert "decision budget ran out" in report.assessment


def test_works_read_counts_distinct_works_with_a_passage_not_retracted() -> None:
    # 4 works but one is retracted: 3 are read, which is enough
    result = _result(works=4, claims={0: "a"}, retracted={3})
    assert _verdict(result, _scored(result, 0.1)).is_novel is True
    # 3 works, one retracted: 2 read, not enough
    result = _result(works=3, claims={0: "a"}, retracted={2})
    assert _verdict(result, _scored(result, 0.1)).is_novel is None


def test_unverified_quotes_are_mentioned() -> None:
    result = _result(claims={0: "a", 1: "b"}, unverified={1})
    report = _verdict(result, _scored(result, 0.1))
    assert report.is_novel is True
    assert "1 of the 2 judged finding(s)" in report.assessment
    anticipated = _verdict(result, _scored(result, 0.9))
    assert "1 of the 2 judged finding(s)" in anticipated.assessment


# -- scoring with a stub decision model ---------------------------------------------


def _by_finding_text(state: object, instructions: str) -> float:
    assert isinstance(state, dict)
    anticipates = instructions.startswith("The finding reports the same")
    if anticipates:
        return 0.9 if ANTICIPATING in str(state["finding"]) else 0.1
    return 0.7 if "latency" in str(state["finding"]) else 0.1


async def test_scoring_judges_each_finding_by_its_text_one_request_each() -> None:
    result = _result(
        claims={
            0: f"The Apollo cache {ANTICIPATING}.",
            1: "SaltLog rotates its files daily.",
        }
    )
    handle, stub = llm_stub.decisions(_by_finding_text)
    store = ResearchStore.in_memory()
    asker = llm_stub.make_asker(decision_model=handle, store=store)
    scored, cut_short = await novelty._score(asker, CLAIM, result)
    assert cut_short is False
    assert len(stub.calls) == 2  # one request per finding, both questions in it
    assert all(set(q) == {"anticipates", "related"} for _, q in stub.calls)
    assert [round(s.anticipates, 1) for s in scored] == [0.9, 0.1]
    assert [round(s.related, 1) for s in scored] == [0.7, 0.1]
    # the state names the claim under review, the finding, its quote, title and year
    first = stub.calls[0][0]
    assert isinstance(first, dict)
    assert first["claim_under_review"] == CLAIM
    assert first["quote"] == result.findings[0].quote
    assert first["work"] == "Work 0"
    assert first["year"] == 2020
    # logged under the step 'novelty'
    assert store.decisions(asker.run_id, novelty.STEP)


async def test_retracted_works_are_never_evidence() -> None:
    result = _result(
        claims={0: f"Prior: {ANTICIPATING}.", 1: "Unrelated."}, retracted={0}
    )
    # a finding of the retracted work is still in the table (hand-built)
    handle, stub = llm_stub.decisions(_by_finding_text)
    asker = llm_stub.make_asker(decision_model=handle)
    scored, _ = await novelty._score(asker, CLAIM, result)
    assert [s.finding.work_id for s in scored] == ["w1"]
    assert len(stub.calls) == 1
    report = _verdict(result, scored)
    assert report.is_novel is True


async def test_a_work_marked_retracted_but_left_in_works_is_not_evidence() -> None:
    result = _result(claims={0: f"Prior: {ANTICIPATING}."})
    retracted = result.works[0].model_copy(update={"retracted": "retracted"})
    result = result.model_copy(update={"works": [retracted, *result.works[1:]]})
    handle, _ = llm_stub.decisions(_by_finding_text)
    asker = llm_stub.make_asker(decision_model=handle)
    scored, _ = await novelty._score(asker, CLAIM, result)
    assert scored == []


async def test_a_spent_decision_budget_cuts_scoring_short() -> None:
    result = _result(claims={0: "a", 1: "b", 2: "c"})
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    asker = llm_stub.make_asker(
        decision_model=handle, budget=Budget(max_decision_calls=2)
    )
    scored, cut_short = await novelty._score(asker, CLAIM, result)
    assert cut_short is True
    assert len(scored) < 3
    assert _verdict(result, scored, cut_short=True).is_novel is None


# -- the whole check, research replaced -----------------------------------------------


class FakeResearch:
    def __init__(self, result: ResearchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, question: str, **kwargs: Any) -> ResearchResult:
        self.calls.append((question, kwargs))
        store = kwargs["store"]
        assert isinstance(store, ResearchStore)
        run_id = store.start_run(
            question=question, mode="scholarly", config={}, models={}
        )
        return self.result.model_copy(update={"run_id": run_id})


def _patch(monkeypatch: pytest.MonkeyPatch, result: ResearchResult) -> FakeResearch:
    fake = FakeResearch(result)
    monkeypatch.setattr(novelty, "research", fake)
    return fake


async def test_the_check_calls_research_in_scholarly_mode_without_a_decision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch(monkeypatch, _result(claims={0: f"Prior: {ANTICIPATING}.", 1: "x"}))
    handle, stub = llm_stub.decisions(_by_finding_text)
    budget = Budget(max_rounds=5)
    report = await run_novelty_check_v2(
        CLAIM, model="test", decision_model=handle, budget=budget
    )
    assert len(fake.calls) == 1
    question, kwargs = fake.calls[0]
    assert CLAIM in question
    assert kwargs["mode"] == "scholarly"
    assert "decision_model" not in kwargs
    assert kwargs["budget"].max_rounds == 5
    assert report.is_novel is False
    assert report.search_queries_used == ["apollo cache latency"]
    assert len(stub.calls) == 2  # one request per finding


async def test_nothing_anticipating_after_a_healthy_search_is_novel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(claims={0: "SaltLog rotates files.", 1: "More."}))
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    report = await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)
    assert report.is_novel is True
    assert report.confidence == pytest.approx(0.9)
    assert ANTICIPATES_AT > 0.1


async def test_a_degraded_search_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(claims={0: "a"}, degraded=["openalex: http 500"]))
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    report = await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)
    assert report.is_novel is None


async def test_a_decision_model_failure_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(claims={0: "a"}))
    real = Asker.decide

    async def failing(self: Asker, step: str, **kwargs: Any):
        if step == novelty.STEP:
            raise DecisionError("the model is down")
        return await real(self, step, **kwargs)

    monkeypatch.setattr(Asker, "decide", failing)
    handle, _ = llm_stub.decisions(lambda s, i: 0.9)
    report = await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)
    assert report.is_novel is None
    assert "the model is down" in report.assessment


async def test_a_language_model_failure_in_the_search_is_undetermined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(question: str, **kwargs: Any) -> ResearchResult:
        raise DecisionError("no model")

    monkeypatch.setattr(novelty, "research", failing)
    handle, _ = llm_stub.decisions(lambda s, i: 0.9)
    report = await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)
    assert report.is_novel is None


async def test_another_error_in_the_search_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing(question: str, **kwargs: Any) -> ResearchResult:
        raise RuntimeError("bug")

    monkeypatch.setattr(novelty, "research", failing)
    handle, _ = llm_stub.decisions(lambda s, i: 0.9)
    with pytest.raises(RuntimeError, match="bug"):
        await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)


async def test_a_decision_model_given_by_name_is_opened_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _result(claims={0: "a"}))
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    closed: list[bool] = []
    real_close = handle.aclose

    async def close() -> None:
        closed.append(True)
        await real_close()

    opened: list[str] = []

    async def fake_open(name: str, **_kw: Any) -> DecisionModel:
        opened.append(name)
        return handle

    monkeypatch.setattr(handle, "aclose", close)
    monkeypatch.setattr(DecisionModel, "open", staticmethod(fake_open))
    await run_novelty_check_v2(CLAIM, model="test", decision_model="stub:1b")
    assert opened == ["stub:1b"]
    assert closed == [True]


async def test_a_handle_is_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _result(claims={0: "a"}))
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    closed: list[bool] = []

    async def close() -> None:
        closed.append(True)

    monkeypatch.setattr(handle, "aclose", close)
    await run_novelty_check_v2(CLAIM, model="test", decision_model=handle)
    assert closed == []


async def test_an_open_store_is_left_open(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _result(claims={0: "a"}))
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    store = ResearchStore.in_memory()
    await run_novelty_check_v2(CLAIM, model="test", decision_model=handle, store=store)
    run_id = store.list_runs()[0].run_id
    assert store.decisions(run_id, novelty.STEP)  # usable, with the audit trail


async def test_an_empty_claim_is_refused() -> None:
    handle, _ = llm_stub.decisions(lambda s, i: 0.1)
    with pytest.raises(ValueError, match="empty"):
        await run_novelty_check_v2("  ", model="m", decision_model=handle)
