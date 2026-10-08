"""The novelty check on deep_research: ``run_novelty_searches``.

``run_novelty_check`` is stubbed at the package boundary; its own behaviour
is tested in ``deep_research/tests/test_novelty.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from andamentum.deep_research import NoveltyResearch, ResearchStore
from andamentum.deep_research.novelty import (
    NoveltyReport,
    Relevance,
    SimilarWork,
)
from andamentum.whetstone.v3.graph import (
    JudgeNovelty,
    RunNoveltySearches,
    V3Deps,
    V3State,
    run_review_v3,
)
from andamentum.whetstone.v3.novelty import (
    NoveltyTarget,
    judge_novelty,
    run_novelty_searches,
)


def _target(n: int) -> NoveltyTarget:
    return NoveltyTarget(
        claim_text=f"claim {n}",
        short_summary=f"claim {n} summary",
        why_load_bearing="why",
    )


@dataclass
class _Ctx:
    state: V3State
    deps: V3Deps


def _setup(**kw) -> NoveltyResearch:
    return NoveltyResearch(**kw)


async def test_the_research_path_runs_every_target_in_order_over_one_store() -> None:
    seen: list[tuple[str, object, object]] = []

    async def stub(claim, *, model, store, sources, budget, **_kw):
        seen.append((claim, store, budget.max_rounds))
        assert model == "stub"  # the review's own model judges
        return NoveltyReport(
            claim=claim,
            is_novel=False,
            confidence=0.9,
            assessment="found",
            similar_work=[
                SimilarWork(
                    title="Prior",
                    url="https://doi.org/10.5555/x",
                    relevance=Relevance.DIRECT,
                    summary="the passage's own words",
                )
            ],
            sources=["https://doi.org/10.5555/x"],
            search_queries_used=["q"],
        )

    setup = _setup()
    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        evidence = await run_novelty_searches(
            [_target(1), _target(2)], agent_model="stub", search_depth=3, research=setup
        )
    assert [e.target.claim_text for e in evidence] == ["claim 1", "claim 2"]
    assert [s[0] for s in seen] == ["claim 1 summary", "claim 2 summary"]
    assert isinstance(seen[0][1], ResearchStore) and seen[0][1] is seen[1][1]
    assert [s[2] for s in seen] == [3, 3]  # the search depth is the round count
    assert evidence[0].is_novel is False
    assert evidence[0].similar_work[0].relevance == "direct"
    assert evidence[0].similar_work[0].summary == "the passage's own words"


async def test_an_undetermined_report_gives_no_verdict_and_no_finding() -> None:
    async def stub(claim, **_kw):
        return NoveltyReport(
            claim=claim,
            is_novel=None,
            confidence=0.0,
            assessment="The search was degraded. Novelty is UNDETERMINED.",
        )

    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        evidence = await run_novelty_searches(
            [_target(1)], agent_model="stub", research=_setup()
        )
    assert evidence[0].error is not None and "UNDETERMINED" in evidence[0].error
    assert judge_novelty(evidence) == []


async def test_a_crash_on_one_target_does_not_stop_the_others() -> None:
    calls = [0]

    async def stub(claim, **_kw):
        calls[0] += 1
        if calls[0] == 1:
            raise RuntimeError("boom")
        return NoveltyReport(
            claim=claim, is_novel=True, confidence=0.8, assessment="none"
        )

    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        evidence = await run_novelty_searches(
            [_target(1), _target(2)], agent_model="stub", research=_setup()
        )
    assert evidence[0].error == "boom"
    assert evidence[1].is_novel is True and evidence[1].error is None


async def test_a_store_the_caller_opened_is_left_open() -> None:
    store = ResearchStore.in_memory()

    async def stub(claim, *, store, **_kw):
        return NoveltyReport(claim=claim, is_novel=True, confidence=0.8, assessment="")

    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        await run_novelty_searches(
            [_target(1)], agent_model="stub", research=_setup(store=store)
        )
    assert store.list_runs() == []  # still usable
    store.close()


async def test_a_store_path_is_opened_and_kept(tmp_path) -> None:
    path = tmp_path / "novelty" / "research.db"

    async def stub(claim, *, store, **_kw):
        assert isinstance(store, ResearchStore)
        return NoveltyReport(claim=claim, is_novel=True, confidence=0.8, assessment="")

    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        await run_novelty_searches(
            [_target(1)], agent_model="stub", research=_setup(store=path)
        )
    assert path.exists()


async def test_without_a_setup_the_defaults_run_in_memory() -> None:
    seen: list[object] = []

    async def stub(claim, *, store, sources, budget, **_kw):
        seen.append((store, sources, budget.max_rounds))
        return NoveltyReport(claim=claim, is_novel=True, confidence=0.8, assessment="")

    with patch("andamentum.deep_research.run_novelty_check", new=stub):
        evidence = await run_novelty_searches(
            [_target(1)], agent_model="stub", search_depth=2
        )
    assert evidence[0].is_novel is True
    store, sources, rounds = seen[0]  # type: ignore[misc]
    assert isinstance(store, ResearchStore)
    assert sources == NoveltyResearch().sources and rounds == 2


async def test_the_node_hands_the_setup_to_the_search() -> None:
    setup = _setup()
    state = V3State(source="...", novelty_targets=[_target(1)])
    deps = V3Deps(agent_model="stub", check_novelty=True, novelty_research=setup)
    with patch(
        "andamentum.whetstone.v3.graph.run_novelty_searches",
        new=AsyncMock(return_value=[]),
    ) as search:
        result = await RunNoveltySearches().run(_Ctx(state, deps))  # type: ignore[arg-type]
    assert isinstance(result, JudgeNovelty)
    assert search.call_args.kwargs["research"] is setup


async def test_the_search_asks_nothing_of_a_decision_model() -> None:
    """The check needs the review's model alone: no decision model anywhere."""
    import inspect

    assert "decision_model" not in inspect.signature(run_novelty_searches).parameters
    assert not hasattr(NoveltyResearch(), "decision_model")


async def test_a_setup_without_check_novelty_is_refused() -> None:
    with pytest.raises(ValueError, match="check_novelty"):
        await run_review_v3("# T\n\ntext", model="stub", novelty_research=_setup())


async def test_a_target_that_could_not_be_assessed_is_listed_in_the_result() -> None:
    """Undetermined and crashed targets are named in ``failed_criteria`` and
    produce no finding, so "not assessed" is distinguishable from "novel"."""
    from andamentum.whetstone.v3.novelty import NoveltyEvidence

    state = V3State(
        source="...",
        novelty_evidence=[
            NoveltyEvidence(
                target=_target(1), error="novelty undetermined: search degraded"
            ),
            NoveltyEvidence(target=_target(2), error="boom"),
            NoveltyEvidence(target=_target(3), is_novel=True, confidence=0.8),
        ],
    )
    deps = V3Deps(agent_model="stub", check_novelty=True)
    await JudgeNovelty().run(_Ctx(state, deps))  # type: ignore[arg-type]
    assert state.findings == []
    assert len(state.failed_criteria) == 2
    assert "claim 1 summary" in state.failed_criteria[0]
    assert "undetermined" in state.failed_criteria[0]
    assert "claim 2 summary" in state.failed_criteria[1]
    assert "boom" in state.failed_criteria[1]
