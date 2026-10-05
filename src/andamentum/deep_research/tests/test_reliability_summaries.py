"""R1 reliability: a failed summary is a recorded failure, never evidence.

Stubbed agents; no model, no network.
"""

from __future__ import annotations

from pydantic_graph import GraphRunContext

from andamentum.deep_research.models import (
    EvidenceReport,
    FetchedPage,
    PageSummary,
    SummaryError,
)
from andamentum.deep_research.nodes import (
    AnalyzeGaps,
    NodeDeps,
    SummarizePages,
    Synthesize,
)
from andamentum.deep_research.state import ResearchState
from andamentum.deep_research.summarize_pages import summarize_pages


class _Result:
    def __init__(self, output: object) -> None:
        self.output = output


class _ScriptedAgent:
    """Answers from a url -> PageSummary or Exception map; records the urls asked."""

    def __init__(self, script: dict[str, PageSummary | Exception]) -> None:
        self._script = script
        self.asked: list[str] = []

    async def run(self, prompt: str, **_kwargs: object) -> _Result:
        url = next(u for u in self._script if u in prompt)
        self.asked.append(url)
        item = self._script[url]
        if isinstance(item, Exception):
            raise item
        return _Result(item.model_copy())


class _NoBackend:
    async def search(self, query: str, max_results: int = 10) -> list:
        raise AssertionError("not used")

    async def fetch_page(self, url: str) -> FetchedPage:
        raise AssertionError("not used")


def _page(url: str) -> FetchedPage:
    return FetchedPage(
        url=url,
        title=f"T {url}",
        content=f"body of {url}",
        word_count=4,
        relevance_score=0.5,
        is_relevant=True,
    )


def _summary(url: str, relevance: float) -> PageSummary:
    return PageSummary(
        url=url, title="t", summary="s", key_points=["k"], relevance_score=relevance
    )


async def test_failed_summary_is_an_error_not_a_placeholder_summary() -> None:
    agent = _ScriptedAgent(
        {
            "https://ok.test/": _summary("https://ok.test/", 0.7),
            "https://bad.test/": RuntimeError("model timeout"),
        }
    )
    outcome = await summarize_pages(
        [_page("https://ok.test/"), _page("https://bad.test/")],
        goal="g",
        model=object(),
        overrides={"page_summarizer": agent},
    )
    assert [s.url for s in outcome.summaries] == ["https://ok.test/"]
    assert [e.url for e in outcome.errors] == ["https://bad.test/"]
    assert "model timeout" in outcome.errors[0].error
    for s in outcome.summaries:
        assert "Failed to summarize" not in s.summary
        assert "Summarization failed" not in s.key_points


def _ctx(
    script: dict[str, PageSummary | Exception],
) -> tuple[GraphRunContext[ResearchState, NodeDeps], _ScriptedAgent]:
    agent = _ScriptedAgent(script)
    deps = NodeDeps(
        backend=_NoBackend(),  # type: ignore[arg-type]
        model=object(),
        agent_overrides={"page_summarizer": agent},
    )
    return GraphRunContext(state=ResearchState(query="q"), deps=deps), agent


async def test_node_records_failures_in_state_and_keeps_good_summaries() -> None:
    ctx, _ = _ctx(
        {
            "https://ok.test/": _summary("https://ok.test/", 0.7),
            "https://bad.test/": RuntimeError("boom"),
        }
    )
    ctx.state.fetched_pages = [_page("https://ok.test/"), _page("https://bad.test/")]
    assert isinstance(await SummarizePages().run(ctx), AnalyzeGaps)
    assert [s.url for s in ctx.state.page_summaries] == ["https://ok.test/"]
    assert [e.url for e in ctx.state.summary_errors] == ["https://bad.test/"]


async def test_node_summarises_only_new_pages_and_keeps_ranking() -> None:
    script: dict[str, PageSummary | Exception] = {
        "https://a.test/": _summary("https://a.test/", 0.4),
        "https://b.test/": RuntimeError("boom"),
        "https://c.test/": _summary("https://c.test/", 0.9),
    }
    ctx, agent = _ctx(script)
    ctx.state.fetched_pages = [_page("https://a.test/"), _page("https://b.test/")]
    await SummarizePages().run(ctx)
    assert sorted(agent.asked) == ["https://a.test/", "https://b.test/"]

    # Next cycle: one new page. Neither the summarised nor the failed page is
    # sent again.
    agent.asked.clear()
    ctx.state.fetched_pages.append(_page("https://c.test/"))
    await SummarizePages().run(ctx)
    assert agent.asked == ["https://c.test/"]
    assert [s.url for s in ctx.state.page_summaries] == [
        "https://c.test/",
        "https://a.test/",
    ]
    assert [e.url for e in ctx.state.summary_errors] == ["https://b.test/"]

    # Nothing new: no agent call at all.
    agent.asked.clear()
    await SummarizePages().run(ctx)
    assert agent.asked == []


async def test_synthesize_is_degraded_when_most_summaries_failed() -> None:
    report = EvidenceReport(
        evidence_summary="partial",
        key_findings=["f"],
        sources=["https://ok.test/"],
        total_searches_performed=0,
        total_pages_fetched=0,
        iterations_required=0,
    )

    class _Lead:
        async def run(self, prompt: str, **_kwargs: object) -> _Result:
            return _Result(report)

    deps = NodeDeps(
        backend=_NoBackend(),  # type: ignore[arg-type]
        model=object(),
        agent_overrides={"lead_agent": _Lead()},
    )
    state = ResearchState(query="q")
    state.page_summaries = [_summary("https://ok.test/", 0.8)]
    state.summary_errors = [
        SummaryError(url=f"https://bad{i}.test/", error="boom") for i in range(3)
    ]
    end = await Synthesize().run(GraphRunContext(state=state, deps=deps))
    assert end.data.degraded is True
    assert "3/4 page summaries failed" in end.data.degraded_reason


async def test_synthesize_with_only_failed_summaries_says_so() -> None:
    deps = NodeDeps(backend=_NoBackend(), model=object())  # type: ignore[arg-type]
    state = ResearchState(query="q")
    state.total_pages_fetched = 2
    state.summary_errors = [
        SummaryError(url=f"https://bad{i}.test/", error="boom") for i in range(2)
    ]
    end = await Synthesize().run(GraphRunContext(state=state, deps=deps))
    assert "no page could be summarised" in end.data.evidence_summary
    assert "summary failures: 2" in end.data.evidence_summary
    assert end.data.degraded is True
