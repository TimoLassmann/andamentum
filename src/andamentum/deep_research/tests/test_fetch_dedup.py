"""Plumbing tests for FetchPhase URL deduplication.

When several search queries return the same URL, the page must only
appear in the fetcher's selection list once — and never be re-fetched
across iterations. Regression observed on the original 'Kalign (MSA)
competitors' run, where ``PMC1538774`` appeared 3 times in the final
Sources list because the same URL surfaced from multiple queries and
no dedup happened before/after the fetcher agent picked link IDs.

No LLM, no network. Stubs the page_fetcher agent and the search
backend; drives FetchPhase directly to assert the deduped state.
"""

from __future__ import annotations

from pydantic_graph import GraphRunContext

from andamentum.deep_research.models import (
    FetchedPage,
    FetchPlan,
    SearchResult,
)
from andamentum.deep_research.nodes import FetchPhase, NodeDeps, SummarizePages
from andamentum.deep_research.state import ResearchState


# ── Stubs ──────────────────────────────────────────────────────────────


class _StubResult:
    def __init__(self, output):
        self.output = output


class StubAgent:
    def __init__(self, output):
        self._output = output
        self.last_prompt: str | None = None

    async def run(self, prompt, **_kwargs):
        self.last_prompt = prompt
        return _StubResult(self._output)


class FakeBackend:
    """Records every URL fetched; returns a dummy FetchedPage per call."""

    def __init__(self):
        self.fetched: list[str] = []

    async def search(self, query: str, max_results: int = 10):
        raise AssertionError("backend.search must not be called here")

    async def fetch_page(self, url: str):
        self.fetched.append(url)
        return FetchedPage(
            url=url,
            title=f"Title for {url}",
            content="...",
            word_count=10,
            relevance_score=0.5,
            is_relevant=True,
        )


def _result(link_id: int, url: str) -> SearchResult:
    return SearchResult(
        link_id=link_id,
        title=f"Title for {url}",
        url=url,
        snippet="...",
        domain="example.test",
        relevance_score=0.5,
    )


def _make_ctx(*, fetcher_output: FetchPlan):
    state = ResearchState(query="test query")
    deps = NodeDeps(
        backend=FakeBackend(),  # type: ignore[arg-type]
        model="stub",
        correlation_id="test",
        max_pages_to_fetch=5,
        agent_overrides={"page_fetcher": StubAgent(fetcher_output)},
    )
    return GraphRunContext(state=state, deps=deps)


# ── Tests ──────────────────────────────────────────────────────────────


async def test_within_cycle_duplicate_urls_collapse_to_one_pick():
    """Three queries returning the same URL should produce one link_id.

    The fetcher agent should never see the same URL under several
    link_ids — that's how the original Kalign run picked PMC1538774
    three times.
    """
    ctx = _make_ctx(fetcher_output=FetchPlan(link_ids=[0, 1], reasoning="r"))
    # Three queries each returned the same URL plus one unique URL.
    ctx.state.all_results = {
        "q1": [
            _result(link_id=0, url="https://shared.test/page"),
            _result(link_id=1, url="https://q1-only.test/"),
        ],
        "q2": [
            _result(link_id=0, url="https://shared.test/page"),
            _result(link_id=2, url="https://q2-only.test/"),
        ],
        "q3": [
            _result(link_id=0, url="https://shared.test/page"),
        ],
    }

    await FetchPhase().run(ctx)

    # The fetcher agent's prompt should list only the 3 unique URLs,
    # each under one link_id.
    overrides = ctx.deps.agent_overrides
    assert overrides is not None
    fetcher_stub = overrides["page_fetcher"]
    prompt = fetcher_stub.last_prompt
    assert prompt.count("shared.test/page") == 1, (
        "Duplicate URL surfaced under multiple link_ids in the fetcher "
        f"prompt:\n{prompt}"
    )
    # url_map should have exactly 3 unique URLs (one per).
    assert len(set(ctx.state.url_map.values())) == len(ctx.state.url_map)
    assert "https://shared.test/page" in ctx.state.url_map.values()
    assert len(ctx.state.url_map) == 3


async def test_already_fetched_urls_excluded_in_next_cycle():
    """URLs in state.fetched_urls (from prior cycles) must be hard-excluded
    from the new cycle's fetcher prompt — not just hinted at via text."""
    # Stub plan; if FetchPhase reaches the agent at all, it'll pick lid 99
    # (which won't be in url_map) and fetch nothing.
    ctx = _make_ctx(fetcher_output=FetchPlan(link_ids=[99], reasoning="r"))
    ctx.state.fetched_urls = {"https://oldcycle.test/done"}
    ctx.state.all_results = {
        "q1": [
            _result(link_id=0, url="https://oldcycle.test/done"),
            _result(link_id=1, url="https://newcycle.test/fresh"),
        ],
    }

    await FetchPhase().run(ctx)

    overrides = ctx.deps.agent_overrides
    assert overrides is not None
    fetcher_stub = overrides["page_fetcher"]
    prompt = fetcher_stub.last_prompt
    assert prompt is not None
    # The old URL must not appear in the "Search Results to Evaluate"
    # block. (It MAY still appear in the "Already Fetched: …" hint line —
    # that's a separate text channel for the agent's awareness.)
    eval_block = prompt.split("Search Results to Evaluate")[1].split("Already Fetched")[
        0
    ]
    assert "oldcycle.test" not in eval_block, (
        f"Already-fetched URL leaked into the fetcher's evaluate list:\n{eval_block}"
    )
    assert "newcycle.test" in eval_block


async def test_agent_picking_same_link_id_twice_only_fetches_once():
    """Defence-in-depth: even if the fetcher agent picks the same link_id
    twice, do_fetch must only run once for that URL."""
    ctx = _make_ctx(
        # Agent picks link_id 0 four times — should fetch once.
        fetcher_output=FetchPlan(link_ids=[0, 0, 0, 0], reasoning="r"),
    )
    ctx.state.all_results = {
        "q1": [_result(link_id=0, url="https://once.test/")],
    }

    await FetchPhase().run(ctx)
    backend = ctx.deps.backend
    assert isinstance(backend, FakeBackend)
    assert backend.fetched == ["https://once.test/"]
    assert len(ctx.state.fetched_pages) == 1


async def test_previously_failed_urls_excluded_in_next_cycle():
    """URLs in ``state.fetch_errors`` (from prior cycles) must be hard-
    excluded from the next cycle's candidate list.

    Regression: in a multi-cycle research run, OECD pages 403'd in cycle
    1 then kept getting re-picked by the fetcher agent in cycles 2+
    because we tracked successful fetches in ``state.fetched_urls`` but
    never excluded URLs we had already failed on.
    """
    ctx = _make_ctx(fetcher_output=FetchPlan(link_ids=[99], reasoning="r"))
    ctx.state.fetch_errors = [
        {
            "url": "https://failed.test/blocked",
            "error": "Client error '403 Forbidden'",
            "is_retryable": "True",
            "link_id": "0",
        },
    ]
    ctx.state.all_results = {
        "q1": [
            _result(link_id=0, url="https://failed.test/blocked"),
            _result(link_id=1, url="https://fresh.test/"),
        ],
    }

    await FetchPhase().run(ctx)

    overrides = ctx.deps.agent_overrides
    assert overrides is not None
    fetcher_stub = overrides["page_fetcher"]
    prompt = fetcher_stub.last_prompt
    assert prompt is not None
    eval_block = prompt.split("Search Results to Evaluate")[1].split("Already Fetched")[
        0
    ]
    assert "failed.test/blocked" not in eval_block, (
        f"Previously-failed URL leaked back into the fetcher's evaluate "
        f"list:\n{eval_block}"
    )
    assert "fresh.test" in eval_block
    # url_map should not contain the failed URL either.
    assert "https://failed.test/blocked" not in ctx.state.url_map.values()


async def test_summarize_pages_returns_when_all_urls_already_fetched():
    """If every search-result URL was already fetched in prior cycles,
    FetchPhase must skip the fetcher LLM call entirely and short-circuit
    to SummarizePages."""
    # Stub plan; if FetchPhase reaches the agent at all, it'll pick lid 99
    # (which won't be in url_map) and fetch nothing.
    ctx = _make_ctx(fetcher_output=FetchPlan(link_ids=[99], reasoning="r"))
    ctx.state.fetched_urls = {
        "https://a.test/",
        "https://b.test/",
    }
    ctx.state.all_results = {
        "q1": [
            _result(link_id=0, url="https://a.test/"),
            _result(link_id=1, url="https://b.test/"),
        ],
    }

    nxt = await FetchPhase().run(ctx)
    assert isinstance(nxt, SummarizePages)
    overrides = ctx.deps.agent_overrides
    assert overrides is not None
    fetcher_stub = overrides["page_fetcher"]
    # The fetcher agent must NOT have been called — there was nothing
    # left to evaluate after the already-fetched filter.
    assert fetcher_stub.last_prompt is None
