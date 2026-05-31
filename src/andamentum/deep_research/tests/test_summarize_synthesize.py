"""Plumbing tests for the post-filter-removal SummarizePages + Synthesize.

Covers the regression that motivated the redesign — a "Kalign competitors"
style query where every page summary scored below 0.3:

- SummarizePages must keep ALL summaries, sorted by relevance descending
  (no silent <0.3 filter).
- Synthesize must NOT bail with "incomplete — no content summaries"; it
  calls the lead_agent with an EVIDENCE QUALITY: LIMITED framing.

No LLM, no network — stubbed agents and backend.
"""

from __future__ import annotations

from pydantic_graph import GraphRunContext

from andamentum.deep_research.models import (
    EvidenceReport,
    FetchedPage,
    PageSummary,
)
from andamentum.deep_research.nodes import (
    AnalyzeGaps,
    NodeDeps,
    SummarizePages,
    Synthesize,
)
from andamentum.deep_research.state import ResearchState


# ── Stubs (mirror the shape used by the search-loop plumbing tests) ──


class _StubResult:
    def __init__(self, output):
        self.output = output


class StubAgent:
    """Returns scripted outputs in order; raises if exhausted."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    async def run(self, prompt, **_kwargs):
        if not self._outputs:
            raise AssertionError(f"StubAgent exhausted; received call: {prompt[:80]!r}")
        self.calls.append(prompt)
        return _StubResult(self._outputs.pop(0))


class _NoBackend:
    async def search(self, query, max_results=10):
        raise AssertionError("backend.search must not be called here")

    async def fetch_page(self, url):
        raise AssertionError("backend.fetch_page must not be called here")


def _make_ctx(*, page_summarizer_outputs=None, lead_agent_outputs=None):
    state = ResearchState(query="What are Kalign's MSA competitors?")
    overrides: dict = {}
    if page_summarizer_outputs is not None:
        overrides["page_summarizer"] = StubAgent(page_summarizer_outputs)
    if lead_agent_outputs is not None:
        overrides["lead_agent"] = StubAgent(lead_agent_outputs)
    deps = NodeDeps(
        backend=_NoBackend(),  # type: ignore[arg-type]
        model="stub",
        correlation_id="test",
        agent_overrides=overrides,
    )
    return GraphRunContext(state=state, deps=deps)


def _fetched_page(url: str, title: str, content: str = "...") -> FetchedPage:
    return FetchedPage(
        url=url,
        title=title,
        content=content,
        word_count=len(content.split()),
        relevance_score=0.5,
        is_relevant=True,
    )


# ── Tests ──────────────────────────────────────────────────────────────


async def test_summarize_keeps_all_summaries_sorted_by_relevance():
    """SummarizePages must store all summaries (no <0.3 filter), sorted desc.

    Regression: the previous behaviour silently dropped any summary scoring
    <=0.3, which caused the "Kalign competitors" run to discard pages that
    actually contained competitor data.
    """
    summary_low = PageSummary(
        url="https://a.test/",
        title="Page A",
        summary="Mostly off-topic page that mentions Kalign once.",
        key_points=["off-topic"],
        relevance_score=0.1,
    )
    summary_mid = PageSummary(
        url="https://b.test/",
        title="Page B",
        summary="Tangential — describes Kalign internals.",
        key_points=["internals"],
        relevance_score=0.25,
    )
    summary_high = PageSummary(
        url="https://c.test/",
        title="Page C",
        summary="Direct competitor list.",
        key_points=["MUSCLE", "MAFFT"],
        relevance_score=0.85,
    )

    ctx = _make_ctx(page_summarizer_outputs=[summary_low, summary_mid, summary_high])
    ctx.state.fetched_pages = [
        _fetched_page("https://a.test/", "Page A"),
        _fetched_page("https://b.test/", "Page B"),
        _fetched_page("https://c.test/", "Page C"),
    ]

    nxt = await SummarizePages().run(ctx)
    assert isinstance(nxt, AnalyzeGaps)

    rels = [s.relevance_score for s in ctx.state.page_summaries]
    assert rels == [0.85, 0.25, 0.1], (
        f"Expected all summaries sorted desc by relevance; got {rels!r}"
    )
    assert len(ctx.state.page_summaries) == 3, (
        "Filter regression — low-relevance summaries were discarded again"
    )


async def test_synthesize_calls_lead_agent_when_only_low_relevance_pages():
    """When max relevance < 0.3, Synthesize must NOT bail — it must call
    the lead_agent with an EVIDENCE QUALITY: LIMITED note in the prompt.

    Regression: the previous behaviour returned an "incomplete - no
    content summaries available" placeholder report, hiding what the
    summariser actually found.
    """
    canned_report = EvidenceReport(
        evidence_summary="Limited evidence — pages found discuss Kalign internals…",
        key_findings=["Limited finding 1", "Limited finding 2"],
        sources=["https://a.test/"],
        total_searches_performed=0,
        total_pages_fetched=0,
        iterations_required=0,
    )
    ctx = _make_ctx(lead_agent_outputs=[canned_report])
    ctx.state.page_summaries = [
        PageSummary(
            url="https://a.test/",
            title="Page A",
            summary="Tangential page about Kalign internals.",
            key_points=["internals"],
            relevance_score=0.20,
        ),
        PageSummary(
            url="https://b.test/",
            title="Page B",
            summary="Off-topic page mentioning Kalign once.",
            key_points=["passing"],
            relevance_score=0.05,
        ),
    ]

    nxt = await Synthesize().run(ctx)
    # Synthesize returns End[EvidenceReport]; the End wrapper has .data.
    assert nxt.data is canned_report, (
        "Synthesize bailed instead of calling the lead agent on low-relevance "
        "summaries — graceful-degrade regression"
    )

    # Confirm the lead-agent prompt actually flagged limited evidence.
    overrides = ctx.deps.agent_overrides
    assert overrides is not None
    lead_stub = overrides["lead_agent"]
    assert isinstance(lead_stub, StubAgent)
    assert len(lead_stub.calls) == 1
    prompt = lead_stub.calls[0]
    assert "EVIDENCE QUALITY: LIMITED" in prompt, (
        f"Synthesize prompt did not flag limited evidence:\n{prompt[:500]}"
    )


async def test_synthesize_short_circuits_only_when_no_pages_at_all():
    """The bail-out path is reserved for the genuine "zero pages fetched"
    case (every search/fetch failed), not for "summaries scored low"."""
    ctx = _make_ctx()  # no agent overrides — bail-out path requires no LLM
    # page_summaries empty, fetched_pages empty, fetch_errors populated.
    ctx.state.fetch_errors = [
        {"url": "https://x.test/", "error": "HTTP 403", "link_id": "0"},
    ]
    ctx.state.total_searches = 3

    nxt = await Synthesize().run(ctx)
    report = nxt.data
    assert "no pages were fetched" in report.evidence_summary.lower()
    # The new path includes search/fetch counts in the message so the user
    # can see why it bailed.
    assert "Searches performed: 3" in report.evidence_summary
    assert "fetch failures: 1" in report.evidence_summary
