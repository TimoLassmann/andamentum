"""Tests for WebSearchGatherer evidence gathering."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from ..evidence_gathering import WebSearchGatherer


def _make_research_result(
    fetched_pages: list | None = None,
    page_summaries: list | None = None,
    evidence_summary: str | None = None,
    sources: list[str] | None = None,
):
    """Build a mock ResearchResult with the fields WebSearchGatherer accesses."""
    from andamentum.deep_research.models import (
        EvidenceReport,
        ResearchErrors,
        ResearchResult,
    )

    if fetched_pages is None:
        fetched_pages = []
    if page_summaries is None:
        page_summaries = []

    report = EvidenceReport(
        evidence_summary=evidence_summary
        if evidence_summary is not None
        else "Fallback summary",
        key_findings=["finding1"],
        sources=sources or ["http://example.com"],
        total_searches_performed=1,
        total_pages_fetched=len(fetched_pages),
        iterations_required=1,
    )

    from andamentum.deep_research.verification import VerificationResult

    return ResearchResult(
        output=report,
        page_summaries=page_summaries,
        fetched_pages=fetched_pages,
        iterations=1,
        searches=1,
        pages_fetched=len(fetched_pages),
        verification=VerificationResult(
            total_cited=0,
            verified_count=0,
            verified=[],
            unverified=[],
            accessed_not_cited=[],
            verification_rate=0.0,
        ),
        errors=ResearchErrors(search_errors=0, fetch_errors=0),
    )


def _make_fetched_page(
    url: str = "http://example.com/page1",
    title: str = "Test Page",
    content: str = "This is the raw page content with real evidence.",
    word_count: int = 50,
    relevance_score: float = 0.8,
    is_relevant: bool = True,
):
    from andamentum.deep_research.models import FetchedPage

    return FetchedPage(
        url=url,
        title=title,
        content=content,
        word_count=word_count,
        relevance_score=relevance_score,
        is_relevant=is_relevant,
        extraction_timestamp=datetime(2026, 4, 7, 12, 0, 0),
    )


def _make_page_summary(
    url: str = "http://example.com/page1",
    title: str = "Test Page",
    summary: str = "AI generated summary of the page.",
    key_points: list[str] | None = None,
    key_excerpts: list[str] | None = None,
    relevance_score: float = 0.8,
):
    from andamentum.deep_research.models import PageSummary

    return PageSummary(
        url=url,
        title=title,
        summary=summary,
        key_points=key_points or ["Point A", "Point B"],
        key_excerpts=key_excerpts or ['"Verbatim quote from source"'],
        relevance_score=relevance_score,
    )


class TestWebSearchGathererRawContent:
    """Test that WebSearchGatherer passes raw page content as primary evidence."""

    @pytest.mark.asyncio
    async def test_raw_content_is_primary(self):
        """Raw FetchedPage.content should be the GatheredEvidence.content, not AI summary."""
        raw_text = "The actual text from the web page with specific data points."
        page = _make_fetched_page(content=raw_text)
        summary = _make_page_summary(summary="AI compressed version of the page.")

        result = _make_research_result(
            fetched_pages=[page],
            page_summaries=[summary],
        )

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        assert len(gathered) == 1
        assert gathered[0].content == raw_text
        assert gathered[0].source_ref == "http://example.com/page1"

    @pytest.mark.asyncio
    async def test_annotations_in_structured_data(self):
        """Annotations from pointers should be in structured_data."""
        page = _make_fetched_page()
        summary = _make_page_summary(
            summary="The AI summary.",
            key_points=["Point 1", "Point 2"],
            key_excerpts=['"raw page content"'],
        )

        result = _make_research_result(fetched_pages=[page], page_summaries=[summary])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        sd = gathered[0].structured_data
        assert "annotations" in sd
        assert len(sd["annotations"]) >= 1
        assert "page_title" in sd

    @pytest.mark.asyncio
    async def test_evidence_kind_set(self):
        """evidence_kind should be 'web_page' for web search results."""
        page = _make_fetched_page()
        summary = _make_page_summary()
        result = _make_research_result(fetched_pages=[page], page_summaries=[summary])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        assert gathered[0].evidence_kind == "web_page"

    @pytest.mark.asyncio
    async def test_quality_score_from_summary_relevance(self):
        """quality_score should come from PageSummary.relevance_score."""
        page = _make_fetched_page(relevance_score=0.5)
        summary = _make_page_summary(relevance_score=0.9)
        result = _make_research_result(fetched_pages=[page], page_summaries=[summary])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        # Should use summary relevance (post-content analysis), not fetch relevance
        assert gathered[0].quality_score == 0.9


class TestWebSearchGathererEdgeCases:
    """Test edge cases and fallback strategies."""

    @pytest.mark.asyncio
    async def test_fetched_page_without_matching_summary_is_skipped(self):
        """FetchedPage with no PageSummary match (filtered at relevance < 0.3) should be skipped."""
        page = _make_fetched_page(url="http://example.com/irrelevant")
        # No matching summary — summarizer filtered it out
        result = _make_research_result(fetched_pages=[page], page_summaries=[])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        # Should fall through to Strategy 2 (EvidenceReport fallback)
        assert len(gathered) == 1
        assert gathered[0].content == "Fallback summary"

    @pytest.mark.asyncio
    async def test_empty_fetched_pages_falls_back_to_summaries(self):
        """When fetched_pages is empty, fall back to page_summaries (backward compat)."""
        summary = _make_page_summary(
            summary="AI summary from older deep-research.",
            key_points=["Old point"],
        )
        result = _make_research_result(fetched_pages=[], page_summaries=[summary])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        assert len(gathered) == 1
        # Falls back to summary-based content (current behavior)
        assert "AI summary from older deep-research." in gathered[0].content

    @pytest.mark.asyncio
    async def test_multiple_pages_produce_multiple_items(self):
        """Each matched FetchedPage+PageSummary pair produces one GatheredEvidence."""
        pages = [
            _make_fetched_page(url="http://a.com", content="Content A"),
            _make_fetched_page(url="http://b.com", content="Content B"),
        ]
        summaries = [
            _make_page_summary(url="http://a.com", relevance_score=0.7),
            _make_page_summary(url="http://b.com", relevance_score=0.9),
        ]
        result = _make_research_result(fetched_pages=pages, page_summaries=summaries)

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        assert len(gathered) == 2
        assert gathered[0].content == "Content A"
        assert gathered[1].content == "Content B"

    @pytest.mark.asyncio
    async def test_no_results_at_all(self):
        """When everything is empty, return a 'no results' GatheredEvidence."""
        result = _make_research_result(
            fetched_pages=[],
            page_summaries=[],
            evidence_summary="",
            sources=[],
        )

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        assert len(gathered) == 1
        assert "no usable results" in gathered[0].content.lower()
        assert gathered[0].quality_score == 0.0

    @pytest.mark.asyncio
    async def test_passage_has_page_title_in_structured_data(self):
        """Page title should be in structured_data for provenance."""
        page = _make_fetched_page(content="Some real content here. " * 50)
        summary = _make_page_summary(key_excerpts=["real content here"])
        result = _make_research_result(fetched_pages=[page], page_summaries=[summary])

        gatherer = WebSearchGatherer(model="test")
        with patch(
            "andamentum.deep_research.orchestrator.run_research",
            new_callable=AsyncMock,
            return_value=result,
        ):
            with patch("andamentum.epistemic.evidence_gathering.ensure_searxng"):
                gathered = await gatherer.gather("web_search", "test query")

        sd = gathered[0].structured_data
        assert sd["page_title"] == "Test Page"
