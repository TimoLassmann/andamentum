"""Tests for deep_research Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from deep_research.models import (
    SearchQuery,
    SearchResult,
    SearchPlan,
    FetchedPage,
    FetchPlan,
    FetchResults,
    PageSummary,
    GapAnalysis,
    EvidenceItem,
    EvidenceReport,
)


class TestSearchQuery:
    def test_construction(self):
        q = SearchQuery(query="test", reasoning="needed")
        assert q.query == "test"
        assert q.reasoning == "needed"
        assert q.iteration == 0

    def test_timestamp_auto(self):
        q = SearchQuery(query="test", reasoning="r")
        assert isinstance(q.timestamp, datetime)

    def test_custom_iteration(self):
        q = SearchQuery(query="test", reasoning="r", iteration=3)
        assert q.iteration == 3


class TestSearchResult:
    def test_construction(self):
        r = SearchResult(link_id=1, title="Page", url="https://example.com", snippet="text", domain="example.com")
        assert r.link_id == 1
        assert r.relevance_score == 0.0

    def test_score_bounds(self):
        r = SearchResult(link_id=1, title="P", url="u", snippet="s", domain="d", relevance_score=0.5)
        assert r.relevance_score == 0.5

    def test_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            SearchResult(link_id=1, title="P", url="u", snippet="s", domain="d", relevance_score=1.5)

        with pytest.raises(ValidationError):
            SearchResult(link_id=1, title="P", url="u", snippet="s", domain="d", relevance_score=-0.1)


class TestSearchPlan:
    def test_construction(self):
        sp = SearchPlan(queries=["q1", "q2"], reasoning="strategy")
        assert len(sp.queries) == 2

    def test_empty_queries_rejected(self):
        with pytest.raises(ValidationError):
            SearchPlan(queries=[], reasoning="r")


class TestFetchedPage:
    def test_construction(self):
        fp = FetchedPage(
            url="https://example.com",
            title="Page",
            content="text content",
            word_count=2,
            relevance_score=0.8,
            is_relevant=True,
        )
        assert fp.word_count == 2
        assert fp.is_relevant is True

    def test_relevance_bounds(self):
        with pytest.raises(ValidationError):
            FetchedPage(url="u", title="t", content="c", word_count=1, relevance_score=2.0, is_relevant=True)


class TestFetchPlan:
    def test_construction(self):
        fp = FetchPlan(link_ids=[1, 2, 3], reasoning="top pages")
        assert len(fp.link_ids) == 3

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            FetchPlan(link_ids=[], reasoning="r")


class TestFetchResults:
    def test_construction(self):
        fr = FetchResults(pages=[], skipped_count=2, error_count=1)
        assert fr.skipped_count == 2


class TestPageSummary:
    def test_construction(self):
        ps = PageSummary(
            url="https://example.com",
            title="Page",
            summary="A summary of the page content.",
            key_points=["point1", "point2", "point3"],
            relevance_score=0.9,
        )
        assert len(ps.key_points) == 3

    def test_relevance_bounds(self):
        with pytest.raises(ValidationError):
            PageSummary(url="u", title="t", summary="s", key_points=["p"], relevance_score=1.5)


class TestGapAnalysis:
    def test_complete(self):
        ga = GapAnalysis(is_complete=True, reasoning="All covered")
        assert ga.is_complete is True
        assert ga.identified_gaps == []
        assert ga.suggested_queries == []

    def test_incomplete(self):
        ga = GapAnalysis(
            is_complete=False,
            identified_gaps=["missing data"],
            reasoning="Need more info",
            suggested_queries=["search for data"],
        )
        assert not ga.is_complete
        assert len(ga.identified_gaps) == 1


class TestEvidenceItem:
    def test_construction(self):
        ei = EvidenceItem(finding="Result X", source_url="https://example.com", source_title="Paper")
        assert ei.confidence == "medium"  # default

    def test_confidence_values(self):
        for level in ("high", "medium", "low"):
            ei = EvidenceItem(finding="f", source_url="u", source_title="t", confidence=level)
            assert ei.confidence == level


class TestEvidenceReport:
    def test_construction(self):
        er = EvidenceReport(
            evidence_summary="Summary of findings",
            key_findings=["finding 1"],
            sources=["https://example.com"],
            total_searches_performed=3,
            total_pages_fetched=5,
            iterations_required=2,
        )
        assert er.total_searches_performed == 3
        assert er.evidence_items == []

    def test_requires_key_findings(self):
        with pytest.raises(ValidationError):
            EvidenceReport(
                evidence_summary="s",
                key_findings=[],
                sources=["u"],
                total_searches_performed=1,
                total_pages_fetched=1,
                iterations_required=1,
            )

    def test_requires_sources(self):
        with pytest.raises(ValidationError):
            EvidenceReport(
                evidence_summary="s",
                key_findings=["f"],
                sources=[],
                total_searches_performed=1,
                total_pages_fetched=1,
                iterations_required=1,
            )

    def test_serialization_roundtrip(self):
        er = EvidenceReport(
            evidence_summary="Summary",
            key_findings=["finding 1", "finding 2"],
            sources=["https://example.com"],
            evidence_items=[EvidenceItem(finding="f", source_url="u", source_title="t")],
            total_searches_performed=3,
            total_pages_fetched=5,
            iterations_required=2,
        )
        data = er.model_dump()
        er2 = EvidenceReport.model_validate(data)
        assert er2.evidence_summary == er.evidence_summary
        assert len(er2.evidence_items) == 1
        assert er2.key_findings == er.key_findings
