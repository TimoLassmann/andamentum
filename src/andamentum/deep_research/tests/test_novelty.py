"""Tests for novelty checking submodule."""

import pytest

from ..models import EvidenceReport
from ..novelty import (
    NoveltyReport,
    NoveltyAssessment,
    SimilarWork,
    Relevance,
)
from ..novelty.checker import _check_novelty_with_deps


def _report(
    *, evidence_summary: str, key_findings: list[str], sources: list[str]
) -> EvidenceReport:
    """Minimal EvidenceReport stub for research_fn.

    Uses ``model_construct`` to bypass the min-length validators so the
    'nothing found' scenarios (empty findings/sources) stay representable.
    """
    return EvidenceReport.model_construct(
        evidence_summary=evidence_summary,
        key_findings=key_findings,
        sources=sources,
        total_searches_performed=0,
        total_pages_fetched=0,
        iterations_required=0,
    )


class TestNoveltyModels:
    def test_relevance_enum_values(self):
        assert Relevance.DIRECT == "direct"
        assert Relevance.PARTIAL == "partial"
        assert Relevance.TANGENTIAL == "tangential"

    def test_similar_work_construction(self):
        sw = SimilarWork(
            title="Paper",
            url="https://example.com",
            relevance=Relevance.DIRECT,
            summary="Related",
        )
        assert sw.title == "Paper"
        assert sw.relevance == Relevance.DIRECT

    def test_novelty_report_construction(self):
        report = NoveltyReport(
            claim="test claim",
            is_novel=True,
            confidence=0.8,
            assessment="Appears novel",
        )
        assert report.claim == "test claim"
        assert report.is_novel is True
        assert report.similar_work == []
        assert report.sources == []
        assert report.search_queries_used == []

    def test_novelty_report_with_similar_work(self):
        sw = SimilarWork(title="P", url="u", relevance=Relevance.PARTIAL, summary="s")
        report = NoveltyReport(
            claim="c", is_novel=False, confidence=0.7, assessment="a", similar_work=[sw]
        )
        assert len(report.similar_work) == 1

    def test_novelty_assessment_pydantic(self):
        na = NoveltyAssessment(
            is_novel=False,
            confidence=0.9,
            assessment="Prior work exists",
            similar_works=[
                {"title": "P", "url": "u", "relevance": "direct", "summary": "s"}
            ],
        )
        assert na.is_novel is False
        assert len(na.similar_works) == 1


class TestCheckNovelty:
    @pytest.mark.asyncio
    async def test_happy_path_prior_work_found(self):
        """Research finds prior work, assessment says not novel."""
        output = _report(
            evidence_summary="Prior work found",
            key_findings=["finding 1"],
            sources=["https://example.com/paper"],
        )

        async def research_fn(**kwargs):
            return output

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            return NoveltyAssessment(
                is_novel=False,
                confidence=0.9,
                assessment="Prior work exists",
                similar_works=[
                    {
                        "title": "Paper",
                        "url": "https://example.com/paper",
                        "relevance": "direct",
                        "summary": "Same claim",
                    }
                ],
            )

        report = await _check_novelty_with_deps("test claim", research_fn, assess_fn)
        assert report.is_novel is False
        assert report.confidence == 0.9
        assert len(report.similar_work) == 1
        assert report.similar_work[0].relevance == Relevance.DIRECT

    @pytest.mark.asyncio
    async def test_empty_research_is_undetermined(self):
        """Research returns no output — novelty is UNDETERMINED, not asserted.

        A failed/empty search must not claim the work is novel (that would
        falsely reassure the author); is_novel is None."""

        async def research_fn(**kwargs):
            return None

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            raise AssertionError("Should not be called")

        report = await _check_novelty_with_deps("novel claim", research_fn, assess_fn)
        assert report.is_novel is None
        assert report.confidence == 0.0

    @pytest.mark.asyncio
    async def test_research_failure_is_undetermined(self):
        """Research function raises — novelty is UNDETERMINED (is_novel=None),
        never silently asserted as novel."""

        async def research_fn(**kwargs):
            raise ConnectionError("SearXNG down")

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            raise AssertionError("Should not be called")

        report = await _check_novelty_with_deps("test claim", research_fn, assess_fn)
        assert report.is_novel is None
        assert report.confidence == 0.0
        assert "Could not complete search" in report.assessment

    @pytest.mark.asyncio
    async def test_assessment_failure_with_sources(self):
        """Assessment fails but research found sources — heuristic fallback."""
        output = _report(
            evidence_summary="Found some evidence",
            key_findings=["finding 1", "finding 2"],
            sources=["https://example.com"],
        )

        async def research_fn(**kwargs):
            return output

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            raise RuntimeError("LLM error")

        report = await _check_novelty_with_deps("test claim", research_fn, assess_fn)
        assert report.is_novel is False
        assert report.confidence == 0.6
        assert "prior work exists" in report.assessment.lower()

    @pytest.mark.asyncio
    async def test_assessment_failure_without_sources(self):
        """Assessment fails and no sources found — novel with low confidence."""
        output = _report(
            evidence_summary="Nothing found",
            key_findings=[],
            sources=[],
        )

        async def research_fn(**kwargs):
            return output

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            raise RuntimeError("LLM error")

        report = await _check_novelty_with_deps("test claim", research_fn, assess_fn)
        assert report.is_novel is True
        assert report.confidence == 0.3

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        """Confidence values outside [0, 1] get clamped."""
        output = _report(
            evidence_summary="s", key_findings=["f"], sources=["u"]
        )

        async def research_fn(**kwargs):
            return output

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            return NoveltyAssessment(
                is_novel=True, confidence=1.5, assessment="a", similar_works=[]
            )

        report = await _check_novelty_with_deps("test", research_fn, assess_fn)
        assert report.confidence == 1.0

    @pytest.mark.asyncio
    async def test_invalid_relevance_defaults_to_tangential(self):
        """Invalid relevance value in similar_works defaults to TANGENTIAL."""
        output = _report(
            evidence_summary="s", key_findings=["f"], sources=["u"]
        )

        async def research_fn(**kwargs):
            return output

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            return NoveltyAssessment(
                is_novel=False,
                confidence=0.8,
                assessment="a",
                similar_works=[
                    {
                        "title": "P",
                        "url": "u",
                        "relevance": "invalid_value",
                        "summary": "s",
                    }
                ],
            )

        report = await _check_novelty_with_deps("test", research_fn, assess_fn)
        assert report.similar_work[0].relevance == Relevance.TANGENTIAL

    @pytest.mark.asyncio
    async def test_search_queries_always_populated(self):
        """Search queries are populated even on failure."""

        async def research_fn(**kwargs):
            raise Exception("fail")

        async def assess_fn(claim, evidence_summary, key_findings, sources):
            raise AssertionError("Should not be called")

        report = await _check_novelty_with_deps("my claim", research_fn, assess_fn)
        assert len(report.search_queries_used) == 3
        assert any("my claim" in q for q in report.search_queries_used)
