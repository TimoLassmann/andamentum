"""Core novelty checking logic — framework-agnostic.

Accepts callable dependencies (research_fn, assess_fn) instead of importing
framework code directly.
"""

from typing import Any
from collections.abc import Callable, Awaitable

from pydantic import BaseModel

from .models import NoveltyReport, SimilarWork, Relevance


class NoveltyAssessment(BaseModel):
    """Structured output from novelty assessment agent."""

    is_novel: bool
    confidence: float
    assessment: str
    similar_works: list[dict[str, str]]  # title, url, relevance, summary


# Type aliases for injectable dependencies
ResearchFn = Callable[..., Awaitable[dict[str, Any]]]
AssessFn = Callable[[str, str, list[str], list[str]], Awaitable[NoveltyAssessment]]


async def check_novelty(
    claim: str,
    research_fn: ResearchFn,
    assess_fn: AssessFn,
    search_depth: int = 2,
    verbose: bool = False,
) -> NoveltyReport:
    """
    Check if a claim is novel by searching for prior work.

    This is the pure function — it accepts callables for research and assessment
    instead of importing framework code.

    Args:
        claim: The claim or statement to check for novelty
        research_fn: Async callable that performs web research.
            Called as: research_fn(query=str, max_iterations=int, verbose=bool) -> dict
        assess_fn: Async callable that assesses novelty from evidence.
            Called as: assess_fn(claim, evidence_summary, key_findings, sources) -> NoveltyAssessment
        search_depth: 1=quick (1 iteration), 2=balanced (2), 3=thorough (3)
        verbose: Show detailed progress

    Returns:
        NoveltyReport with assessment, similar work, and sources
    """
    # Generate targeted search queries
    search_queries = [
        f"prior research {claim}",
        f"existing work {claim}",
        f"{claim} state of the art literature",
    ]

    # Select primary query based on claim
    primary_query = f"Find prior work, existing research, and publications related to: {claim}"

    if verbose:
        print(f"Searching for prior work on: {claim}")
        print(f"Search depth: {search_depth} (iterations)")

    # Run deep research to find prior work
    try:
        research_result = await research_fn(
            query=primary_query,
            max_iterations=search_depth,
            verbose=verbose,
        )
    except Exception as e:
        return NoveltyReport(
            claim=claim,
            is_novel=True,
            confidence=0.2,
            assessment=f"Could not complete search for prior work: {e}. "
            "Claim may or may not be novel - manual verification recommended.",
            similar_work=[],
            sources=[],
            search_queries_used=search_queries,
        )

    # Extract evidence from research
    output = research_result.get("output")
    if output is None:
        return NoveltyReport(
            claim=claim,
            is_novel=True,
            confidence=0.3,
            assessment="Research completed but no evidence was gathered. "
            "Claim appears novel based on empty search results.",
            similar_work=[],
            sources=[],
            search_queries_used=search_queries,
        )

    evidence_summary = getattr(output, "evidence_summary", "No evidence found.")
    key_findings = getattr(output, "key_findings", [])
    sources = getattr(output, "sources", [])

    # Use assessment function to evaluate novelty
    try:
        assessment_data = await assess_fn(claim, evidence_summary, key_findings, sources)
    except Exception:
        # If assessment fails, provide a report based on research results
        has_sources = len(sources) > 0
        has_findings = len(key_findings) > 0

        if has_sources or has_findings:
            is_novel = False
            confidence = 0.6 if has_findings else 0.5
            assessment_text = (
                f"Research found {len(sources)} relevant sources and {len(key_findings)} key findings. "
                f"This suggests prior work exists on this topic. "
                f"Evidence summary: {evidence_summary[:300]}..."
            )
        else:
            is_novel = True
            confidence = 0.3
            assessment_text = (
                "No prior work found in web search, but this may be due to search limitations. "
                "Manual verification recommended."
            )

        return NoveltyReport(
            claim=claim,
            is_novel=is_novel,
            confidence=confidence,
            assessment=assessment_text,
            similar_work=[],
            sources=sources,
            search_queries_used=search_queries,
        )

    # Convert assessment to NoveltyReport
    similar_work = []
    for work in assessment_data.similar_works:
        try:
            relevance = Relevance(work.get("relevance", "tangential"))
        except ValueError:
            relevance = Relevance.TANGENTIAL

        similar_work.append(
            SimilarWork(
                title=work.get("title", "Unknown"),
                url=work.get("url", ""),
                relevance=relevance,
                summary=work.get("summary", ""),
            )
        )

    return NoveltyReport(
        claim=claim,
        is_novel=assessment_data.is_novel,
        confidence=max(0.0, min(1.0, assessment_data.confidence)),
        assessment=assessment_data.assessment,
        similar_work=similar_work,
        sources=sources,
        search_queries_used=search_queries,
    )
