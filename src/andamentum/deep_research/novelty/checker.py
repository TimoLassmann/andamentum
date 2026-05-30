"""Core novelty checking logic.

Public entry point: :func:`run_novelty_check` — composes the deep-research
pipeline with the novelty-assessor agent into a single one-shot.

Internal helper :func:`_check_novelty_with_deps` accepts callable
dependencies for ``research_fn`` and ``assess_fn`` so the failure-mode
tests can exercise the orchestration logic with stubs (no live model,
no SearXNG). Production code calls ``run_novelty_check``.
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


# Type aliases for injectable dependencies (used by _check_novelty_with_deps).
ResearchFn = Callable[..., Awaitable[dict[str, Any]]]
AssessFn = Callable[[str, str, list[str], list[str]], Awaitable[NoveltyAssessment]]


async def run_novelty_check(
    claim: str,
    *,
    model: str,
    search_depth: int = 2,
    verbose: bool = False,
) -> NoveltyReport:
    """Check whether ``claim`` is novel by running web research + LLM assessment.

    One-shot entry point. Internally:
      1. Runs :func:`run_research` to retrieve evidence on the claim.
      2. Runs the registered ``novelty_assessor`` agent on that evidence
         to decide whether the claim is novel.
      3. Returns a :class:`NoveltyReport` summarising the verdict, prior
         work, and sources.

    Args:
        claim: The claim or statement to check for novelty.
        model: pydantic-ai model identifier (e.g.
            ``"anthropic:claude-haiku-4-5"``). Used for both the research
            agents and the novelty-assessment agent.
        search_depth: Number of search-analyze iterations. 1=quick,
            2=balanced, 3=thorough.
        verbose: Print progress.

    Returns:
        :class:`NoveltyReport` with verdict, prior work, and sources.
    """
    from ..orchestrator import run_research
    from ..agents.novelty import build_assessment_prompt

    async def research_fn(
        *, query: str, max_iterations: int, verbose: bool
    ) -> dict[str, Any]:
        result = await run_research(
            query=query,
            max_iterations=max_iterations,
            model=model,
            verbose=verbose,
        )
        return {"output": result.output}

    async def assess_fn(
        claim_text: str,
        evidence_summary: str,
        key_findings: list[str],
        sources: list[str],
    ) -> NoveltyAssessment:
        prompt = build_assessment_prompt(
            claim_text, evidence_summary, key_findings, sources
        )
        from andamentum.core.agents import build_pydantic_ai_agent
        from ..agents import get_agent

        agent = build_pydantic_ai_agent(get_agent("novelty_assessor"), model)
        result = await agent.run(prompt)
        return result.output  # type: ignore[no-any-return]

    return await _check_novelty_with_deps(
        claim=claim,
        research_fn=research_fn,
        assess_fn=assess_fn,
        search_depth=search_depth,
        verbose=verbose,
    )


async def _check_novelty_with_deps(
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
    primary_query = (
        f"Find prior work, existing research, and publications related to: {claim}"
    )

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
            is_novel=None,  # undetermined: the search did not complete
            confidence=0.0,
            assessment=f"Could not complete search for prior work: {e}. "
            "Novelty is UNDETERMINED — manual verification required.",
            similar_work=[],
            sources=[],
            search_queries_used=search_queries,
        )

    # Extract evidence from research
    output = research_result.get("output")
    if output is None:
        return NoveltyReport(
            claim=claim,
            is_novel=None,  # undetermined: no evidence was gathered
            confidence=0.0,
            assessment="Research completed but no evidence was gathered. "
            "Novelty is UNDETERMINED — manual verification required.",
            similar_work=[],
            sources=[],
            search_queries_used=search_queries,
        )

    evidence_summary = getattr(output, "evidence_summary", "No evidence found.")
    key_findings = getattr(output, "key_findings", [])
    sources = getattr(output, "sources", [])

    # Use assessment function to evaluate novelty
    try:
        assessment_data = await assess_fn(
            claim, evidence_summary, key_findings, sources
        )
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
