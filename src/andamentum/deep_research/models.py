"""Pydantic models for structured agent communication."""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

from .verification import VerificationResult

__all__ = [
    "SearchQuery",
    "SearchResult",
    "SearchError",
    "SearchOutcome",
    "GeneratorOutput",
    "VerifierOutput",
    "FetchedPage",
    "FetchError",
    "FetchOutcome",
    "FetchResults",
    "FetchPlan",
    "PageSummary",
    "FetchSummary",
    "GapAnalysis",
    "EvidenceReport",
    "ResearchErrors",
    "ResearchResult",
]


# Search phase models
class SearchQuery(BaseModel):
    """A search query with metadata."""

    query: str = Field(..., description="The search query string")
    reasoning: str = Field(..., description="Why this query is needed")
    iteration: int = Field(default=0, description="Which search iteration")
    timestamp: datetime = Field(default_factory=datetime.now)


class SearchResult(BaseModel):
    """Single search result."""

    link_id: int = Field(..., description="Numeric ID for this result")
    title: str = Field(..., description="Page title")
    url: str = Field(..., description="Full URL")
    snippet: str = Field(..., description="Result snippet/summary")
    domain: str = Field(..., description="Domain name")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchError(BaseModel):
    """One failed search query, recorded at the ParallelSearch join."""

    query: str = Field(..., description="The query that failed")
    error: str = Field(..., description="Backend error message")
    is_retryable: bool = Field(
        default=True, description="Whether a retry could plausibly succeed"
    )


class SearchOutcome(BaseModel):
    """Result of one query in the parallel-search fan-out.

    Either ``results`` is populated (success) or ``error`` carries the
    failure message (soft failure — one bad query does not abort the
    others). Produced by the ``run_searches`` worker; reduced into
    ``ResearchState`` at the ParallelSearch join.
    """

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    error: str | None = None


class GeneratorOutput(BaseModel):
    """Output of the per-slot query generator."""

    query: str = Field(
        ...,
        description="A single search query (3-8 keywords, no operators)",
    )
    rationale: str = Field(
        ...,
        description="One sentence: what angle this query covers",
    )


class VerifierOutput(BaseModel):
    """Output of the topic verifier — accept/reject one query."""

    on_topic: bool = Field(
        ...,
        description="True if the query helps answer the research goal",
    )
    reason: str = Field(
        ...,
        description="One sentence: why on-topic, or why drifting",
    )


# Fetch phase models
class FetchedPage(BaseModel):
    """Content from an opened page."""

    url: str
    title: str
    content: str = Field(
        ...,
        description="Cleaned, sanitized page content (may be truncated; see truncated/original_length)",
    )
    word_count: int
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    is_relevant: bool = Field(..., description="Whether content is relevant to query")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
    original_length: int = Field(
        default=0, description="Original character length before any truncation"
    )
    truncated: bool = Field(
        default=False, description="True if content was truncated from a longer source"
    )


class FetchPlan(BaseModel):
    """Simplified output from PageFetcher - just the link IDs to fetch.

    The actual page fetching and result accumulation happens in the graph node,
    not in the LLM output. This avoids complex nested JSON generation that breaks
    with smaller local models.
    """

    link_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Link IDs to fetch (1-5 most relevant pages)",
    )
    reasoning: str = Field(..., description="Why these pages were selected")


class FetchError(BaseModel):
    """One failed page fetch, recorded at the FetchPhase join.

    URLs recorded here are treated as session-permanent failures — later
    cycles exclude them from the fetch-candidate list (within a 1-3
    minute research run, retrying the same URL almost never changes the
    outcome).
    """

    url: str = Field(..., description="The URL that failed to fetch")
    error: str = Field(..., description="Fetch/extraction error message")
    link_id: int | None = Field(
        default=None, description="link_id the fetcher agent picked, if known"
    )


class FetchOutcome(BaseModel):
    """Result of the select-and-fetch worker for one cycle.

    Produced by the ``fetch_pages`` worker; reduced into
    ``ResearchState`` at the FetchPhase join. Empty (all defaults) when
    the cycle had no fetch candidates left after dedup.
    """

    url_map: dict[int, str] = Field(default_factory=dict)
    pages: list[FetchedPage] = Field(default_factory=list)
    errors: list[FetchError] = Field(default_factory=list)


class FetchResults(BaseModel):
    """Output from PageFetcher subagent."""

    pages: list[FetchedPage] = Field(..., description="Pages successfully fetched")
    skipped_count: int = Field(
        ..., description="Number of pages skipped (low relevance)"
    )
    error_count: int = Field(..., description="Number of fetch errors")


# One-shot fetch (research-question-free counterpart of PageSummary)
class FetchSummary(BaseModel):
    """Structured summary of a single fetched page.

    Returned by :func:`run_fetch`. Distinct from :class:`PageSummary`
    (which carries a research-question relevance score) — this is the
    research-question-free counterpart for one-shot URL summarisation.
    """

    url: str = Field(default="", description="Source URL")
    title: str = Field(default="", description="Page title")
    summary: str = Field(
        ..., description="~200-word faithful summary of the page's main content"
    )
    key_points: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="3-5 substantive points from the page",
    )


# Summary phase models
class PageSummary(BaseModel):
    """Condensed summary of a fetched page's key points."""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Page title")
    summary: str = Field(
        ..., description="200-word summary of key points relevant to research question"
    )
    key_points: list[str] = Field(
        ..., min_length=1, max_length=5, description="3-5 main points from the page"
    )
    key_excerpts: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="1-3 verbatim quotes from the page that support the key points",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How relevant this page is to the research question",
    )


# Analysis phase models
class EvidenceItem(BaseModel):
    """Single piece of evidence extracted from research."""

    finding: str = Field(..., description="The actual evidence/fact")
    source_url: str
    source_title: str
    confidence: Literal["high", "medium", "low"] = "medium"


class GapAnalysis(BaseModel):
    """Output from GapAnalyzer subagent.

    IMPORTANT: ALL four fields are ALWAYS required, even when research is complete.
    When research is complete, use empty lists for identified_gaps and suggested_queries.
    """

    is_complete: bool = Field(
        ...,
        description="True if research comprehensively answers the question with sufficient evidence from credible sources. False if critical information is missing or sources are insufficient.",
    )
    identified_gaps: list[str] = Field(
        default_factory=list,
        description="Specific, concrete gaps in current research. Each gap should identify what information is missing (e.g., 'Missing treatment success rates', 'No data on side effects'). Empty list if research is complete.",
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of why research is complete OR what critical gaps exist. If complete, explain what question aspects are covered. If incomplete, explain which aspects lack sufficient evidence.",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="3-5 keyword search queries to fill identified gaps. Each query should be SHORT (3-5 words) and target a specific gap. Empty list if research is complete. Example: 'treatment success rates', 'drug side effects study'",
    )


# Final output model
class EvidenceReport(BaseModel):
    """Final research output from Lead Agent."""

    evidence_summary: str = Field(..., description="Comprehensive summary of findings")
    key_findings: list[str] = Field(
        ..., min_length=1, description="List of key evidence points"
    )
    sources: list[str] = Field(
        ..., min_length=1, description="List of credible source URLs"
    )
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    total_searches_performed: int
    total_pages_fetched: int
    iterations_required: int
    # L7 aggregate loudness: a run that skipped most of its work is not
    # green. Stamped deterministically at synthesis when the soft-failure
    # rate (failed searches or fetches over attempts) crosses
    # ``NodeDeps.soft_failure_threshold``.
    degraded: bool = Field(
        default=False,
        description=(
            "True when the run's search/fetch soft-failure rate crossed the "
            "configured threshold — treat findings as partial, not green"
        ),
    )
    degraded_reason: str = Field(
        default="",
        description="Why the run was marked degraded; empty when healthy",
    )


class ResearchErrors(BaseModel):
    """Error counts from a research session."""

    search_errors: int
    fetch_errors: int


class ResearchResult(BaseModel):
    """Complete result from a research session."""

    output: EvidenceReport
    page_summaries: list[PageSummary] = Field(default_factory=list)
    fetched_pages: list[FetchedPage] = Field(default_factory=list, exclude=True)
    iterations: int
    searches: int
    pages_fetched: int
    verification: VerificationResult
    errors: ResearchErrors
