"""Research state maintained across graph execution."""

from dataclasses import dataclass, field
from typing import Literal
from .models import (
    EvidenceItem,
    FetchedPage,
    FetchError,
    PageSummary,
    SearchError,
    SearchQuery,
    SearchResult,
)


@dataclass
class SearchCycleState:
    """Per-cycle state for the generate→verify→search loop.

    Reinitialised by ``PrepareSearchCycle`` at the start of every search
    cycle (initial entry from ``PlanResearch`` and every loop-back from
    ``RefineSearch``). Read+mutated by ``GenerateOne`` and ``Verify``;
    consumed by ``ParallelSearch`` on cycle exit.

    ``mode`` and ``target_count`` distinguish initial-broad-coverage
    cycles from gap-targeted refinement cycles. ``slot_attempts`` is the
    per-slot retry counter that triggers skip-and-tighten when it exceeds
    ``MAX_SLOT_RETRIES`` (constant in ``nodes.py``).

    ``slot_rejected_queries`` holds the queries the verifier rejected for
    the *current* slot. It's threaded into the generator's prompt so the
    generator can see (and avoid paraphrasing) what it already tried.
    Resets to empty whenever the slot resolves (acceptance or
    skip-and-tighten).
    """

    mode: Literal["initial", "gap"] = "initial"
    target_count: int = 0
    gaps: list[str] = field(default_factory=list)
    validated_queries: list[str] = field(default_factory=list)
    slot_attempts: int = 0
    slot_rejected_queries: list[str] = field(default_factory=list)


@dataclass
class ResearchState:
    """Shared state for research workflow."""

    # Input (required at start)
    query: str

    # Search tracking
    search_history: list[SearchQuery] = field(default_factory=list)
    all_results: dict[str, list[SearchResult]] = field(
        default_factory=dict
    )  # query -> results
    url_map: dict[int, str] = field(
        default_factory=dict
    )  # link_id -> URL for tool lookups

    # Content tracking
    fetched_pages: list[FetchedPage] = field(default_factory=list)
    page_summaries: list[PageSummary] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)

    # Gap analysis tracking
    identified_gaps: list[str] = field(default_factory=list)
    is_complete: bool = False

    # Search cycle (generate→verify→search) — reset every cycle.
    cycle: SearchCycleState = field(default_factory=SearchCycleState)

    # Flow control
    iteration_count: int = 0
    current_phase: Literal[
        "plan", "search", "fetch", "summarize", "analyze", "refine", "synthesize"
    ] = "plan"

    # Metrics
    total_searches: int = 0
    total_pages_fetched: int = 0

    # Source verification tracking
    searched_urls: set[str] = field(default_factory=set)
    fetched_urls: set[str] = field(default_factory=set)

    # Error tracking (Phase 2 fix: surface search failures)
    search_errors: list[SearchError] = field(default_factory=list)
    fetch_errors: list[FetchError] = field(default_factory=list)
