"""Graph nodes for research workflow orchestration.

These nodes implement the research cycle: plan → search-cycle → fetch →
summarize → analyze gaps → (refine | synthesize). Per the dialect (L2:
thin orchestrator, fat worker) every node's ``run()`` is thin: read the
surfaces, call one engine-free worker (``generate_query`` /
``verify_query`` / ``run_searches`` / ``fetch_pages`` /
``summarize_pages`` / ``analyze_gaps`` / ``synthesize_report``), assign
the result to State at the join, return a typed successor. The workers
never see ``ctx`` / State / Deps — they take explicit narrow inputs plus
the ``SearchBackend`` Port and the ``SearchReporter`` sink.

Search-cycle internals (post-2026-04 redesign): ``PrepareSearchCycle`` →
``GenerateOne`` ⇄ ``Verify`` (per-slot loop, bounded by
``MAX_SLOT_RETRIES``) → ``ParallelSearch``. Generation and verification
are separate LLM calls; the parallel search is pure Python over
``state.cycle.validated_queries``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Union

from pydantic_graph import BaseNode, End, GraphRunContext

from .analyze_gaps import analyze_gaps
from .backends import SearchBackend
from .build_agent import AgentOverrides
from .fetch_pages import fetch_pages
from .generate_query import generate_query
from .models import EvidenceReport, SearchError, SearchQuery
from .reporter import NoopReporter, SearchReporter
from .run_searches import run_searches
from .state import ResearchState, SearchCycleState
from .summarize_pages import summarize_pages
from .synthesize_report import build_iteration_limit_report, synthesize_report
from .verify_query import verify_query

logger = logging.getLogger(__name__)


# Slot-level retry budget for the per-slot generate→verify loop. When a
# slot exhausts this many rejections, ``Verify`` decrements
# ``state.cycle.target_count`` and proceeds to the next slot (or to
# ``ParallelSearch`` if the lowered target is already met).
#
# A previous Phase-1-efficiency cut reduced this to 2 to halve
# wasted retries. Reverted (2026-05-02) — gave the generator one
# fewer chance to recover from a bad first draft, contributing to
# fewer validated queries reaching ParallelSearch and weaker
# evidence pools downstream.
MAX_SLOT_RETRIES = 3

# Validated-query targets per cycle mode (L5: fan-out width traces to a
# named constant). Initial cycles cast a wider net; gap cycles are
# narrower because they target specific identified gaps.
INITIAL_QUERY_TARGET = 3
GAP_QUERY_TARGET = 2


# ── Node Deps ──────────────────────────────────────────────────────────


def _default_reporter() -> SearchReporter:
    return NoopReporter()


@dataclass(frozen=True)
class NodeDeps:
    """Dependencies available to graph nodes.

    Frozen (L1): what the run was given — read everywhere, rebound
    nowhere. Config lives here, not in State.
    """

    backend: SearchBackend
    model: Any  # pydantic-ai model instance
    correlation_id: str = ""
    # Maximum search-analyze cycles before the run ends with an
    # iteration-limit report (L5: the outer loop bound).
    max_iterations: int = 3
    max_pages_to_fetch: int = 5
    max_results_per_search: int = 10
    # L7 aggregate loudness: when the fraction of failed searches (or
    # fetches) over attempts crosses this threshold, the final report is
    # stamped degraded — a run that skipped most of its work is not green.
    soft_failure_threshold: float = 0.5
    # Test-only: maps agent name → pydantic-ai Agent instance. Honoured by
    # the worker layer's ``build_agent`` to substitute a stub Agent for
    # the registry lookup. Production code MUST leave this as ``None``.
    agent_overrides: AgentOverrides | None = None
    # Progress reporter for the search cycle. Defaults to a NoopReporter
    # so nodes and workers can call methods unconditionally; the CLI
    # installs a RichReporter when ``--verbose`` is set.
    reporter: SearchReporter = field(default_factory=_default_reporter)


# ── PlanResearch ───────────────────────────────────────────────────────


@dataclass
class PlanResearch(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Entry node: initialize research."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "PrepareSearchCycle":
        ctx.state.current_phase = "plan"
        ctx.state.iteration_count = 0
        return PrepareSearchCycle()


# ── PrepareSearchCycle ─────────────────────────────────────────────────


@dataclass
class PrepareSearchCycle(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Initialise per-cycle state and bump the iteration counter.

    Runs at the start of every search cycle (initial entry from
    ``PlanResearch`` and every loop-back from ``RefineSearch``). Replaces
    ``state.cycle`` with a fresh ``SearchCycleState`` so prior-cycle data
    (validated queries, slot attempts) doesn't leak forward.
    """

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> Union["GenerateOne", End[EvidenceReport]]:
        ctx.state.current_phase = "search"
        ctx.state.iteration_count += 1

        if ctx.state.iteration_count > ctx.deps.max_iterations:
            return End(
                build_iteration_limit_report(
                    fetched_pages=ctx.state.fetched_pages,
                    total_searches=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iteration_count=ctx.state.iteration_count,
                )
            )

        is_gap = ctx.state.iteration_count > 1 and bool(ctx.state.identified_gaps)
        ctx.state.cycle = SearchCycleState(
            mode="gap" if is_gap else "initial",
            target_count=GAP_QUERY_TARGET if is_gap else INITIAL_QUERY_TARGET,
            gaps=list(ctx.state.identified_gaps) if is_gap else [],
        )
        ctx.deps.reporter.cycle_starting(
            iteration=ctx.state.iteration_count,
            mode=ctx.state.cycle.mode,
            target_count=ctx.state.cycle.target_count,
            gaps=ctx.state.cycle.gaps,
        )
        ctx.deps.reporter.slot_starting(slot=1)
        return GenerateOne()


# ── GenerateOne ────────────────────────────────────────────────────────


@dataclass
class GenerateOne(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Produce one search query via the ``generate_query`` worker.

    Called once per slot; on a retry within the same slot the previous
    verifier's reason is passed in via ``feedback`` so the generator can
    correct the rejected query without losing strategic context.
    """

    feedback: str | None = None

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "Verify":
        c = ctx.state.cycle
        out = await generate_query(
            ctx.state.query,
            validated_queries=c.validated_queries,
            gaps=c.gaps,
            rejected_queries=c.slot_rejected_queries,
            feedback=self.feedback,
            slot=len(c.validated_queries) + 1,
            attempt=c.slot_attempts + 1,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
            reporter=ctx.deps.reporter,
        )
        return Verify(query=out.query)


# ── Verify ─────────────────────────────────────────────────────────────


@dataclass
class Verify(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Judge a single query against the goal via the ``verify_query`` worker.

    On accept: appends to ``cycle.validated_queries``, resets
    ``slot_attempts``, and routes either back to ``GenerateOne`` for the
    next slot or to ``ParallelSearch`` when ``target_count`` is reached.

    On reject: increments ``slot_attempts``. If under
    ``MAX_SLOT_RETRIES``, retries the slot via ``GenerateOne`` with the
    verifier's reason as feedback. If the budget is exhausted, applies
    skip-and-tighten — drops ``target_count`` by one and proceeds.
    """

    query: str

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> Union["GenerateOne", "ParallelSearch"]:
        verdict = await verify_query(
            self.query,
            goal=ctx.state.query,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
        )
        c = ctx.state.cycle

        slot = len(c.validated_queries) + 1

        if verdict.on_topic:
            c.validated_queries.append(self.query)
            c.slot_attempts = 0
            c.slot_rejected_queries.clear()
            ctx.deps.reporter.query_accepted(
                slot=slot, query=self.query, reason=verdict.reason
            )
            if len(c.validated_queries) >= c.target_count:
                return ParallelSearch()
            ctx.deps.reporter.slot_starting(slot=slot + 1)
            return GenerateOne()

        # Rejected.
        c.slot_attempts += 1
        c.slot_rejected_queries.append(self.query)
        ctx.deps.reporter.query_rejected(
            slot=slot,
            attempt=c.slot_attempts,
            query=self.query,
            reason=verdict.reason,
        )
        if c.slot_attempts >= MAX_SLOT_RETRIES:
            logger.warning(
                "[%s] Search-cycle slot exhausted retries; tightening target_count "
                "from %d to %d (validated=%d, last reason=%r)",
                ctx.deps.correlation_id,
                c.target_count,
                c.target_count - 1,
                len(c.validated_queries),
                verdict.reason,
            )
            c.slot_attempts = 0
            c.slot_rejected_queries.clear()
            c.target_count -= 1
            ctx.deps.reporter.slot_exhausted(slot=slot, new_target_count=c.target_count)
            if len(c.validated_queries) >= c.target_count:
                return ParallelSearch()
            ctx.deps.reporter.slot_starting(slot=slot + 1)
            return GenerateOne()

        return GenerateOne(feedback=verdict.reason)


# ── ParallelSearch ─────────────────────────────────────────────────────


@dataclass
class ParallelSearch(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Run the validated queries in parallel via the ``run_searches`` worker.

    No LLM. The worker owns the bounded fan-out; this node is the join —
    the sole State-write site for the cycle's search results (L3).
    Errors per query are recorded in ``state.search_errors``; one failed
    query does not abort the others.
    """

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "FetchPhase":
        c = ctx.state.cycle
        if not c.validated_queries:
            logger.warning(
                "[%s] Search cycle produced no validated queries; "
                "ParallelSearch running with empty list "
                "(outer max_iterations will eventually terminate).",
                ctx.deps.correlation_id,
            )

        outcomes = await run_searches(
            c.validated_queries,
            backend=ctx.deps.backend,
            max_results=ctx.deps.max_results_per_search,
            correlation_id=ctx.deps.correlation_id,
            reporter=ctx.deps.reporter,
        )

        cycle_reasoning = (
            f"Generated via per-slot generate/verify; "
            f"mode={c.mode}, validated={len(c.validated_queries)}/{c.target_count}"
        )

        # Join — the sole State-write site for this fan-out (L3).
        for outcome in outcomes:
            if outcome.error:
                ctx.state.search_errors.append(
                    SearchError(
                        query=outcome.query,
                        error=outcome.error,
                        is_retryable=True,
                    )
                )
            ctx.state.search_history.append(
                SearchQuery(
                    query=outcome.query,
                    reasoning=cycle_reasoning,
                    iteration=ctx.state.iteration_count,
                )
            )
            ctx.state.all_results[outcome.query] = outcome.results
            ctx.state.total_searches += 1
            for r in outcome.results:
                ctx.state.searched_urls.add(r.url)

        ctx.deps.reporter.cycle_complete()
        return FetchPhase()


# ── FetchPhase ─────────────────────────────────────────────────────────


@dataclass
class FetchPhase(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Fetch relevant pages via the ``fetch_pages`` worker.

    The worker owns dedup, agent selection, and the parallel fetch; this
    node is the join — the sole State-write site for the fetched pages
    and fetch errors (L3).
    """

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "SummarizePages":
        ctx.state.current_phase = "fetch"

        if not ctx.state.all_results:
            return SummarizePages()

        outcome = await fetch_pages(
            goal=ctx.state.query,
            search_results=ctx.state.all_results,
            fetched_urls=ctx.state.fetched_urls,
            failed_urls={e.url for e in ctx.state.fetch_errors},
            max_pages=ctx.deps.max_pages_to_fetch,
            backend=ctx.deps.backend,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
            reporter=ctx.deps.reporter,
        )

        # Join — the sole State-write site for this fan-out (L3).
        ctx.state.url_map = outcome.url_map
        for page in outcome.pages:
            # Final defence-in-depth: skip any page whose URL is
            # already in fetched_urls (would only fire if state
            # was mutated concurrently, but cheap to check).
            if page.url in ctx.state.fetched_urls:
                continue
            ctx.state.fetched_pages.append(page)
            ctx.state.total_pages_fetched += 1
            ctx.state.fetched_urls.add(page.url)
        ctx.state.fetch_errors.extend(outcome.errors)

        return SummarizePages()


# ── SummarizePages ─────────────────────────────────────────────────────


@dataclass
class SummarizePages(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Summarize each fetched page via the ``summarize_pages`` worker."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "AnalyzeGaps":
        ctx.state.current_phase = "summarize"

        if not ctx.state.fetched_pages:
            return AnalyzeGaps()

        ctx.state.page_summaries = await summarize_pages(
            ctx.state.fetched_pages,
            goal=ctx.state.query,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
            reporter=ctx.deps.reporter,
        )
        return AnalyzeGaps()


# ── AnalyzeGaps ────────────────────────────────────────────────────────


@dataclass
class AnalyzeGaps(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Evaluate research completeness via the ``analyze_gaps`` worker."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> Union["RefineSearch", "Synthesize"]:
        ctx.state.current_phase = "analyze"

        gap_analysis = await analyze_gaps(
            goal=ctx.state.query,
            summaries=ctx.state.page_summaries,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
        )

        ctx.state.is_complete = gap_analysis.is_complete
        ctx.state.identified_gaps = gap_analysis.identified_gaps

        if gap_analysis.is_complete:
            return Synthesize()
        elif ctx.state.iteration_count >= ctx.deps.max_iterations:
            return Synthesize()
        else:
            return RefineSearch(gaps=gap_analysis.identified_gaps)


# ── RefineSearch ───────────────────────────────────────────────────────


@dataclass
class RefineSearch(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Loop back to search with identified gaps."""

    gaps: list[str]

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "PrepareSearchCycle":
        ctx.state.current_phase = "refine"
        return PrepareSearchCycle()


# ── Synthesize ─────────────────────────────────────────────────────────


@dataclass
class Synthesize(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Final synthesis via the ``synthesize_report`` worker.

    The worker owns the lead-agent call, the zero-pages bail-out, the
    synthesis-failure fallback, and the L7 degradation stamp (aggregate
    soft-failure rate vs ``deps.soft_failure_threshold``).
    """

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> End[EvidenceReport]:
        ctx.state.current_phase = "synthesize"

        report = await synthesize_report(
            goal=ctx.state.query,
            summaries=ctx.state.page_summaries,
            iteration_count=ctx.state.iteration_count,
            total_searches=ctx.state.total_searches,
            total_pages_fetched=ctx.state.total_pages_fetched,
            n_search_errors=len(ctx.state.search_errors),
            n_fetch_errors=len(ctx.state.fetch_errors),
            soft_failure_threshold=ctx.deps.soft_failure_threshold,
            model=ctx.deps.model,
            overrides=ctx.deps.agent_overrides,
            reporter=ctx.deps.reporter,
        )
        return End(report)
