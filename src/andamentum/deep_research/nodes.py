"""Graph nodes for research workflow orchestration.

These nodes implement the research cycle: plan → search-cycle → fetch →
summarize → analyze gaps → (refine | synthesize). They use the
SearchBackend protocol for search/fetch operations and pydantic-ai agents
for LLM decisions.

Search-cycle internals (post-2026-04 redesign): ``PrepareSearchCycle`` →
``GenerateOne`` ⇄ ``Verify`` (per-slot loop, bounded by
``MAX_SLOT_RETRIES``) → ``ParallelSearch``. Generation and verification
are separate LLM calls; the parallel search is pure Python over
``state.cycle.validated_queries``.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Union

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .state import ResearchState, SearchCycleState
from .models import (
    EvidenceReport,
    FetchedPage,
    FetchPlan,
    GapAnalysis,
    GeneratorOutput,
    PageSummary,
    SearchQuery,
    SearchResult,
    VerifierOutput,
)
from .agents import get_agent
from .backends import SearchBackend
from .reporter import NoopReporter, SearchReporter

logger = logging.getLogger(__name__)


# Slot-level retry budget for the per-slot generate→verify loop. When a
# slot exhausts this many rejections, ``Verify`` decrements
# ``state.cycle.target_count`` and proceeds to the next slot (or to
# ``ParallelSearch`` if the lowered target is already met).
MAX_SLOT_RETRIES = 3


def _build_agent(
    name: str,
    model: Any,
    overrides: dict[str, Any] | None = None,
) -> Agent[Any, Any]:
    """Create a pydantic-ai ``Agent`` from a registry definition.

    If ``overrides`` contains ``name``, the override (typically a stub
    Agent for tests) is returned instead of building from the registry.
    Production code never sets ``overrides``; tests pass it via
    ``NodeDeps.agent_overrides``.

    Delegates to ``andamentum.core.agents.build_pydantic_ai_agent`` so
    every node-based caller shares one Agent-construction recipe.
    """
    if overrides and name in overrides:
        return overrides[name]
    from andamentum.core.agents import build_pydantic_ai_agent

    return build_pydantic_ai_agent(get_agent(name), model)


# ── Node Deps ──────────────────────────────────────────────────────────


def _default_reporter() -> SearchReporter:
    return NoopReporter()


@dataclass
class NodeDeps:
    """Dependencies available to graph nodes."""

    backend: SearchBackend
    model: Any  # pydantic-ai model instance
    correlation_id: str = ""
    max_pages_to_fetch: int = 5
    max_results_per_search: int = 10
    # Test-only: maps agent name → pydantic-ai Agent instance. Honoured by
    # ``_build_agent`` to substitute a stub Agent for the registry lookup.
    # Production code MUST leave this as ``None``.
    agent_overrides: dict[str, Any] | None = None
    # Progress reporter for the search cycle. Defaults to a NoopReporter
    # so nodes can call methods unconditionally; the CLI installs a
    # RichReporter when ``--verbose`` is set.
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

        if ctx.state.iteration_count > ctx.state.max_iterations:
            sources = [
                f"{p.title} - {p.url}"
                for p in ctx.state.fetched_pages
                if p.is_relevant
            ]
            if not sources:
                sources = ["Research incomplete - max iterations reached"]
            return End(
                EvidenceReport(
                    evidence_summary="Max iterations reached. Research incomplete.",
                    key_findings=["Incomplete research - iteration limit reached"],
                    sources=sources,
                    total_searches_performed=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iterations_required=ctx.state.iteration_count,
                )
            )

        is_gap = ctx.state.iteration_count > 1 and bool(ctx.state.identified_gaps)
        ctx.state.cycle = SearchCycleState(
            mode="gap" if is_gap else "initial",
            target_count=2 if is_gap else 3,
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
    """Produce one search query via the ``query_generator`` agent.

    Called once per slot; on a retry within the same slot the previous
    verifier's reason is passed in via ``feedback`` so the generator can
    correct the rejected query without losing strategic context.
    """

    feedback: str | None = None

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "Verify":
        agent = _build_agent(
            "query_generator", ctx.deps.model, ctx.deps.agent_overrides
        )
        result = await agent.run(
            self._prompt(ctx),
            usage_limits=UsageLimits(request_limit=10),
        )
        out: GeneratorOutput = result.output
        c = ctx.state.cycle
        ctx.deps.reporter.query_generated(
            slot=len(c.validated_queries) + 1,
            attempt=c.slot_attempts + 1,
            query=out.query,
            rationale=out.rationale,
        )
        return Verify(query=out.query)

    def _prompt(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> str:
        c = ctx.state.cycle
        parts = [f"research_goal: {ctx.state.query}"]
        if c.validated_queries:
            parts.append(
                f"validated_queries: {', '.join(c.validated_queries)}"
            )
        else:
            parts.append("validated_queries: (none yet)")
        if c.gaps:
            parts.append(f"gaps: {', '.join(c.gaps)}")
        if c.slot_rejected_queries:
            parts.append(
                "already_rejected_in_this_slot: "
                + ", ".join(c.slot_rejected_queries)
            )
        if self.feedback:
            parts.append(f"feedback: {self.feedback}")
        return "\n".join(parts)


# ── Verify ─────────────────────────────────────────────────────────────


@dataclass
class Verify(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Judge a single query against the goal via ``topic_verifier``.

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
        agent = _build_agent(
            "topic_verifier", ctx.deps.model, ctx.deps.agent_overrides
        )
        result = await agent.run(
            f"research_goal: {ctx.state.query}\nquery: {self.query}",
            usage_limits=UsageLimits(request_limit=5),
        )
        verdict: VerifierOutput = result.output
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
            ctx.deps.reporter.slot_exhausted(
                slot=slot, new_target_count=c.target_count
            )
            if len(c.validated_queries) >= c.target_count:
                return ParallelSearch()
            ctx.deps.reporter.slot_starting(slot=slot + 1)
            return GenerateOne()

        return GenerateOne(feedback=verdict.reason)


# ── ParallelSearch ─────────────────────────────────────────────────────


@dataclass
class ParallelSearch(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Run the validated queries in parallel against the search backend.

    Pure orchestration — no LLM. Bounded concurrency via a semaphore.
    Errors per query are recorded in ``state.search_errors``; one failed
    query does not abort the others.
    """

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "FetchPhase":
        c = ctx.state.cycle
        if not c.validated_queries:
            logger.warning(
                "[%s] Search cycle produced no validated queries; "
                "ParallelSearch running with empty list "
                "(outer max_iterations will eventually terminate).",
                ctx.deps.correlation_id,
            )

        ctx.deps.reporter.parallel_search_starting(
            queries=list(c.validated_queries)
        )

        sem = asyncio.Semaphore(3)

        async def do_search(
            q: str,
        ) -> tuple[str, list[SearchResult], str | None]:
            async with sem:
                try:
                    results = await ctx.deps.backend.search(
                        q, max_results=ctx.deps.max_results_per_search
                    )
                    return (q, results, None)
                except Exception as e:
                    return (q, [], str(e))

        search_results = await asyncio.gather(
            *[do_search(q) for q in c.validated_queries]
        )

        cycle_reasoning = (
            f"Generated via per-slot generate/verify; "
            f"mode={c.mode}, validated={len(c.validated_queries)}/{c.target_count}"
        )

        for query_str, results, error_msg in search_results:
            ctx.deps.reporter.query_search_complete(
                query=query_str,
                n_results=len(results),
                error=error_msg,
            )
            if error_msg:
                ctx.state.search_errors.append(
                    {
                        "query": query_str,
                        "error": error_msg,
                        "is_retryable": "True",
                    }
                )
                logger.error(
                    "[%s] Search failed for %r: %s",
                    ctx.deps.correlation_id,
                    query_str,
                    error_msg,
                )

            query_obj = SearchQuery(
                query=query_str,
                reasoning=cycle_reasoning,
                iteration=ctx.state.iteration_count,
            )
            ctx.state.search_history.append(query_obj)
            ctx.state.all_results[query_str] = results
            ctx.state.total_searches += 1
            for r in results:
                ctx.state.searched_urls.add(r.url)

        ctx.deps.reporter.cycle_complete()
        return FetchPhase()


# ── FetchPhase ─────────────────────────────────────────────────────────


@dataclass
class FetchPhase(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Fetch relevant pages via page_fetcher agent."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "SummarizePages":
        ctx.state.current_phase = "fetch"

        if not ctx.state.all_results:
            return SummarizePages()

        # Flatten across queries, dedupe by URL, exclude already-fetched.
        # Multiple queries often surface the same URL — without this dedup,
        # the page_fetcher agent saw the same URL under several link_ids
        # and could pick it more than once, leading to duplicate fetches,
        # duplicate summaries, and the same source being listed 2-3 times
        # in the final report (regression observed on the Kalign query).
        # Cross-cycle, ``state.fetched_urls`` is a hard exclusion — not
        # just a text hint to the agent — so the fetcher cannot pick a
        # URL we already have content for.
        seen_urls: dict[str, SearchResult] = {}
        gid = 0
        for _query, results in ctx.state.all_results.items():
            for r in results:
                if r.url in ctx.state.fetched_urls:
                    continue  # already fetched in a previous cycle
                if r.url in seen_urls:
                    continue  # dupe within this cycle
                seen_urls[r.url] = SearchResult(
                    link_id=gid,
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    domain=r.domain,
                    relevance_score=r.relevance_score,
                )
                gid += 1

        all_search_results: list[SearchResult] = list(seen_urls.values())

        if not all_search_results:
            return SummarizePages()

        url_map = {r.link_id: r.url for r in all_search_results}
        ctx.state.url_map = url_map

        agent = _build_agent(
            "page_fetcher", ctx.deps.model, ctx.deps.agent_overrides
        )
        already_fetched = sorted(ctx.state.fetched_urls)

        prompt = f"""Research Question: {ctx.state.query}

Search Results to Evaluate ({len(all_search_results)}):
{chr(10).join(f"[{r.link_id}] {r.title} - {r.domain}" for r in all_search_results)}

Already Fetched ({len(already_fetched)}): {", ".join(already_fetched) if already_fetched else "None"}

Your budget: Maximum {ctx.deps.max_pages_to_fetch} pages.

Select the top {ctx.deps.max_pages_to_fetch} most relevant link IDs."""

        result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=5))
        fetch_plan: FetchPlan = result.output

        # Parallel fetch
        async def do_fetch(lid: int, url: str) -> tuple[str, int, str, FetchedPage | None, str | None]:
            try:
                page = await ctx.deps.backend.fetch_page(url)
                return ("success", lid, url, page, None)
            except Exception as e:
                return ("error", lid, url, None, str(e))

        # De-duplicate the agent's picks too — even after the URL-set
        # filter above, an LLM might pick the same link_id twice or
        # produce a list with repeats. Do not rely on agent obedience.
        seen_link_ids: set[int] = set()
        unique_picks = []
        for lid in fetch_plan.link_ids:
            if lid in url_map and lid not in seen_link_ids:
                seen_link_ids.add(lid)
                unique_picks.append((lid, url_map[lid]))
        tasks = unique_picks
        if tasks:
            ctx.deps.reporter.fetch_starting(n_pages=len(tasks))
            results = await asyncio.gather(*[do_fetch(lid, url) for lid, url in tasks])
            for status, lid, url, page, err in results:
                if status == "success" and page is not None:
                    # Final defence-in-depth: skip any page whose URL is
                    # already in fetched_urls (would only fire if state
                    # was mutated concurrently, but cheap to check).
                    if page.url in ctx.state.fetched_urls:
                        continue
                    ctx.state.fetched_pages.append(page)
                    ctx.state.total_pages_fetched += 1
                    ctx.state.fetched_urls.add(page.url)
                    ctx.deps.reporter.fetch_complete(
                        url=page.url,
                        success=True,
                        n_words=page.word_count,
                        error=None,
                    )
                elif err is not None:
                    ctx.state.fetch_errors.append({"url": url, "error": err, "link_id": str(lid)})
                    ctx.deps.reporter.fetch_complete(
                        url=url, success=False, n_words=0, error=err
                    )

        return SummarizePages()


# ── SummarizePages ─────────────────────────────────────────────────────


@dataclass
class SummarizePages(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Summarize each fetched page in parallel."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "AnalyzeGaps":
        ctx.state.current_phase = "summarize"

        if not ctx.state.fetched_pages:
            return AnalyzeGaps()

        agent = _build_agent(
            "page_summarizer", ctx.deps.model, ctx.deps.agent_overrides
        )

        async def summarize(page: FetchedPage) -> PageSummary:
            truncation_note = (
                f"\n[NOTE: Page was {page.original_length:,} chars; showing first 50,000 (truncated).]"
                if page.truncated
                else ""
            )
            prompt = f"""Question: {ctx.state.query}

Page Content ({page.word_count} words):
{page.content}{truncation_note}

Follow the process in your instructions: extract usable facts first,
then derive a relevance score from the scale."""
            try:
                result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=10))
                summary: PageSummary = result.output
                summary.url = page.url
                summary.title = page.title
                return summary
            except Exception as e:
                logger.warning(f"Failed to summarize {page.title[:50]}...: {e}")
                return PageSummary(
                    url=page.url,
                    title=page.title,
                    summary=f"Failed to summarize: {e}",
                    key_points=["Summarization failed"],
                    relevance_score=0.0,
                )

        ctx.deps.reporter.summarize_starting(n_pages=len(ctx.state.fetched_pages))
        summaries = await asyncio.gather(*[summarize(p) for p in ctx.state.fetched_pages])

        # Keep ALL summaries, sorted by relevance descending. Relevance is a
        # *sort key* (higher first), not a *gate* (drop low). Synthesize will
        # frame low-relevance results as 'limited evidence' rather than
        # silently discarding them — see the previous Kalign/competitors
        # failure mode where every summary scored <0.3 and the system bailed
        # with "no content summaries available" despite the pages containing
        # competitor data.
        sorted_summaries = sorted(
            summaries, key=lambda s: s.relevance_score, reverse=True
        )
        for s in sorted_summaries:
            ctx.deps.reporter.page_summarized(
                url=s.url, relevance=s.relevance_score, summary=s.summary
            )
        ctx.state.page_summaries = sorted_summaries
        return AnalyzeGaps()


# ── AnalyzeGaps ────────────────────────────────────────────────────────


@dataclass
class AnalyzeGaps(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Evaluate research completeness via gap_analyzer agent."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> Union["RefineSearch", "Synthesize"]:
        ctx.state.current_phase = "analyze"

        agent = _build_agent(
            "gap_analyzer", ctx.deps.model, ctx.deps.agent_overrides
        )

        evidence = [
            f"{s.title} (relevance {s.relevance_score:.2f}): {s.summary}"
            for s in ctx.state.page_summaries
        ]
        sources = [s.url for s in ctx.state.page_summaries]

        prompt = f"""Research Question: {ctx.state.query}

Evidence Gathered ({len(evidence)} items):
{chr(10).join(f"- {e}" for e in evidence)}

Sources Consulted ({len(sources)}):
{chr(10).join(f"- {s}" for s in sources)}

Evaluate:
1. Does this evidence comprehensively answer the research question?
2. What specific information is missing?
3. What targeted searches would fill the gaps?

If research is complete, explain why. If gaps exist, be specific about what's missing."""

        result = await agent.run(prompt)
        gap_analysis: GapAnalysis = result.output

        ctx.state.is_complete = gap_analysis.is_complete
        ctx.state.identified_gaps = gap_analysis.identified_gaps

        if gap_analysis.is_complete:
            return Synthesize()
        elif ctx.state.iteration_count >= ctx.state.max_iterations:
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
    """Final synthesis: lead agent creates comprehensive evidence report."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> End[EvidenceReport]:
        ctx.state.current_phase = "synthesize"

        # Empty only when zero pages were fetched at all (search/fetch
        # failures across the whole run). After the SummarizePages
        # filter-removal change, low-relevance summaries are NO LONGER
        # discarded — they're ranked and surfaced via the limited-evidence
        # path below.
        max_relevance = (
            max((s.relevance_score for s in ctx.state.page_summaries), default=0.0)
        )
        ctx.deps.reporter.synthesis_starting(
            n_summaries=len(ctx.state.page_summaries),
            max_relevance=max_relevance,
        )

        if not ctx.state.page_summaries:
            return End(
                EvidenceReport(
                    evidence_summary=(
                        f"Research on '{ctx.state.query}' could not gather "
                        f"evidence — no pages were fetched. "
                        f"Searches performed: {ctx.state.total_searches}; "
                        f"fetch failures: {len(ctx.state.fetch_errors)}."
                    ),
                    key_findings=["No pages were fetched"],
                    sources=["No sources"],
                    total_searches_performed=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iterations_required=ctx.state.iteration_count,
                )
            )

        agent = _build_agent(
            "lead_agent", ctx.deps.model, ctx.deps.agent_overrides
        )

        # Summaries are already sorted by relevance descending in
        # SummarizePages; preserve that order in the synthesis prompt.
        summaries_text = []
        for i, s in enumerate(ctx.state.page_summaries, 1):
            excerpts = ""
            if s.key_excerpts:
                excerpt_lines = "\n".join('  "' + e + '"' for e in s.key_excerpts)
                excerpts = "\n\nVerbatim Excerpts:\n" + excerpt_lines
            summaries_text.append(f"""
Source {i}: {s.title} ({s.url})
Relevance: {s.relevance_score:.2f}

Summary:
{s.summary}

Key Points:
{chr(10).join(f"  • {point}" for point in s.key_points)}{excerpts}
""")

        # Evidence-quality framing — tell the lead agent how confident the
        # underlying summarisation was, so it can frame the output
        # appropriately rather than over- or under-claiming.
        if max_relevance >= 0.6:
            quality_note = ""
        elif max_relevance >= 0.3:
            quality_note = (
                "\nEVIDENCE QUALITY: MODERATE. The page summaries are "
                "topically relevant but no single source directly answers "
                "the question. Synthesise carefully and flag any gaps.\n"
            )
        else:
            quality_note = (
                "\nEVIDENCE QUALITY: LIMITED. Every page summary scored "
                "below the relevance threshold (max relevance "
                f"{max_relevance:.2f}). The pages found may be tangential "
                "to the research question. Frame your output as 'partial "
                "findings' or 'limited evidence', acknowledge what the "
                "available pages actually cover, and explicitly note that "
                "the question was not directly answered by any source. "
                "Do NOT pad the answer with speculation — report only what "
                "the sources actually say.\n"
            )

        prompt = f"""Question: {ctx.state.query}
{quality_note}
Research Process:
- Iterations: {ctx.state.iteration_count}
- Total Searches: {ctx.state.total_searches}
- Pages Fetched: {ctx.state.total_pages_fetched}
- Pages Summarized: {len(ctx.state.page_summaries)}
- Max Relevance Score: {max_relevance:.2f}

Page Summaries (sorted by relevance, highest first):
{"".join(summaries_text)}

Synthesise these into an EvidenceReport. Apply the writing rules in your
instructions, with hedging calibrated to the max relevance score above.
Sources must contain UNIQUE URLs only — if the same URL appears in
several summaries, list it once."""

        try:
            result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=25))
            report: EvidenceReport = result.output

            if not report.sources or report.sources == ["No sources"]:
                report.sources = list(dict.fromkeys([s.url for s in ctx.state.page_summaries]))
            else:
                # Defence-in-depth: even if the agent populated sources, dedupe
                # — the synthesis prompt instructs unique URLs, but small models
                # sometimes repeat. Order is preserved (dict.fromkeys keeps
                # first-occurrence order).
                report.sources = list(dict.fromkeys(report.sources))

            report.total_searches_performed = ctx.state.total_searches
            report.total_pages_fetched = ctx.state.total_pages_fetched
            report.iterations_required = ctx.state.iteration_count

            return End(report)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            key_findings: list[str] = []
            for s in ctx.state.page_summaries[:10]:
                key_findings.extend(s.key_points[:2])

            return End(
                EvidenceReport(
                    evidence_summary=f'Research on "{ctx.state.query}" completed. Automatic synthesis failed: {e}',
                    key_findings=key_findings if key_findings else ["Synthesis failed"],
                    sources=list(dict.fromkeys([s.url for s in ctx.state.page_summaries])),
                    total_searches_performed=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iterations_required=ctx.state.iteration_count,
                )
            )
