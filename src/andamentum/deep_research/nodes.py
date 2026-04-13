"""Graph nodes for research workflow orchestration.

These nodes implement the research cycle: plan → search → fetch → summarize →
analyze gaps → (refine | synthesize). They use the SearchBackend protocol for
search/fetch operations and pydantic-ai agents for LLM decisions.

Requires the [llm] optional extra: ``pip install andamentum[llm]``
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Union, Any

from pydantic_graph import BaseNode, GraphRunContext, End
from pydantic_ai.usage import UsageLimits
from pydantic_ai import Agent

from .state import ResearchState
from .models import (
    SearchQuery,
    SearchResult,
    SearchPlan,
    FetchPlan,
    PageSummary,
    FetchedPage,
    GapAnalysis,
    EvidenceReport,
)
from .text_utils import guard_queries_against_drift
from .agents import get_agent
from .backends import SearchBackend

logger = logging.getLogger(__name__)


def _build_agent(name: str, model: Any) -> Agent[Any, Any]:
    """Create a pydantic-ai Agent from a registry definition."""
    defn = get_agent(name)
    return Agent(
        model,
        output_type=defn.output_model,
        instructions=defn.prompt,
        retries=defn.retries,
        output_retries=defn.output_retries,
    )


# ── Node Deps ──────────────────────────────────────────────────────────


@dataclass
class NodeDeps:
    """Dependencies available to graph nodes."""

    backend: SearchBackend
    model: Any  # pydantic-ai model instance
    correlation_id: str = ""
    max_pages_to_fetch: int = 5
    max_results_per_search: int = 10


# ── PlanResearch ───────────────────────────────────────────────────────


@dataclass
class PlanResearch(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Entry node: initialize research."""

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "SearchPhase":
        ctx.state.current_phase = "plan"
        ctx.state.iteration_count = 0
        return SearchPhase()


# ── SearchPhase ────────────────────────────────────────────────────────


@dataclass
class SearchPhase(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Execute search queries via search_planner agent."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> Union["FetchPhase", End[EvidenceReport]]:
        ctx.state.current_phase = "search"
        ctx.state.iteration_count += 1

        # Guard: max iterations
        if ctx.state.iteration_count > ctx.state.max_iterations:
            sources = [
                f"{p.title} - {p.url}" for p in ctx.state.fetched_pages if p.is_relevant
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

        agent = _build_agent("search_planner", ctx.deps.model)

        previous_queries = [q.query for q in ctx.state.search_history]
        target_gaps = ctx.state.identified_gaps if ctx.state.iteration_count > 1 else []

        prompt_parts = [
            f"Research Question: {ctx.state.query}",
            f"Previous Queries ({len(previous_queries)}): {', '.join(previous_queries) if previous_queries else 'None'}",
        ]
        if target_gaps:
            prompt_parts.append(f"Target Gaps: {', '.join(target_gaps)}")
            prompt_parts.append(
                "Generate 1-2 queries specifically targeting these gaps."
            )
        else:
            prompt_parts.append(
                "Generate 2-3 diverse initial search queries covering different aspects."
            )

        result = await agent.run(
            "\n".join(prompt_parts),
            usage_limits=UsageLimits(request_limit=15),
        )
        search_plan: SearchPlan = result.output

        # Topic guard
        search_plan.queries = guard_queries_against_drift(
            queries=search_plan.queries,
            objective=ctx.state.query,
        )

        # Parallel search
        sem = asyncio.Semaphore(3)

        async def do_search(q: str) -> tuple[str, list[SearchResult], str | None]:
            async with sem:
                try:
                    results = await ctx.deps.backend.search(
                        q, max_results=ctx.deps.max_results_per_search
                    )
                    return (q, results, None)
                except Exception as e:
                    return (q, [], str(e))

        search_results = await asyncio.gather(
            *[do_search(q) for q in search_plan.queries]
        )

        for query_str, results, error_msg in search_results:
            if error_msg:
                ctx.state.search_errors.append(
                    {"query": query_str, "error": error_msg, "is_retryable": "True"}
                )
                logger.error(
                    f"[{ctx.deps.correlation_id}] Search failed for '{query_str}': {error_msg}"
                )

            query_obj = SearchQuery(
                query=query_str,
                reasoning=search_plan.reasoning,
                iteration=ctx.state.iteration_count,
            )
            ctx.state.search_history.append(query_obj)
            ctx.state.all_results[query_str] = results
            ctx.state.total_searches += 1
            for r in results:
                ctx.state.searched_urls.add(r.url)

        return FetchPhase()


# ── FetchPhase ─────────────────────────────────────────────────────────


@dataclass
class FetchPhase(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Fetch relevant pages via page_fetcher agent."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> "SummarizePages":
        ctx.state.current_phase = "fetch"

        if not ctx.state.all_results:
            return SummarizePages()

        # Flatten and renumber link_ids globally
        all_search_results: list[SearchResult] = []
        gid = 0
        for _query, results in ctx.state.all_results.items():
            for r in results:
                all_search_results.append(
                    SearchResult(
                        link_id=gid,
                        title=r.title,
                        url=r.url,
                        snippet=r.snippet,
                        domain=r.domain,
                        relevance_score=r.relevance_score,
                    )
                )
                gid += 1

        if not all_search_results:
            return SummarizePages()

        url_map = {r.link_id: r.url for r in all_search_results}
        ctx.state.url_map = url_map

        agent = _build_agent("page_fetcher", ctx.deps.model)
        already_fetched = [p.url for p in ctx.state.fetched_pages]

        prompt = f"""Research Question: {ctx.state.query}

Search Results to Evaluate ({len(all_search_results)}):
{chr(10).join(f"[{r.link_id}] {r.title} - {r.domain}" for r in all_search_results)}

Already Fetched ({len(already_fetched)}): {", ".join(already_fetched) if already_fetched else "None"}

Your budget: Maximum {ctx.deps.max_pages_to_fetch} pages.

Select the top {ctx.deps.max_pages_to_fetch} most relevant link IDs."""

        result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=5))
        fetch_plan: FetchPlan = result.output

        # Parallel fetch
        async def do_fetch(
            lid: int, url: str
        ) -> tuple[str, int, str, FetchedPage | None, str | None]:
            try:
                page = await ctx.deps.backend.fetch_page(url)
                return ("success", lid, url, page, None)
            except Exception as e:
                return ("error", lid, url, None, str(e))

        tasks = [(lid, url_map[lid]) for lid in fetch_plan.link_ids if lid in url_map]
        if tasks:
            results = await asyncio.gather(*[do_fetch(lid, url) for lid, url in tasks])
            for status, lid, url, page, err in results:
                if status == "success" and page is not None:
                    ctx.state.fetched_pages.append(page)
                    ctx.state.total_pages_fetched += 1
                    ctx.state.fetched_urls.add(page.url)
                elif err is not None:
                    ctx.state.fetch_errors.append(
                        {"url": url, "error": err, "link_id": str(lid)}
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

        agent = _build_agent("page_summarizer", ctx.deps.model)

        async def summarize(page: FetchedPage) -> PageSummary:
            prompt = f"""Research Question: {ctx.state.query}

Page Content ({page.word_count} words):
{page.content[:20000]}{"..." if len(page.content) > 20000 else ""}

Extract ONLY information that DIRECTLY answers or informs the research question.
Create a 200-word summary, identify 3-5 key points, and include 1-3 verbatim quotes from the page.
If this page only mentions the research topic in passing, set relevance_score below 0.3."""
            try:
                result = await agent.run(
                    prompt, usage_limits=UsageLimits(request_limit=10)
                )
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

        summaries = await asyncio.gather(
            *[summarize(p) for p in ctx.state.fetched_pages]
        )
        ctx.state.page_summaries = [s for s in summaries if s.relevance_score > 0.3]
        return AnalyzeGaps()


# ── AnalyzeGaps ────────────────────────────────────────────────────────


@dataclass
class AnalyzeGaps(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Evaluate research completeness via gap_analyzer agent."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> Union["RefineSearch", "Synthesize"]:
        ctx.state.current_phase = "analyze"

        agent = _build_agent("gap_analyzer", ctx.deps.model)

        evidence = [
            f"{p.title}: {p.content[:2000]}..."
            if len(p.content) > 2000
            else f"{p.title}: {p.content}"
            for p in ctx.state.fetched_pages
        ]
        sources = [p.url for p in ctx.state.fetched_pages]

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

    async def run(self, ctx: GraphRunContext[ResearchState, NodeDeps]) -> "SearchPhase":
        ctx.state.current_phase = "refine"
        return SearchPhase()


# ── Synthesize ─────────────────────────────────────────────────────────


@dataclass
class Synthesize(BaseNode[ResearchState, NodeDeps, EvidenceReport]):
    """Final synthesis: lead agent creates comprehensive evidence report."""

    async def run(
        self, ctx: GraphRunContext[ResearchState, NodeDeps]
    ) -> End[EvidenceReport]:
        ctx.state.current_phase = "synthesize"

        if not ctx.state.page_summaries:
            return End(
                EvidenceReport(
                    evidence_summary=f"Research on '{ctx.state.query}' incomplete - no content summaries available.",
                    key_findings=["No summaries generated"],
                    sources=[p.url for p in ctx.state.fetched_pages]
                    if ctx.state.fetched_pages
                    else ["No sources"],
                    total_searches_performed=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iterations_required=ctx.state.iteration_count,
                )
            )

        agent = _build_agent("lead_agent", ctx.deps.model)

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

        prompt = f"""Research Question: {ctx.state.query}

Research Process:
- Iterations: {ctx.state.iteration_count}
- Total Searches: {ctx.state.total_searches}
- Pages Fetched: {ctx.state.total_pages_fetched}
- Pages Summarized: {len(ctx.state.page_summaries)}

Page Summaries:
{"".join(summaries_text)}

Your Task:
Create a comprehensive EvidenceReport that synthesizes these summaries into:

1. Evidence Summary (2-3 paragraphs):
   - What did the research reveal?
   - What are the main themes and findings?
   - How comprehensive is the evidence?

2. Key Findings (5-10 bullet points):
   - Each finding MUST be substantiated by a specific claim from at least one source
   - Include specific details: numbers, dates, names, statistics
   - Do NOT promote passing mentions or general background into findings
   - If something is only mentioned in a list or sidebar but not discussed in detail, it is NOT a finding

3. Sources:
   - List all {len(ctx.state.page_summaries)} source URLs

CRITICAL: Only report what the sources DIRECTLY state. Do not inflate passing mentions into findings.
If a company or product is just listed alongside others but nothing specific is reported about it, leave it out."""

        try:
            result = await agent.run(prompt, usage_limits=UsageLimits(request_limit=25))
            report: EvidenceReport = result.output

            if not report.sources or report.sources == ["No sources"]:
                report.sources = list(
                    dict.fromkeys([s.url for s in ctx.state.page_summaries])
                )

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
                    sources=list(
                        dict.fromkeys([s.url for s in ctx.state.page_summaries])
                    ),
                    total_searches_performed=ctx.state.total_searches,
                    total_pages_fetched=ctx.state.total_pages_fetched,
                    iterations_required=ctx.state.iteration_count,
                )
            )
