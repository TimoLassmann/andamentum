"""Standalone research orchestrator.

Standalone deep research entry point. Uses the package's own
graph, agents, and backends.

Usage::

    from andamentum.deep_research.orchestrator import run_research

    result = await run_research("What is quantum computing?", model="anthropic:claude-haiku-4-5")
    print(result.output.evidence_summary)
"""

from __future__ import annotations

import uuid
from typing import Any

from andamentum.core.decisions import DecisionModel

from .models import ResearchResult, ResearchErrors


async def run_research(
    query: str,
    *,
    max_iterations: int = 3,
    model: str,
    searxng_url: str = "http://127.0.0.1:4070",
    max_results: int = 10,
    max_pages: int = 5,
    backend: Any = None,  # SearchBackend — typed loosely to avoid import at module level
    verbose: bool = False,
    reporter: Any = None,  # SearchReporter — typed loosely to avoid mandatory import
    tdm_allowed_hosts: frozenset[str] = frozenset(),
    decision_model: DecisionModel | None = None,
) -> "ResearchResult":
    """Run a complete research session.

    Args:
        query: Research question
        max_iterations: Maximum search-analyze cycles (1-5)
        model: pydantic-ai model string (e.g. "anthropic:claude-haiku-4-5", "openai:gpt-4o")
        searxng_url: SearXNG instance URL for default backend
        max_results: Max search results per query
        max_pages: Max pages to fetch per iteration
        backend: Optional SearchBackend override (when supplied, ``tdm_allowed_hosts``
            does not apply — the caller-provided backend is responsible for its own gating)
        verbose: Print progress
        tdm_allowed_hosts: Hostnames the caller attests they hold a TDM licence
            for. Disarms the paywalled-publisher tripwire for those hosts. Default
            empty: tripwire fully active. Only consulted when ``backend`` is None.
        decision_model: Optional opened local decision model
            (``DecisionModel.open``) that judges candidate queries instead of
            the ``topic_verifier`` agent. Default None: the agent judges.

    Returns:
        ResearchResult with output, iterations, searches, pages_fetched, verification, errors
    """
    from .backends import HttpxSearchBackend
    from .graph import research_graph
    from .nodes import NodeDeps
    from .reporter import NoopReporter
    from .runner import _resolve_model
    from .state import ResearchState
    from .verification import verify_sources

    # Set up backend
    owns_backend = backend is None
    if owns_backend:
        backend = HttpxSearchBackend(
            searxng_url=searxng_url,
            tdm_allowed_hosts=tdm_allowed_hosts,
        )

    # Resolve model (provides localhost default for ollama: prefix)
    model_instance = _resolve_model(model)

    correlation_id = uuid.uuid4().hex[:8]

    state = ResearchState(query=query)

    # Config lives on Deps (L1: given → Deps), not State.
    deps = NodeDeps(
        backend=backend,
        model=model_instance,
        correlation_id=correlation_id,
        max_iterations=max_iterations,
        max_pages_to_fetch=max_pages,
        max_results_per_search=max_results,
        reporter=reporter if reporter is not None else NoopReporter(),
        decision_model=decision_model,
    )

    try:
        # Run the graph. The entry node is constructed by the graph's own
        # entry step, so no start node is passed here.
        output = await research_graph.run(
            state=state,
            deps=deps,
        )
    finally:
        if owns_backend:
            await backend.close()

    # Source verification
    verification = verify_sources(
        cited_sources=output.sources if output else [],
        searched_urls=state.searched_urls,
        fetched_urls=state.fetched_urls,
    )

    return ResearchResult(
        output=output,
        page_summaries=state.page_summaries,
        fetched_pages=state.fetched_pages,
        iterations=state.iteration_count,
        searches=state.total_searches,
        pages_fetched=state.total_pages_fetched,
        verification=verification,
        errors=ResearchErrors(
            search_errors=len(state.search_errors),
            fetch_errors=len(state.fetch_errors),
            summary_errors=len(state.summary_errors),
        ),
    )
