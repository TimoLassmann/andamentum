"""Standalone research orchestrator.

Standalone deep research entry point. Uses the package's own
graph, agents, and backends.

Requires the [llm] optional extra: ``pip install andamentum[llm]``

Usage::

    from andamentum.deep_research.orchestrator import run_research

    result = await run_research("What is quantum computing?", model="bedrock:claude-haiku-4-5")
    print(result.output.evidence_summary)
"""

from __future__ import annotations

import uuid
from typing import Any

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
) -> "ResearchResult":
    """Run a complete research session.

    Args:
        query: Research question
        max_iterations: Maximum search-analyze cycles (1-5)
        model: pydantic-ai model string (e.g. "anthropic:claude-haiku-4-5", "openai:gpt-4o")
        searxng_url: SearXNG instance URL for default backend
        max_results: Max search results per query
        max_pages: Max pages to fetch per iteration
        backend: Optional SearchBackend override
        verbose: Print progress

    Returns:
        ResearchResult with output, iterations, searches, pages_fetched, verification, errors
    """
    # Load .env from CWD so importing repos get their API keys
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from .backends import HttpxSearchBackend
    from .graph import research_graph
    from .nodes import PlanResearch, NodeDeps
    from .runner import _resolve_model
    from .state import ResearchState
    from .verification import verify_sources

    # Auto-start SearXNG if using localhost and no backend provided
    if backend is None and searxng_url.startswith(
        ("http://127.0.0.1:", "http://localhost:")
    ):
        try:
            from .searxng import SearxngManager

            import re as _re

            port_match = _re.search(r":(\d+)", searxng_url)
            port = int(port_match.group(1)) if port_match else 4070
            manager = SearxngManager(host_port=port)
            if not manager.is_running():
                if verbose:
                    print(f"Starting SearXNG on port {port}...")
                manager.ensure_running()
                if verbose:
                    print(f"SearXNG started at {searxng_url}")
        except Exception as e:
            if verbose:
                print(f"Could not auto-start SearXNG: {e}")

    # Set up backend
    owns_backend = backend is None
    if owns_backend:
        backend = HttpxSearchBackend(searxng_url=searxng_url)

    # Resolve model (provides localhost default for ollama: prefix)
    model_instance = _resolve_model(model)

    correlation_id = uuid.uuid4().hex[:8]

    state = ResearchState(
        query=query,
        max_iterations=max_iterations,
    )

    deps = NodeDeps(
        backend=backend,
        model=model_instance,
        correlation_id=correlation_id,
        max_pages_to_fetch=max_pages,
        max_results_per_search=max_results,
    )

    if verbose:
        print(f"Starting research: {query}")
        print(f"Model: {model}, Max iterations: {max_iterations}")

    try:
        # Run the graph
        result = await research_graph.run(
            PlanResearch(),
            state=state,
            deps=deps,
        )
    finally:
        if owns_backend:
            await backend.close()

    output = result.output

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
        ),
    )
