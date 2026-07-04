"""Worker: run the validated queries against the search backend in parallel.

Engine-free (L2): pure fan-out over the ``SearchBackend`` Port — no LLM.
Bounded concurrency via a semaphore (L5). One failed query does not
abort the others: its outcome carries the error string and an empty
result list (L7 soft failure — the ParallelSearch join records it in
State).
"""

from __future__ import annotations

import asyncio
import logging

from .backends import SearchBackend
from .models import SearchOutcome
from .reporter import NOOP_REPORTER, SearchReporter

logger = logging.getLogger(__name__)

# Concurrent-search bound for the fan-out semaphore (L5).
SEARCH_CONCURRENCY = 3


async def run_searches(
    queries: list[str],
    *,
    backend: SearchBackend,
    max_results: int,
    correlation_id: str = "",
    reporter: SearchReporter = NOOP_REPORTER,
) -> list[SearchOutcome]:
    """Search every query concurrently; outcomes preserve query order."""
    reporter.parallel_search_starting(queries=list(queries))

    sem = asyncio.Semaphore(SEARCH_CONCURRENCY)

    async def do_search(q: str) -> SearchOutcome:
        async with sem:
            try:
                results = await backend.search(q, max_results=max_results)
                return SearchOutcome(query=q, results=results)
            except Exception as e:
                return SearchOutcome(query=q, results=[], error=str(e))

    outcomes = list(await asyncio.gather(*[do_search(q) for q in queries]))

    for outcome in outcomes:
        reporter.query_search_complete(
            query=outcome.query,
            n_results=len(outcome.results),
            error=outcome.error,
        )
        if outcome.error:
            logger.error(
                "[%s] Search failed for %r: %s",
                correlation_id,
                outcome.query,
                outcome.error,
            )
    return outcomes
