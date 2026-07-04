"""Worker: select and fetch relevant pages for the current cycle.

Engine-free (L2). Three sub-steps sharing one invariant (the
``link_id`` → URL map), so they live in one module (a deep module, not
three shallow ones): dedupe the cycle's search results into unique fetch
candidates, ask the ``page_fetcher`` agent to pick within budget, then
fetch the picks in parallel. One failed fetch does not abort the others —
it lands in ``FetchOutcome.errors`` (L7 soft failure; the FetchPhase
join records it in State).
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic_ai.usage import UsageLimits

from .backends import SearchBackend
from .build_agent import AgentOverrides, build_agent
from .models import FetchedPage, FetchError, FetchOutcome, FetchPlan, SearchResult
from .reporter import NOOP_REPORTER, SearchReporter

# In-agent (client-level) retry ceiling for the page-selection call.
FETCHER_REQUEST_LIMIT = 5


async def fetch_pages(
    *,
    goal: str,
    search_results: dict[str, list[SearchResult]],
    fetched_urls: set[str],
    failed_urls: set[str],
    max_pages: int,
    backend: SearchBackend,
    model: Any,
    overrides: AgentOverrides | None = None,
    reporter: SearchReporter = NOOP_REPORTER,
) -> FetchOutcome:
    """Dedupe candidates, let the fetcher agent pick, fetch in parallel.

    Returns an empty ``FetchOutcome`` (without calling the agent) when
    nothing is left to evaluate after the dedup filters.
    """
    # Flatten across queries, dedupe by URL, exclude both
    # already-fetched (``fetched_urls``) and already-failed
    # (``failed_urls``). Multiple queries often surface the same
    # URL — without dedup, the page_fetcher agent saw the same URL
    # under several link_ids and could pick it more than once.
    # Without failure exclusion, an authoritative-looking URL that
    # 403'd in cycle 1 would keep getting re-picked in cycles 2/3
    # (observed: OECD, oxfordeconomics, on a multi-cycle run). We
    # treat every prior failure as session-permanent: within a 1-3
    # minute research run, retrying the same URL almost never
    # changes the outcome, and we have other candidates to fall back
    # on.
    seen_urls: dict[str, SearchResult] = {}
    gid = 0
    for _query, results in search_results.items():
        for r in results:
            if r.url in fetched_urls:
                continue  # already fetched in a previous cycle
            if r.url in failed_urls:
                continue  # failed in a previous cycle (don't retry)
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

    candidates: list[SearchResult] = list(seen_urls.values())
    if not candidates:
        return FetchOutcome()

    url_map = {r.link_id: r.url for r in candidates}

    agent = build_agent("page_fetcher", model, overrides)
    already_fetched = sorted(fetched_urls)

    prompt = f"""Research Question: {goal}

Search Results to Evaluate ({len(candidates)}):
{chr(10).join(f"[{r.link_id}] {r.title} - {r.domain}" for r in candidates)}

Already Fetched ({len(already_fetched)}): {", ".join(already_fetched) if already_fetched else "None"}

Your budget: Maximum {max_pages} pages.

Select the top {max_pages} most relevant link IDs."""

    result = await agent.run(
        prompt, usage_limits=UsageLimits(request_limit=FETCHER_REQUEST_LIMIT)
    )
    fetch_plan: FetchPlan = result.output

    # De-duplicate the agent's picks too — even after the URL-set
    # filter above, an LLM might pick the same link_id twice or
    # produce a list with repeats. Do not rely on agent obedience.
    seen_link_ids: set[int] = set()
    picks: list[tuple[int, str]] = []
    for lid in fetch_plan.link_ids:
        if lid in url_map and lid not in seen_link_ids:
            seen_link_ids.add(lid)
            picks.append((lid, url_map[lid]))

    pages: list[FetchedPage] = []
    errors: list[FetchError] = []
    if picks:
        reporter.fetch_starting(n_pages=len(picks))

        async def do_fetch(
            lid: int, url: str
        ) -> tuple[int, str, FetchedPage | None, str | None]:
            try:
                page = await backend.fetch_page(url)
                return (lid, url, page, None)
            except Exception as e:
                return (lid, url, None, str(e))

        results = await asyncio.gather(*[do_fetch(lid, url) for lid, url in picks])
        for lid, url, page, err in results:
            if page is not None:
                pages.append(page)
                reporter.fetch_complete(
                    url=page.url,
                    success=True,
                    n_words=page.word_count,
                    error=None,
                )
            elif err is not None:
                errors.append(FetchError(url=url, error=err, link_id=lid))
                reporter.fetch_complete(url=url, success=False, n_words=0, error=err)

    return FetchOutcome(url_map=url_map, pages=pages, errors=errors)
