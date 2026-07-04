"""Worker: summarize every fetched page in parallel via ``page_summarizer``.

Engine-free (L2). Per-page summarization failure is the one allowed soft
failure (L7): the page gets a zero-relevance placeholder summary and the
run continues.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic_ai.usage import UsageLimits

from .build_agent import AgentOverrides, build_agent
from .models import FetchedPage, PageSummary
from .reporter import NOOP_REPORTER, SearchReporter

logger = logging.getLogger(__name__)

# In-agent (client-level) retry ceiling per page-summary call.
SUMMARIZER_REQUEST_LIMIT = 10


async def summarize_pages(
    pages: list[FetchedPage],
    *,
    goal: str,
    model: Any,
    overrides: AgentOverrides | None = None,
    reporter: SearchReporter = NOOP_REPORTER,
) -> list[PageSummary]:
    """Summarize ``pages`` concurrently; returns summaries sorted by relevance.

    Keeps ALL summaries, sorted by relevance descending. Relevance is a
    *sort key* (higher first), not a *gate* (drop low). Synthesis frames
    low-relevance results as 'limited evidence' rather than silently
    discarding them — see the previous Kalign/competitors failure mode
    where every summary scored <0.3 and the system bailed with "no
    content summaries available" despite the pages containing competitor
    data.
    """
    if not pages:
        return []

    agent = build_agent("page_summarizer", model, overrides)

    async def summarize(page: FetchedPage) -> PageSummary:
        truncation_note = (
            f"\n[NOTE: Page was {page.original_length:,} chars; showing first 50,000 (truncated).]"
            if page.truncated
            else ""
        )
        prompt = f"""Question: {goal}

Page Content ({page.word_count} words):
{page.content}{truncation_note}

Follow the process in your instructions: extract usable facts first,
then derive a relevance score from the scale."""
        try:
            result = await agent.run(
                prompt, usage_limits=UsageLimits(request_limit=SUMMARIZER_REQUEST_LIMIT)
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

    reporter.summarize_starting(n_pages=len(pages))
    summaries = await asyncio.gather(*[summarize(p) for p in pages])

    sorted_summaries = sorted(summaries, key=lambda s: s.relevance_score, reverse=True)
    for s in sorted_summaries:
        reporter.page_summarized(
            url=s.url, relevance=s.relevance_score, summary=s.summary
        )
    return sorted_summaries
