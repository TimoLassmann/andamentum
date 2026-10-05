"""Worker: summarize the given pages via ``page_summarizer``, in parallel for cloud
models and strictly one at a time for a local (Ollama) model.

Engine-free (L2). A per-page summarization failure is a soft failure (L7)
that is recorded, not disguised: the page comes back as a
:class:`SummaryError` in the outcome and contributes no summary, so no
placeholder text is ever read downstream as evidence. The caller passes
only pages it has not summarised yet.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai.usage import UsageLimits

from andamentum.core.agents import gather_llm, llm_concurrency

from .build_agent import AgentOverrides, build_agent
from .models import FetchedPage, PageSummary, SummaryError
from .reporter import NOOP_REPORTER, SearchReporter

logger = logging.getLogger(__name__)

# In-agent (client-level) retry ceiling per page-summary call.
SUMMARIZER_REQUEST_LIMIT = 10


class SummarizeOutcome(BaseModel):
    """What one ``summarize_pages`` call produced."""

    summaries: list[PageSummary] = Field(default_factory=list)
    errors: list[SummaryError] = Field(default_factory=list)


async def summarize_pages(
    pages: list[FetchedPage],
    *,
    goal: str,
    model: Any,
    overrides: AgentOverrides | None = None,
    reporter: SearchReporter = NOOP_REPORTER,
) -> SummarizeOutcome:
    """Summarize ``pages``; summaries come back sorted by relevance.

    Keeps ALL summaries, sorted by relevance descending. Relevance is a
    *sort key* (higher first), not a *gate* (drop low). Synthesis frames
    low-relevance results as 'limited evidence' rather than silently
    discarding them — see the previous Kalign/competitors failure mode
    where every summary scored <0.3 and the system bailed with "no
    content summaries available" despite the pages containing competitor
    data.
    """
    if not pages:
        return SummarizeOutcome()

    agent = build_agent("page_summarizer", model, overrides)

    async def summarize(page: FetchedPage) -> PageSummary | SummaryError:
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
            return SummaryError(url=page.url, error=f"{type(e).__name__}: {e}")

    reporter.summarize_starting(n_pages=len(pages))
    results = await gather_llm(
        [summarize(p) for p in pages], concurrency=llm_concurrency(model)
    )

    summaries = [r for r in results if isinstance(r, PageSummary)]
    errors = [r for r in results if isinstance(r, SummaryError)]
    sorted_summaries = sorted(summaries, key=lambda s: s.relevance_score, reverse=True)
    for s in sorted_summaries:
        reporter.page_summarized(
            url=s.url, relevance=s.relevance_score, summary=s.summary
        )
    return SummarizeOutcome(summaries=sorted_summaries, errors=errors)
