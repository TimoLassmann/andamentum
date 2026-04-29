"""One-shot URL → structured summary.

Public entry point: :func:`run_fetch`. Composes the unified
``andamentum.harvest.extract`` pipeline (URL fetch + format detection +
extraction via trafilatura/Docling/etc.) with the
``general_page_summarizer`` agent.

Distinct from :func:`run_research` (which performs multi-iteration
search + fetch + synthesis against a research question) — ``run_fetch``
takes one URL and returns one structured summary.
"""

from __future__ import annotations

from .models import FetchSummary


async def run_fetch(url: str, *, model: str) -> FetchSummary:
    """Fetch a URL, extract its content, and return a structured summary.

    Args:
        url: The URL to fetch and summarise.
        model: pydantic-ai model identifier
            (e.g. ``"anthropic:claude-haiku-4-5"``).

    Returns:
        :class:`FetchSummary` with url, title, summary, and key points.

    Raises:
        andamentum.harvest.HarvestError: If the URL cannot be fetched
            or content cannot be extracted.
    """
    from andamentum.harvest import extract
    from andamentum.core.agents import build_pydantic_ai_agent

    from .agents import get_agent

    markdown = await extract(url)
    title = _first_heading(markdown)

    prompt = (
        f"PAGE URL: {url}\n\n"
        f"PAGE CONTENT:\n{markdown}\n\n"
        "Produce a faithful structured summary of this page."
    )

    agent = build_pydantic_ai_agent(get_agent("general_page_summarizer"), model)
    result = await agent.run(prompt)
    summary: FetchSummary = result.output

    # Authoritative url + title from the fetch — agents sometimes leave
    # these blank or hallucinate them from the content.
    summary.url = url
    if not summary.title.strip():
        summary.title = title or url
    return summary


def _first_heading(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""
