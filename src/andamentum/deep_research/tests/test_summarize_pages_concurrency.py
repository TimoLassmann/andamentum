"""summarize_pages must not run concurrent LLM calls against a local (Ollama) model.

Ollama serialises inference per process, so overlapping requests only queue and risk
timeout cascades. Cloud models keep the parallel fan-out. Stub agents record how many
summarisation calls are in flight; no model, no network.
"""

from __future__ import annotations

import asyncio

from andamentum.deep_research.models import FetchedPage, PageSummary
from andamentum.deep_research.summarize_pages import summarize_pages


class OllamaModel:
    """Stands in for pydantic-ai's resolved ``OllamaModel`` (detected by class name, so no underscore prefix)."""


class _OpenAIModel:
    pass


class _Result:
    def __init__(self, output: PageSummary) -> None:
        self.output = output


class _CountingAgent:
    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def run(self, prompt: str, **_kwargs):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return _Result(
            PageSummary(
                url="", title="", summary="s", key_points=["k"], relevance_score=0.5
            )
        )


def _pages(n: int) -> list[FetchedPage]:
    return [
        FetchedPage(
            url=f"https://example.org/{i}",
            title=f"Page {i}",
            content="text",
            word_count=1,
            relevance_score=0.5,
            is_relevant=True,
        )
        for i in range(n)
    ]


async def test_local_model_summarises_one_page_at_a_time() -> None:
    agent = _CountingAgent()
    outcome = await summarize_pages(
        _pages(5),
        goal="g",
        model=OllamaModel(),
        overrides={"page_summarizer": agent},
    )
    assert len(outcome.summaries) == 5
    assert agent.peak == 1


async def test_cloud_model_keeps_parallel_fan_out() -> None:
    agent = _CountingAgent()
    await summarize_pages(
        _pages(5),
        goal="g",
        model=_OpenAIModel(),
        overrides={"page_summarizer": agent},
    )
    assert agent.peak > 1
