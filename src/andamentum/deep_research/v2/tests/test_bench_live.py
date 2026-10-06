"""Benchmark checks that need the network. Deselected by default
(``uv run pytest -m research_eval``); they use no model."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from andamentum.deep_research.v2.bench.download import Downloader
from andamentum.deep_research.v2.bench.litsearch import (
    download_litsearch,
    load_litsearch,
)
from andamentum.deep_research.v2.bench.web_set import fetch_page, verify_sources

pytestmark = pytest.mark.research_eval

PAUSE_S = 1.0


async def test_every_web_question_source_still_states_its_answer() -> None:
    async def fetch(url: str) -> str:
        text = await fetch_page(url)
        await asyncio.sleep(PAUSE_S)
        return text

    checks = await verify_sources(fetch)
    assert [c for c in checks if not c.states_answer] == []


async def test_litsearch_downloads_and_loads_a_three_query_subset(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        await download_litsearch(tmp_path, downloader=Downloader(client), n=3, seed=0)
    data = load_litsearch(tmp_path, n=3, seed=0)
    assert data.n_queries_total == 597
    assert len(data.items) + data.n_dropped_queries == 3
    assert all(item.gold for item in data.items)


async def test_an_any_term_arxiv_query_and_a_keyless_semantic_scholar_search_find_works() -> (
    None
):
    """Round 3's two new search routes against the live services, one request each:
    arXiv with the words joined by OR (no key exists for it), Semantic Scholar with no
    key. Semantic Scholar's shared keyless pool may answer 429 (the freeze waits and
    gives up; here that is a skip, not a pass)."""
    from andamentum.deep_research.v2.sources import (
        ArxivSource,
        SemanticScholarSource,
        SourceClient,
        SourceError,
    )
    from andamentum.deep_research.v2.store import ResearchStore

    with ResearchStore.in_memory() as store:
        async with SourceClient(store) as client:
            arxiv = ArxivSource(client=client, query_form="any_term")
            found = await arxiv.search(
                "conformal risk control for language model outputs", limit=5
            )
            assert found and all(c.work.arxiv_id for c in found)
            assert " OR " in next(iter(store.iter_cache())).url
            try:
                scholar = await SemanticScholarSource(client=client).search(
                    "conformal risk control", limit=5
                )
            except SourceError as error:
                if error.kind == "rate_limited":
                    pytest.skip(f"keyless Semantic Scholar is rate limited: {error}")
                raise
            assert scholar and all(not c.work.redistributable for c in scholar)
