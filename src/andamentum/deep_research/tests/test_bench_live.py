"""Checks that need the network. Deselected by default
(``uv run pytest -m research_eval``); they use no model."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.research_eval


async def test_an_any_term_arxiv_query_and_a_keyless_semantic_scholar_search_find_works() -> (
    None
):
    """Round 3's two new search routes against the live services, one request each:
    arXiv with the words joined by OR (no key exists for it), Semantic Scholar with no
    key. Semantic Scholar's shared keyless pool may answer 429 (the freeze waits and
    gives up; here that is a skip, not a pass)."""
    from andamentum.deep_research.sources import (
        ArxivSource,
        SemanticScholarSource,
        SourceClient,
        SourceError,
    )
    from andamentum.deep_research.store import ResearchStore

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
