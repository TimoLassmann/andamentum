"""Tests for how documents still in the ingestion queue surface in search.

Three properties, in tension, are what these pin down:

1. A ``pending_source`` row holds no markdown — only a title derived from the
   filename. It must never appear in results: on a title-shaped query BM25
   ranks a zero-length document top, and the caller gets a confident-looking
   hit with nothing behind it.
2. A ``pending_enrich`` document holds real markdown and *must* stay findable,
   but it can only ever match the keyword signal. Fusing it with documents that
   also carry chunk embeddings, a document embedding and a cluster reads its
   missing index as three rejections, so it is appended below the fused page
   instead — within ``limit``, and labelled.
3. A database with an empty queue must rank exactly as it did before any of
   this existed. The tail costs nothing when there is nothing to put in it.

These tests deliberately avoid embeddings: with no query embedding only the
keyword signals run, which is precisely the path a queued document travels.
"""

from __future__ import annotations

import pytest

from andamentum.document_store import public
from andamentum.document_store import queue as q
from andamentum.document_store.api import DocumentStore
from andamentum.document_store.search import (
    _pending_quota,
    search_fts5,
    search_pending_keyword,
    search_unified,
)

BODY = "Mitochondrial dysfunction in skeletal muscle tissue. " * 40


@pytest.fixture()
async def store(tmp_path):
    s = DocumentStore(database_name="testvis", db_dir=str(tmp_path))
    await s.initialize()
    return s


async def _add(store, title: str, content: str, status: str) -> str:
    """Register a document in ``status``.

    ``title`` is folded into the content because ``register_document`` dedups
    by content hash — a corpus of byte-identical bodies would collapse to a
    single row and quietly hollow out every assertion below.
    """
    doc_id = await store.register_document(
        title=title,
        content=f"{content} Distinguishing marker: {title}.",
        metadata={"title": title},
    )
    await q.set_ingest_status(str(store.db_path), doc_id, status)  # type: ignore[arg-type]
    return doc_id


async def _populate(store, *, n_complete: int = 30) -> dict[str, str]:
    """A corpus of fully-indexed documents plus one of each queue state."""
    ids: dict[str, str] = {}
    for i in range(n_complete):
        doc_id = await _add(
            store, f"Real paper {i} on mitochondrial biology", BODY, q.COMPLETE
        )
        ids[f"complete-{i}"] = doc_id
    ids["pending_enrich"] = await _add(
        store, "Deferred mitochondrial draft", BODY, q.PENDING_ENRICH
    )
    ids["failed"] = await _add(store, "Broken mitochondrial import", BODY, q.FAILED)
    ids["pending_source"] = await q.register_pending_source(
        str(store.db_path), "https://example.org/mitochondrial-review-2026.pdf"
    )
    return ids


class TestPlaceholdersNeverSurface:
    async def test_queued_source_absent_from_title_shaped_query(self, store):
        """The regression: a contentless row winning on its own filename.

        A ``pending_source`` row's only indexed text is its title, and FTS5's
        BM25 length normalisation rewards a zero-length document heavily. Before
        the content predicate this query returned the placeholder at rank 1 with
        an empty snippet.
        """
        ids = await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial review 2026", limit=10
        )

        assert ids["pending_source"] not in [r.doc_id for r in results]

    async def test_queued_source_absent_from_fused_signal(self, store):
        ids = await _populate(store)

        fts = await search_fts5(str(store.db_path), "mitochondrial", 50)

        assert ids["pending_source"] not in [doc_id for doc_id, _ in fts]

    async def test_queued_source_absent_from_the_tail_too(self, store):
        """Excluded by content, not by status — so neither path can readmit it."""
        ids = await _populate(store)

        tail = await search_pending_keyword(str(store.db_path), "mitochondrial", 10)

        assert ids["pending_source"] not in [doc_id for doc_id, _, _ in tail]

    async def test_no_result_is_ever_contentless(self, store):
        await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial review 2026", limit=10
        )

        assert all(r.snippet for r in results)


class TestQueuedDocumentsStaySearchable:
    async def test_deferred_document_survives_a_populated_corpus(self, store):
        """The burial case: 30 fully-indexed competitors, and it still appears."""
        ids = await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        assert ids["pending_enrich"] in [r.doc_id for r in results]

    async def test_deferred_document_carries_its_queue_status_as_tier(self, store):
        ids = await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        hit = next(r for r in results if r.doc_id == ids["pending_enrich"])
        assert hit.tier == q.PENDING_ENRICH

    async def test_failed_enrichment_with_content_is_still_retrievable(self, store):
        """A failed *enrichment* kept its markdown, so it stays findable."""
        ids = await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        hit = next((r for r in results if r.doc_id == ids["failed"]), None)
        assert hit is not None
        assert hit.tier == q.FAILED

    async def test_tail_snippet_is_bounded(self, store):
        """These documents have no chunks, so an unbounded fallback would
        return the entire converted source as the 'snippet'."""
        ids = await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        hit = next(r for r in results if r.doc_id == ids["pending_enrich"])
        assert 0 < len(hit.snippet) < len(BODY)


class TestPageContract:
    @pytest.mark.parametrize("limit", [1, 3, 5, 10, 25])
    async def test_limit_is_never_exceeded(self, store, limit):
        await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=limit
        )

        assert len(results) <= limit

    @pytest.mark.parametrize("limit", [3, 5, 10, 25])
    async def test_page_stays_sorted_by_score(self, store, limit):
        """A caller re-sorting by score must not interleave the tail into the
        fused results — so tail scores sit strictly below the fused floor."""
        await _populate(store)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=limit
        )

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_tail_never_outranks_a_fused_result(self, store):
        ids = await _populate(store)
        queued = {ids["pending_enrich"], ids["failed"]}

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        tiers = [r.doc_id in queued for r in results]
        # Every fused result precedes every tail result.
        assert tiers == sorted(tiers)

    async def test_tail_is_bounded_by_its_quota(self, store):
        """Many queued documents must not crowd out the fused page."""
        for i in range(20):
            await _add(store, f"Deferred draft {i}", BODY, q.PENDING_ENRICH)
        for i in range(20):
            await _add(store, f"Real paper {i}", BODY, q.COMPLETE)

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        queued = sum(1 for r in results if r.tier == q.PENDING_ENRICH)
        assert queued == _pending_quota(10)


class TestEmptyQueueIsUntouched:
    async def test_ranking_is_unchanged_when_nothing_is_queued(self, store):
        """The whole point of appending rather than re-weighting: a database
        with an empty queue behaves exactly as it did before."""
        for i in range(30):
            await _add(
                store, f"Real paper {i} on mitochondrial biology", BODY, q.COMPLETE
            )

        results = await search_unified(
            db_path=str(store.db_path), query="mitochondrial", limit=10
        )

        assert len(results) == 10
        assert all(
            r.tier not in {q.PENDING_ENRICH, q.PENDING_SOURCE, q.FAILED}
            for r in results
        )


class TestWarningText:
    def test_queued_status_gets_a_caveat(self):
        assert "process_pending" in public._result_warning(q.PENDING_ENRICH, "")

    def test_failed_status_points_at_retry(self):
        assert "retry_failed" in public._result_warning(q.FAILED, "")

    def test_ordinary_tier_is_left_alone(self):
        assert public._result_warning("chunks", "") == ""

    def test_existing_warning_is_preserved_not_replaced(self):
        """search() already uses this field for the empty-filter notice."""
        combined = public._result_warning(
            q.PENDING_ENRICH, "Filter produced no results."
        )

        assert combined.startswith("Filter produced no results.")
        assert "process_pending" in combined
