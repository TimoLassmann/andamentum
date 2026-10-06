"""Tests for chunk hits travelling through the four-signal fusion, and for the
promise that search never reranks.

Two properties are load-bearing here.

The first is that fusion collapses to one row per document, and chunk hits must
not collapse with it. The chunk signal emits one result per matching *chunk*, so
a document with three good passages arrives at the fusion three times; keeping
one of those rows and discarding the other two would silently lose real
passages, and the loss would be invisible because the document still ranks
correctly.

The second is that :func:`search` stays a pure retrieval call. A default rerank
would charge every caller — CLIs, and pipelines that run search many times per
task — a 0.6B model's import, its multi-second load and its resident memory, and
would return different results depending on which optional extras happen to be
installed. These tests fail if anyone wires the reranker into the search path.
"""

from __future__ import annotations

import sys

import pytest

from andamentum.document_store import queue as q
from andamentum.document_store import reranking
from andamentum.document_store.api import DocumentStore
from andamentum.document_store.chunk_hits import chunk_id_for
from andamentum.document_store.search import (
    UnifiedSearchResult,
    search_unified,
)

EMBEDDING_DIM = 768
BODY = "Mitochondrial dysfunction in skeletal muscle tissue. " * 40


def _embedding(seed: float) -> list[float]:
    """A unit-ish vector that differs per seed, so cosine distance varies."""
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = 1.0
    vector[1] = seed
    return vector


@pytest.fixture()
async def store(tmp_path):
    s = DocumentStore(database_name="testhits", db_dir=str(tmp_path))
    await s.initialize()
    return s


async def _document_with_chunks(store, title: str, texts: list[str]) -> str:
    doc_id = await store.register_document(
        title=title,
        content=f"{BODY} Distinguishing marker: {title}.",
        metadata={"title": title},
    )
    await q.set_ingest_status(str(store.db_path), doc_id, q.COMPLETE)
    for index, text in enumerate(texts):
        await store.store_chunk(
            doc_id,
            text,
            _embedding(0.1 * index),
            chunk_index=index,
            start_char=index * 100,
            end_char=index * 100 + len(text),
        )
    return doc_id


async def test_every_matching_chunk_of_one_document_survives_fusion(store):
    """The regression: fusion is per document, chunk hits are per chunk."""
    doc_id = await _document_with_chunks(
        store,
        "muscle",
        [
            "mitochondrial dysfunction one",
            "mitochondrial dysfunction two",
            "mitochondrial dysfunction three",
        ],
    )

    results = await search_unified(
        db_path=str(store.db_path),
        query="mitochondrial dysfunction",
        limit=10,
        query_embedding=_embedding(0.05),
    )

    hit = next(r for r in results if r.doc_id == doc_id)
    assert len(hit.chunks) == 3
    assert {c.chunk_index for c in hit.chunks} == {0, 1, 2}


async def test_chunk_hits_carry_a_resolvable_id(store):
    doc_id = await _document_with_chunks(store, "muscle", ["mitochondrial one"])

    results = await search_unified(
        db_path=str(store.db_path),
        query="mitochondrial",
        limit=10,
        query_embedding=_embedding(0.05),
    )

    hit = next(r for r in results if r.doc_id == doc_id)
    assert hit.chunks[0].chunk_id == chunk_id_for(doc_id=doc_id, chunk_index=0)


async def test_chunk_hits_are_ordered_best_first(store):
    doc_id = await _document_with_chunks(
        store, "muscle", [f"mitochondrial passage {i}" for i in range(4)]
    )

    results = await search_unified(
        db_path=str(store.db_path),
        query="mitochondrial passage",
        limit=10,
        query_embedding=_embedding(0.05),
    )

    hit = next(r for r in results if r.doc_id == doc_id)
    scores = [c.score for c in hit.chunks]
    assert all(score is not None for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


async def test_a_keyword_only_match_reports_no_chunks(store):
    """Empty means 'this signal found no passage', never 'this document has none'.

    With no query embedding only the keyword signals run, so nothing looked at
    a chunk. Reporting a passage here would be an invention.
    """
    doc_id = await _document_with_chunks(store, "muscle", ["mitochondrial one"])

    results = await search_unified(
        db_path=str(store.db_path),
        query="mitochondrial",
        limit=10,
        query_embedding=None,
    )

    hit = next(r for r in results if r.doc_id == doc_id)
    assert hit.chunks == []


def test_chunks_defaults_to_empty_not_none():
    """So every caller can iterate without a None check."""
    assert UnifiedSearchResult(doc_id="d", score=1.0, tier="fts5").chunks == []


async def test_search_never_loads_a_cross_encoder(store):
    """Search stays a pure retrieval call — no model import, no rerank.

    Fails loudly if anyone wires the reranker into the search path, which would
    make every search pay for a model most callers never asked for.
    """
    await _document_with_chunks(store, "muscle", ["mitochondrial one"])
    already_loaded = "transformers" in sys.modules

    await search_unified(
        db_path=str(store.db_path),
        query="mitochondrial",
        limit=10,
        query_embedding=_embedding(0.05),
    )

    if not already_loaded:
        assert "transformers" not in sys.modules

    assert reranking._model is None, "search must not populate the reranker cache"
