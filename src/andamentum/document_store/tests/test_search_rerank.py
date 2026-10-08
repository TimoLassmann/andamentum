"""Tests for the cross-encoder rerank inside :func:`public.search`.

``search`` fuses four signals and then reorders the fused top
:data:`~andamentum.document_store.public.RERANK_POOL` by cross-encoder
relevance, scoring each document by its best passage. These tests stub the
planner, the embedder and the cross-encoder, so they pin the wiring: what is
sent to the reranker, how its scores become the order and ``score``, and that
the queued tail stays below the reranked head.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from andamentum.document_store import public
from andamentum.document_store import query_planner
from andamentum.document_store import queue as q
from andamentum.document_store import reranking
from andamentum.document_store.api import DocumentStore

EMBEDDING_DIM = 768
FILLER = "Notes on tidal patterns along the Apollo coastline. " * 20


def _embedding(seed: float) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[0] = 1.0
    vector[1] = seed
    return vector


class _Embedder:
    def __init__(self, *, model: str) -> None:
        self.model = model

    async def embed_text(self, text: str) -> list[float]:
        return _embedding(0.05)

    async def close(self) -> None:
        return None


class _Reranker:
    """Scores a text by the first marker it contains; records every call."""

    def __init__(self, marks: dict[str, float]) -> None:
        self.marks = marks
        self.calls: list[tuple[str, list[str], str]] = []

    async def __call__(self, query, texts, *, reranker_model):
        self.calls.append((query, list(texts), reranker_model))
        return [
            next((score for mark, score in self.marks.items() if mark in text), 0.01)
            for text in texts
        ]

    @property
    def texts(self) -> list[str]:
        return [text for _query, texts, _model in self.calls for text in texts]


@pytest.fixture()
async def store(tmp_path, monkeypatch):
    s = DocumentStore(database_name="testrerank", db_dir=str(tmp_path))
    await s.initialize()

    async def get_store(_database):
        return s

    async def preflight(_database, **_kwargs):
        return None

    async def plan(query, *, model):
        return query_planner.SearchPlan(semantic_query=query)

    monkeypatch.setattr(public, "_get_store", get_store)
    monkeypatch.setattr(public, "_preflight", preflight)
    monkeypatch.setattr(query_planner, "plan_search", plan)
    monkeypatch.setattr(public, "OllamaEmbedder", _Embedder)
    return s


@pytest.fixture()
def reranker(monkeypatch):
    def install(marks: dict[str, float]) -> _Reranker:
        stub = _Reranker(marks)
        monkeypatch.setattr(reranking, "rerank", stub)
        return stub

    return install


async def _document(store, title: str, body: str, *, chunks: Sequence[str] = ()) -> str:
    doc_id = await store.register_document(
        title=title, content=f"{FILLER} {body}", metadata={"title": title}
    )
    for index, text in enumerate(chunks):
        await store.store_chunk(
            doc_id,
            text,
            _embedding(0.1 * index),
            chunk_index=index,
            start_char=index * 100,
            end_char=index * 100 + len(text),
        )
    await q.set_ingest_status(str(store.db_path), doc_id, q.COMPLETE)
    return doc_id


async def _search(query: str, limit: int = 10):
    return await public.search(
        "testrerank",
        query,
        limit,
        model="stub-planner",
        embedding_model="stub-embedder",
        reranker_model="stub-reranker",
    )


async def test_order_and_score_come_from_the_reranker(store, reranker):
    low = await _document(store, "low", "seiche resonance MARK_LOW")
    high = await _document(store, "high", "seiche resonance MARK_HIGH")
    mid = await _document(store, "mid", "seiche resonance MARK_MID")
    stub = reranker({"MARK_HIGH": 0.9, "MARK_MID": 0.5, "MARK_LOW": 0.1})

    results = await _search("seiche resonance")

    assert [r.doc_id for r in results] == [high, mid, low]
    assert [r.score for r in results] == [0.9, 0.5, 0.1]
    assert len(stub.calls) == 1


async def test_query_and_model_are_passed_through(store, reranker):
    await _document(store, "one", "seiche resonance")
    stub = reranker({})

    await _search("seiche resonance")

    [(query, _texts, model)] = stub.calls
    assert query == "seiche resonance"
    assert model == "stub-reranker"


async def test_a_document_scores_as_its_best_matched_passage(store, reranker):
    doc_id = await _document(
        store,
        "chunked",
        "seiche resonance",
        chunks=[
            "seiche resonance weak passage",
            "seiche resonance PASSAGE_BEST",
            "seiche resonance PASSAGE_OK",
        ],
    )
    stub = reranker({"PASSAGE_BEST": 0.8, "PASSAGE_OK": 0.4})

    results = await _search("seiche resonance")

    hit = next(r for r in results if r.doc_id == doc_id)
    assert hit.score == 0.8
    assert any("PASSAGE_BEST" in text for text in stub.texts)


async def test_without_a_matched_chunk_the_document_text_is_scored(store, reranker):
    """A keyword-only match has no passage, so a window of the document goes in."""
    doc_id = await _document(store, "plain", "seiche resonance MARK_DOC")
    stub = reranker({"MARK_DOC": 0.7})

    results = await _search("seiche resonance")

    assert next(r for r in results if r.doc_id == doc_id).score == 0.7
    assert all(len(text) <= public._RERANK_WINDOW_CHARS + 3 for text in stub.texts)


async def test_queued_tail_is_not_reranked_and_stays_below(store, reranker):
    await _document(store, "done", "seiche resonance MARK_DONE")
    queued = await store.register_document(
        title="queued",
        content="seiche resonance MARK_QUEUED, not yet chunked",
        metadata={"title": "queued"},
    )
    await q.set_ingest_status(str(store.db_path), queued, q.PENDING_ENRICH)
    stub = reranker({"MARK_DONE": 0.2, "MARK_QUEUED": 0.99})

    results = await _search("seiche resonance")

    assert results[-1].doc_id == queued
    assert results[-1].match_type == q.PENDING_ENRICH
    assert results[-1].score < min(r.score for r in results[:-1])
    assert not any("MARK_QUEUED" in text for text in stub.texts)


async def test_limit_is_applied_after_the_rerank(store, reranker):
    """The pool is wider than the page, so a low-fused document can reach the top."""
    ids = [
        await _document(store, f"doc{i}", f"seiche resonance MARK_{i:02d}")
        for i in range(8)
    ]
    reranker({"MARK_07": 0.95})

    results = await _search("seiche resonance", limit=3)

    assert len(results) == 3
    assert results[0].doc_id == ids[7]


async def test_no_candidates_never_loads_a_model(store, monkeypatch):
    await _document(store, "one", "seiche resonance")

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("rerank called with no fused candidates")

    monkeypatch.setattr(reranking, "rerank", must_not_run)

    assert await _search("zzzunmatchedterm") == []
