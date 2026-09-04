"""Tests for the public chunk-level surface: ChunkHit, get_chunks, get_chunk.

These are what make two-stage retrieval possible from outside the store: search
narrows to documents, then a caller enumerates a document's passages, scores
them, and can resolve any one of them again later. The properties under test
are the id's contract and the honesty of the empty and ``None`` cases — a
caller records a chunk id as provenance, so a coordinate that silently resolves
to the wrong passage would be worse than one that fails.
"""

from __future__ import annotations

import pytest

from andamentum.document_store import public
from andamentum.document_store.api import DocumentStore
from andamentum.document_store.chunk_hits import (
    ChunkHit,
    chunk_id_for,
    parse_chunk_id,
)

EMBEDDING = [0.0] * 768


async def _noop_preflight(*_a, **_kw) -> None:
    return None


@pytest.fixture()
async def db(tmp_path, monkeypatch):
    """An initialized store, with the module-level store cache pointed at it."""
    store = DocumentStore(database_name="chunkdb", db_dir=str(tmp_path))
    await store.initialize()
    monkeypatch.setattr(public, "_stores", {"chunkdb": store})
    monkeypatch.setattr(public, "_preflight", _noop_preflight)
    return store


async def _document_with_chunks(store: DocumentStore, texts: list[str]) -> str:
    doc_id = await store.register_document(
        content="\n\n".join(texts), title="Test document"
    )
    offset = 0
    for index, text in enumerate(texts):
        await store.store_chunk(
            doc_id,
            text,
            EMBEDDING,
            chunk_index=index,
            start_char=offset,
            end_char=offset + len(text),
        )
        offset += len(text) + 2
    return doc_id


# ── the identifier's contract ────────────────────────────────────────────


def test_chunk_id_round_trips():
    chunk_id = chunk_id_for(doc_id="abc-123", chunk_index=7)
    assert parse_chunk_id(chunk_id) == ("abc-123", 7)


def test_chunk_id_round_trips_when_the_doc_id_contains_the_separator():
    """Splitting on the LAST separator, so an unusual document id survives."""
    chunk_id = chunk_id_for(doc_id="urn:uuid:abc", chunk_index=3)
    assert parse_chunk_id(chunk_id) == ("urn:uuid:abc", 3)


@pytest.mark.parametrize(
    "bad", ["", "no-separator", ":4", "abc:", "abc:notanumber", "abc:1.5"]
)
def test_malformed_chunk_id_raises_rather_than_returning_a_guess(bad: str):
    """A caller bug must not disappear into an ordinary 'not found' branch."""
    with pytest.raises(ValueError, match="Not a chunk id"):
        parse_chunk_id(bad)


def test_chunk_id_arguments_are_keyword_only():
    """Two confusable values a caller reads off one row."""
    with pytest.raises(TypeError):
        chunk_id_for("abc", 1)  # type: ignore[misc]


# ── get_chunks ───────────────────────────────────────────────────────────


async def test_get_chunks_returns_every_chunk_in_document_order(db):
    doc_id = await _document_with_chunks(db, ["first", "second", "third"])

    chunks = await public.get_chunks("chunkdb", doc_id=doc_id)

    assert [c.content for c in chunks] == ["first", "second", "third"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(isinstance(c, ChunkHit) for c in chunks)


async def test_get_chunks_ids_resolve_back_to_the_same_chunk(db):
    doc_id = await _document_with_chunks(db, ["alpha", "beta", "gamma"])

    for chunk in await public.get_chunks("chunkdb", doc_id=doc_id):
        resolved = await public.get_chunk("chunkdb", chunk_id=chunk.chunk_id)
        assert resolved is not None
        assert resolved.content == chunk.content
        assert resolved.chunk_index == chunk.chunk_index


async def test_get_chunks_scores_nothing(db):
    """None, not 0.0 — this function ranks nothing and must not imply it did."""
    doc_id = await _document_with_chunks(db, ["alpha", "beta"])

    chunks = await public.get_chunks("chunkdb", doc_id=doc_id)

    assert [c.score for c in chunks] == [None, None]


async def test_get_chunks_carries_offsets_into_the_document(db):
    doc_id = await _document_with_chunks(db, ["alpha", "beta"])

    chunks = await public.get_chunks("chunkdb", doc_id=doc_id)

    assert (chunks[0].start_char, chunks[0].end_char) == (0, 5)
    assert (chunks[1].start_char, chunks[1].end_char) == (7, 11)


async def test_get_chunks_on_an_unknown_document_is_empty(db):
    assert await public.get_chunks("chunkdb", doc_id="no-such-document") == []


async def test_get_chunks_on_a_queued_document_is_empty(db):
    """A deferred document has no chunks yet — that is the queue, not an error."""
    doc_id = await public.ingest("chunkdb", "# Title\n\nbody", process="defer")

    assert await public.get_chunks("chunkdb", doc_id=doc_id) == []


async def test_get_chunks_hides_a_soft_deleted_document(db):
    doc_id = await _document_with_chunks(db, ["alpha", "beta"])
    assert await public.get_chunks("chunkdb", doc_id=doc_id)

    await db.delete(doc_id)

    assert await public.get_chunks("chunkdb", doc_id=doc_id) == []


async def test_get_chunks_doc_id_is_keyword_only(db):
    """`database` and `doc_id` are both strings; a swap must not type-check."""
    doc_id = await _document_with_chunks(db, ["alpha"])
    with pytest.raises(TypeError):
        await public.get_chunks("chunkdb", doc_id)  # type: ignore[misc]


# ── get_chunk ────────────────────────────────────────────────────────────


async def test_get_chunk_returns_none_for_a_coordinate_that_names_nothing(db):
    doc_id = await _document_with_chunks(db, ["alpha", "beta"])

    assert await public.get_chunk("chunkdb", chunk_id=f"{doc_id}:99") is None
    assert await public.get_chunk("chunkdb", chunk_id="missing-doc:0") is None


async def test_get_chunk_raises_on_a_malformed_id_rather_than_returning_none(db):
    """A malformed id is a caller bug, not a miss."""
    with pytest.raises(ValueError, match="Not a chunk id"):
        await public.get_chunk("chunkdb", chunk_id="garbage")


async def test_chunk_id_survives_deleting_and_rewriting_the_same_chunks(db):
    """The documented guarantee: a coordinate, stable under re-embedding.

    Re-embedding drops and rewrites the chunk rows, so every AUTOINCREMENT row
    id changes while the text does not. The public id is derived from the
    document and the position instead, so it still resolves — which is exactly
    why the row id is not the public id.
    """
    doc_id = await _document_with_chunks(db, ["alpha", "beta", "gamma"])
    before = await public.get_chunks("chunkdb", doc_id=doc_id)
    target = before[1].chunk_id

    from andamentum.document_store.chunks import delete_chunks_for_document

    delete_chunks_for_document(doc_id, db_path=db.db_path)
    for index, text in enumerate(["alpha", "beta", "gamma"]):
        await db.store_chunk(doc_id, text, EMBEDDING, chunk_index=index)

    resolved = await public.get_chunk("chunkdb", chunk_id=target)
    assert resolved is not None
    assert resolved.content == "beta"


async def test_chunk_id_is_a_coordinate_not_a_content_hash(db):
    """The documented non-guarantee, pinned so nobody promises more later.

    Re-chunk a document differently and the same id names different text. A
    caller that must detect this compares the content it kept — nothing in the
    store can detect it for them.
    """
    doc_id = await _document_with_chunks(db, ["alpha", "beta", "gamma"])
    target = chunk_id_for(doc_id=doc_id, chunk_index=1)
    assert (await public.get_chunk("chunkdb", chunk_id=target)).content == "beta"  # type: ignore[union-attr]

    from andamentum.document_store.chunks import delete_chunks_for_document

    delete_chunks_for_document(doc_id, db_path=db.db_path)
    for index, text in enumerate(["alpha beta", "gamma delta"]):
        await db.store_chunk(doc_id, text, EMBEDDING, chunk_index=index)

    resolved = await public.get_chunk("chunkdb", chunk_id=target)
    assert resolved is not None
    assert resolved.content == "gamma delta"
