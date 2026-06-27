"""Tests for in-place document editing: title updates and single-doc reembed.

The store's write side was historically append-then-metadata-only. These cover
the two capabilities that make a document editable in place:
  - ``DocumentStore.update(new_title=...)`` (cheap, FTS-only, no reindex)
  - ``reembed_document()`` (targeted phase-2 rebuild after a content edit)
"""

from __future__ import annotations

import uuid

import pytest

try:
    import sqlite_vec as _sv  # type: ignore[import-not-found]  # noqa: F401

    _HAS_SQLITE_VEC = True
except ImportError:
    _HAS_SQLITE_VEC = False

pytestmark = pytest.mark.skipif(not _HAS_SQLITE_VEC, reason="sqlite_vec not installed")


@pytest.fixture
async def db():
    from andamentum.document_store import DocumentStore

    db_name = f"test_title_{uuid.uuid4().hex[:8]}"
    store = DocumentStore.for_database(db_name)
    await store.initialize()
    yield db_name


class TestUpdateTitle:
    @pytest.mark.asyncio
    async def test_new_title_changes_the_title_column(self, db):
        from andamentum.document_store import DocumentStore

        store = DocumentStore.for_database(db)
        doc_id = await store.register_document(
            title="Original title",
            content="Some body text.",
            metadata={"record_type": "note"},
        )

        result = await store.update(doc_id, new_title="Revised title")
        assert result.success is True

        doc = await store.read(doc_id)
        assert doc is not None
        assert doc.metadata.title == "Revised title"
        # Content is untouched by a title-only update.
        assert doc.content == "Some body text."

    @pytest.mark.asyncio
    async def test_update_title_and_content_together(self, db):
        from andamentum.document_store import DocumentStore

        store = DocumentStore.for_database(db)
        doc_id = await store.register_document(
            title="Untitled",
            content="draft one",
            metadata={"record_type": "note", "status": "draft"},
        )

        result = await store.update(
            doc_id,
            new_content="draft two — more words",
            new_title="A real title",
            metadata={"status": "note"},
        )
        assert result.success is True
        assert result.metadata_updated is True

        doc = await store.read(doc_id)
        assert doc is not None
        assert doc.metadata.title == "A real title"
        assert doc.content == "draft two — more words"
        assert doc.metadata.metadata.get("status") == "note"

    @pytest.mark.asyncio
    async def test_empty_update_is_a_noop_failure(self, db):
        from andamentum.document_store import DocumentStore

        store = DocumentStore.for_database(db)
        doc_id = await store.register_document(title="t", content="c")

        result = await store.update(doc_id)
        assert result.success is False
        assert "new_title" in result.message


class TestReembedDocument:
    @pytest.mark.ollama
    @pytest.mark.asyncio
    async def test_reembed_after_content_edit(self, db):
        """Edit content cheaply, then reembed so semantic search reflects it."""
        from andamentum.document_store import (
            DocumentStore,
            ingest,
            reembed_document,
        )
        from andamentum.document_store.public import search

        model = "ollama:gpt-oss:20b"
        embedding_model = "embeddinggemma:latest"

        doc_id = await ingest(
            db,
            "The original note is about marine biology and coral reefs.",
            title="Note",
            model=model,
            embedding_model=embedding_model,
        )

        store = DocumentStore.for_database(db)
        await store.update(
            doc_id,
            new_content="The note now discusses orbital mechanics and satellites.",
        )

        ok = await reembed_document(
            db, doc_id, model=model, embedding_model=embedding_model
        )
        assert ok is True

        results = await search(
            db, "spacecraft in orbit", model=model, embedding_model=embedding_model
        )
        assert any(r.doc_id == doc_id for r in results)

    @pytest.mark.ollama
    @pytest.mark.asyncio
    async def test_reembed_unknown_doc_returns_false(self, db):
        from andamentum.document_store import reembed_document

        ok = await reembed_document(
            db,
            "missing-doc-id",
            model="ollama:gpt-oss:20b",
            embedding_model="embeddinggemma:latest",
        )
        assert ok is False
