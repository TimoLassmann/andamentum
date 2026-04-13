"""Test hash-based document lookup."""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture
async def db():
    from document_store import DocumentStore

    db_name = f"test_hash_{uuid.uuid4().hex[:8]}"
    store = DocumentStore.for_database(db_name)
    await store.initialize()
    yield db_name


class TestExistsByHash:
    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_hash(self, db):
        from document_store import DocumentStore

        store = DocumentStore.for_database(db)
        assert await store.exists_by_hash("deadbeef" * 8) is False

    @pytest.mark.asyncio
    async def test_returns_true_for_known_hash(self, db):
        import hashlib

        from document_store import DocumentStore

        store = DocumentStore.for_database(db)
        content = "Test document content for hashing"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        await store.register_document(
            title="Hash test doc",
            content=content,
            metadata={"record_type": "note"},
        )

        assert await store.exists_by_hash(expected_hash) is True

    @pytest.mark.asyncio
    async def test_returns_false_after_delete(self, db):
        import hashlib

        from document_store import DocumentStore

        store = DocumentStore.for_database(db)
        content = "Document to be deleted"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        doc_id = await store.register_document(
            title="Delete me",
            content=content,
            metadata={},
        )

        assert await store.exists_by_hash(expected_hash) is True

        await store.delete(doc_id)
        assert await store.exists_by_hash(expected_hash) is False
