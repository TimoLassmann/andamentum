"""Storage backend protocol for epistemic repository.

Defines the interface that storage backends must implement.
andamentum's document_store subpackage provides a rich DocumentStore implementation;
standalone users can use InMemoryStorageBackend for testing and lightweight use.
"""
from typing import Protocol, Any, Optional, runtime_checkable
from dataclasses import dataclass, field
import uuid


@dataclass
class DocumentMetadata:
    """Metadata wrapper for stored documents."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredDocument:
    """A document retrieved from storage."""
    doc_id: str
    content: str
    metadata: Optional[DocumentMetadata] = None


@dataclass
class DocumentRef:
    """A reference to a stored document (from find/query results)."""
    doc_id: str


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for epistemic storage backends.

    andamentum's DocumentStore satisfies this protocol naturally.
    For standalone use, SimpleStorageBackend provides basic persistence.
    """

    async def add(
        self,
        file_path: str,
        content: str,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a document to storage. Returns doc_id."""
        ...

    async def read(self, doc_id: str) -> Optional[StoredDocument]:
        """Read a document by ID."""
        ...

    async def find_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int | None = None,
    ) -> list[DocumentRef]:
        """Find documents matching metadata filters."""
        ...

    async def update(
        self,
        doc_id: str,
        new_content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update document content and/or metadata."""
        ...

    async def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        ...


class InMemoryStorageBackend:
    """In-memory storage backend for testing and standalone use.

    Implements the StorageBackend protocol using a plain dict.
    No persistence — data lives only for the lifetime of the object.

    None-filter semantics: when a filter value is None, it matches
    documents where the metadata field is absent OR explicitly None.
    This mirrors SQL IS NULL behaviour used by andamentum's DocumentStore.
    """

    def __init__(self) -> None:
        self._docs: dict[str, StoredDocument] = {}
        self._counter = 0

    async def add(
        self,
        file_path: str,
        content: str,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._counter += 1
        doc_id = f"doc_{self._counter:04d}_{uuid.uuid4().hex[:8]}"
        self._docs[doc_id] = StoredDocument(
            doc_id=doc_id,
            content=content,
            metadata=DocumentMetadata(metadata=metadata or {}),
        )
        return doc_id

    async def read(self, doc_id: str) -> Optional[StoredDocument]:
        return self._docs.get(doc_id)

    async def find_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int | None = None,
    ) -> list[DocumentRef]:
        results: list[DocumentRef] = []
        for doc in self._docs.values():
            meta = doc.metadata.metadata if doc.metadata else {}
            if self._matches(meta, filters):
                results.append(DocumentRef(doc_id=doc.doc_id))
                if limit is not None and len(results) >= limit:
                    break
        return results

    async def update(
        self,
        doc_id: str,
        new_content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        doc = self._docs.get(doc_id)
        if doc is None:
            return False
        if new_content is not None:
            doc.content = new_content
        if metadata is not None:
            if doc.metadata is None:
                doc.metadata = DocumentMetadata(metadata=metadata)
            else:
                doc.metadata.metadata.update(metadata)
        return True

    async def delete(self, doc_id: str) -> bool:
        return self._docs.pop(doc_id, None) is not None

    @staticmethod
    def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            if expected is None:
                # None means "field absent or None" (SQL IS NULL)
                if key in meta and meta[key] is not None:
                    return False
            else:
                if meta.get(key) != expected:
                    return False
        return True


class DocumentStoreAdapter:
    """Wraps andamentum's DocumentStore to satisfy StorageBackend.

    Translates between DocumentStore's return types (Document, DocumentMetadata,
    UpdateResult) and the types expected by EpistemicRepository (StoredDocument,
    DocumentRef, bool).

    Usage:
        from andamentum.document_store import DocumentStore
        raw_store = DocumentStore.for_database("epistemic_research")
        await raw_store.initialize()
        adapter = DocumentStoreAdapter(raw_store)
        repo = EpistemicRepository(adapter)
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    @property
    def raw_store(self) -> Any:
        """Access the underlying DocumentStore for direct operations (trace recording, stats)."""
        return self._store

    async def add(
        self,
        file_path: str,
        content: str,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self._store.add(file_path, content=content, title=title, metadata=metadata)

    async def read(self, doc_id: str) -> Optional[StoredDocument]:
        doc = await self._store.read(doc_id)
        if doc is None:
            return None
        return StoredDocument(
            doc_id=doc.metadata.doc_id,
            content=doc.content,
            metadata=DocumentMetadata(metadata=doc.metadata.metadata),
        )

    async def find_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int | None = None,
    ) -> list[DocumentRef]:
        results = await self._store.find_by_metadata(filters, limit=limit or 100)
        return [DocumentRef(doc_id=r.doc_id) for r in results]

    async def update(
        self,
        doc_id: str,
        new_content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        result = await self._store.update(doc_id, new_content=new_content, metadata=metadata)
        return result.success if hasattr(result, "success") else bool(result)

    async def delete(self, doc_id: str) -> bool:
        return await self._store.delete(doc_id)
