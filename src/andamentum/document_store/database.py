"""Low-level database operations for Document Store.

Handles SQLite operations for document metadata using unified documents table.
"""

import hashlib
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

import aiosqlite

from .models import DocumentMetadata, DocumentType

logger = logging.getLogger(__name__)


@asynccontextmanager
async def get_async_connection(db_path: str):
    """Get async database connection with WAL mode and busy timeout.

    This ensures concurrent requests don't immediately fail with 'database is locked'.
    WAL mode allows concurrent reads and writes.
    Busy timeout waits up to 5 seconds for locks to be released.

    Args:
        db_path: Path to SQLite database

    Yields:
        aiosqlite connection configured for concurrent access
    """
    async with aiosqlite.connect(db_path) as db:
        # Enable WAL mode for better concurrent access
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        yield db


async def init_document_store_tables(db_path: str) -> None:
    """DEPRECATED: Use init_all_tables() from utilities.init_database instead.

    This function creates a partial schema (document_tiers + working_documents_fts)
    which is now obsolete. The unified schema uses the documents table from
    utilities.documents with full RAG support.

    Kept for backward compatibility only. Will be removed in future version.
    """
    import warnings

    warnings.warn(
        "init_document_store_tables() is deprecated. Use init_all_tables() from utilities.init_database instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # No-op: Modern initialization uses init_all_tables()
    pass


async def register_document(
    db_path: str,
    doc_id: str,
    title: str,
    content: str,
    metadata: dict,
    document_type: Optional[DocumentType] = None,
    file_path: Optional[str] = None,
    file_format: str = "md",
) -> DocumentMetadata:
    """Register a new document in the unified documents table.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier (from DocumentStore)
        title: Document title
        content: Document content (stored as markdown_content for FTS5)
        metadata: Additional metadata (JSON)
        document_type: Optional tier classification (defaults to WORKING for backward compat)
        file_path: Optional file path. If None, a synthetic path is generated.
        file_format: File format (default "md")

    Returns:
        DocumentMetadata with registration details
    """
    if document_type is None:
        document_type = DocumentType.WORKING
    if file_path is None:
        file_path = f"memory://{doc_id}.md"

    content_hash = hashlib.sha256(content.encode()).hexdigest()
    file_size_bytes = len(content.encode())
    created_at = datetime.now().isoformat()
    updated_at = created_at

    async with get_async_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO documents (
                doc_uuid, dc_title, document_tier, file_path, markdown_content,
                file_hash, file_size, dc_format, created_date, updated_date, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title,
                document_type.value,
                file_path,
                content,
                content_hash,
                file_size_bytes,
                file_format,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        await db.commit()

    return DocumentMetadata(
        doc_id=doc_id,
        title=title,
        document_type=document_type,
        file_path=file_path,
        content_hash=content_hash,
        file_format=file_format,
        file_size_bytes=file_size_bytes,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
        indexed_at=None,
        metadata=metadata,
    )


async def get_document_metadata(
    db_path: str, doc_id: str
) -> Optional[DocumentMetadata]:
    """Retrieve document metadata by UUID.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier

    Returns:
        DocumentMetadata if found, None otherwise
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM documents WHERE doc_uuid = ? AND deleted_at IS NULL
            """,
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

            return DocumentMetadata(
                doc_id=row["doc_uuid"],  # UUID
                title=row["dc_title"],
                document_type=DocumentType(row["document_tier"]),
                file_path=row["file_path"],
                content_hash=row["file_hash"],
                file_format=row["dc_format"],
                file_size_bytes=row["file_size"],
                created_at=datetime.fromisoformat(row["created_date"]),
                updated_at=datetime.fromisoformat(row["updated_date"]),
                indexed_at=datetime.fromisoformat(row["indexed_at"])
                if row["indexed_at"]
                else None,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )


async def update_document_metadata(
    db_path: str,
    doc_id: str,
    metadata: dict,
    merge: bool = True,
) -> dict:
    """Update document metadata (JSON field).

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier
        metadata: New metadata dict
        merge: If True, merge with existing. If False, replace entirely.

    Returns:
        Updated metadata dict
    """
    async with get_async_connection(db_path) as db:
        # Get existing metadata if merging
        if merge:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT metadata FROM documents WHERE doc_uuid = ?", (doc_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row and row["metadata"]:
                    existing = json.loads(row["metadata"])
                    metadata = {**existing, **metadata}

        updated_at = datetime.now().isoformat()

        await db.execute(
            """UPDATE documents SET metadata = ?, updated_date = ? WHERE doc_uuid = ?""",
            (json.dumps(metadata), updated_at, doc_id),
        )
        await db.commit()

    return metadata


async def update_document_content(
    db_path: str, doc_id: str, new_content: str
) -> tuple[str, str]:
    """Update document content and hash.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier
        new_content: New document content

    Returns:
        Tuple of (previous_hash, new_hash)
    """
    # Get previous hash
    metadata = await get_document_metadata(db_path, doc_id)
    if not metadata:
        raise ValueError(f"Document {doc_id} not found")

    previous_hash = metadata.content_hash
    new_hash = hashlib.sha256(new_content.encode()).hexdigest()
    updated_at = datetime.now().isoformat()

    async with get_async_connection(db_path) as db:
        await db.execute(
            """
            UPDATE documents
            SET file_hash = ?, updated_date = ?, file_size = ?, markdown_content = ?
            WHERE doc_uuid = ?
            """,
            (new_hash, updated_at, len(new_content.encode()), new_content, doc_id),
        )
        await db.commit()

    return previous_hash, new_hash


async def mark_document_indexed(db_path: str, doc_id: str) -> None:
    """Mark document as indexed with current timestamp.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier
    """
    indexed_at = datetime.now().isoformat()
    async with get_async_connection(db_path) as db:
        await db.execute(
            """
            UPDATE documents SET indexed_at = ? WHERE doc_uuid = ?
            """,
            (indexed_at, doc_id),
        )
        await db.commit()


async def soft_delete_document(db_path: str, doc_id: str) -> bool:
    """Soft-delete a document by setting deleted_at timestamp.

    The document remains in the database but is excluded from search and listing.
    Use restore_document() to undo, or purge_deleted() to permanently remove.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier

    Returns:
        True if soft-deleted, False if not found
    """
    deleted_at = datetime.now().isoformat()
    async with get_async_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE documents SET deleted_at = ? WHERE doc_uuid = ? AND deleted_at IS NULL",
            (deleted_at, doc_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def restore_document(db_path: str, doc_id: str) -> bool:
    """Restore a soft-deleted document.

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier

    Returns:
        True if restored, False if not found or not deleted
    """
    async with get_async_connection(db_path) as db:
        cursor = await db.execute(
            "UPDATE documents SET deleted_at = NULL WHERE doc_uuid = ? AND deleted_at IS NOT NULL",
            (doc_id,),
        )
        await db.commit()
        return cursor.rowcount > 0


async def purge_deleted(db_path: str, older_than_days: int = 30) -> int:
    """Permanently delete soft-deleted documents older than N days.

    Args:
        db_path: Path to SQLite database
        older_than_days: Only purge documents deleted more than this many days ago.
            Use 0 to purge all soft-deleted documents immediately.

    Returns:
        Number of documents permanently deleted
    """
    from datetime import timedelta

    cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()
    async with get_async_connection(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        # Clean up vec0 tables first (not covered by FK CASCADE)
        await db.execute(
            """
            DELETE FROM chunk_embeddings WHERE chunk_id IN (
                SELECT c.id FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.deleted_at IS NOT NULL AND d.deleted_at < ?
            )
        """,
            (cutoff,),
        )
        await db.execute(
            """
            DELETE FROM doc_embeddings WHERE doc_id IN (
                SELECT id FROM documents
                WHERE deleted_at IS NOT NULL AND deleted_at < ?
            )
        """,
            (cutoff,),
        )

        # Now delete documents (CASCADE handles chunks, FTS triggers handle FTS5)
        cursor = await db.execute(
            "DELETE FROM documents WHERE deleted_at IS NOT NULL AND deleted_at < ?",
            (cutoff,),
        )
        await db.commit()
        return cursor.rowcount


async def list_deleted_documents(
    db_path: str, limit: int = 50
) -> list[DocumentMetadata]:
    """List soft-deleted documents.

    Args:
        db_path: Path to SQLite database
        limit: Maximum results

    Returns:
        List of DocumentMetadata for deleted documents
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM documents WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                DocumentMetadata(
                    doc_id=row["doc_uuid"],
                    title=row["dc_title"],
                    document_type=DocumentType(row["document_tier"]),
                    file_path=row["file_path"],
                    content_hash=row["file_hash"],
                    file_format=row["dc_format"],
                    file_size_bytes=row["file_size"],
                    created_at=datetime.fromisoformat(row["created_date"]),
                    updated_at=datetime.fromisoformat(row["updated_date"]),
                    indexed_at=datetime.fromisoformat(row["indexed_at"])
                    if row["indexed_at"]
                    else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]


async def delete_document_record(db_path: str, doc_id: str) -> bool:
    """Permanently delete a document record. Use soft_delete_document() instead for normal operations.

    CASCADE deletes will automatically remove:
    - chunks (via documents.id foreign key)
    - document_tags (via documents.id foreign key)
    - document_entity_mentions (via documents.id foreign key)
    - FTS5 entries (via triggers)

    Args:
        db_path: Path to SQLite database
        doc_id: UUID identifier

    Returns:
        True if deleted, False if not found
    """
    async with get_async_connection(db_path) as db:
        cursor = await db.execute(
            "DELETE FROM documents WHERE doc_uuid = ?",
            (doc_id,),
        )
        await db.commit()
        return cursor.rowcount > 0


async def list_documents_by_type(
    db_path: str, document_type: Optional[DocumentType] = None
) -> list[DocumentMetadata]:
    """List all documents, optionally filtered by tier.

    Args:
        db_path: Path to SQLite database
        document_type: Optional tier filter (WORKING/REFERENCE/GENERATED)

    Returns:
        List of DocumentMetadata objects
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        if document_type:
            query = "SELECT * FROM documents WHERE document_tier = ? AND deleted_at IS NULL ORDER BY updated_date DESC"
            params = (document_type.value,)
        else:
            query = "SELECT * FROM documents WHERE deleted_at IS NULL ORDER BY updated_date DESC"
            params = ()

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

            return [
                DocumentMetadata(
                    doc_id=row["doc_uuid"],
                    title=row["dc_title"],
                    document_type=DocumentType(row["document_tier"]),
                    file_path=row["file_path"],
                    content_hash=row["file_hash"],
                    file_format=row["dc_format"],
                    file_size_bytes=row["file_size"],
                    created_at=datetime.fromisoformat(row["created_date"]),
                    updated_at=datetime.fromisoformat(row["updated_date"]),
                    indexed_at=datetime.fromisoformat(row["indexed_at"])
                    if row["indexed_at"]
                    else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]


async def find_by_metadata(
    db_path: str,
    filters: dict[str, Any],
    limit: int = 100,
) -> list[DocumentMetadata]:
    """Find documents by metadata field values.

    Uses SQLite JSON functions to query the metadata column.

    Args:
        db_path: Path to SQLite database
        filters: Dict of {field_name: expected_value} to match (values can be str, bool, int, etc.)
        limit: Maximum results to return

    Returns:
        List of matching DocumentMetadata objects

    Example:
        # Find all objectives
        results = await find_by_metadata(db_path, {"epistemic_type": "objective"})

        # Find specific objective by ID
        results = await find_by_metadata(db_path, {
            "epistemic_type": "objective",
            "objective_id": "abc-123"
        })
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Build WHERE clause with JSON extraction
        conditions = ["deleted_at IS NULL"]
        params: list[Any] = []
        for field, value in filters.items():
            if field.startswith("_"):
                continue  # Skip internal fields like _history
            if value is None:
                conditions.append(f"json_extract(metadata, '$.{field}') IS NULL")
            else:
                conditions.append(f"json_extract(metadata, '$.{field}') = ?")
                params.append(value)

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT * FROM documents
            WHERE {where_clause}
            ORDER BY updated_date DESC
            LIMIT ?
        """
        params.append(limit)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

            return [
                DocumentMetadata(
                    doc_id=row["doc_uuid"],
                    title=row["dc_title"],
                    document_type=DocumentType(row["document_tier"]),
                    file_path=row["file_path"],
                    content_hash=row["file_hash"],
                    file_format=row["dc_format"],
                    file_size_bytes=row["file_size"],
                    created_at=datetime.fromisoformat(row["created_date"]),
                    updated_at=datetime.fromisoformat(row["updated_date"]),
                    indexed_at=datetime.fromisoformat(row["indexed_at"])
                    if row["indexed_at"]
                    else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                for row in rows
            ]


async def find_doc_uuids_by_filters(
    db_path: str,
    filters: list[dict],
) -> set[str]:
    """Find document UUIDs matching metadata filters.

    Supports closed-set fields only:
      doc_type, source (equals) → documents.metadata JSON
      created_at (after/before) → documents.created_date column
      has_decision, has_action_item (is_true) → chunks.metadata JSON booleans

    Args:
        db_path: Path to SQLite database
        filters: List of filter dicts with keys: field, operator, value.

    Returns:
        Set of matching doc_uuid strings. Empty set if no matches.
    """
    if not filters:
        return set()

    doc_conditions: list[str] = []
    doc_params: list[str] = []
    chunk_conditions: list[str] = []
    chunk_params: list[str] = []

    for f in filters:
        field = f["field"]
        operator = f["operator"]
        value = f.get("value", "")

        if field == "created_at":
            if operator == "after":
                doc_conditions.append("created_date >= ?")
                doc_params.append(value)
            elif operator == "before":
                doc_conditions.append("created_date <= ?")
                doc_params.append(value)
        elif field in ("doc_type", "source"):
            doc_conditions.append(f"json_extract(metadata, '$.{field}') = ?")
            doc_params.append(value)
        elif field in ("has_decision", "has_action_item"):
            # Boolean flags in chunk metadata: json_extract returns 1/0 for true/false
            chunk_conditions.append(f"json_extract(c.metadata, '$.{field}') = 1")

    result_uuids: set[str] = set()
    got_doc = False
    got_chunk = False

    async with get_async_connection(db_path) as db:
        if doc_conditions:
            where = " AND ".join(doc_conditions)
            async with db.execute(
                f"SELECT doc_uuid FROM documents WHERE {where}", doc_params
            ) as cursor:
                rows = await cursor.fetchall()
                result_uuids = {row[0] for row in rows}
                got_doc = True

        if chunk_conditions:
            where = " AND ".join(chunk_conditions)
            query = f"""
                SELECT DISTINCT d.doc_uuid
                FROM chunks c JOIN documents d ON c.document_id = d.id
                WHERE {where}
            """
            async with db.execute(query, chunk_params) as cursor:
                rows = await cursor.fetchall()
                chunk_uuids = {row[0] for row in rows}
                result_uuids = (result_uuids & chunk_uuids) if got_doc else chunk_uuids
                got_chunk = True

    if not got_doc and not got_chunk:
        return set()

    return result_uuids


async def store_doc_embedding(
    db_path: str, doc_id: str, embedding: list[float]
) -> None:
    """Store document-level embedding in both doc_embedding column and vec0 table.

    The doc_embedding TEXT column is the persistent store (survives re-clustering, portable).
    The doc_embeddings vec0 table is the search index (fast cosine distance queries).

    Args:
        db_path: Path to SQLite database
        doc_id: Document UUID
        embedding: 768-dimensional embedding vector
    """
    embedding_json = json.dumps(embedding)

    # 1. Store in doc_embedding TEXT column (async — persistent store)
    async with get_async_connection(db_path) as db:
        await db.execute(
            "UPDATE documents SET doc_embedding = ? WHERE doc_uuid = ?",
            (embedding_json, doc_id),
        )
        await db.commit()

    # 2. Store in doc_embeddings vec0 table (sync — requires sqlite-vec extension)
    from .connection import get_connection
    from pathlib import Path
    import struct

    with get_connection(Path(db_path)) as conn:
        # Get the integer id for this doc_uuid
        cursor = conn.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_id,))
        row = cursor.fetchone()
        if row is None:
            logger.warning(f"Document {doc_id} not found when storing vec0 embedding")
            return

        int_id = row[0] if isinstance(row, tuple) else row["id"]

        # Pack embedding as binary float32 for vec0
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        # Delete existing entry if present (upsert for vec0)
        conn.execute("DELETE FROM doc_embeddings WHERE doc_id = ?", (int_id,))
        conn.execute(
            "INSERT INTO doc_embeddings (doc_id, embedding) VALUES (?, ?)",
            (int_id, embedding_bytes),
        )
        conn.commit()


async def get_doc_embedding(db_path: str, doc_id: str) -> Optional[list[float]]:
    """Retrieve document-level embedding.

    Reads from the doc_embedding TEXT column (the persistent store).

    Args:
        db_path: Path to SQLite database
        doc_id: Document UUID

    Returns:
        768-dimensional embedding vector, or None if not set
    """
    async with get_async_connection(db_path) as db:
        async with db.execute(
            "SELECT doc_embedding FROM documents WHERE doc_uuid = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
            return None
