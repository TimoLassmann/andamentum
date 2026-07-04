"""Low-level database operations for Document Store.

Handles SQLite operations for document metadata using unified documents table.
"""

import asyncio
import hashlib
import json
import logging
import re
import struct
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import aiosqlite

from .connection import get_connection
from .models import DocumentMetadata

# A metadata field name is interpolated into a SQLite JSON path
# (``json_extract(metadata, '$.<field>')``) which cannot be parameterised.
# Restrict it to identifier characters + dots (for nested paths) so a caller
# cannot inject SQL via a crafted key. Values are always bound as parameters.
_SAFE_METADATA_FIELD = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.]*$")

logger = logging.getLogger(__name__)


def _row_to_document_metadata(row: aiosqlite.Row) -> DocumentMetadata:
    """Map a ``documents`` table row to :class:`DocumentMetadata`.

    Shared by every read path (get / list / list_deleted / find_by_metadata)
    so the column→field mapping lives in exactly one place. Requires the
    cursor's ``row_factory`` to be :class:`aiosqlite.Row`.
    """
    return DocumentMetadata(
        doc_id=row["doc_uuid"],
        title=row["dc_title"],
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


async def register_document(
    db_path: str,
    doc_id: str,
    title: str,
    content: str,
    metadata: dict,
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
        file_path: Optional file path. If None, a synthetic path is generated.
        file_format: File format (default "md")

    Returns:
        DocumentMetadata with registration details
    """
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
                doc_uuid, dc_title, file_path, markdown_content,
                file_hash, file_size, dc_format, created_date, updated_date, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title,
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

            return _row_to_document_metadata(row)


async def get_documents_metadata(
    db_path: str, doc_ids: list[str]
) -> dict[str, DocumentMetadata]:
    """Batch-fetch metadata for many documents in a single query.

    Returns a ``{doc_uuid: DocumentMetadata}`` mapping for the non-deleted
    documents among ``doc_ids`` (missing / soft-deleted ids are simply absent).
    Callers that need a specific order re-order by their own id list. This is
    the batched replacement for calling :func:`get_document_metadata` in a loop.
    """
    if not doc_ids:
        return {}

    placeholders = ", ".join("?" for _ in doc_ids)
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM documents WHERE doc_uuid IN ({placeholders}) AND deleted_at IS NULL",
            list(doc_ids),
        ) as cursor:
            rows = await cursor.fetchall()

    return {row["doc_uuid"]: _row_to_document_metadata(row) for row in rows}


async def get_documents_content(db_path: str, doc_ids: list[str]) -> dict[str, str]:
    """Batch-fetch ``markdown_content`` for many documents in a single query.

    Returns a ``{doc_uuid: content}`` mapping for the non-deleted documents
    among ``doc_ids`` that have content. The batched replacement for calling
    :meth:`DocumentStore.read` in a loop when only the content is needed.
    """
    if not doc_ids:
        return {}

    placeholders = ", ".join("?" for _ in doc_ids)
    async with get_async_connection(db_path) as db:
        async with db.execute(
            f"SELECT doc_uuid, markdown_content FROM documents "
            f"WHERE doc_uuid IN ({placeholders}) AND deleted_at IS NULL",
            list(doc_ids),
        ) as cursor:
            rows = await cursor.fetchall()

    return {row[0]: row[1] for row in rows if row[1] is not None}


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
            return [_row_to_document_metadata(row) for row in rows]


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


async def list_documents(db_path: str) -> list[DocumentMetadata]:
    """List all non-deleted documents, most-recently-updated first.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of DocumentMetadata objects
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM documents WHERE deleted_at IS NULL ORDER BY updated_date DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [_row_to_document_metadata(row) for row in rows]


def _build_metadata_conditions(
    filters: Mapping[str, Any],
) -> tuple[list[str], list[Any]]:
    """Build SQL conditions + bound params from a metadata filter dict.

    Shared by :func:`find_by_metadata` and :func:`describe_metadata` so both
    honour identical matching semantics:

      * scalar (str / int / bool) → exact equality (``field = value``)
      * ``None`` → SQL NULL (``field IS NULL``)
      * list / tuple / set → set-membership (``field IN (...)``); an empty
        collection emits a never-true condition so it matches nothing rather
        than silently dropping the predicate.

    Internal fields (keys starting with ``_``, e.g. ``_history``) are skipped.
    Field names are validated against :data:`_SAFE_METADATA_FIELD` because they
    are interpolated into a JSON path that cannot be parameterised; values are
    always bound as parameters.

    Returns:
        A ``(conditions, params)`` tuple. ``conditions`` is a list of SQL
        fragments to be ANDed by the caller; ``params`` are the bound values.
    """
    conditions: list[str] = []
    params: list[Any] = []
    for field, value in filters.items():
        if field.startswith("_"):
            continue  # Skip internal fields like _history
        if not _SAFE_METADATA_FIELD.match(field):
            raise ValueError(
                f"Unsafe metadata field name {field!r}: only letters, "
                "digits, underscore and dot are allowed."
            )
        col = f"json_extract(metadata, '$.{field}')"
        if value is None:
            conditions.append(f"{col} IS NULL")
        elif isinstance(value, (list, tuple, set)):
            # Set-membership: ``field IN (...)``. A string is deliberately
            # NOT treated as a collection — it stays an exact-match scalar.
            members = list(value)
            if not members:
                # Empty set matches nothing; fail closed rather than silently
                # dropping the predicate (which would match every document).
                conditions.append("0")
                continue
            placeholders = ", ".join("?" for _ in members)
            conditions.append(f"{col} IN ({placeholders})")
            params.extend(members)
        else:
            conditions.append(f"{col} = ?")
            params.append(value)
    return conditions, params


async def find_by_metadata(
    db_path: str,
    filters: Mapping[str, Any],
    limit: int = 100,
) -> list[DocumentMetadata]:
    """Find documents by metadata field values.

    Uses SQLite JSON functions to query the metadata column. All conditions are
    ANDed together.

    A filter value matches in one of three ways:
      * scalar (str / int / bool) → exact equality (``field = value``)
      * ``None`` → SQL NULL (``field IS NULL``)
      * list / tuple / set → set-membership (``field IN (...)``); an empty
        collection matches nothing.

    Args:
        db_path: Path to SQLite database
        filters: Dict of {field_name: predicate} to match (see above)
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

        # Set-membership: any of several statuses in a single query
        results = await find_by_metadata(db_path, {
            "record_type": "task",
            "status": ["todo", "in_progress", "blocked"],
        })
    """
    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row

        # Build WHERE clause with JSON extraction
        filter_conditions, params = _build_metadata_conditions(filters)
        conditions = ["deleted_at IS NULL", *filter_conditions]
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

            return [_row_to_document_metadata(row) for row in rows]


async def describe_metadata(
    db_path: str,
    filters: Optional[Mapping[str, Any]] = None,
) -> dict[str, dict[str, int]]:
    """Profile the metadata schema actually present in the database.

    Walks every non-deleted document's top-level metadata object and tallies,
    for each field, how many times each distinct value occurs. This is how a
    caller discovers the (schema-less) metadata vocabulary without prior
    knowledge of which fields or values exist.

    Optionally scoped to the subset matching ``filters`` (identical matching
    semantics to :func:`find_by_metadata`), so a caller can drill in — e.g.
    profile the whole database, then re-profile just ``record_type = "task"``
    to see which fields tasks carry.

    Internal fields (keys starting with ``_``, e.g. ``_history``) are excluded.

    Returns:
        Mapping of ``field -> {value: count}``. Values are stringified for a
        stable, JSON-friendly key type. Presentation concerns — such as hiding
        the per-value breakdown for high-cardinality fields — are left to the
        caller (see the public ``describe_metadata`` wrapper).
    """
    where = ["d.deleted_at IS NULL", "d.metadata IS NOT NULL"]
    params: list[Any] = []
    if filters:
        filter_conditions, params = _build_metadata_conditions(filters)
        where.extend(filter_conditions)
    where_clause = " AND ".join(where)

    # json_each expands each document's top-level metadata object into one row
    # per (key, value); GROUP BY collapses that to a per-value occurrence count.
    query = f"""
        SELECT je.key AS field, je.value AS value, COUNT(*) AS n
        FROM documents d, json_each(d.metadata) je
        WHERE {where_clause}
        GROUP BY je.key, je.value
    """

    async with get_async_connection(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

    profiles: dict[str, dict[str, int]] = {}
    for row in rows:
        field = row["field"]
        if field.startswith("_"):
            continue  # internal fields like _history
        profiles.setdefault(field, {})[str(row["value"])] = row["n"]
    return profiles


async def find_doc_uuids_by_filters(
    db_path: str,
    filters: list[dict],
) -> set[str]:
    """Find document UUIDs matching metadata filters.

    Supports closed-set fields only:
      source (equals) → documents.metadata JSON
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
        elif field == "source":
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


# ---------------------------------------------------------------------------
# Document-level embeddings — the doc_embeddings vec0 table is the SINGLE home.
#
# vec0 virtual tables are only visible on a connection that has loaded the
# sqlite-vec extension (aiosqlite does not), so every access goes through the
# sync `get_connection()` and is offloaded with `asyncio.to_thread` in the
# async wrappers. Embeddings are stored as float32 — the width Ollama already
# produces — and read back from the raw vector BLOB, which is bit-exact
# (unlike `vec_to_json`, whose text form truncates to ~6 significant figures).
# ---------------------------------------------------------------------------


def _unpack_embedding(blob: bytes) -> list[float]:
    """Decode a vec0 ``FLOAT[N]`` blob back to a Python float list (exact float32)."""
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _store_doc_embedding_sync(
    db_path: str, doc_id: str, embedding: list[float]
) -> None:
    with get_connection(Path(db_path)) as conn:
        cursor = conn.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_id,))
        row = cursor.fetchone()
        if row is None:
            logger.warning(f"Document {doc_id} not found when storing vec0 embedding")
            return

        int_id = row[0] if isinstance(row, tuple) else row["id"]
        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

        # Upsert (vec0 has no ON CONFLICT): delete any existing row first.
        conn.execute("DELETE FROM doc_embeddings WHERE doc_id = ?", (int_id,))
        conn.execute(
            "INSERT INTO doc_embeddings (doc_id, embedding) VALUES (?, ?)",
            (int_id, embedding_bytes),
        )
        conn.commit()


async def store_doc_embedding(
    db_path: str, doc_id: str, embedding: list[float]
) -> None:
    """Store a document-level embedding in the ``doc_embeddings`` vec0 table.

    The vec0 table is the single home for doc-level embeddings — both the
    search index and the source of truth for re-clustering and duplicate
    detection. Values are stored as float32 (the width Ollama produces).

    Args:
        db_path: Path to SQLite database
        doc_id: Document UUID
        embedding: embedding vector (``EMBEDDING_DIM`` floats)
    """
    await asyncio.to_thread(_store_doc_embedding_sync, db_path, doc_id, embedding)


def _load_doc_embeddings_sync(
    db_path: str, *, include_deleted: bool
) -> list[tuple[str, str, list[float], dict, str]]:
    where = "" if include_deleted else "WHERE d.deleted_at IS NULL"
    with get_connection(Path(db_path)) as conn:
        rows = conn.execute(
            f"""
            SELECT d.doc_uuid, d.dc_title, de.embedding, d.metadata, d.created_date
            FROM doc_embeddings de
            JOIN documents d ON de.doc_id = d.id
            {where}
            ORDER BY d.created_date ASC
            """
        ).fetchall()

    return [
        (
            row["doc_uuid"],
            row["dc_title"] or "",
            _unpack_embedding(row["embedding"]),
            json.loads(row["metadata"]) if row["metadata"] else {},
            row["created_date"],
        )
        for row in rows
    ]


async def load_doc_embeddings(
    db_path: str, *, include_deleted: bool = False
) -> list[tuple[str, str, list[float], dict, str]]:
    """Load every document that has a doc-level embedding, read from vec0.

    Returns ``(doc_uuid, title, embedding, metadata, created_date)`` tuples in
    ascending ``created_date`` order. ``include_deleted=False`` (default) skips
    soft-deleted documents; re-clustering passes ``True`` to keep its historical
    behaviour of clustering over all embedded documents.
    """
    return await asyncio.to_thread(
        _load_doc_embeddings_sync, db_path, include_deleted=include_deleted
    )


def _docs_missing_doc_embedding_sync(db_path: str) -> list[tuple[str, str, str]]:
    with get_connection(Path(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT d.doc_uuid, d.dc_title, d.markdown_content
            FROM documents d
            WHERE d.markdown_content IS NOT NULL
              AND d.id NOT IN (SELECT doc_id FROM doc_embeddings)
            ORDER BY d.created_date ASC
            """
        ).fetchall()
    return [(row["doc_uuid"], row["dc_title"], row["markdown_content"]) for row in rows]


async def docs_missing_doc_embedding(db_path: str) -> list[tuple[str, str, str]]:
    """Return ``(doc_uuid, title, content)`` for documents with no doc-level
    embedding in vec0 — the backfill work-list for :meth:`reembed_all`."""
    return await asyncio.to_thread(_docs_missing_doc_embedding_sync, db_path)


async def count_doc_embeddings(db_path: str) -> int:
    """Count documents that have a doc-level embedding stored in vec0."""

    def _count() -> int:
        with get_connection(Path(db_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM doc_embeddings").fetchone()
            return row[0] if row else 0

    return await asyncio.to_thread(_count)
