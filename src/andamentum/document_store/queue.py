"""The durable ingestion queue — status transitions on the ``documents`` table.

Ingestion has two expensive stages: converting a source to markdown, and
enriching markdown (chunk + embed). Either can be deferred and
drained later. The queue that makes that possible is not a separate table — it
is three columns on ``documents``, because the unit of work *is* a document:

    pending_source ──convert──► pending_enrich ──enrich──► complete
           │                          │
           └──────── failed ◄─────────┘

Each arrow is committed in the same transaction as the work it represents, so
the state on disk is always truthful. That single property is what makes the
drain resumable: there is no in-memory cursor to lose. A hard kill mid-document
leaves that document in its *previous* state, and the stage re-runs from there —
so the worst case is ever losing one document's compute, never the run's.

``pending_source`` rows carry a *reference* to the source (path or URL) in
``source_file_path`` and hold empty markdown until conversion runs. They are
deliberately not deduplicated by content hash — every enqueued source is its own
unit of work until it has content to compare.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from .database import get_async_connection

#: A document's position in the ingestion queue.
IngestStatus = Literal["pending_source", "pending_enrich", "complete", "failed"]

PENDING_SOURCE: IngestStatus = "pending_source"
PENDING_ENRICH: IngestStatus = "pending_enrich"
COMPLETE: IngestStatus = "complete"
FAILED: IngestStatus = "failed"

#: States that represent outstanding work, in drain order.
PENDING_STATUSES: tuple[IngestStatus, ...] = (PENDING_SOURCE, PENDING_ENRICH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def set_ingest_status(
    db_path: str,
    doc_id: str,
    status: IngestStatus,
    *,
    error: str | None = None,
) -> None:
    """Move a document to ``status``, stamping the time and any failure reason.

    Clearing an error is explicit: any non-``failed`` status resets
    ``ingest_error`` to NULL, so a document that fails and later succeeds does
    not carry a stale reason.
    """
    async with get_async_connection(db_path) as db:
        await db.execute(
            """
            UPDATE documents
            SET ingest_status = ?, ingest_error = ?, ingest_updated_at = ?
            WHERE doc_uuid = ?
            """,
            (status, error if status == FAILED else None, _now(), doc_id),
        )
        await db.commit()


async def register_pending_source(
    db_path: str,
    source: str,
    *,
    title: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Enqueue a source (path or URL) for later conversion. Returns the doc_id.

    Writes a row with empty markdown and ``ingest_status='pending_source'``.
    Deliberately bypasses the content-hash dedup in
    :meth:`DocumentStore.register_document`: every pending source has the same
    (empty) content, so hashing would collapse the whole queue into one row.
    """
    doc_id = str(uuid.uuid4())
    now = _now()
    # Hash/size of the (empty) placeholder content, so the row is well-formed
    # the moment it exists — readers construct DocumentMetadata from it, which
    # requires both to be non-NULL. Conversion overwrites them with real values.
    empty_hash = hashlib.sha256(b"").hexdigest()
    async with get_async_connection(db_path) as db:
        await db.execute(
            """
            INSERT INTO documents (
                doc_uuid, dc_title, file_path, markdown_content,
                file_hash, file_size, source_file_path, dc_format,
                created_date, updated_date,
                metadata, ingest_status, ingest_updated_at
            ) VALUES (?, ?, ?, '', ?, 0, ?, 'md', ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                title or _title_from_source(source),
                # Synthetic and unique: file_path is UNIQUE NOT NULL, and the
                # same source may legitimately be enqueued more than once.
                f"source://{doc_id}",
                empty_hash,
                source,
                now,
                now,
                json.dumps(metadata or {}),
                PENDING_SOURCE,
                now,
            ),
        )
        await db.commit()
    return doc_id


def _title_from_source(source: str) -> str:
    """Placeholder title for a not-yet-converted source: its final path segment."""
    cleaned = source.rstrip("/")
    tail = cleaned.rsplit("/", 1)[-1] if "/" in cleaned else cleaned
    return tail or source


async def list_pending(
    db_path: str,
    *,
    statuses: tuple[IngestStatus, ...] = PENDING_STATUSES,
    limit: int | None = None,
) -> list[dict]:
    """Return outstanding work, oldest first, ``pending_source`` before ``pending_enrich``.

    Ordering matters: draining conversions first means a paused run leaves
    behind markdown (cheap to finish) rather than unconverted sources.
    """
    if not statuses:
        return []

    placeholders = ",".join("?" * len(statuses))
    # CASE orders the statuses explicitly rather than relying on alphabetical
    # ordering of the status strings, which would be coincidental.
    sql = f"""
        SELECT doc_uuid, dc_title, ingest_status, source_file_path, markdown_content
        FROM documents
        WHERE ingest_status IN ({placeholders}) AND deleted_at IS NULL
        ORDER BY CASE ingest_status WHEN '{PENDING_SOURCE}' THEN 0 ELSE 1 END,
                 created_date ASC
    """
    params: list[object] = list(statuses)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    async with get_async_connection(db_path) as db:
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

    return [
        {
            "doc_id": r[0],
            "title": r[1],
            "status": r[2],
            "source": r[3],
            "content": r[4],
        }
        for r in rows
    ]


async def count_by_status(db_path: str) -> dict[str, int]:
    """Count active documents per ingest status. Always reports all four keys."""
    counts: dict[str, int] = {
        PENDING_SOURCE: 0,
        PENDING_ENRICH: 0,
        COMPLETE: 0,
        FAILED: 0,
    }
    async with get_async_connection(db_path) as db:
        async with db.execute(
            """
            SELECT ingest_status, COUNT(*) FROM documents
            WHERE deleted_at IS NULL GROUP BY ingest_status
            """
        ) as cursor:
            for status, count in await cursor.fetchall():
                counts[status] = count
    return counts
