"""Documents table schema and initialization.

The documents table stores metadata for all indexed documents.
Other modules reference this table via foreign keys.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


def init_documents_table(cursor: sqlite3.Cursor) -> None:
    """Initialize documents table with unified schema.

    Supports both RAG system (integer ID, Dublin Core) and
    DocumentStore (UUID, tier classification).

    This is the base table referenced by:
    - chunks (document chunking)
    - chunk_embeddings (vector similarity search)
    - documents_fts (full-text search)
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            -- Primary keys (both systems)
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),

            -- Core identification
            file_path TEXT UNIQUE NOT NULL,

            -- Metadata (unified title field)
            dc_title TEXT,
            dc_format TEXT,
            dc_creator TEXT,
            dc_subject TEXT,

            -- Content
            markdown_content TEXT,

            -- Deduplication
            file_hash TEXT,
            file_size INTEGER,
            file_mtime REAL,

            -- File management
            source_file_path TEXT,

            -- Timestamps
            created_date TEXT NOT NULL,
            updated_date TEXT NOT NULL,
            indexed_at TEXT,

            -- Extensible metadata
            metadata TEXT DEFAULT '{}',

            -- Document-level embeddings live in the doc_embeddings vec0 table
            -- (the single home), not a column here.

            -- Cluster assignment (Phase 2: DHP temporal clustering)
            cluster_id INTEGER,

            -- Deferred-ingestion queue (see queue.py). 'complete' means all
            -- stages ran; 'pending_source'/'pending_enrich' are deliberate
            -- todo states drained by process_pending(); 'failed' records a
            -- stage that raised, kept for inspection rather than dropped.
            ingest_status TEXT NOT NULL DEFAULT 'complete',
            ingest_error TEXT,
            ingest_updated_at TEXT,

            -- Soft delete (NULL = active, timestamp = deleted)
            deleted_at TEXT DEFAULT NULL
        )
    """)

    # Column migrations MUST run before the indexes below. On a pre-existing
    # database the CREATE TABLE above is a no-op, so any column added after the
    # original schema was written is missing until these ALTERs run — and
    # CREATE INDEX on a missing column raises "no such column".
    cursor.execute("PRAGMA table_info(documents)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    # Migration: add deleted_at column to existing databases.
    # Migration: deferred-ingestion queue columns. Existing rows predate the
    # queue and are, by definition, already fully ingested — 'complete'.
    for column, ddl in (
        (
            "deleted_at",
            "ALTER TABLE documents ADD COLUMN deleted_at TEXT DEFAULT NULL",
        ),
        (
            "ingest_status",
            "ALTER TABLE documents ADD COLUMN ingest_status TEXT NOT NULL "
            "DEFAULT 'complete'",
        ),
        ("ingest_error", "ALTER TABLE documents ADD COLUMN ingest_error TEXT"),
        (
            "ingest_updated_at",
            "ALTER TABLE documents ADD COLUMN ingest_updated_at TEXT",
        ),
    ):
        if column not in existing_columns:
            cursor.execute(ddl)

    # Indexes — after the migrations above, so every indexed column exists.
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(file_path)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(dc_title)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_uuid ON documents(doc_uuid)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_ingest_status "
        "ON documents(ingest_status)"
    )
