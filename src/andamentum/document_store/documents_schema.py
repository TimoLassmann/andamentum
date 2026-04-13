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

            -- Tier classification
            document_tier TEXT DEFAULT 'working',

            -- Extensible metadata
            metadata TEXT DEFAULT '{}',

            -- Document-level embedding (JSON-encoded 768-dim float vector)
            -- Stored as TEXT for portability and debuggability (~6KB per doc)
            doc_embedding TEXT,

            -- Cluster assignment (Phase 2: DHP temporal clustering)
            cluster_id INTEGER,

            -- Soft delete (NULL = active, timestamp = deleted)
            deleted_at TEXT DEFAULT NULL
        )
    """)

    # Indexes
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
        "CREATE INDEX IF NOT EXISTS idx_documents_tier ON documents(document_tier)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id)"
    )

    # Migration: add deleted_at column to existing databases
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN deleted_at TEXT DEFAULT NULL")
    except Exception:
        pass  # Column already exists
