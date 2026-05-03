"""Idempotent schema migration for the unified DocumentStore tables.

Extends the base ``documents`` table with the columns DocumentStore needs
(doc_uuid, document_tier, indexed_at, metadata, doc_embedding, cluster_id)
and creates the ``doc_embeddings`` vec0 virtual table.
"""

from __future__ import annotations

from pathlib import Path

from .database import get_connection


def migrate_to_unified_schema(db_path: Path) -> None:
    """Add DocumentStore columns to existing documents table.

    Extends the documents table with columns needed for DocumentStore:
    - doc_uuid: Unique identifier for cross-database document references
    - document_tier: Classification (working/reference/generated)
    - indexed_at: Timestamp of last indexing
    - metadata: JSON field for extensible metadata

    Safe to run multiple times (idempotent).

    Args:
        db_path: Path to database file
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Check which columns exist
        cursor.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add missing columns (idempotent)
        if "doc_uuid" not in existing_columns:
            try:
                # First add column as nullable with default
                cursor.execute("""
                    ALTER TABLE documents
                    ADD COLUMN doc_uuid TEXT
                """)
                # Generate UUIDs for existing rows
                cursor.execute("""
                    UPDATE documents
                    SET doc_uuid = lower(hex(randomblob(16)))
                    WHERE doc_uuid IS NULL
                """)
                # Create unique index (enforces uniqueness)
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_uuid ON documents(doc_uuid)"
                )
            except Exception:
                pass  # Column already exists

        if "document_tier" not in existing_columns:
            try:
                cursor.execute("""
                    ALTER TABLE documents
                    ADD COLUMN document_tier TEXT DEFAULT 'working'
                """)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_tier ON documents(document_tier)"
                )
            except Exception:
                pass  # Column already exists

        if "indexed_at" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN indexed_at TEXT")
            except Exception:
                pass  # Column already exists

        if "metadata" not in existing_columns:
            try:
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN metadata TEXT DEFAULT '{}'"
                )
            except Exception:
                pass  # Column already exists

        # Phase 1: Document-level embeddings (DHP temporal clustering)
        if "doc_embedding" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN doc_embedding TEXT")
            except Exception:
                pass  # Column already exists

        if "cluster_id" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN cluster_id INTEGER")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id)"
                )
            except Exception:
                pass  # Column already exists

        # Create doc_embeddings vec0 table if it doesn't exist
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings USING vec0(
                    doc_id INTEGER PRIMARY KEY,
                    embedding FLOAT[768]
                )
            """)
        except Exception:
            pass  # Table already exists or sqlite-vec not loaded

        conn.commit()
