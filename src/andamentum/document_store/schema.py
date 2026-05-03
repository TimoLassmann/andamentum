"""Master database initialization for document-store.

Initializes the database with all required tables:
- documents: Base document table
- chunks + chunk_embeddings: chunk storage and vector index
- documents_fts + chunks_fts: Full-text search
- agent_audit_log: Agent action tracking
- doc_embeddings + clusters: DHP temporal clustering
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .connection import get_connection, DEFAULT_DB_PATH


def init_all_tables(db_path: Optional[Path] = None) -> None:
    """Initialize all database tables.

    Creates the unified database schema by calling:
    - documents table (base table referenced by all others)
    - chunks + chunk_embeddings (vector storage)
    - FTS5 tables (documents_fts, chunks_fts with triggers)
    - Agent audit log
    - doc_embeddings + cluster tables

    Args:
        db_path: Path to database file (uses default if None)
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Documents table (base table referenced by all others)
        from .documents_schema import init_documents_table

        init_documents_table(cursor)

        # 2. Chunks + chunk_embeddings
        from .chunks_schema import init_chunk_tables

        init_chunk_tables(cursor)

        # 3. Search tables (FTS5 virtual table + triggers)
        from .fts_schema import init_fts_tables

        init_fts_tables(cursor)

        # 4. Agent audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # 5. Document-level embeddings vec0 table (for semantic search across ALL tiers)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings USING vec0(
                doc_id INTEGER PRIMARY KEY,
                embedding FLOAT[768]
            )
        """)

        # 6. DHP temporal clustering tables
        from .clusters_schema import init_cluster_tables

        init_cluster_tables(cursor)

        conn.commit()

    # 7. Idempotent migration for existing databases that pre-date the
    #    unified schema columns above.
    _migrate_to_unified_schema(db_path)


def _migrate_to_unified_schema(db_path: Path) -> None:
    """Idempotent extension of legacy ``documents`` rows to the unified schema.

    Adds doc_uuid / document_tier / indexed_at / metadata / doc_embedding /
    cluster_id columns and the doc_embeddings vec0 table when missing. New
    databases get these from ``init_all_tables`` directly; this only does
    work on databases created before that DDL existed.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "doc_uuid" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN doc_uuid TEXT")
                cursor.execute(
                    "UPDATE documents SET doc_uuid = lower(hex(randomblob(16))) WHERE doc_uuid IS NULL"
                )
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_uuid ON documents(doc_uuid)"
                )
            except Exception:
                pass

        if "document_tier" not in existing_columns:
            try:
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN document_tier TEXT DEFAULT 'working'"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_tier ON documents(document_tier)"
                )
            except Exception:
                pass

        if "indexed_at" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN indexed_at TEXT")
            except Exception:
                pass

        if "metadata" not in existing_columns:
            try:
                cursor.execute(
                    "ALTER TABLE documents ADD COLUMN metadata TEXT DEFAULT '{}'"
                )
            except Exception:
                pass

        if "doc_embedding" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN doc_embedding TEXT")
            except Exception:
                pass

        if "cluster_id" not in existing_columns:
            try:
                cursor.execute("ALTER TABLE documents ADD COLUMN cluster_id INTEGER")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_documents_cluster ON documents(cluster_id)"
                )
            except Exception:
                pass

        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS doc_embeddings USING vec0(
                    doc_id INTEGER PRIMARY KEY,
                    embedding FLOAT[768]
                )
            """)
        except Exception:
            pass

        conn.commit()
