"""Master database initialization for document-store.

Initializes the database with all required tables:
- documents: Base document table
- chunks + embeddings: RAG storage
- documents_fts + chunks_fts: Full-text search
- agent_audit_log: Agent action tracking
- database_metadata: Access control metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .connection import get_connection, DEFAULT_DB_PATH


def init_all_tables(db_path: Optional[Path] = None) -> None:
    """Initialize all database tables.

    Creates the unified database schema by calling:
    - documents table (base table referenced by all others)
    - RAG tables (chunks, chunk_embeddings)
    - FTS5 tables (documents_fts, chunks_fts with triggers)
    - Agent audit log
    - Database metadata tables

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

        # 2. RAG tables (chunks, chunk_embeddings)
        from .rag.database import _init_rag_tables

        _init_rag_tables(cursor)

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

    # 5. Ensure all DocumentStore extensions applied (idempotent migration)
    from .rag.migration import migrate_to_unified_schema

    migrate_to_unified_schema(db_path)
