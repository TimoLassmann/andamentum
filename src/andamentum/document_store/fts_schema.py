"""FTS5 full-text search database schema and initialization."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


def init_fts_tables(cursor: sqlite3.Cursor) -> None:
    """Initialize FTS5 full-text search tables and triggers.

    Creates:
    - documents_fts: Document-level FTS5 virtual table (legacy)
    - chunks_fts: Chunk-level FTS5 virtual table (accurate BM25)
    - Auto-sync triggers to keep FTS indexes updated
    """
    # Document-level FTS5 virtual table (legacy)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            title,
            content,
            content_rowid=id,
            tokenize='porter unicode61'
        )
    """)

    # Auto-sync triggers for documents_fts
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, title, content)
            VALUES (new.id, new.dc_title, new.markdown_content);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_au AFTER UPDATE ON documents BEGIN
            UPDATE documents_fts
            SET title = new.dc_title, content = new.markdown_content
            WHERE rowid = new.id;
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_fts_ad AFTER DELETE ON documents BEGIN
            DELETE FROM documents_fts WHERE rowid = old.id;
        END
    """)

    # Chunk-level FTS5 virtual table (for accurate BM25 on full chunk corpus)
    # Aligns with Anthropic's contextual retrieval approach
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            content,
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            tokenize='porter unicode61'
        )
    """)

    # Auto-sync triggers for chunks_fts
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, content, chunk_id, document_id)
            VALUES (new.id, new.content, new.id, new.document_id);
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
            UPDATE chunks_fts
            SET content = new.content
            WHERE rowid = new.id;
        END
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            DELETE FROM chunks_fts WHERE rowid = old.id;
        END
    """)
