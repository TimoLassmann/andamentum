"""Chunks table schema and initialization.

The chunks table stores document chunks for semantic search. The
chunk_embeddings vec0 virtual table stores the corresponding 768-dim
embeddings (sized for ``embeddinggemma:latest``).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


def init_chunk_tables(cursor: sqlite3.Cursor) -> None:
    """Initialize chunks + chunk_embeddings tables and the chunks index."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            metadata TEXT,  -- JSON metadata
            token_count INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, chunk_index)
        )
    """)

    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[768]
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
    )
