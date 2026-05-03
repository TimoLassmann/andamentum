"""Chunk-level database operations: store, delete, vector search.

Operates on the chunks + chunk_embeddings tables (DDL in chunks_schema.py).
Uses sqlite-vec for cosine-distance search; embedding storage is the
binary vec0 representation, not JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .connection import get_connection


def store_chunk_for_document(
    doc_uuid: str,
    chunk_text: str,
    embedding: List[float],
    chunk_index: int = 0,
    start_char: int = 0,
    end_char: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Store a single chunk with embedding for an already-registered document.

    Assumes the parent document has been registered via
    DocumentStore.register_document().

    Args:
        doc_uuid: Document UUID (from register_document)
        chunk_text: The chunk text content
        embedding: Embedding vector (768-dim)
        chunk_index: Position within the document (0-based)
        start_char: Start character position in original document
        end_char: End character position in original document
        metadata: Optional chunk metadata (JSON-serializable dict)
        db_path: Path to database file (uses default if None)

    Returns:
        The chunk integer ID

    Raises:
        ValueError: If document with given UUID not found
    """
    import struct

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_uuid,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Document with UUID {doc_uuid} not found")
        document_id = row["id"] if isinstance(row, dict) else row[0]

        cursor.execute(
            """
            INSERT INTO chunks (document_id, chunk_index, content, start_char, end_char, metadata, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                chunk_index,
                chunk_text,
                start_char,
                end_char,
                json.dumps(metadata) if metadata else None,
                max(1, len(chunk_text) // 4),  # rough token estimate
            ),
        )
        chunk_id: int = cursor.lastrowid  # type: ignore[assignment]

        embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
        cursor.execute(
            "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, embedding_bytes),
        )

        conn.commit()
        return chunk_id


def delete_chunks_for_document(
    doc_uuid: str,
    db_path: Optional[Path] = None,
) -> int:
    """Delete all chunks and chunk embeddings for a document.

    chunk_embeddings is a vec0 virtual table without FK cascade,
    so must be deleted explicitly before chunks.

    Args:
        doc_uuid: Document UUID
        db_path: Path to database file (uses default if None)

    Returns:
        Number of chunks deleted
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_uuid,))
        row = cursor.fetchone()
        if row is None:
            return 0
        document_id = row["id"] if isinstance(row, dict) else row[0]

        cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
        chunk_ids = [
            r["id"] if isinstance(r, dict) else r[0] for r in cursor.fetchall()
        ]

        if not chunk_ids:
            return 0

        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )

        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        conn.commit()
        return len(chunk_ids)


def search_chunks(
    query_embedding: List[float], limit: int = 10, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Search for similar chunks using vector similarity.

    Args:
        query_embedding: Query vector
        limit: Maximum number of results
        db_path: Path to database file (uses default if None)

    Returns:
        List of dicts with keys: chunk_id, content, distance, file_path, dc_title, metadata
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        query_blob = json.dumps(query_embedding)

        query = """
            SELECT
                c.id as chunk_id,
                c.content,
                c.start_char,
                c.end_char,
                c.metadata,
                c.token_count,
                d.doc_uuid as doc_id,
                d.file_path,
                d.dc_title,
                d.dc_format,
                d.dc_creator,
                d.dc_subject,
                vec_distance_cosine(ce.embedding, ?) as distance
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.chunk_id = c.id
            JOIN documents d ON c.document_id = d.id
            WHERE d.deleted_at IS NULL
            ORDER BY distance
            LIMIT ?
        """
        cursor.execute(query, (query_blob, limit))

        results = []
        for row in cursor.fetchall():
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            dc_subject = json.loads(row["dc_subject"]) if row["dc_subject"] else None

            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "start_char": row["start_char"],
                    "end_char": row["end_char"],
                    "token_count": row["token_count"],
                    "doc_id": row["doc_id"],
                    "file_path": row["file_path"],
                    "dc_title": row["dc_title"],
                    "dc_format": row["dc_format"],
                    "dc_creator": row["dc_creator"],
                    "dc_subject": dc_subject,
                    "distance": row["distance"],
                    "metadata": metadata,
                }
            )

        return results
