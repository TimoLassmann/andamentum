"""Vector database operations using SQLite + sqlite-vec.

Low-level database operations for RAG system.
Part of the standalone RAG package.

This utility provides vector similarity search for document chunks using:
- SQLite for structured data storage
- sqlite-vec for efficient vector similarity search
- Dublin Core metadata integration
- FTS5 full-text search

Standalone package — no external application framework dependencies.

Usage:
    from andamentum.document_store.rag.database import search_chunks

    # Search for similar chunks
    results = search_chunks(query_embedding, limit=5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import from canonical location
from ..connection import DEFAULT_DB_PATH, get_connection  # noqa: F401


def _init_rag_tables(cursor) -> None:
    """Initialize RAG-specific tables (chunks, chunk_embeddings)."""
    # Chunks table
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

    # Vector embeddings table (using sqlite-vec)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[768]  -- embeddinggemma:latest dimension
        )
    """)

    # Indexes for performance
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)"
    )


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

    Unlike add_document_chunks() which creates/upserts a document row, this
    assumes the document already exists (registered via DocumentStore.register_document()).

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

        # Look up integer id from doc_uuid
        cursor.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_uuid,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Document with UUID {doc_uuid} not found")
        document_id = row["id"] if isinstance(row, dict) else row[0]

        # Insert chunk
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

        # Insert embedding into vec0 table
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

        # Look up integer id
        cursor.execute("SELECT id FROM documents WHERE doc_uuid = ?", (doc_uuid,))
        row = cursor.fetchone()
        if row is None:
            return 0
        document_id = row["id"] if isinstance(row, dict) else row[0]

        # Get chunk IDs
        cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
        chunk_ids = [
            r["id"] if isinstance(r, dict) else r[0] for r in cursor.fetchall()
        ]

        if not chunk_ids:
            return 0

        # Delete from vec0 first (no FK cascade)
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(
            f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )

        # Delete chunks (chunks_fts trigger auto-fires)
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


def get_document_by_hash(
    file_hash: str, db_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Check if document with given hash exists.

    Args:
        file_hash: SHA256 hash of file content
        db_path: Path to database file (uses default if None)

    Returns:
        Document record if exists, None otherwise.
        Record includes: id, file_path, file_hash, file_size, file_mtime, updated_date
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, file_path, file_hash, file_size, file_mtime, updated_date
            FROM documents
            WHERE file_hash = ?
        """,
            (file_hash,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_document_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get database statistics.

    Args:
        db_path: Path to database file (uses default if None)

    Returns:
        Dict with document_count, chunk_count, tag_count, entity_count
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Total documents
        cursor.execute("SELECT COUNT(*) as total FROM documents")
        document_count = cursor.fetchone()["total"]

        # Total chunks
        cursor.execute("SELECT COUNT(*) as total FROM chunks")
        chunk_count = cursor.fetchone()["total"]

        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
        }


def delete_document(
    file_path: str, db_path: Optional[Path] = None, context_root: Optional[Path] = None
) -> dict[str, Any]:
    """Delete document, chunks, embeddings, and associated files.

    Args:
        file_path: Relative path to document (markdown path)
        db_path: Path to database file (uses default if None)
        context_root: Context root for file operations (default: ~/.local/share/document-store/context)

    Returns:
        Dict with deletion details:
        - success: bool
        - database_deleted: bool
        - markdown_deleted: bool
        - original_deleted: bool
        - source_file_path: str (if original existed)
    """
    from pathlib import Path

    if context_root is None:
        context_root = Path.home() / ".local" / "share" / "document-store" / "context"

    result = {
        "success": False,
        "database_deleted": False,
        "markdown_deleted": False,
        "original_deleted": False,
        "source_file_path": None,
    }

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Get document ID and source file path
        cursor.execute(
            "SELECT id, source_file_path FROM documents WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()

        if not row:
            return result  # Document not found

        document_id = row["id"]
        source_file_path = row["source_file_path"]
        result["source_file_path"] = source_file_path

        # Get chunk IDs for embedding deletion
        cursor.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,))
        chunk_ids = [row["id"] for row in cursor.fetchall()]

        # Delete embeddings (not cascade-protected, virtual table)
        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            cursor.execute(
                f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )

        # Delete chunks (CASCADE will handle, but explicit for clarity)
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        # Delete document
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        conn.commit()
        result["database_deleted"] = True

    # Delete markdown file
    markdown_path = context_root / file_path
    if markdown_path.exists():
        try:
            markdown_path.unlink()
            result["markdown_deleted"] = True
        except Exception as e:
            print(f"⚠️  Failed to delete markdown {markdown_path}: {e}")

    # Delete original source file
    if source_file_path:
        original_path = context_root / source_file_path
        if original_path.exists():
            try:
                original_path.unlink()
                result["original_deleted"] = True
            except Exception as e:
                print(f"⚠️  Failed to delete original {original_path}: {e}")

    result["success"] = result["database_deleted"]
    return result
