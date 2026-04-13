"""Database migration utilities for RAG system.

Handles schema migrations and backfilling existing data with new metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .database import get_connection


def migrate_add_hash_columns(db_path: Optional[Path] = None) -> None:
    """Add hash-related columns to existing documents table.

    This migration is idempotent - safe to run multiple times.
    Uses ALTER TABLE IF NOT EXISTS pattern (SQLite 3.35+).

    Args:
        db_path: Path to database file (uses default if None)
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Add hash columns (idempotent - won't fail if already exists)
        try:
            cursor.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
        except Exception:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE documents ADD COLUMN file_size INTEGER")
        except Exception:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE documents ADD COLUMN file_mtime REAL")
        except Exception:
            pass  # Column already exists

        # Add index on file_hash for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)
        """)

        conn.commit()


def backfill_document_hashes(
    db_path: Optional[Path] = None,
    context_root: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Backfill hash metadata for existing documents.

    Calculates hashes for documents that don't have them yet.
    Only works for files that still exist on disk.

    Args:
        db_path: Path to database file (uses default if None)
        context_root: Root directory to resolve relative paths
        verbose: Print progress messages

    Returns:
        Dict with:
        - updated_count: Number of documents updated with hashes
        - missing_count: Number of documents with missing source files
        - error_count: Number of documents that failed to hash
        - errors: List of error messages
    """
    from ..hashing import calculate_file_hash

    if context_root is None:
        context_root = Path.cwd()

    updated_count = 0
    missing_count = 0
    error_count = 0
    errors = []

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Get all documents without hashes
        cursor.execute("""
            SELECT id, file_path
            FROM documents
            WHERE file_hash IS NULL
        """)
        docs = cursor.fetchall()

        if verbose and docs:
            print(f"Found {len(docs)} documents without hash metadata")

        for doc_id, file_path_str in docs:
            try:
                # Resolve file path
                file_path = context_root / file_path_str

                if not file_path.exists():
                    if verbose:
                        print(f"⚠️  File not found: {file_path_str}")
                    missing_count += 1
                    continue

                # Calculate hash and metadata
                file_hash = calculate_file_hash(file_path)
                stat = file_path.stat()

                # Update document
                cursor.execute(
                    """
                    UPDATE documents
                    SET file_hash = ?, file_size = ?, file_mtime = ?
                    WHERE id = ?
                """,
                    (file_hash, stat.st_size, stat.st_mtime, doc_id),
                )

                updated_count += 1
                if verbose:
                    print(f"✅ Updated hash for: {file_path_str}")

            except Exception as e:
                error_count += 1
                error_msg = f"{file_path_str}: {str(e)}"
                errors.append(error_msg)
                if verbose:
                    print(f"❌ Error: {error_msg}")

        conn.commit()

    if verbose:
        print("\nBackfill complete:")
        print(f"  Updated: {updated_count}")
        print(f"  Missing: {missing_count}")
        print(f"  Errors: {error_count}")

    return {
        "updated_count": updated_count,
        "missing_count": missing_count,
        "error_count": error_count,
        "errors": errors,
    }


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


def migrate_database(
    db_path: Optional[Path] = None,
    context_root: Optional[Path] = None,
    backfill: bool = True,
    verbose: bool = True,
) -> dict:
    """Run all migrations for RAG database.

    Args:
        db_path: Path to database file (uses default if None)
        context_root: Root directory to resolve relative paths (for backfill)
        backfill: Whether to backfill hashes for existing documents
        verbose: Print progress messages

    Returns:
        Dict with migration results
    """
    if verbose:
        print("Running RAG database migrations...")

    # Step 1: Add columns
    if verbose:
        print("\n1. Adding hash columns...")
    migrate_add_hash_columns(db_path)
    if verbose:
        print("   ✅ Hash columns added")

    # Step 2: Backfill (optional)
    backfill_result = {
        "updated_count": 0,
        "missing_count": 0,
        "error_count": 0,
        "errors": [],
    }

    if backfill:
        if verbose:
            print("\n2. Backfilling hashes for existing documents...")
        backfill_result = backfill_document_hashes(db_path, context_root, verbose)

    if verbose:
        print("\n✅ Migration complete!")

    return {"schema_migrated": True, "backfill_result": backfill_result}
