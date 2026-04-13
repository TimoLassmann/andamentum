"""Shared database connection for document-store.

For document operations, use DocumentStore.for_database(name) which
stores documents in named databases at ~/.local/share/document-store/{name}.db
(override with DOCUMENT_STORE_DIR env var).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .lifecycle import get_databases_dir

# Legacy default path — used by RAG internals that haven't been migrated to DocumentStore
DEFAULT_DB_PATH = get_databases_dir() / "default.db"


@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Get database connection with sqlite-vec loaded.

    Args:
        db_path: Path to database file (uses default if None).
                 Special value ":memory:" creates in-memory database.

    Yields:
        sqlite3.Connection with sqlite-vec extension loaded
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    # Handle in-memory database
    if str(db_path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        # Ensure parent directory exists for file-based databases
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))

    conn.row_factory = sqlite3.Row  # Enable column access by name

    # Enable WAL mode for better concurrent access
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout for locked database

    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    try:
        import sqlite_vec
        sqlite_vec.load(conn)
    except Exception as e:
        print(f"Warning: Could not load sqlite-vec extension: {e}")

    try:
        yield conn
    finally:
        conn.close()
