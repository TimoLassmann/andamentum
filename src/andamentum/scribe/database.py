"""Database path resolution and connection management.

Reuses sqlite3 directly. We intentionally do NOT load sqlite-vec here —
scribe has no embedding needs in v1. If that changes, swap to
`document_store.connection.get_connection`.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .schema import init_schema

DB_DIR_ENV_VAR = "SCRIBE_DIR"
_DEFAULT_SUBDIR = ".local/share/scribe"


def get_databases_dir() -> Path:
    """Return the directory that holds scribe databases.

    Honours $SCRIBE_DIR; otherwise falls back to ~/.local/share/scribe/.
    """
    override = os.environ.get(DB_DIR_ENV_VAR)
    if override:
        return Path(override)
    return Path.home() / _DEFAULT_SUBDIR


def get_database_path(name: str) -> Path:
    """Return the .db path for a named scribe database."""
    return get_databases_dir() / f"{name}.db"


@contextmanager
def open_db(name_or_path: str) -> Iterator[sqlite3.Connection]:
    """Open (and migrate, if needed) a scribe database.

    `name_or_path` is either a logical database name ("my_paper") or
    the literal ":memory:" for an in-memory store. File-based DBs are
    created on first use. The connection is closed even if `init_schema`
    raises during setup.
    """
    if name_or_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = get_database_path(name_or_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))

    try:
        conn.row_factory = sqlite3.Row
        init_schema(conn)
        yield conn
    finally:
        conn.close()
