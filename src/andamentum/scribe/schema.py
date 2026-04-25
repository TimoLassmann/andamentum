"""SQL DDL and schema migration for scribe.

All scribe tables are prefixed `scribe_` so a future merge into the
document_store database remains collision-free. Schema version is
recorded in `scribe_meta` so migrations can be applied conditionally.
"""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 1

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS scribe_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        template TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_blocks (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES scribe_documents(id) ON DELETE CASCADE,
        type TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        position INTEGER NOT NULL,
        parent_id TEXT REFERENCES scribe_blocks(id) ON DELETE CASCADE,
        metadata TEXT NOT NULL DEFAULT '{}',
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS scribe_blocks_doc_position "
    "ON scribe_blocks(doc_id, position)",
    """
    CREATE TABLE IF NOT EXISTS scribe_references (
        id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES scribe_documents(id) ON DELETE CASCADE,
        cite_key TEXT NOT NULL,
        bibtex_entry TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        UNIQUE(doc_id, cite_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scribe_revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        block_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        previous_content TEXT NOT NULL,
        new_content TEXT NOT NULL,
        reason TEXT,
        created_at TEXT NOT NULL
    )
    """,
]


def init_schema(conn: sqlite3.Connection) -> None:
    """Create scribe tables and record the schema version. Idempotent."""
    conn.execute("PRAGMA foreign_keys = ON")
    for stmt in _DDL:
        conn.execute(stmt)
    conn.execute(
        "INSERT OR IGNORE INTO scribe_meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
