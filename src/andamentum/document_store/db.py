"""The SQLite connection and the fixed source schema."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

#: Version of the source tables below. The source schema is meant never to
#: change; if it ever must, this number says which layout a file holds.
SCHEMA_VERSION = 1

SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  name TEXT PRIMARY KEY,
  value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  key TEXT,
  created_at TEXT NOT NULL,
  deleted_at TEXT);
-- A key names at most one live document; a deleted one gives its key up.
CREATE UNIQUE INDEX IF NOT EXISTS documents_live_key ON documents(key)
  WHERE key IS NOT NULL AND deleted_at IS NULL;
CREATE TABLE IF NOT EXISTS blobs (
  hash TEXT PRIMARY KEY,
  bytes BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS revisions (
  doc_id TEXT NOT NULL REFERENCES documents(id),
  rev INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  media_type TEXT NOT NULL,
  blob_hash TEXT REFERENCES blobs(hash),
  text TEXT NOT NULL,
  metadata TEXT NOT NULL,
  PRIMARY KEY (doc_id, rev));
CREATE TABLE IF NOT EXISTS changes (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  rev INTEGER,
  op TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS indexes (
  name TEXT PRIMARY KEY,
  config TEXT NOT NULL,
  last_seq INTEGER NOT NULL,
  built_at TEXT);
"""

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """A ULID: 48 bits of milliseconds then 80 random bits, Crockford base32."""
    n = ((time.time_ns() // 1_000_000) << 80) | int.from_bytes(os.urandom(10))
    return "".join(_CROCKFORD[(n >> (5 * i)) & 31] for i in reversed(range(26)))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: Path) -> sqlite3.Connection:
    """Open ``path`` with sqlite-vec loaded, WAL on, and the source schema present.

    The connection is in autocommit mode; every write runs inside an explicit
    ``BEGIN IMMEDIATE`` transaction (see ``Store._write``). A file written by a
    different source-schema version raises.
    """
    import sqlite_vec  # type: ignore[import-not-found]

    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    try:
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        db.executescript(SOURCE_SCHEMA)
        row = db.execute("SELECT value FROM meta WHERE name = 'schema_version'").fetchone()
        if row is None:
            db.execute(
                "INSERT OR IGNORE INTO meta (name, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"{path} holds source schema version {row[0]}; this code reads "
                f"version {SCHEMA_VERSION}"
            )
    except BaseException:
        db.close()
        raise
    return db
