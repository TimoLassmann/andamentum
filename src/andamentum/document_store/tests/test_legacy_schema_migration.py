"""Regression tests: init_all_tables must upgrade a pre-existing database.

On an existing database ``CREATE TABLE IF NOT EXISTS documents`` is a no-op, so
any column introduced after that database was created exists only once the
ALTER TABLE migrations have run. Indexes over those columns must therefore be
created *after* the migrations, never before -- otherwise opening a legacy
database dies with ``sqlite3.OperationalError: no such column: ingest_status``.
"""

from __future__ import annotations

import sqlite3

from andamentum.document_store.schema import init_all_tables

# The documents table as it existed before deleted_at / the deferred-ingestion
# queue columns were added. Deliberately verbatim-legacy: do not "modernise".
LEGACY_DOCUMENTS_DDL = """
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_uuid TEXT UNIQUE NOT NULL DEFAULT (lower(hex(randomblob(16)))),
        file_path TEXT UNIQUE NOT NULL,
        dc_title TEXT,
        dc_format TEXT,
        dc_creator TEXT,
        dc_subject TEXT,
        markdown_content TEXT,
        file_hash TEXT,
        file_size INTEGER,
        file_mtime REAL,
        source_file_path TEXT,
        created_date TEXT NOT NULL,
        updated_date TEXT NOT NULL,
        indexed_at TEXT,
        metadata TEXT DEFAULT '{}',
        cluster_id INTEGER
    )
"""

MIGRATED_COLUMNS = ("deleted_at", "ingest_status", "ingest_error", "ingest_updated_at")


def _make_legacy_db(path, row_count: int = 3) -> None:
    conn = sqlite3.connect(path)
    conn.execute(LEGACY_DOCUMENTS_DDL)
    for i in range(row_count):
        conn.execute(
            "INSERT INTO documents (file_path, dc_title, created_date, updated_date, "
            "metadata) VALUES (?, ?, '2024-01-01', '2024-01-01', '{}')",
            (f"/legacy/doc_{i}.md", f"Legacy {i}"),
        )
    conn.commit()
    conn.close()


def _columns(path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    finally:
        conn.close()


def test_init_all_tables_upgrades_legacy_database(tmp_path):
    """A database predating the queue columns must open, not raise."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    assert MIGRATED_COLUMNS[0] not in _columns(db_path), "fixture is not legacy"

    init_all_tables(db_path)  # must not raise "no such column: ingest_status"

    columns = _columns(db_path)
    for column in MIGRATED_COLUMNS:
        assert column in columns, f"{column} was not added to the legacy table"


def test_legacy_rows_survive_and_default_to_complete(tmp_path):
    """Migration must preserve existing rows and mark them fully ingested."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path, row_count=5)

    init_all_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 5
        statuses = conn.execute(
            "SELECT DISTINCT ingest_status FROM documents"
        ).fetchall()
        assert statuses == [("complete",)]
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM documents WHERE deleted_at IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_ingest_status_index_exists_after_legacy_migration(tmp_path):
    """The index over the migrated column must be created, not skipped."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    init_all_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        conn.close()
    assert "idx_documents_ingest_status" in indexes


def test_init_all_tables_is_idempotent_on_legacy_database(tmp_path):
    """Re-opening an already-migrated legacy database must stay quiet."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_db(db_path)

    init_all_tables(db_path)
    first = _columns(db_path)
    init_all_tables(db_path)

    assert _columns(db_path) == first
