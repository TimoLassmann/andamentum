"""Schema creation and migration tests."""

import sqlite3

from andamentum.scribe.schema import init_schema, SCHEMA_VERSION


def test_init_schema_creates_all_tables():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)

    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name LIKE 'scribe_%' ORDER BY name"
    )
    tables = [row[0] for row in cur.fetchall()]
    assert tables == [
        "scribe_blocks",
        "scribe_documents",
        "scribe_meta",
        "scribe_references",
        "scribe_revisions",
    ]


def test_init_schema_records_version():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    cur = conn.execute("SELECT value FROM scribe_meta WHERE key='schema_version'")
    assert cur.fetchone()[0] == str(SCHEMA_VERSION)


def test_init_schema_is_idempotent():
    conn = sqlite3.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # must not raise
    cur = conn.execute("SELECT count(*) FROM scribe_meta WHERE key='schema_version'")
    assert cur.fetchone()[0] == 1
