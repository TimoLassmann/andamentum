"""Database path resolution and connection tests."""

from pathlib import Path

from andamentum.scribe.database import (
    get_databases_dir,
    get_database_path,
    open_db,
)


def test_get_databases_dir_default(monkeypatch):
    monkeypatch.delenv("SCRIBE_DIR", raising=False)
    monkeypatch.setenv("HOME", "/tmp/fake_home")
    assert get_databases_dir() == Path("/tmp/fake_home/.local/share/scribe")


def test_get_databases_dir_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    assert get_databases_dir() == tmp_path


def test_get_database_path_appends_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    assert get_database_path("my_paper") == tmp_path / "my_paper.db"


def test_open_db_creates_file_and_initialises_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRIBE_DIR", str(tmp_path))
    with open_db("test_doc") as conn:
        cur = conn.execute("SELECT value FROM scribe_meta WHERE key='schema_version'")
        assert cur.fetchone() is not None
    assert (tmp_path / "test_doc.db").exists()


def test_open_db_in_memory():
    with open_db(":memory:") as conn:
        cur = conn.execute("SELECT value FROM scribe_meta WHERE key='schema_version'")
        assert cur.fetchone() is not None
