"""The public surface is exactly __all__, and no SQLite remains."""

import andamentum.scribe as scribe


def test_all_names_resolve():
    for name in scribe.__all__:
        assert hasattr(scribe, name)
    assert "Bundle" in scribe.__all__ and "Document" not in scribe.__all__


def test_no_sqlite_in_scribe_sources():
    from pathlib import Path

    root = Path(scribe.__file__).parent
    for path in root.glob("*.py"):
        assert "sqlite3" not in path.read_text(encoding="utf-8"), path.name
