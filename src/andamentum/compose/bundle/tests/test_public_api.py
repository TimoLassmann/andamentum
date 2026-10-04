"""The public surface is exactly __all__, and no SQLite remains."""

import andamentum.compose.bundle as bundle_pkg


def test_all_names_resolve():
    for name in bundle_pkg.__all__:
        assert hasattr(bundle_pkg, name)
    assert "Bundle" in bundle_pkg.__all__ and "Document" not in bundle_pkg.__all__


def test_no_sqlite_in_bundle_sources():
    from pathlib import Path

    root = Path(bundle_pkg.__file__).parent
    for path in root.glob("*.py"):
        assert "sqlite3" not in path.read_text(encoding="utf-8"), path.name
