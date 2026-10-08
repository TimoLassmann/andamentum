"""Regression: ask databases are deleted where they were created.

``ask`` opened its database at ``resolve_db_path(name, db_dir)`` (for an
``ask_*`` name, the ephemeral directory) but deleted it with
``delete_database(name)``, which looked in the main directory, so nothing was
deleted. ``cleanup`` scanned only the main directory and never found them.
"""

from __future__ import annotations

import os
import time

from andamentum.epistemic.cli_handlers import handle_cleanup
from andamentum.epistemic.databases import (
    delete_store_files,
    get_ephemeral_dir,
    open_store,
    resolve_db_path,
)


async def _make(name: str, db_dir=None):
    store = await open_store(name, db_dir)
    await store.put("x")
    await store.close()
    return resolve_db_path(name, db_dir)


async def test_an_ask_database_is_deleted_where_it_was_opened(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path / "dbs"))
    path = await _make("ask_0123abcd")
    assert path.parent == get_ephemeral_dir()
    (path.parent / "ask_0123abcd_raw").mkdir()
    assert delete_store_files(resolve_db_path("ask_0123abcd")) is True
    assert list(path.parent.iterdir()) == []
    assert delete_store_files(resolve_db_path("ask_0123abcd")) is False


async def test_an_ask_database_in_a_given_directory_is_deleted_there(tmp_path):
    path = await _make("ask_feedbeef", tmp_path / "mine")
    assert path.parent == tmp_path / "mine"
    assert delete_store_files(resolve_db_path("ask_feedbeef", tmp_path / "mine")) is True
    assert not path.exists()


async def test_cleanup_finds_old_ask_databases_in_the_ephemeral_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCUMENT_STORE_DIR", str(tmp_path / "dbs"))
    old = await _make("ask_00000001")
    new = await _make("ask_00000002")
    long_ago = time.time() - 30 * 86400
    for f in old.parent.glob("ask_00000001*"):
        os.utime(f, (long_ago, long_ago))
    result = await handle_cleanup(older_than_days=7)
    assert result.deleted == 1
    assert not any(old.parent.glob("ask_00000001*"))
    assert new.exists()
