"""The rung-2 Store (forge.runtime.Store): keyed CRUD, in-memory vs durable.

The store is the cross-run memory Port (dialect L1). These tests prove the five
operations, that ``add`` is create-or-update (idempotent by key, L8), and that a file
path actually persists across separate Store instances while ``None`` does not.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.forge.runtime import Store


def test_add_get_roundtrip_in_memory() -> None:
    s = Store()
    s.add("note", "n1", {"text": "hello", "done": False})
    assert s.get("note", "n1") == {"text": "hello", "done": False}


def test_get_missing_is_none() -> None:
    assert Store().get("note", "nope") is None


def test_list_returns_all_ordered_by_key() -> None:
    s = Store()
    s.add("note", "b", {"text": "second"})
    s.add("note", "a", {"text": "first"})
    assert s.list("note") == [{"text": "first"}, {"text": "second"}]  # key order
    assert s.list("empty") == []


def test_add_is_create_or_update() -> None:
    s = Store()
    s.add("list", "_", {"items": ["a"]})
    s.add("list", "_", {"items": ["a", "b"]})  # same key overwrites
    assert s.get("list", "_") == {"items": ["a", "b"]}
    assert len(s.list("list")) == 1  # not two rows


def test_remove_deletes_and_is_safe_when_absent() -> None:
    s = Store()
    s.add("note", "n1", {"text": "x"})
    s.remove("note", "n1")
    assert s.get("note", "n1") is None
    s.remove("note", "n1")  # no error on a missing key


def test_collections_are_isolated() -> None:
    s = Store()
    s.add("a", "k", {"v": 1})
    s.add("b", "k", {"v": 2})
    assert s.get("a", "k") == {"v": 1}
    assert s.get("b", "k") == {"v": 2}


def test_file_path_persists_across_instances(tmp_path: Path) -> None:
    db = str(tmp_path / "store.db")
    writer = Store(db)
    writer.add("note", "n1", {"text": "durable"})

    reader = Store(db)  # a fresh instance against the same file
    assert reader.get("note", "n1") == {"text": "durable"}


def test_in_memory_does_not_persist_across_instances() -> None:
    Store().add("note", "n1", {"text": "ephemeral"})
    assert Store().get("note", "n1") is None  # a new in-memory store starts empty
