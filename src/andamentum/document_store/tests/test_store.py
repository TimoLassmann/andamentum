"""The fixed source layer and API: put/get/update/delete/list/count/where/read."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from andamentum.document_store import (
    DocumentNotFound,
    KeyExists,
    Store,
    group_by_document,
    line_range,
)
from andamentum.document_store.db import SCHEMA_VERSION
from andamentum.document_store.types import Hit


@pytest.fixture
async def store(tmp_path):
    s = await Store.open(tmp_path / "s.db")
    yield s
    await s.close()


async def test_put_get_round_trip_keeps_text_and_metadata(store):
    text = "# Apollo\n\nThe SaltLog ships logs.\n"
    doc_id = await store.put(text, key="doc:a", metadata={"kind": "note", "n": 3})
    doc = await store.get(doc_id)
    assert doc is not None
    assert (doc.text, doc.key, doc.rev, doc.media_type) == (text, "doc:a", 1, "text/markdown")
    assert doc.metadata == {"kind": "note", "n": 3}
    assert doc.original_hash is None


async def test_put_never_deduplicates(store):
    a = await store.put("same text")
    b = await store.put("same text")
    assert a != b
    assert await store.count() == 2


async def test_put_with_a_taken_key_raises(store):
    await store.put("one", key="k")
    with pytest.raises(KeyExists):
        await store.put("two", key="k")


async def test_a_deleted_document_gives_its_key_up(store):
    doc_id = await store.put("one", key="k")
    await store.delete(doc_id)
    assert await store.resolve("k") is None
    new_id = await store.put("two", key="k")
    assert await store.resolve("k") == new_id
    with pytest.raises(KeyExists):
        await store.restore(doc_id)  # the key is taken now
    await store.delete(new_id)
    assert await store.restore(doc_id) is True
    assert await store.resolve("k") == doc_id


async def test_resolve_finds_the_live_document_by_key(store):
    doc_id = await store.put("x", key="claim:1234")
    assert await store.resolve("claim:1234") == doc_id
    assert await store.resolve("claim:nope") is None


async def test_update_writes_a_new_revision_and_keeps_the_old(store):
    doc_id = await store.put("v1", metadata={"a": 1})
    assert await store.update(doc_id, text="v2") == 2
    assert await store.update(doc_id, metadata={"b": 2}) == 3
    head = await store.get(doc_id)
    assert head is not None
    assert (head.rev, head.text, head.metadata) == (3, "v2", {"b": 2})
    first = await store.get(doc_id, rev=1)
    assert first is not None
    assert (first.text, first.metadata) == ("v1", {"a": 1})
    assert await store.get(doc_id, rev=9) is None


async def test_update_replaces_metadata_as_a_whole(store):
    doc_id = await store.put("x", metadata={"a": 1, "b": 2})
    await store.update(doc_id, metadata={"a": 5})
    doc = await store.get(doc_id)
    assert doc is not None and doc.metadata == {"a": 5}


async def test_update_needs_something_and_a_live_document(store):
    doc_id = await store.put("x")
    with pytest.raises(ValueError):
        await store.update(doc_id)
    await store.delete(doc_id)
    with pytest.raises(DocumentNotFound):
        await store.update(doc_id, text="y")
    with pytest.raises(DocumentNotFound):
        await store.update("01NOPE", text="y")


async def test_delete_restore_purge(store):
    doc_id = await store.put("x", original=b"%PDF bytes")
    assert await store.delete(doc_id) is True
    assert await store.delete(doc_id) is False
    assert await store.get(doc_id) is None
    assert await store.count() == 0
    assert await store.restore(doc_id) is True
    assert await store.restore(doc_id) is False
    assert await store.get(doc_id) is not None
    assert await store.purge(doc_id) is True
    assert await store.purge(doc_id) is False
    assert await store.get(doc_id) is None
    assert store.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0


async def test_original_bytes_are_kept_and_shared(store):
    a = await store.put("text a", original=b"same bytes")
    b = await store.put("text b", original=b"same bytes")
    assert await store.original(a) == b"same bytes"
    assert store.db.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1
    await store.purge(a)
    assert await store.original(b) == b"same bytes"  # still referenced
    assert await store.original(await store.put("no original")) is None


async def test_read_returns_a_span_of_a_revision(store):
    doc_id = await store.put("0123456789")
    await store.update(doc_id, text="abcdefghij")
    assert await store.read(doc_id, rev=1, start=2, end=5) == "234"
    assert await store.read(doc_id, rev=2, start=2, end=5) == "cde"
    with pytest.raises(ValueError):
        await store.read(doc_id, rev=2, start=5, end=50)
    with pytest.raises(DocumentNotFound):
        await store.read(doc_id, rev=7, start=0, end=1)


async def test_list_is_newest_first_and_has_no_silent_limit(store):
    ids = [await store.put(f"doc {i}", metadata={"type": "claim"}) for i in range(130)]
    refs = await store.list(where={"type": "claim"}, limit=None)
    assert len(refs) == 130
    assert [r.doc_id for r in refs] == list(reversed(ids))
    await store.update(ids[0], text="touched")
    assert (await store.list(limit=1))[0].doc_id == ids[0]


async def test_list_pages_with_a_cursor(store):
    ids = [await store.put(f"doc {i}") for i in range(7)]
    seen: list[str] = []
    cursor = None
    while True:
        page = await store.list(limit=3, cursor=cursor)
        if not page:
            break
        seen += [r.doc_id for r in page]
        cursor = page[-1].cursor
    assert seen == list(reversed(ids))


async def test_list_requires_an_explicit_limit(store):
    with pytest.raises(TypeError):
        await store.list()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        await store.list(limit=0)


async def test_where_equality_null_membership_nested_and_booleans(store):
    await store.put("a", metadata={"t": "claim", "stage": 2, "ok": True, "bib": {"year": 2024}})
    await store.put("b", metadata={"t": "claim", "stage": 3, "ok": False, "resolution": None})
    await store.put("c", metadata={"t": "evidence"})
    assert await store.count(where={"t": "claim"}) == 2
    assert await store.count(where={"t": "claim", "stage": 3}) == 1
    assert await store.count(where={"stage": [2, 3]}) == 2
    assert await store.count(where={"ok": True}) == 1
    assert await store.count(where={"ok": False}) == 1
    assert await store.count(where={"resolution": None}) == 3  # missing or null
    assert await store.count(where={"bib.year": 2024}) == 1
    assert await store.count(where={"t": ["claim", "evidence"], "stage": None}) == 1


async def test_where_rejects_bad_values(store):
    with pytest.raises(TypeError):
        await store.count(where={"x": {"nested": 1}})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        await store.count(where={"a..b": 1})
    with pytest.raises(TypeError):
        await store.count(where={"x": [1, None]})  # type: ignore[list-item]


async def test_where_empty_list_matches_nothing_and_tuples_work(store):
    await store.put("a", metadata={"t": "claim"})
    assert await store.count(where={"t": []}) == 0
    assert await store.count(where={"t": ("claim", "x")}) == 1  # type: ignore[dict-item]


async def test_where_keeps_booleans_and_numbers_apart(store):
    await store.put("a", metadata={"v": True})
    await store.put("b", metadata={"v": 1})
    await store.put("c", metadata={"v": "1"})
    assert await store.count(where={"v": True}) == 1
    assert await store.count(where={"v": 1}) == 1
    assert await store.count(where={"v": "1"}) == 1
    assert await store.count(where={"v": [True, "1"]}) == 2


async def test_where_reaches_non_ascii_field_names(store):
    await store.put("a", metadata={"é": 1})
    assert await store.count(where={"é": 1}) == 1


async def test_paging_survives_an_update_of_the_cursor_document(store):
    ids = [await store.put(f"doc {i}") for i in range(6)]
    first = await store.list(limit=2)
    await store.update(first[-1].doc_id, text="changed")
    rest = await store.list(limit=None, cursor=first[-1].cursor)
    assert [r.doc_id for r in rest] == list(reversed(ids))[2:]


async def test_where_field_names_cannot_inject_sql(store):
    await store.put("a", metadata={"weird'name": 1})
    assert await store.count(where={"weird'name": 1}) == 1
    assert await store.count(where={'x") OR 1=1 --': 1}) == 0


async def test_metadata_must_be_a_dict(store):
    with pytest.raises(TypeError):
        await store.put("x", metadata=["not", "a", "dict"])  # type: ignore[arg-type]


async def test_two_stores_on_one_file_write_concurrently(tmp_path):
    path = tmp_path / "shared.db"
    a, b = await Store.open(path), await Store.open(path)
    try:

        async def writer(s: Store, tag: str) -> None:
            for i in range(40):
                await s.put(f"{tag} {i}", key=f"{tag}:{i}")
                await asyncio.sleep(0)

        await asyncio.gather(writer(a, "a"), writer(b, "b"))
        assert await a.count() == 80
    finally:
        await a.close()
        await b.close()


async def test_a_file_of_another_schema_version_is_refused(tmp_path):
    path = tmp_path / "old.db"
    s = await Store.open(path)
    await s.close()
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value = ? WHERE name = 'schema_version'", (str(SCHEMA_VERSION + 1),))
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="schema version"):
        await Store.open(path)


async def test_every_write_is_in_the_change_log(store):
    doc_id = await store.put("x")
    await store.update(doc_id, text="y")
    await store.delete(doc_id)
    await store.restore(doc_id)
    await store.purge(doc_id)
    ops = [r[0] for r in store.db.execute("SELECT op FROM changes ORDER BY seq")]
    assert ops == ["put", "update", "delete", "restore", "purge"]


async def test_metadata_is_stored_as_json(store):
    doc_id = await store.put("x", metadata={"é": "ü"})
    raw = store.db.execute("SELECT metadata FROM revisions WHERE doc_id = ?", (doc_id,)).fetchone()[0]
    assert json.loads(raw) == {"é": "ü"}


def test_line_range():
    text = "one\ntwo\nthree\n"
    assert line_range(text, start=0, end=3) == (1, 1)
    assert line_range(text, start=4, end=13) == (2, 3)
    assert line_range(text, start=4, end=8) == (2, 2)  # the newline ends line 2
    with pytest.raises(ValueError):
        line_range(text, start=3, end=99)


def test_group_by_document_keeps_rank_order_and_takes_the_best_score():
    hits = [
        Hit("b", 1, 0, 1, "x", 0.9),
        Hit("a", 1, 0, 1, "y", 0.5),
        Hit("b", 1, 2, 3, "z", 0.95),
    ]
    docs = group_by_document(hits)
    assert [d.doc_id for d in docs] == ["b", "a"]
    assert docs[0].score == 0.95 and len(docs[0].hits) == 2
