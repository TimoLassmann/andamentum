"""Keyword search: BM25 only, and no model call of any kind."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from andamentum.document_store import HybridIndex, KeywordSearcher, Store
from andamentum.document_store import reranking
from andamentum.document_store.cli import main

PERTH = timezone(timedelta(hours=8))

DOCS = {
    "salt.md": "# SaltLog\n\nThe SaltLog service stores salt records. Salt records are kept for a year.",
    "apollo.md": "# Apollo\n\nApollo launched toward the moon. A salt shaker flew too.",
    "garden.md": "# Garden\n\nNotes about the garden and its soil.",
}


@pytest.fixture(autouse=True)
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any embedding or rerank call fails the test."""

    async def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a keyword search called a model")

    monkeypatch.setattr(HybridIndex, "embed", refuse)
    monkeypatch.setattr(reranking, "rerank", refuse)


async def _store(path: Path) -> tuple[Store, dict[str, str]]:
    s = await Store.open_with_keyword_search(path)
    ids = {k: await s.put(t, key=k, metadata={"title": k}) for k, t in DOCS.items()}
    return s, ids


async def test_keyword_search_ranks_by_bm25_and_calls_no_model(tmp_path: Path) -> None:
    s, ids = await _store(tmp_path / "s.db")
    hits = await s.search("salt records", limit=10)
    assert [h.doc_id for h in hits] == [ids["salt.md"], ids["apollo.md"]]
    assert hits[0].score > hits[1].score > 0
    assert hits[0].text.startswith("# SaltLog") and hits[0].start == 0
    assert await s.search("moon", limit=10) and await s.search("zebra", limit=10) == []
    await s.close()


async def test_limit_cuts_the_bm25_order(tmp_path: Path) -> None:
    s, ids = await _store(tmp_path / "s.db")
    (hit,) = await s.search("salt", limit=1)
    assert hit.doc_id == ids["salt.md"]
    await s.close()


async def test_keyword_search_keeps_to_where_and_the_window(tmp_path: Path) -> None:
    s, ids = await _store(tmp_path / "s.db")
    await s.update(ids["apollo.md"], metadata={"title": "apollo.md", "type": "note"})
    hits = await s.search("salt", limit=10, where={"type": "note"})
    assert [h.doc_id for h in hits] == [ids["apollo.md"]]
    s.db.execute(
        "UPDATE documents SET created_at = ? WHERE id = ?",
        ("2026-11-18T00:00:00.000000+00:00", ids["salt.md"]),
    )
    since = datetime(2026, 11, 19, tzinfo=PERTH)
    assert [h.doc_id for h in await s.search("salt", limit=10, since=since)] == [ids["apollo.md"]]
    assert [h.doc_id for h in await s.search("salt", limit=10, until=since)] == [ids["salt.md"]]
    await s.close()


async def test_the_keyword_index_sits_beside_a_hybrid_index(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    s, _ = await _store(db)
    await s.close()
    # a store whose hybrid index was built earlier keeps it
    s = await Store.open(db, indexes=[HybridIndex(embedding_model=None)])
    await s.sync()
    await s.close()
    s = await Store.open_with_keyword_search(db)
    assert (await s.sync()).ok
    tables = {r[0] for r in s.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"ix_keyword_chunks", "ix_hybrid_chunks"} <= tables
    await s.close()


def test_keyword_searcher_refuses_an_index_with_an_embedding_model() -> None:
    with pytest.raises(ValueError, match="without an embedding model"):
        KeywordSearcher(index=HybridIndex(embedding_model="e"))


# --- the command -------------------------------------------------------------------


def test_the_command_syncs_and_searches_with_no_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio

    db = tmp_path / "s.db"

    async def make() -> None:
        s = await Store.open(db)
        for k, t in DOCS.items():
            await s.put(t, key=k)
        await s.close()

    asyncio.run(make())
    assert main(["sync", str(db), "--index", "keyword", "--json"]) == 0
    capsys.readouterr()
    assert main(["search", str(db), "salt records", "--index", "keyword", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["doc"] for r in rows] == ["salt.md", "apollo.md"]
    # a small store gives tiny BM25 scores; the JSON keeps them, so the order shows
    assert rows[0]["score"] > rows[1]["score"] > 0


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "Q", "--index", "keyword", "--reranker-model", "r"],
        ["search", "Q", "--index", "keyword", "--embedding-model", "e"],
        ["sync", "--index", "keyword", "--embedding-model", "e"],
        ["search", "Q", "--embedding-model", "e"],  # hybrid needs the reranker
        ["sync"],  # hybrid needs the embedding model
    ],
)
def test_model_flags_must_match_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    db = tmp_path / "s.db"
    db.touch()
    with pytest.raises(SystemExit) as exc:
        main([argv[0], str(db), *argv[1:]])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
