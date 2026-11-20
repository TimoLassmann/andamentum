"""andamentum.serve: the routes, the model queue and the records, with stubbed procedures."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import andamentum.whetstone as whetstone
from andamentum.document_store import Hit, Store
from andamentum.serve import ServeError, create_app
from andamentum.serve import app as app_mod
from andamentum.serve.cli import SECRET_ENV, main
from andamentum.serve.queue import is_local
from andamentum.whetstone import Edit

SRC = Path(__file__).resolve().parents[2]
NOTE = "# Apollo\n\nThe rover moved slow across the the crater.\n"
SECRET = "s" * 32
AUTH = {"Authorization": f"Bearer {SECRET}"}


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    return TestClient(create_app(out=tmp_path / "records", secret=SECRET, **kwargs), headers=AUTH)


def _records(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "records").iterdir())


# --- bodies and errors -------------------------------------------------------------


def test_an_unknown_body_field_is_a_422_with_one_error_sentence(tmp_path: Path) -> None:
    reply = _client(tmp_path).post(
        "/procedure/review_document", json={"note": NOTE, "model": "m", "path": "/etc/hosts"}
    )
    assert reply.status_code == 422
    assert set(reply.json()) == {"error"}
    assert "path" in reply.json()["error"]


def test_a_procedure_without_a_model_is_a_422(tmp_path: Path) -> None:
    reply = _client(tmp_path).post("/procedure/research", json={"question": "q", "mode": "scholarly"})
    assert reply.status_code == 422
    assert "model" in reply.json()["error"]


def test_an_unknown_route_is_a_404_in_the_error_shape(tmp_path: Path) -> None:
    reply = _client(tmp_path).post("/procedure/decide", json={})
    assert reply.status_code == 404
    assert reply.json() == {"error": "no such route"}


def test_web_research_without_a_web_source_is_refused(tmp_path: Path) -> None:
    reply = _client(tmp_path).post(
        "/procedure/research", json={"question": "q", "mode": "web", "model": "m"}
    )
    assert reply.status_code == 422
    assert "--searxng-url" in reply.json()["error"]


def test_a_store_without_its_models_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ServeError):
        create_app(out=tmp_path, secret=SECRET, store=tmp_path / "s.db", embedding_model="e")


# --- the shared secret ---------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer " + "x" * 32}, {"Authorization": SECRET},
     {"Authorization": f"Bearer {SECRET}x"}],
    ids=["none", "wrong", "no-scheme", "longer"],
)
@pytest.mark.parametrize(
    "method,route", [("get", "/health"), ("post", "/procedure/accept_edits"), ("post", "/nowhere")]
)
def test_a_request_without_the_secret_is_a_401_and_runs_nothing(
    tmp_path: Path, headers: dict[str, str], method: str, route: str
) -> None:
    """Mutation: drop the middleware, or compare only a prefix, and this fails."""
    client = TestClient(create_app(out=tmp_path / "records", secret=SECRET))
    body = {"note": NOTE, "edits": []} if method == "post" else None
    reply = client.request(method.upper(), route, json=body, headers=headers)
    assert reply.status_code == 401
    assert set(reply.json()) == {"error"}
    assert reply.headers["www-authenticate"] == "Bearer"
    assert _records(tmp_path) == []


def test_a_request_with_the_secret_is_answered(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/health").status_code == 200


def test_the_secret_is_never_recorded(tmp_path: Path) -> None:
    _client(tmp_path).post("/procedure/accept_edits", json={"note": NOTE, "edits": []})
    (folder,) = _records(tmp_path)
    assert SECRET not in (folder / "request.json").read_text()


def test_a_short_secret_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ServeError, match="32"):
        create_app(out=tmp_path, secret="s" * 31)


# --- review: the note is text, never a path ---------------------------------------


class _Result:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return self.data


def test_a_note_that_names_a_file_is_reviewed_as_that_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("SECRET CONTENT\n")
    seen: dict[str, Any] = {}

    async def fake_review(document: Any, *, model: str, reviewer: Any) -> _Result:
        seen["text"], seen["model"] = document.text, model
        return _Result({"edits": []})

    monkeypatch.setattr(whetstone, "review_document", fake_review)
    reply = _client(tmp_path).post(
        "/procedure/review_document", json={"note": str(secret), "model": "ollama:x-cloud"}
    )
    assert reply.status_code == 200, reply.json()
    assert str(secret) in seen["text"]
    assert "SECRET" not in seen["text"]
    assert seen["model"] == "ollama:x-cloud"


def test_an_unknown_reviewer_is_a_422(tmp_path: Path) -> None:
    reply = _client(tmp_path).post(
        "/procedure/review_document", json={"note": NOTE, "model": "m", "reviewer": "nobody"}
    )
    assert reply.status_code == 422


# --- accept_edits ------------------------------------------------------------------


def _edit(original: str, new: str, at: int) -> dict[str, Any]:
    edit = Edit(
        title="t", rationale="r", section_id="s1", char_start=at,
        char_end=at + len(original), original_text=original, new_text=new,
    )
    return edit.model_dump(mode="json")


def test_accept_edits_names_every_edit_once_and_needs_no_model(tmp_path: Path) -> None:
    edits = [_edit("moved slow", "moved slowly", 20), _edit("nowhere", "x", 0),
             _edit("the the", "the", 40)]
    reply = _client(tmp_path).post("/procedure/accept_edits", json={"note": NOTE, "edits": edits})
    assert reply.status_code == 200
    result = reply.json()
    named = result["applied"] + [n["id"] for n in result["not_applied"]]
    assert sorted(named) == sorted(e["id"] for e in edits)
    assert "moved slowly across the crater" in result["text"]


def test_accept_edits_with_a_malformed_edit_is_a_422(tmp_path: Path) -> None:
    reply = _client(tmp_path).post(
        "/procedure/accept_edits", json={"note": NOTE, "edits": [{"title": "t"}]}
    )
    assert reply.status_code == 422
    assert reply.json()["error"].startswith("edits:")


# --- records under --out -----------------------------------------------------------


def test_each_procedure_call_is_recorded_with_its_caller(tmp_path: Path) -> None:
    edits = [_edit("moved slow", "moved slowly", 20)]
    _client(tmp_path).post(
        "/procedure/accept_edits",
        json={"note": NOTE, "edits": edits},
        headers={"X-Agent": "agent:m", "X-Run-Session": "s1", "X-Scope": "home"},
    )
    (folder,) = _records(tmp_path)
    request = json.loads((folder / "request.json").read_text())
    assert request["procedure"] == "accept_edits"
    assert request["caller"] == {"x-agent": "agent:m", "x-run-session": "s1", "x-scope": "home"}
    assert json.loads((folder / "result.json").read_text())["applied"] == [edits[0]["id"]]


def test_a_failed_run_is_a_500_and_leaves_error_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(document: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the model went away")

    monkeypatch.setattr(whetstone, "review_document", broken)
    reply = _client(tmp_path).post(
        "/procedure/review_document", json={"note": NOTE, "model": "ollama:x-cloud"}
    )
    assert reply.status_code == 500
    assert "the model went away" in reply.json()["error"]
    (folder,) = _records(tmp_path)
    assert not (folder / "result.json").exists()
    assert "the model went away" in json.loads((folder / "error.json").read_text())["error"]


# --- the model queue ---------------------------------------------------------------


def test_which_models_are_local() -> None:
    assert is_local("ollama:gemma4:31b")
    assert not is_local("ollama:gemma4:31b-cloud")
    assert not is_local("openai:gpt-5.4-nano")


async def _two_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []

    async def slow_review(document: Any, **kwargs: Any) -> _Result:
        start = time.monotonic()
        await asyncio.sleep(0.2)
        spans.append((start, time.monotonic()))
        return _Result({"edits": []})

    monkeypatch.setattr(whetstone, "review_document", slow_review)
    app = create_app(out=tmp_path / "records", secret=SECRET)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", headers=AUTH) as client:
        replies = await asyncio.gather(*[
            client.post("/procedure/review_document", json={"note": NOTE, "model": model})
            for _ in range(2)
        ])
    assert [r.status_code for r in replies] == [200, 200]
    return sorted(spans)


async def test_two_calls_on_a_local_model_run_one_after_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = await _two_calls(tmp_path, monkeypatch, "ollama:gemma4:31b")
    assert second[0] >= first[1]


async def test_two_calls_on_a_cloud_model_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = await _two_calls(tmp_path, monkeypatch, "ollama:gemma4:31b-cloud")
    assert second[0] < first[1]


# --- the library ---------------------------------------------------------------------


class _Searcher:
    def indexes(self) -> list[Any]:
        return []

    async def search(
        self, store: Store, query: str, *, limit: int, where: Any, since: Any = None, until: Any = None
    ) -> list[Hit]:
        doc = (await store.list(limit=1))[0]
        return [Hit(doc_id=doc.doc_id, rev=1, start=0, end=8, text="# Apollo", score=0.9)][:limit]


def test_the_library_answers_503_without_a_store(tmp_path: Path) -> None:
    reply = _client(tmp_path).post("/library/search", json={"query": "rover"})
    assert reply.status_code == 503
    assert "--store" in reply.json()["error"]


def test_library_search_and_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "library.db"

    async def fake_open(path: Path, **kwargs: Any) -> Store:
        store = await Store.open(path, searcher=_Searcher())
        if not await store.list(limit=1):
            await store.put(NOTE, key="apollo.md", metadata={"title": "Apollo"})
        return store

    monkeypatch.setattr(app_mod, "_open_store", fake_open)
    with _client(tmp_path, store=db, embedding_model="e", reranker_model="r") as client:
        hits = client.post("/library/search", json={"query": "rover", "limit": 5}).json()["hits"]
        assert [(h["key"], h["title"], h["text"]) for h in hits] == [("apollo.md", "Apollo", "# Apollo")]
        doc_id = hits[0]["doc_id"]
        part = client.post("/library/read", json={"doc_id": doc_id, "start": 2, "end": 8}).json()
        assert part["text"] == "Apollo" and part["length"] == len(NOTE)
        assert client.post("/library/read", json={"doc_id": "nope"}).status_code == 404
        assert client.post("/library/read", json={"doc_id": doc_id, "end": 10_000}).status_code == 422
        assert client.post("/library/search", json={"query": "x", "limit": 51}).status_code == 422
        assert client.get("/health").json()["library"] is True


class _WhereSearcher:
    """One hit per live document that matches `where`: filtered before ranking, as the real searcher."""

    def indexes(self) -> list[Any]:
        return []

    async def search(
        self, store: Store, query: str, *, limit: int, where: Any, since: Any = None, until: Any = None
    ) -> list[Hit]:
        refs = await store.list(where=where, since=since, until=until, limit=None)
        return [
            Hit(doc_id=r.doc_id, rev=r.rev, start=0, end=8, text="# Apollo", score=0.9)
            for r in refs
        ][:limit]


def test_a_where_limits_the_library_search_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "library.db"

    async def fake_open(path: Path, **kwargs: Any) -> Store:
        store = await Store.open(path, searcher=_WhereSearcher())
        if not await store.list(limit=1):
            await store.put(NOTE, key="open.md", metadata={"title": "Open", "public": True})
            await store.put(NOTE, key="closed.md", metadata={"title": "Closed"})
        return store

    monkeypatch.setattr(app_mod, "_open_store", fake_open)
    public = {"public": True}
    with _client(tmp_path, store=db, embedding_model="e", reranker_model="r") as client:
        every = client.post("/library/search", json={"query": "rover"}).json()["hits"]
        assert sorted(h["key"] for h in every) == ["closed.md", "open.md"]
        ids = {h["key"]: h["doc_id"] for h in every}
        hits = client.post("/library/search", json={"query": "rover", "where": public}).json()["hits"]
        assert [h["key"] for h in hits] == ["open.md"]
        assert client.post("/library/read", json={"doc_id": ids["open.md"], "where": public}).status_code == 200
        refused = client.post("/library/read", json={"doc_id": ids["closed.md"], "where": public, "end": 10_000})
        assert refused.status_code == 403
        assert refused.json() == {
            "error": f"document {ids['closed.md']} is private, and this run may read public documents only"
        }
        assert client.post("/library/read", json={"doc_id": ids["closed.md"]}).status_code == 200
        assert client.post("/library/read", json={"doc_id": "nope", "where": public}).status_code == 404
        assert client.post("/library/search", json={"query": "rover", "where": "public"}).status_code == 422


def _dated_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A library of three notes, first put at 07:00, 09:00 and 11:00 Perth time."""
    from datetime import datetime, timedelta, timezone

    perth = timezone(timedelta(hours=8))

    async def fake_open(path: Path, **kwargs: Any) -> Store:
        store = await Store.open(path, searcher=_WhereSearcher())
        if not await store.list(limit=1):
            for hour in (7, 9, 11):
                doc_id = await store.put(
                    NOTE, key=f"n{hour}.md", metadata={"title": f"N{hour}", "type": "note"}
                )
                at = datetime(2026, 11, 19, hour, tzinfo=perth).astimezone(timezone.utc)
                store.db.execute(
                    "UPDATE documents SET created_at = ? WHERE id = ?",
                    (at.isoformat(timespec="microseconds"), doc_id),
                )
        return store

    monkeypatch.setattr(app_mod, "_open_store", fake_open)


def test_library_list_filters_by_window_and_where_and_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _dated_open(monkeypatch)
    with _client(tmp_path, store=tmp_path / "library.db", embedding_model="e", reranker_model="r") as client:
        reply = client.post("/library/list", json={"since": "2026-11-19T09:00:00+08:00"}).json()
        assert [d["key"] for d in reply["documents"]] == ["n11.md", "n9.md"]
        assert reply["cursor"] is None  # fewer than limit: the last page
        first = reply["documents"][0]
        assert first["title"] == "N11" and first["metadata"]["type"] == "note"
        assert first["created_at"].startswith("2026-11-19T03:00:00")
        window = {"since": "2026-11-19T07:00:00+08:00", "until": "2026-11-19T11:00:00+08:00"}
        assert [d["key"] for d in client.post("/library/list", json=window).json()["documents"]] == [
            "n9.md", "n7.md"
        ]
        assert client.post("/library/list", json={"where": {"type": "other"}}).json()["documents"] == []
        page1 = client.post("/library/list", json={"limit": 2}).json()
        assert len(page1["documents"]) == 2 and page1["cursor"] is not None
        page2 = client.post("/library/list", json={"limit": 2, "cursor": page1["cursor"]}).json()
        assert [d["key"] for d in page2["documents"]] == ["n7.md"]
        assert client.post("/library/list", json={"cursor": "x"}).status_code == 422
        assert client.post("/library/list", json={"limit": 201}).status_code == 422


@pytest.mark.parametrize(
    "window",
    [
        {"since": "2026-11-19"},  # a bare date
        {"since": "2026-11-19T09:00:00"},  # no zone
        {"since": "2026-11-19T09:00:00+08:00", "until": "2026-11-19T01:00:00Z"},  # the same moment
    ],
)
def test_library_list_and_search_refuse_a_time_without_a_zone_or_an_empty_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: dict[str, str]
) -> None:
    _dated_open(monkeypatch)
    with _client(tmp_path, store=tmp_path / "library.db", embedding_model="e", reranker_model="r") as client:
        assert client.post("/library/list", json=window).status_code == 422
        assert client.post("/library/search", json={"query": "rover", **window}).status_code == 422


def test_library_search_keeps_to_the_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _dated_open(monkeypatch)
    with _client(tmp_path, store=tmp_path / "library.db", embedding_model="e", reranker_model="r") as client:
        hits = client.post(
            "/library/search", json={"query": "rover", "until": "2026-11-19T09:00:00+08:00"}
        ).json()["hits"]
        assert [h["key"] for h in hits] == ["n7.md"]


# --- the command and the layering -----------------------------------------------------


def test_the_command_refuses_a_store_without_models(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--out", "records", "--store", "library.db"])
    assert exc.value.code == 2


def test_the_command_refuses_to_start_without_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(SECRET_ENV, raising=False)
    assert main(["--out", str(tmp_path / "r")]) == 2
    assert SECRET_ENV in capsys.readouterr().err


def test_the_command_refuses_a_short_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(SECRET_ENV, "short")
    assert main(["--out", str(tmp_path / "r")]) == 2
    assert "32" in capsys.readouterr().err


def test_the_command_reports_a_missing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(SECRET_ENV, SECRET)
    code = main(["--out", str(tmp_path / "r"), "--store", str(tmp_path / "none.db"),
                 "--embedding-model", "e", "--reranker-model", "r"])
    assert code == 3
    assert "no store" in capsys.readouterr().err


def test_nothing_imports_serve() -> None:
    """Mutation: import andamentum.serve from any other module and this fails."""
    imports = re.compile(r"^\s*(?:from|import)\s+(?:andamentum\.serve\b|\.+serve\b)", re.M)
    importers = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.relative_to(SRC).parts[0] != "serve" and imports.search(path.read_text())
    ]
    assert importers == []


# --- keyword search: no model at all -----------------------------------------------


def test_a_keyword_store_opens_and_searches_with_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from andamentum.document_store import HybridIndex, reranking

    async def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("a keyword search called a model")

    monkeypatch.setattr(HybridIndex, "embed", refuse)
    monkeypatch.setattr(reranking, "rerank", refuse)
    db = tmp_path / "library.db"

    async def make() -> None:
        s = await Store.open(db)
        await s.put(NOTE, key="apollo.md", metadata={"title": "Apollo"})
        await s.put("# Garden\n\nSoil notes.\n", key="garden.md")
        await s.close()

    asyncio.run(make())
    with _client(tmp_path, store=db, index="keyword") as client:
        hits = client.post("/library/search", json={"query": "rover crater"}).json()["hits"]
        assert [(h["key"], h["title"]) for h in hits] == [("apollo.md", "Apollo")]
        assert client.get("/health").json()["library"] is True


def test_a_keyword_store_refuses_a_model_and_a_hybrid_store_needs_both(tmp_path: Path) -> None:
    with pytest.raises(ServeError, match="calls no model"):
        create_app(out=tmp_path / "r", secret=SECRET, store=tmp_path / "x.db",
                   index="keyword", embedding_model="e")
    with pytest.raises(ServeError, match="needs both"):
        create_app(out=tmp_path / "r", secret=SECRET, store=tmp_path / "x.db")


@pytest.mark.parametrize(
    "argv",
    [
        ["--out", "records", "--store", "x.db", "--index", "keyword", "--embedding-model", "e"],
        ["--out", "records", "--store", "x.db"],
    ],
)
def test_the_command_checks_the_models_against_the_index(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
