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
from andamentum.serve.cli import main
from andamentum.serve.queue import is_local
from andamentum.whetstone import Edit

SRC = Path(__file__).resolve().parents[2]
NOTE = "# Apollo\n\nThe rover moved slow across the the crater.\n"


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    return TestClient(create_app(out=tmp_path / "records", **kwargs))


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
        create_app(out=tmp_path, store=tmp_path / "s.db", embedding_model="e")


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
    app = create_app(out=tmp_path / "records")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
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

    async def search(self, store: Store, query: str, *, limit: int, where: Any) -> list[Hit]:
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


# --- the command and the layering -----------------------------------------------------


def test_the_command_refuses_a_store_without_models(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--out", "records", "--store", "library.db"])
    assert exc.value.code == 2


def test_the_command_reports_a_missing_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
