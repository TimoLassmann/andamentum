"""``--store`` validation in ``run_draft`` and the real cite tools' outage capture."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import httpx
import pytest

from andamentum.compose.harness.draft import search
from andamentum.compose.harness.literature import Sources
from andamentum.deep_research import FinderResult
from andamentum.references import ReferenceHttp

from ..draft.run import DraftUsageError
from ..draft.verb import DraftParams, run_draft
from ._draft_helpers import install_script, make_project, no_cite_tools

TASK = "Write the SaltLog pilot article"


async def _draft(tmp_path: Path, **extra: object) -> object:
    notes, results = make_project(tmp_path)
    return await run_draft(
        DraftParams(
            task=TASK,
            out=tmp_path / "paper.docx",
            model="stub:model",
            source=[str(notes)],
            results=results,
            **extra,  # type: ignore[arg-type]
        ),
        transport=httpx.MockTransport(lambda request: httpx.Response(404)),
        url_check=lambda url: (True, ""),
        open_cite=no_cite_tools(tmp_path),
    )


async def test_store_with_offline_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    with pytest.raises(DraftUsageError):
        await _draft(tmp_path, store=tmp_path / "s.db", offline=True)


async def test_store_that_is_a_folder_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    folder = tmp_path / "store_dir"
    folder.mkdir()
    with pytest.raises(DraftUsageError):
        await _draft(tmp_path, store=folder)


async def test_store_that_is_not_sqlite_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("just text, not a database\n")
    with pytest.raises(DraftUsageError, match="notes.txt"):
        await _draft(tmp_path, store=bogus)


async def test_store_of_another_sqlite_schema_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_script(monkeypatch)
    other = tmp_path / "other.db"
    con = sqlite3.connect(other)
    con.execute("CREATE TABLE orders (id INTEGER)")
    con.commit()
    con.close()
    with pytest.raises(DraftUsageError, match="other.db"):
        await _draft(tmp_path, store=other)


async def test_the_finders_search_outage_lands_in_source_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = "openalex search failed for 'x': HTTP 503"
    calls = 0

    async def fake_find(sentence: str, **_kwargs: object) -> FinderResult:
        nonlocal calls
        calls += 1
        return FinderResult(
            sentence=sentence, follow_citations=True, works=[], degraded=[line]
        )

    monkeypatch.setattr("andamentum.deep_research.find_works", fake_find)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(404))
    )
    http = ReferenceHttp(client=client)
    async with search.cite_tools(
        sources=Sources(),
        http=http,
        model="stub:m",
        store=tmp_path / "papers.db",
        cache=tmp_path / "cache.db",
        mailto=None,
    ) as tools:
        assert await tools.finder("A sentence.", paragraph="A sentence.") == []
        await tools.finder("A sentence.", paragraph="A sentence.")  # repeated
        assert tools.source_notes == [f"search outage: {line}"]
    assert calls == 2
    await client.aclose()


async def test_an_existing_document_store_is_accepted(tmp_path: Path) -> None:
    from andamentum.document_store import Store

    from ..draft.verb import _check_store_file

    path = tmp_path / "papers.db"
    async with await Store.open(path) as papers:
        await papers.put("An invented paper.", key="doi:10.5555/x.1")
    _check_store_file(path)  # no error: a real store is reused
