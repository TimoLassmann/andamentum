"""The v2 MCP server through the SDK's in-memory client, over the real engine.

Real adapters, store and reader; the network is the invented world of
``engine_world.py`` behind a mock transport, the language model is
``ScriptedLLM``.
"""

from __future__ import annotations

import functools
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, types
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research import mcp_server
from andamentum.deep_research.sources import SourceConfig
from andamentum.deep_research.store import ResearchStore
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from .engine_world import (
    RETRACTED_DOI,
    SEARXNG,
    SUBQ_APOLLO,
    World,
    world_llm,
)

ALL_TOOLS = {
    "search_works",
    "search_web",
    "expand_citations",
    "read_work",
    "record_finding",
    "get_run",
    "research",
}


@pytest.fixture(autouse=True)
def _public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


@pytest.fixture
def store() -> ResearchStore:
    return ResearchStore.in_memory()


def serve(
    *,
    model: bool = True,
    searxng: bool = True,
    world: World | None = None,
):
    """Run an async test with a connected in-memory client as its first argument.

    The session is opened inside the test's own task: an async-generator fixture
    would enter and leave the SDK's cancel scopes in different tasks.
    """

    def wrap(test):
        @functools.wraps(test)
        async def wrapper(*args: Any, store: ResearchStore, **kwargs: Any):
            here = world or World()
            tools = mcp_server.ResearchTools(
                store,
                model=world_llm().model if model else None,
                sources=SourceConfig(
                    searxng_url=SEARXNG if searxng else None, scholarly=("openalex",)
                ),
                client=here.client(store),
            )
            server = mcp_server.build_server(tools)
            async with create_connected_server_and_client_session(server) as session:
                await test(session, *args, store=store, **kwargs)

        parameters = list(inspect.signature(test).parameters.values())[1:]
        wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
        return wrapper

    return wrap


def data(result: types.CallToolResult) -> dict[str, Any]:
    assert not result.isError, result.content
    assert result.structuredContent is not None, result.content
    return result.structuredContent


def message(result: types.CallToolResult) -> str:
    (block,) = result.content
    assert isinstance(block, types.TextContent)
    return block.text


async def _apollo_works(client: ClientSession) -> dict[str, str]:
    """Search the world for the Apollo papers; their work ids by DOI."""
    found = data(await client.call_tool("search_works", {"query": "apollo latency"}))
    return {w["doi"]: w["work_id"] for w in found["works"]}


# ------------------------------------------------------------------ listing


@serve()
async def test_lists_every_tool_with_schemas(
    client: ClientSession, store: ResearchStore
):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == ALL_TOOLS
    assert tools["search_works"].inputSchema["required"] == ["query"]
    assert set(tools["record_finding"].inputSchema["required"]) == {
        "run_id",
        "subquestion_id",
        "work_id",
        "claim",
        "quote",
    }
    assert all(t.description for t in tools.values())


# ------------------------------------------------------------------- search


@serve()
async def test_search_works_keeps_the_works_in_the_store(
    client: ClientSession, store: ResearchStore
):
    result = data(await client.call_tool("search_works", {"query": "apollo latency"}))
    dois = {w["doi"] for w in result["works"]}
    assert dois == {"10.5555/apollo-a", "10.5555/apollo-b", RETRACTED_DOI}
    assert result["source_errors"] == {}
    assert {w.doi for w in store.iter_works()} == dois


@serve(world=World(searxng_down=True))
async def test_search_web_reports_a_dead_searxng_as_an_error(
    client: ClientSession, store: ResearchStore
):
    result = await client.call_tool("search_web", {"query": "apollo"})
    assert result.isError and "500" in message(result)


@serve()
async def test_search_web_keeps_results_as_works(
    client: ClientSession, store: ResearchStore
):
    result = data(await client.call_tool("search_web", {"query": "apollo"}))
    assert len(result["works"]) == 4
    assert all(w["url"].startswith("https://") for w in result["works"])


@serve(searxng=False)
async def test_search_web_needs_an_address(client: ClientSession, store: ResearchStore):
    result = await client.call_tool("search_web", {"query": "apollo"})
    text = message(result)
    assert result.isError and "--searxng-url" in text and "--browser" in text


@serve()
async def test_unknown_source_is_refused(client: ClientSession, store: ResearchStore):
    result = await client.call_tool("search_works", {"query": "x", "sources": ["nope"]})
    assert result.isError and "invalid arguments" in message(result)


@serve()
async def test_extra_arguments_are_refused(client: ClientSession, store: ResearchStore):
    result = await client.call_tool("search_works", {"query": "x", "bogus": 1})
    assert result.isError and "invalid arguments" in message(result)


@serve()
async def test_expand_citations_follows_the_graph_one_hop(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    result = data(
        await client.call_tool("expand_citations", {"work_id": ids["10.5555/apollo-a"]})
    )
    assert result["references"] == []
    assert [w["doi"] for w in result["citations"]] == ["10.5555/apollo-f"]
    assert any(w.doi == "10.5555/apollo-f" for w in store.iter_works())


@serve()
async def test_expand_citations_of_an_unknown_work(
    client: ClientSession, store: ResearchStore
):
    result = await client.call_tool("expand_citations", {"work_id": "w_missing"})
    assert result.isError and "no work" in message(result)


# --------------------------------------------------------------------- read


@serve(model=False)
async def test_read_work_fetch_stores_passages_without_a_model(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    work_id = ids["10.5555/apollo-a"]
    result = data(
        await client.call_tool("read_work", {"work_id": work_id, "fetch": True})
    )
    assert result["work"]["access"] == "abstract"
    assert result["work"]["retracted"] == "no"
    assert result["passages"] and "Apollo cache" in result["passages"][0]["text"]
    assert store.passages_for_work(work_id)


@serve(model=False)
async def test_read_work_of_a_restricted_work_has_no_abstract_anywhere(
    client: ClientSession, store: ResearchStore
):
    work = store.upsert_work(
        Work(
            doi="10.5555/restricted-read",
            title="A restricted record",
            abstract="The restricted abstract says Apollo halves latency.",
            licence=LICENCE_NO_REDISTRIBUTE,
        )
    )
    result = data(await client.call_tool("read_work", {"work_id": work.work_id}))
    assert "restricted abstract" not in json.dumps(result)
    assert result["work"]["abstract"] == ""
    assert result["abstract"] == ""


@serve(model=False)
async def test_read_work_fetch_excludes_a_retracted_work(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    work_id = ids[RETRACTED_DOI]
    result = data(
        await client.call_tool("read_work", {"work_id": work_id, "fetch": True})
    )
    assert result["work"]["retracted"] == "retracted"
    assert result["passages"] == []
    assert any(n.startswith("retracted") for n in result["notes"])


@serve()
async def test_read_work_for_a_subquestion_records_checked_findings(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    run_id = store.start_run(question="q", mode="mcp", config={}, models={})
    result = data(
        await client.call_tool(
            "read_work",
            {
                "work_id": ids["10.5555/apollo-a"],
                "subquestion": SUBQ_APOLLO,
                "run_id": run_id,
            },
        )
    )
    (finding,) = result["findings"]
    assert finding["finding_id"] == "F1"
    passages = {p["passage_id"]: p["text"] for p in result["passages"]}
    assert finding["quote"] in passages[finding["passage_id"]]
    assert [i.ref_id for i in store.run_items(run_id, "finding")] == ["F1"]


@serve()
async def test_read_work_for_a_subquestion_needs_a_run(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    result = await client.call_tool(
        "read_work",
        {"work_id": ids["10.5555/apollo-a"], "subquestion": SUBQ_APOLLO},
    )
    assert result.isError and "run_id" in message(result)


# ------------------------------------------------------------------ findings


@serve()
async def test_record_finding_checks_the_quote_and_numbers_findings(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    work_id = ids["10.5555/apollo-a"]
    run_id = store.start_run(question="q", mode="mcp", config={}, models={})
    first = data(
        await client.call_tool(
            "record_finding",
            {
                "run_id": run_id,
                "subquestion_id": "Q1",
                "work_id": work_id,
                "claim": "Apollo cut median latency to 45 ms.",
                # case and spacing differ from the abstract
                "quote": "enabling the  Apollo cache reduced median request latency",
            },
        )
    )
    assert first["finding"]["finding_id"] == "F1"
    assert first["finding"]["quote"] == (
        "Enabling the Apollo cache reduced median request latency"
    )
    second = data(
        await client.call_tool(
            "record_finding",
            {
                "run_id": run_id,
                "subquestion_id": "Q1",
                "work_id": work_id,
                "claim": "It held for every workload.",
                "quote": "in every tested workload",
            },
        )
    )
    assert second["finding"]["finding_id"] == "F2"


@serve()
async def test_record_finding_refuses_a_quote_not_in_the_work(
    client: ClientSession, store: ResearchStore
):
    ids = await _apollo_works(client)
    run_id = store.start_run(question="q", mode="mcp", config={}, models={})
    result = await client.call_tool(
        "record_finding",
        {
            "run_id": run_id,
            "subquestion_id": "Q1",
            "work_id": ids["10.5555/apollo-a"],
            "claim": "Apollo is free.",
            "quote": "the Apollo cache costs nothing at all",
        },
    )
    assert result.isError and "does not occur" in message(result)


@serve()
async def test_record_finding_refuses_the_abstract_of_a_restricted_work(
    client: ClientSession, store: ResearchStore
):
    work = store.upsert_work(
        Work(
            doi="10.5555/restricted",
            title="A restricted record",
            abstract="The restricted abstract says Apollo halves latency.",
            licence=LICENCE_NO_REDISTRIBUTE,
        )
    )
    run_id = store.start_run(question="q", mode="mcp", config={}, models={})
    result = await client.call_tool(
        "record_finding",
        {
            "run_id": run_id,
            "subquestion_id": "Q1",
            "work_id": work.work_id,
            "claim": "Apollo halves latency.",
            "quote": "Apollo halves latency",
        },
    )
    assert result.isError and "does not occur" in message(result)
    assert store.passages_for_work(work.work_id) == []


@serve()
async def test_record_finding_unknown_run(client: ClientSession, store: ResearchStore):
    result = await client.call_tool(
        "record_finding",
        {
            "run_id": "r_missing",
            "subquestion_id": "Q1",
            "work_id": "w",
            "claim": "c",
            "quote": "q",
        },
    )
    assert result.isError and "no run" in message(result)


# ---------------------------------------------------------------- runs


@serve()
async def test_get_run_lists_runs_and_reports_unknown(
    client: ClientSession, store: ResearchStore
):
    run_id = store.start_run(question="Why?", mode="mcp", config={}, models={})
    listed = data(await client.call_tool("get_run", {}))
    assert [r["run_id"] for r in listed["runs"]] == [run_id]
    missing = await client.call_tool("get_run", {"run_id": "r_missing"})
    assert missing.isError


@serve()
async def test_research_runs_the_engine_against_the_servers_store(
    client: ClientSession, store: ResearchStore
):
    result = data(
        await client.call_tool(
            "research",
            {
                "question": "How do the Apollo cache and SaltLog logging affect "
                "request latency?",
                "mode": "scholarly",
            },
        )
    )
    assert result["stop_reason"] == "answered"
    assert result["findings"]
    record = store.get_run(result["run_id"])
    assert record is not None and record.finished_at is not None


@serve(model=False)
async def test_research_needs_a_model(client: ClientSession, store: ResearchStore):
    result = await client.call_tool("research", {"question": "Why?"})
    assert result.isError and "--model" in message(result)


# ---------------------------------------------------------------- resources


@serve()
async def test_runs_are_resources(client: ClientSession, store: ResearchStore):
    run_id = store.start_run(question="Why Apollo?", mode="mcp", config={}, models={})
    store.add_run_item(run_id, "subquestion", "Q1", {"id": "Q1", "text": "Why?"})
    listed = (await client.list_resources()).resources
    assert [str(r.uri) for r in listed] == [f"research://run/{run_id}"]
    contents = await client.read_resource(AnyUrl(f"research://run/{run_id}"))
    (block,) = contents.contents
    assert isinstance(block, types.TextResourceContents)
    document = json.loads(block.text)
    assert document["run"]["question"] == "Why Apollo?"
    assert document["items"][0]["ref_id"] == "Q1"


# ---------------------------------------------------------------------- cli


def test_the_research_cli_dispatches_the_mcp_subcommand(
    monkeypatch: pytest.MonkeyPatch,
):
    from andamentum.deep_research import cli

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_server, "main", seen.append)
    monkeypatch.setattr(sys, "argv", ["andamentum-research", "mcp", "--store", "x.db"])
    cli.main()
    assert seen == [["--store", "x.db"]]


def test_main_opens_the_store_at_the_given_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    opened: dict[str, Any] = {}

    async def fake_serve(tools: mcp_server.ResearchTools) -> None:
        opened["path"] = tools.store.path
        opened["model"] = tools.model
        opened["searxng"] = tools.config.searxng_url
        opened["browser"] = tools.config.browser
        await tools.aclose()

    monkeypatch.setattr(mcp_server, "serve_stdio", fake_serve)
    path = tmp_path / "lab" / "research.db"
    mcp_server.main(
        ["--store", str(path), "--model", "m", "--searxng-url", "http://searx.test"]
        + ["--browser"]
    )
    assert opened == {
        "path": path,
        "model": "m",
        "searxng": "http://searx.test",
        "browser": True,
    }
    assert path.exists()


def test_main_without_a_store_keeps_nothing(monkeypatch: pytest.MonkeyPatch):
    opened: dict[str, Any] = {}

    async def fake_serve(tools: mcp_server.ResearchTools) -> None:
        opened["path"] = tools.store.path
        await tools.aclose()

    monkeypatch.setattr(mcp_server, "serve_stdio", fake_serve)
    mcp_server.main([])
    assert opened == {"path": None}
