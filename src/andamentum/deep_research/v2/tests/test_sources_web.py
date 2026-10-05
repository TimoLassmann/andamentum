"""Web source: SearxNG search and safe fetch, and build_sources."""

from __future__ import annotations

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.sources import (
    ArxivSource,
    OpenAlexSource,
    SemanticScholarSource,
    SourceConfig,
    SourceError,
    WebSource,
    build_sources,
)
from andamentum.deep_research.v2.sources.base import (
    ExpandingSource,
    Source,
    TextSource,
)
from andamentum.references import Work

from .source_stub import SourceKit, ok

SEARX = "http://127.0.0.1:4070"


def results(request: httpx.Request) -> httpx.Response:
    if request.url.path != "/search":
        return httpx.Response(404)
    if request.url.params.get("pageno", "1") != "1":
        return ok("searxng_empty.json")
    return ok("searxng_results.json")


def web(kit: SourceKit, url: str | None = SEARX) -> WebSource:
    return WebSource(searxng_url=url, client=kit.client)


async def test_search_returns_unique_pages_in_rank_order() -> None:
    kit = SourceKit(results)
    found = await web(kit).search("apollo lattice", limit=10)
    request = kit.requests[0]
    assert request.url.params["q"] == "apollo lattice"
    assert request.url.params["format"] == "json"
    assert [c.work.title for c in found] == [
        "Apollo lattice overview",
        "SaltLog documentation",
    ]
    first = found[0]
    assert (
        first.work.url == "https://example.org/apollo"
    )  # the work holds the canonical URL
    assert first.work.year == 2023 and first.work.venue == "example.org"
    assert first.work.access == "metadata"
    assert first.snippet.startswith("An overview of the Apollo lattice")
    assert first.sources == ["web"] and first.queries == ["apollo lattice"]
    assert [c.rank for c in found] == [0, 1]
    assert found[1].work.year is None


async def test_search_asks_for_more_pages_only_while_short_of_the_limit() -> None:
    kit = SourceKit(results)
    await web(kit).search("apollo lattice", limit=1)
    assert len(kit.requests) == 1
    await web(kit).search("apollo lattice", limit=5)
    # page 1 is in the cache from the first search; only page 2 is asked for
    assert [r.url.params["pageno"] for r in kit.requests[1:]] == ["2"]


async def test_no_results_with_dead_engines_is_an_error_and_is_not_cached() -> None:
    kit = SourceKit(lambda _r: ok("searxng_unanswered.json"))
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert "google (timeout)" in caught.value.reason
    assert "bing (HTTP error 429)" in caught.value.reason
    assert list(kit.store.iter_cache()) == []


async def test_no_results_from_healthy_engines_is_an_honest_empty_list() -> None:
    kit = SourceKit(lambda _r: ok("searxng_empty.json"))
    assert await web(kit).search("zzzz qqqq", limit=5) == []


async def test_a_searxng_that_refuses_json_raises() -> None:
    kit = SourceKit(lambda _r: httpx.Response(403, text="forbidden"))
    with pytest.raises(SourceError) as caught:
        await web(kit).search("apollo", limit=5)
    assert caught.value.status == 403


def test_web_search_needs_an_address() -> None:
    kit = SourceKit(results)
    with pytest.raises(SourceError) as caught:
        web(kit, url=None)
    assert caught.value.kind == "config"
    assert (
        WebSource(searxng_url=SEARX + "/", client=kit.client)._endpoint
        == f"{SEARX}/search"
    )


async def test_a_page_is_fetched_through_the_gate_and_returned_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    fetch_gate._robots_cache_clear()
    body = ("Apollo lattices are sampled sparsely in practice. " * 8).encode()
    kit = SourceKit(
        lambda _r: httpx.Response(
            200, content=body, headers={"content-type": "text/plain"}
        )
    )
    reader = web(kit)
    fetched = await reader.fetch_text(
        Work(
            url="https://docs.example.net/page", title="An invented documentation page"
        )
    )
    assert (
        fetched is not None
        and fetched.text == body.decode()
        and fetched.source == "web"
    )
    assert (
        await reader.fetch_text(
            Work(arxiv_id="2101.00001", title="A work with no web page")
        )
        is None
    )


# -- build_sources -----------------------------------------------------------


def names(sources: object) -> list[str]:
    return [s.name for s in sources]  # type: ignore[attr-defined]


def test_scholarly_mode_searches_the_scholarly_sources_only() -> None:
    kit = SourceKit(results)
    built = build_sources(
        SourceConfig(), store=kit.store, mode="scholarly", client=kit.client
    )
    assert names(built.search) == ["openalex", "europepmc", "arxiv"]
    assert names(built.text) == [
        "europepmc",
        "arxiv",
        "openalex",
    ]  # the order to try reading in
    assert names(built.expand) == ["openalex", "europepmc"]  # arXiv has no links
    assert built.crossref.name == "crossref"
    assert built.client is kit.client


def test_web_mode_needs_an_address_and_searches_the_web_alone() -> None:
    kit = SourceKit(results)
    with pytest.raises(SourceError, match="SearxNG"):
        build_sources(SourceConfig(), store=kit.store, mode="web", client=kit.client)
    built = build_sources(
        SourceConfig(searxng_url=SEARX), store=kit.store, mode="web", client=kit.client
    )
    assert names(built.search) == ["web"]
    assert names(built.text)[-1] == "web"


def test_both_mode_searches_everything_with_the_web_last() -> None:
    kit = SourceKit(results)
    config = SourceConfig(searxng_url=SEARX, scholarly=("openalex", "semantic_scholar"))
    built = build_sources(config, store=kit.store, mode="both", client=kit.client)
    assert names(built.search) == ["openalex", "semantic_scholar", "web"]
    assert names(built.expand) == ["openalex", "semantic_scholar"]
    assert all(isinstance(s, Source) for s in built.search)


def test_the_config_reaches_the_adapters() -> None:
    kit = SourceKit(results)
    config = SourceConfig(
        openalex_api_key="OA",
        mailto="me@example.org",
        openalex_search="both",
        semantic_scholar_api_key="S2",
        scholarly=("openalex", "arxiv", "semantic_scholar"),
        arxiv_pdf_interval_s=5.0,
    )
    built = build_sources(config, store=kit.store, mode="scholarly", client=kit.client)
    openalex, arxiv, s2 = built.search
    assert isinstance(openalex, OpenAlexSource) and openalex._auth == {
        "api_key": "OA",
        "mailto": "me@example.org",
    }
    assert openalex._mode == "both"
    assert isinstance(arxiv, ArxivSource)
    assert kit.client._intervals["arxiv:pdf"] == 5.0
    assert isinstance(s2, SemanticScholarSource) and s2._headers == {"x-api-key": "S2"}


def test_the_protocols_tell_what_an_adapter_can_do() -> None:
    kit = SourceKit(results)
    built = build_sources(
        SourceConfig(), store=kit.store, mode="scholarly", client=kit.client
    )
    arxiv = next(s for s in built.search if s.name == "arxiv")
    openalex = next(s for s in built.search if s.name == "openalex")
    assert isinstance(arxiv, TextSource) and not isinstance(arxiv, ExpandingSource)
    assert isinstance(openalex, TextSource) and isinstance(openalex, ExpandingSource)
