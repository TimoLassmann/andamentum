"""Semantic Scholar adapter, against synthetic responses of the documented shape."""

from __future__ import annotations

import json

import httpx
import pytest

from andamentum.deep_research.sources.base import SearchFilters, SourceError
from andamentum.deep_research.sources.semantic_scholar import (
    SemanticScholarSource,
    work_from_paper,
)
from andamentum.references import LICENCE_NO_REDISTRIBUTE, Work

from .source_stub import SourceKit, fixture, ok

APOLLO = "0123456789abcdef0123456789abcdef01234567"


def routes(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/paper/search"):
        return ok("semantic_scholar_search.json")
    if path.endswith("/references"):
        return ok("semantic_scholar_references.json")
    if path.endswith("/citations"):
        return ok("semantic_scholar_citations.json")
    return httpx.Response(404)


def source(kit: SourceKit, **kwargs: object) -> SemanticScholarSource:
    return SemanticScholarSource(client=kit.client, **kwargs)  # type: ignore[arg-type]


async def test_search_maps_papers_and_tags_every_work_no_redistribute() -> None:
    kit = SourceKit(routes)
    found = await source(kit, api_key="KEY").search("apollo lattice", limit=5)
    request = kit.requests[0]
    assert request.headers["x-api-key"] == "KEY"
    assert request.url.params["query"] == "apollo lattice"
    assert request.url.params["limit"] == "5"
    work = found[0].work
    assert work.s2_id == APOLLO
    assert (work.doi, work.arxiv_id, work.pmid, work.pmcid) == (
        "10.1000/apollo.1",
        "2101.00001",
        "11111111",
        "PMC7777777",
    )
    assert work.authors == ["Jane Doe", "Richard Roe"] and work.year == 2021
    assert work.oa_url == "https://example.org/apollo.pdf"
    assert work.access == "abstract"
    assert all(c.work.licence == LICENCE_NO_REDISTRIBUTE for c in found)
    bare = found[1].work
    assert bare.access == "metadata" and bare.year is None and bare.oa_url is None
    assert bare.doi is None


async def test_its_answers_are_cached_but_never_exported() -> None:
    kit = SourceKit(routes)
    await source(kit).search("apollo lattice", limit=5)
    await source(kit).search("apollo lattice", limit=5)
    assert len(kit.requests) == 1
    assert list(kit.store.iter_cache()) == []
    assert len(list(kit.store.iter_cache(exportable_only=False))) == 1


async def test_no_key_means_no_key_header() -> None:
    kit = SourceKit(routes)
    await source(kit).search("apollo lattice", limit=5)
    assert "x-api-key" not in kit.requests[0].headers


async def test_filters_become_year_and_open_access_parameters() -> None:
    kit = SourceKit(routes)
    await source(kit).search(
        "x",
        limit=5,
        filters=SearchFilters(year_from=2019, year_to=2021, open_access_only=True),
    )
    params = kit.requests[0].url.params
    assert params["year"] == "2019-2021"
    assert "openAccessPdf" in params
    await source(kit).search("y", limit=5, filters=SearchFilters(year_from=2020))
    assert kit.requests[1].url.params["year"] == "2020-"


async def test_search_pages_by_offset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(fixture("semantic_scholar_search.json"))
        if request.url.params["offset"] == "0":
            body["next"] = 2
        return httpx.Response(200, json=body)

    kit = SourceKit(handler)
    found = await source(kit).search("apollo lattice", limit=4)
    assert [r.url.params["offset"] for r in kit.requests] == ["0", "2"]
    assert len(found) == 4 and [c.rank for c in found] == [0, 1, 2, 3]


async def test_references_ask_by_the_strongest_identifier() -> None:
    kit = SourceKit(routes)
    work = Work(
        doi="10.1000/apollo.1", arxiv_id="2101.00001", title="Apollo lattice sampling"
    )
    found = await source(kit).references(work, limit=10)
    assert kit.requests[0].url.path.endswith("/paper/DOI:10.1000/apollo.1/references")
    assert [c.work.title for c in found] == [
        "A cited Apollo method"
    ]  # the unmatched one is skipped
    only_arxiv = Work(arxiv_id="2101.00001", title="Apollo lattice sampling")
    await source(kit).references(only_arxiv, limit=10)
    assert kit.requests[1].url.path.endswith("/paper/ARXIV:2101.00001/references")
    held = Work(s2_id=APOLLO, doi="10.1000/apollo.1", title="Apollo lattice sampling")
    await source(kit).citations(held, limit=10)
    assert kit.requests[2].url.path.endswith(f"/paper/{APOLLO}/citations")


async def test_citations_map_the_citing_papers() -> None:
    kit = SourceKit(routes)
    found = await source(kit).citations(
        Work(s2_id=APOLLO, title="Apollo lattice sampling"), limit=10
    )
    assert [c.work.doi for c in found] == ["10.1000/apollo.cites1"]


async def test_a_reference_list_the_publisher_withheld_is_empty_not_an_error() -> None:
    kit = SourceKit(lambda _r: ok("semantic_scholar_references_elided.json"))
    found = await source(kit).references(
        Work(s2_id=APOLLO, title="Apollo lattice sampling"), limit=10
    )
    assert found == []


async def test_a_429_is_a_rate_limit_error() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(429, json={"message": "Too Many Requests"}), retries=0
    )
    with pytest.raises(SourceError) as caught:
        await source(kit).search("apollo", limit=5)
    assert caught.value.kind == "rate_limited" and caught.value.status == 429


async def test_a_work_with_no_usable_identifier_cannot_be_placed() -> None:
    kit = SourceKit(routes)
    with pytest.raises(SourceError) as caught:
        await source(kit).references(
            Work(title="An invented work with no identifiers"), limit=5
        )
    assert caught.value.kind == "config"


async def test_a_paper_semantic_scholar_does_not_hold_cannot_be_placed() -> None:
    def unknown(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Paper with id DOI:x not found"})

    kit = SourceKit(unknown)
    work = Work(title="An invented unindexed work", doi="10.1000/invented.404")
    for follow in (source(kit).references, source(kit).citations):
        with pytest.raises(SourceError) as caught:
            await follow(work, limit=5)
        assert caught.value.kind == "config"
        assert "DOI:10.1000/invented.404 not found" in caught.value.reason


async def test_a_server_error_on_references_stays_an_outage() -> None:
    kit = SourceKit(lambda _r: httpx.Response(500), retries=0)
    with pytest.raises(SourceError) as caught:
        await source(kit).references(
            Work(title="An invented work", doi="10.1000/invented.500"), limit=5
        )
    assert caught.value.kind == "http"


def test_a_paper_without_a_title_or_id_is_skipped() -> None:
    assert work_from_paper({"paperId": APOLLO, "title": ""}) is None
    assert work_from_paper({"paperId": None, "title": "No id"}) is None
    assert work_from_paper({"paperId": "not-an-id", "title": "A bad id"}) is None
