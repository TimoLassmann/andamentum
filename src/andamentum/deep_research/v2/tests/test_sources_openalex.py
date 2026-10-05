"""OpenAlex adapter, against recorded responses."""

from __future__ import annotations

import json

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.sources.base import SearchFilters, SourceError
from andamentum.deep_research.v2.sources.openalex import (
    OpenAlexSource,
    abstract_from_index,
    work_from_item,
)
from andamentum.references import Work

from .source_stub import SourceKit, fixture, ok

FIRST = "W2955341903"


def routes(request: httpx.Request) -> httpx.Response:
    path, params = request.url.path, request.url.params
    if path == "/works" and "search.semantic" in params:
        return ok("openalex_semantic.json")
    if path == "/works" and params.get("filter", "").startswith("cites:"):
        return ok("openalex_cites.json")
    if path == "/works" and params.get("filter", "").startswith("openalex:"):
        return ok("openalex_works_batch.json")
    if path == "/works":
        return ok("openalex_search.json")
    if path.startswith("/works/doi:"):
        return httpx.Response(200, json={"id": f"https://openalex.org/{FIRST}"})
    if path == f"/works/{FIRST}":
        return ok("openalex_one.json")
    return httpx.Response(404, text=f"unrouted {path}")


def source(kit: SourceKit, **kwargs: object) -> OpenAlexSource:
    return OpenAlexSource(client=kit.client, **kwargs)  # type: ignore[arg-type]


async def test_keyword_search_maps_a_record_to_a_work() -> None:
    kit = SourceKit(routes)
    found = await source(kit, api_key="K", mailto="me@example.org").search(
        "crispr off-target", limit=2
    )
    request = kit.requests[0]
    assert request.url.params["search.title_abstract_keywords"] == "crispr off-target"
    assert request.url.params["per_page"] == "2"
    assert request.url.params["cursor"] == "*"
    assert request.url.params["api_key"] == "K"
    assert request.url.params["mailto"] == "me@example.org"
    assert [c.rank for c in found] == [0, 1]
    first = found[0]
    assert first.sources == ["openalex"]
    assert first.queries == ["crispr off-target"]
    work = first.work
    assert work.openalex_id == FIRST
    assert work.doi == "10.1093/bib/bbz069"
    assert work.title == "Data imbalance in CRISPR off-target prediction"
    assert work.year == 2019
    assert work.venue == "Briefings in Bioinformatics"
    assert work.authors and all(work.authors)
    assert work.abstract.startswith("For genome-wide CRISPR off-target cleavage sites")
    assert work.access == "abstract"
    assert work.retracted == "unknown"
    assert first.snippet == work.abstract[:500]


async def test_the_open_access_url_is_read() -> None:
    kit = SourceKit(routes)
    found = await source(kit).search("x", limit=2)
    oa = [c.work for c in found if c.work.oa_url]
    assert oa and (oa[0].oa_url or "").startswith("https://")


async def test_search_pages_by_cursor_until_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["cursor"] == "*":
            return ok("openalex_search.json")
        body = json.loads(fixture("openalex_semantic.json"))
        body["meta"]["next_cursor"] = None
        return httpx.Response(200, json=body)

    kit = SourceKit(handler)
    found = await source(kit).search("crispr", limit=3)
    assert len(kit.requests) == 2
    assert kit.requests[1].url.params["cursor"] == "abc123=="
    assert kit.requests[1].url.params["per_page"] == "1"  # only what is still missing
    assert len(found) == 3
    assert [c.rank for c in found] == [0, 1, 2]


async def test_search_stops_when_the_results_run_out() -> None:
    body = json.loads(fixture("openalex_search.json"))
    body["meta"]["next_cursor"] = None
    kit = SourceKit(lambda _r: httpx.Response(200, json=body))
    found = await source(kit).search("crispr", limit=50)
    assert len(found) == 2
    assert len(kit.requests) == 1


async def test_filters_become_an_openalex_filter() -> None:
    kit = SourceKit(routes)
    await source(kit).search(
        "x",
        limit=2,
        filters=SearchFilters(year_from=2019, year_to=2021, open_access_only=True),
    )
    assert kit.param(0, "filter") == (
        "from_publication_date:2019-01-01,to_publication_date:2021-12-31,open_access.is_oa:true"
    )


async def test_semantic_mode_uses_the_semantic_endpoint_alone() -> None:
    kit = SourceKit(routes)
    found = await source(kit, mode="semantic").search(
        "how do off-target effects arise", limit=80
    )
    assert len(kit.requests) == 1
    assert kit.param(0, "search.semantic") == "how do off-target effects arise"
    assert kit.param(0, "per_page") == "50"  # capped
    assert kit.param(0, "cursor") is None
    assert found[0].work.title.startswith("Which Factors Affect")


async def test_both_mode_does_not_repeat_a_work_found_twice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # the semantic endpoint answers with a work the keyword search already found
        if "search.semantic" in request.url.params:
            return ok("openalex_search.json")
        return ok("openalex_search.json")

    kit = SourceKit(handler)
    found = await source(kit, mode="both").search("crispr", limit=2)
    assert len(kit.requests) == 2
    assert len(found) == 2
    assert [c.rank for c in found] == [0, 1]


async def test_a_failure_of_the_semantic_endpoint_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "search.semantic" in request.url.params:
            return httpx.Response(504, json={"reason": "query_timeout"})
        return ok("openalex_search.json")

    kit = SourceKit(handler, retries=0)
    with pytest.raises(SourceError) as caught:
        await source(kit, mode="both").search("crispr", limit=2)
    assert caught.value.status == 504


async def test_references_batch_fetch_the_cited_works_by_id() -> None:
    kit = SourceKit(routes)
    work = Work(
        openalex_id=FIRST, title="Data imbalance in CRISPR off-target prediction"
    )
    found = await source(kit).references(work, limit=10)
    assert kit.requests[0].url.path == f"/works/{FIRST}"
    assert kit.param(0, "select") == "referenced_works"
    batch = kit.requests[1].url.params
    assert batch["filter"].startswith("openalex:")
    assert len(batch["filter"].removeprefix("openalex:").split("|")) == 3
    assert len(found) == 3
    assert all(c.sources == ["openalex"] for c in found)


async def test_references_respect_the_limit() -> None:
    kit = SourceKit(routes)
    work = Work(
        openalex_id=FIRST, title="Data imbalance in CRISPR off-target prediction"
    )
    await source(kit).references(work, limit=2)
    assert (
        len(kit.requests[1].url.params["filter"].removeprefix("openalex:").split("|"))
        == 2
    )


async def test_a_work_with_only_a_doi_is_placed_by_a_free_lookup_first() -> None:
    kit = SourceKit(routes)
    work = Work(
        doi="10.1093/bib/bbz069", title="Data imbalance in CRISPR off-target prediction"
    )
    await source(kit).citations(work, limit=5)
    assert kit.requests[0].url.path == "/works/doi:10.1093/bib/bbz069"
    cites = kit.requests[1].url.params
    assert cites["filter"] == f"cites:{FIRST}"
    assert cites["sort"] == "cited_by_count:desc"


async def test_citations_map_to_candidates() -> None:
    kit = SourceKit(routes)
    work = Work(
        openalex_id=FIRST, title="Data imbalance in CRISPR off-target prediction"
    )
    found = await source(kit).citations(work, limit=5)
    assert [c.work.title for c in found][0].startswith("Computational approaches")


async def test_a_work_that_cannot_be_placed_raises() -> None:
    kit = SourceKit(routes)
    with pytest.raises(SourceError) as caught:
        await source(kit).references(
            Work(title="Some invented title for a work"), limit=5
        )
    assert caught.value.kind == "config"
    assert kit.requests == []


async def test_a_lookup_miss_raises_with_the_status() -> None:
    kit = SourceKit(lambda _r: httpx.Response(404, text="not found"))
    with pytest.raises(SourceError) as caught:
        await source(kit).references(
            Work(doi="10.1000/none", title="An invented missing work"), limit=5
        )
    assert caught.value.status == 404


async def test_text_is_fetched_from_the_open_access_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    fetch_gate._robots_cache_clear()
    paper = (
        "Sparse sampling recovered the constant in 94 of 100 trials. " * 6
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=paper, headers={"content-type": "text/plain"}
        )

    kit = SourceKit(handler)
    reader = source(kit)
    assert (
        await reader.fetch_text(
            Work(openalex_id=FIRST, title="A work with no open copy")
        )
        is None
    )
    fetched = await reader.fetch_text(
        Work(
            openalex_id=FIRST,
            title="A work with an open copy",
            oa_url="https://open.example/p.txt",
        )
    )
    assert fetched is not None and fetched.source == "openalex"
    assert fetched.text == paper.decode()


async def test_a_response_that_is_not_an_object_is_a_parse_error() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, json=[1, 2]))
    with pytest.raises(SourceError) as caught:
        await source(kit).search("x", limit=2)
    assert caught.value.kind == "parse"


def test_an_abstract_is_rebuilt_from_its_inverted_index() -> None:
    assert (
        abstract_from_index({"world": [1], "Hello": [0], "again": [2, 3]})
        == "Hello world again again"
    )
    assert abstract_from_index(None) == ""
    assert abstract_from_index({"x": ["bad", True]}) == ""


def test_a_record_without_a_title_is_skipped_and_a_retraction_is_kept() -> None:
    assert work_from_item({"id": "https://openalex.org/W1", "title": None}) is None
    work = work_from_item(
        {
            "id": "https://openalex.org/W9",
            "title": "A retracted invented study",
            "is_retracted": True,
            "ids": {
                "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678",
                "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321",
                "doi": "https://doi.org/10.1000/Invented.9",
            },
            "publication_year": 2020,
        }
    )
    assert work is not None
    assert (work.pmid, work.pmcid, work.doi) == (
        "12345678",
        "PMC7654321",
        "10.1000/invented.9",
    )
    assert work.retracted == "retracted"
    assert work.access == "metadata"
