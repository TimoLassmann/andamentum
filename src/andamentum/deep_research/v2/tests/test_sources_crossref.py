"""Crossref adapter: search and lookup over references' client, retraction status."""

from __future__ import annotations

import httpx
import pytest

from andamentum.deep_research.v2.sources.base import SearchFilters, SourceError
from andamentum.deep_research.v2.sources.crossref import CrossrefSource
from andamentum.references import ReferenceHttp

from .source_stub import SourceKit, ok


def message(**fields: object) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "message": fields})


def source(kit: SourceKit, **kwargs: object) -> CrossrefSource:
    return CrossrefSource(client=kit.client, **kwargs)  # type: ignore[arg-type]


async def test_search_maps_records_and_skips_those_without_doi_and_title() -> None:
    kit = SourceKit(lambda _r: ok("crossref_search.json"))
    found = await source(kit, mailto="me@example.org").search(
        "apollo lattice", limit=10
    )
    request = kit.requests[0]
    assert request.url.path == "/works"
    assert request.url.params["query"] == "apollo lattice"
    assert request.url.params["rows"] == "10"
    assert request.url.params["mailto"] == "me@example.org"
    assert [c.work.doi for c in found] == [
        "10.1000/apollo.crossref.1",
        "10.1000/apollo.crossref.2",
    ]
    first = found[0].work
    assert first.title == "Apollo lattice sampling in practice"
    assert first.authors == ["Jane Doe", "The Apollo Consortium"]
    assert first.year == 2022 and first.venue == "Journal of Invented Results"
    assert first.abstract == "Sparse sampling of the Apollo lattice works."
    assert first.access == "abstract"
    bare = found[1].work
    assert bare.access == "metadata" and bare.year is None and bare.abstract == ""
    assert [c.rank for c in found] == [0, 1]


async def test_search_goes_through_the_stores_cache() -> None:
    kit = SourceKit(lambda _r: ok("crossref_search.json"))
    crossref = source(kit)
    await crossref.search("apollo lattice", limit=10)
    await crossref.search("apollo lattice", limit=10)
    assert len(kit.requests) == 1
    assert kit.store.cache_hits >= 1


async def test_year_filters_become_crossref_filters() -> None:
    kit = SourceKit(lambda _r: ok("crossref_search.json"))
    await source(kit).search(
        "x", limit=3, filters=SearchFilters(year_from=2019, year_to=2021)
    )
    assert kit.param(0, "filter") == "from-pub-date:2019,until-pub-date:2021"


async def test_a_failed_search_raises() -> None:
    kit = SourceKit(lambda _r: httpx.Response(403, text="forbidden"))
    with pytest.raises(SourceError) as caught:
        await source(kit).search("apollo", limit=3)
    assert "403" in caught.value.reason


async def test_get_work_reads_a_doi_and_returns_none_for_an_unknown_one() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("10.1000/apollo.known"):
            return message(
                DOI="10.1000/apollo.known",
                title=["A known invented work"],
                issued={"date-parts": [[2020]]},
            )
        return httpx.Response(404)

    kit = SourceKit(handler)
    crossref = source(kit)
    found = await crossref.get_work("https://doi.org/10.1000/Apollo.Known")
    assert found is not None and found.work.title == "A known invented work"
    assert found.work.year == 2020
    assert await crossref.get_work("10.1000/apollo.unknown") is None
    with pytest.raises(SourceError) as caught:
        await crossref.get_work("not a doi")
    assert caught.value.kind == "config"


async def test_retraction_status_reads_the_update_notices() -> None:
    def notice(kind: str) -> dict[str, object]:
        return {"DOI": "10.1000/notice", "type": kind, "source": "retraction-watch"}

    notices = {
        "retracted": [notice("retraction")],
        "concern": [notice("expression_of_concern")],
        "correction": [notice("erratum")],
        "mixed": [notice("correction"), notice("retraction")],
        "clean": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.url.path.rsplit("/", 1)[-1]
        return message(DOI=f"10.1000/{key}", **{"updated-by": notices[key]})

    kit = SourceKit(handler)
    crossref = source(kit)
    got = {key: await crossref.retraction_status(f"10.1000/{key}") for key in notices}
    assert got == {
        "retracted": "retracted",
        "concern": "concern",
        "correction": "correction",
        "mixed": "retracted",  # the most severe wins
        "clean": "no",
    }


async def test_an_unknown_doi_is_unknown_and_a_failed_check_raises() -> None:
    kit = SourceKit(lambda _r: httpx.Response(404))
    assert await source(kit).retraction_status("10.1000/none") == "unknown"
    broken = SourceKit(lambda _r: httpx.Response(403, text="forbidden"))
    with pytest.raises(SourceError):  # "could not check" is never "not retracted"
        await source(broken).retraction_status("10.1000/x")


async def test_an_injected_reference_client_is_used() -> None:
    kit = SourceKit(lambda _r: ok("crossref_search.json"))
    http = ReferenceHttp(client=kit.http, cache=None)
    found = await CrossrefSource(client=kit.client, http=http).search("apollo", limit=2)
    assert len(found) == 2
