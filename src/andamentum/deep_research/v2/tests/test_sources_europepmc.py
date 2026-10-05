"""Europe PMC adapter and the JATS reader, against recorded and synthetic responses."""

from __future__ import annotations

import json

import httpx
import pytest

from andamentum.deep_research.v2.sources.base import SearchFilters, SourceError
from andamentum.deep_research.v2.sources.europepmc import (
    EuropePmcSource,
    work_from_core,
    work_from_link,
)
from andamentum.deep_research.v2.sources.jats import JatsError, jats_to_text
from andamentum.references import Work

from .source_stub import SourceKit, fixture, ok

PMC = "PMC13245737"


def routes(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/search"):
        if request.url.params.get("resultType") == "lite":
            return httpx.Response(
                200,
                json={"resultList": {"result": [{"id": "31290501", "source": "MED"}]}},
            )
        return ok("europepmc_search.json")
    if path.endswith("/references"):
        return ok("europepmc_references.json")
    if path.endswith("/citations"):
        return ok("europepmc_citations.json")
    if path.endswith("/fullTextXML"):
        return ok("europepmc_fulltext.xml")
    return httpx.Response(404, text=f"unrouted {path}")


def source(kit: SourceKit) -> EuropePmcSource:
    return EuropePmcSource(client=kit.client)


async def test_search_maps_a_core_record_to_a_work() -> None:
    kit = SourceKit(routes)
    found = await source(kit).search("crispr off-target", limit=2)
    params = kit.requests[0].url.params
    assert params["query"] == "crispr off-target"
    assert (params["format"], params["resultType"], params["pageSize"]) == (
        "json",
        "core",
        "2",
    )
    assert params["cursorMark"] == "*"
    work = found[0].work
    assert (work.pmid, work.pmcid, work.doi) == ("42261185", PMC, "10.1093/bib/bbag216")
    assert work.title.startswith("CRISPR-MBTF: a multi-branch transformer")
    assert not work.title.endswith(".")
    assert work.authors[0] == "Jahangiri-Sisakht A"
    assert work.year == 2026
    assert work.venue == "Briefings in bioinformatics"
    assert work.access == "abstract" and work.abstract
    assert work.oa_url == f"https://europepmc.org/articles/{PMC}"
    assert found[0].sources == ["europepmc"] and found[0].rank == 0


async def test_filters_extend_the_query() -> None:
    kit = SourceKit(routes)
    await source(kit).search(
        "lattice",
        limit=2,
        filters=SearchFilters(year_from=2019, year_to=2024, open_access_only=True),
    )
    assert (
        kit.param(0, "query")
        == "(lattice) AND PUB_YEAR:[2019 TO 2024] AND OPEN_ACCESS:y"
    )
    await source(kit).search("lattice", limit=2, filters=SearchFilters(year_from=2019))
    assert kit.param(1, "query") == "(lattice) AND PUB_YEAR:[2019 TO 9999]"


async def test_search_follows_the_cursor_and_stops_when_it_stops_moving() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(fixture("europepmc_search.json"))
        if request.url.params["cursorMark"] != "*":
            body["nextCursorMark"] = request.url.params["cursorMark"]  # no further page
        return httpx.Response(200, json=body)

    kit = SourceKit(handler)
    found = await source(kit).search("crispr", limit=10)
    assert len(kit.requests) == 2
    assert kit.requests[1].url.params["cursorMark"] == "AoIIQF6ygig1NTYyNDEwMA=="
    assert len(found) == 4
    assert [c.rank for c in found] == [0, 1, 2, 3]


async def test_an_error_body_in_a_200_raises_and_is_not_cached() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(200, json={"errCode": 400, "errMsg": "bad query"})
    )
    with pytest.raises(SourceError, match="bad query"):
        await source(kit).search("((", limit=2)
    assert list(kit.store.iter_cache()) == []


async def test_references_map_the_listed_entries() -> None:
    kit = SourceKit(routes)
    work = Work(
        pmid="42261185",
        title="CRISPR-MBTF: a multi-branch transformer fusion framework",
    )
    found = await source(kit).references(work, limit=2)
    assert kit.requests[0].url.path.endswith("/MED/42261185/references")
    assert (kit.param(0, "page"), kit.param(0, "pageSize")) == ("1", "2")
    first = found[0].work
    assert first.pmid == "27380939"
    assert first.title.startswith("Evaluation of off-target and on-target scoring")
    assert (
        first.authors[0] == "Haeussler M"
        and first.year == 2016
        and first.venue == "Genome Biol"
    )
    assert first.abstract == ""


async def test_citations_map_the_listed_entries() -> None:
    kit = SourceKit(routes)
    found = await source(kit).citations(
        Work(pmcid=PMC, title="A work that is cited here"), limit=5
    )
    assert kit.requests[0].url.path.endswith(f"/PMC/{PMC}/citations")
    assert [c.work.pmid for c in found] == ["33333524"]


async def test_a_work_with_only_a_doi_is_located_by_searching_for_it() -> None:
    kit = SourceKit(routes)
    work = Work(
        doi="10.1093/bib/bbz069", title="Data imbalance in CRISPR off-target prediction"
    )
    await source(kit).references(work, limit=2)
    assert kit.param(0, "query") == 'DOI:"10.1093/bib/bbz069"'
    assert kit.requests[1].url.path.endswith("/MED/31290501/references")


async def test_a_work_europe_pmc_cannot_place_raises() -> None:
    kit = SourceKit(routes)
    with pytest.raises(SourceError) as caught:
        await source(kit).references(
            Work(title="An invented work with no identifiers"), limit=2
        )
    assert caught.value.kind == "config"

    empty = SourceKit(
        lambda _r: httpx.Response(200, json={"resultList": {"result": []}})
    )
    with pytest.raises(SourceError):
        await source(empty).citations(
            Work(doi="10.1000/none", title="An invented unknown work"), limit=2
        )


async def test_full_text_is_read_from_the_jats_and_cached() -> None:
    kit = SourceKit(routes)
    reader = source(kit)
    work = Work(pmcid=PMC, title="Sparse sampling of invented lattices")
    fetched = await reader.fetch_text(work)
    assert fetched is not None
    assert fetched.access == "full_text" and fetched.source == "europepmc"
    assert fetched.url.endswith(f"/{PMC}/fullTextXML")
    assert (
        "Sparse sampling recovered the lattice constant in 94 of 100 trials."
        in fetched.text
    )
    await reader.fetch_text(work)
    assert len(kit.requests) == 1


async def test_no_full_text_when_the_article_is_not_open_access() -> None:
    kit = SourceKit(
        lambda _r: httpx.Response(404, text="not in the open access subset")
    )
    assert (
        await source(kit).fetch_text(Work(pmcid=PMC, title="A closed invented article"))
        is None
    )
    assert (
        await source(kit).fetch_text(Work(pmid="123", title="A work without a PMCID"))
        is None
    )


async def test_an_unreadable_full_text_is_a_parse_error() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, content=b"<html>not jats</html>"))
    with pytest.raises(SourceError) as caught:
        await source(kit).fetch_text(
            Work(pmcid=PMC, title="A work whose text is not JATS")
        )
    assert caught.value.kind == "parse"


async def test_a_server_error_on_full_text_is_raised_not_read_as_absent() -> None:
    kit = SourceKit(lambda _r: httpx.Response(500, text="boom"), retries=0)
    with pytest.raises(SourceError) as caught:
        await source(kit).fetch_text(
            Work(pmcid=PMC, title="A work behind a failing server")
        )
    assert caught.value.status == 500


def test_records_without_a_title_are_skipped_and_retracted_ones_flagged() -> None:
    assert work_from_core({"id": "1", "source": "MED"}) is None
    assert work_from_link({"id": "1", "source": "MED", "title": ""}) is None
    retracted = work_from_core(
        {
            "title": "<i>An</i> invented paper.",
            "pubTypeList": {"pubType": ["Retracted Publication"]},
        }
    )
    assert retracted is not None
    assert retracted.title == "An invented paper"
    assert retracted.retracted == "retracted"
    assert retracted.access == "metadata"


# -- JATS ----------------------------------------------------------------------


def test_jats_gives_title_abstract_and_sections_as_markdown() -> None:
    text = jats_to_text(fixture("europepmc_fulltext.xml"))
    blocks = text.split("\n\n")
    assert blocks[0] == "# Sparse sampling of invented lattices"
    assert blocks[1] == "## Abstract"
    assert (
        "## Introduction" in blocks
        and "### Background" in blocks
        and "## Results" in blocks
    )
    assert "Earlier work used dense sampling at a cost of 4 hours per run." in blocks
    assert "The recovery rate did not depend on the grid size." in blocks


def test_jats_leaves_out_what_carries_no_quotable_sentence() -> None:
    text = jats_to_text(fixture("europepmc_fulltext.xml"))
    for left_out in ("must not appear", "Graphical summary"):
        assert left_out not in text
    assert (
        "The constant is in all cases." in text
    )  # the inline formula is gone, the sentence stays
    assert "invented materials[1]." in text


def test_jats_does_not_resolve_external_entities() -> None:
    xml = b"""<?xml version="1.0"?>
<!DOCTYPE article [<!ENTITY xxe SYSTEM "file:///etc/hosts">]>
<article><front><article-meta><title-group><article-title>Entity test &xxe;</article-title></title-group></article-meta></front>
<body><sec><title>S</title><p>Body &xxe; text.</p></sec></body></article>"""
    text = jats_to_text(xml)
    assert (
        text == "# Entity test\n\n## S\n\nBody text."
    )  # the entity is dropped, not read


def test_jats_rejects_what_is_not_jats() -> None:
    with pytest.raises(JatsError, match="well-formed"):
        jats_to_text(b"<article><p>unclosed")
    with pytest.raises(JatsError, match="no <article>"):
        jats_to_text(b"<html><body><p>hello</p></body></html>")
