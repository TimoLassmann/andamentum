"""arXiv adapter, against a recorded Atom feed."""

from __future__ import annotations

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.v2.sources.arxiv import (
    ArxivSource,
    to_search_query,
    works_from_feed,
)
from andamentum.deep_research.v2.sources.base import SearchFilters, SourceError
from andamentum.references import Work

from .source_stub import SourceKit, fixture, ok

ERROR_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/api/errors#incorrect_id_format_for_1234</id>
<title>Error</title><summary>incorrect id format for 1234</summary></entry></feed>"""


def feed(_request: httpx.Request) -> httpx.Response:
    return ok("arxiv_feed.xml")


def test_a_plain_query_becomes_an_and_of_its_significant_words() -> None:
    assert (
        to_search_query("How does retrieval-augmented generation reduce hallucination?")
        == "all:retrieval-augmented AND all:generation AND all:reduce AND all:hallucination"
    )
    assert to_search_query("the of and") == ""
    many = to_search_query(
        "alpha beta gamma delta epsilon zeta theta iota kappa lambda"
    )
    assert many.count("all:") == 8  # capped


def test_a_query_with_field_syntax_is_sent_as_written() -> None:
    assert (
        to_search_query('ti:"graph neural" AND cat:cs.LG')
        == 'ti:"graph neural" AND cat:cs.LG'
    )


async def test_search_maps_atom_entries_to_works() -> None:
    kit = SourceKit(feed)
    found = await ArxivSource(client=kit.client).search(
        "retrieval augmented generation", limit=2
    )
    params = kit.requests[0].url.params
    assert (
        params["search_query"] == "all:retrieval AND all:augmented AND all:generation"
    )
    assert (params["start"], params["max_results"], params["sortBy"]) == (
        "0",
        "2",
        "relevance",
    )
    work = found[0].work
    assert work.arxiv_id == "2411.18583"  # the version is dropped
    assert work.title.startswith("Automated Literature Review Using NLP Techniques")
    assert work.authors[:2] == ["Nurshat Fateh Ali", "Md. Mahdi Mohtasim"]
    assert work.year == 2024
    assert work.venue == "arXiv"
    assert work.url == "https://arxiv.org/abs/2411.18583"
    assert work.oa_url == "https://arxiv.org/pdf/2411.18583"
    assert work.access == "abstract" and work.abstract.startswith(
        "This research presents"
    )
    assert "\n" not in work.abstract
    assert [c.rank for c in found] == [0, 1]


async def test_a_year_filter_is_applied_to_the_entries() -> None:
    kit = SourceKit(feed)
    found = await ArxivSource(client=kit.client).search(
        "retrieval augmented generation", limit=5, filters=SearchFilters(year_from=2025)
    )
    assert [c.work.year for c in found] == [2025]
    assert [c.rank for c in found] == [0]


async def test_a_query_with_no_searchable_words_raises_without_a_request() -> None:
    kit = SourceKit(feed)
    with pytest.raises(SourceError) as caught:
        await ArxivSource(client=kit.client).search("the of and", limit=5)
    assert caught.value.kind == "config"
    assert kit.requests == []


async def test_an_error_entry_in_a_200_feed_raises_and_is_not_cached() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, content=ERROR_FEED))
    with pytest.raises(SourceError, match="incorrect id format"):
        await ArxivSource(client=kit.client).search("ti:bad", limit=5)
    assert list(kit.store.iter_cache()) == []


async def test_a_feed_that_is_not_xml_is_a_parse_error() -> None:
    kit = SourceKit(lambda _r: httpx.Response(200, content=b"<feed>"))
    with pytest.raises(SourceError) as caught:
        await ArxivSource(client=kit.client).search("lattice sampling", limit=5)
    assert caught.value.kind == "parse"


async def test_api_requests_are_spaced_three_seconds_apart() -> None:
    kit = SourceKit(feed, intervals={"arxiv": 3.0})
    source = ArxivSource(client=kit.client)
    await source.search("lattice sampling", limit=2)
    await source.search("graph networks", limit=2)
    assert kit.time.sleeps == [pytest.approx(3.0)]


async def test_the_pdf_is_read_with_its_own_crawl_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    fetch_gate._robots_cache_clear()
    paper = (
        "Retrieval augmented generation grounds answers in documents. " * 6
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=paper, headers={"content-type": "text/plain"}
        )

    kit = SourceKit(handler)
    source = ArxivSource(client=kit.client, pdf_interval_s=15.0)
    first = await source.fetch_text(
        Work(arxiv_id="2411.18583", title="Automated literature review")
    )
    second = await source.fetch_text(
        Work(arxiv_id="2502.00306", title="Riddle me this stealthy")
    )
    assert first is not None and second is not None
    assert [r.url.path for r in kit.requests if r.url.path.startswith("/pdf")] == [
        "/pdf/2411.18583",
        "/pdf/2502.00306",
    ]
    assert kit.time.sleeps == [pytest.approx(15.0)]
    assert (
        await source.fetch_text(
            Work(doi="10.1000/x.y", title="A work without an arXiv id")
        )
        is None
    )


def test_feed_parsing_keeps_the_doi_and_journal_reference_when_given() -> None:
    xml = fixture("arxiv_feed.xml").replace(
        b"<arxiv:comment>",
        b"<arxiv:doi>10.1000/Invented.Pub</arxiv:doi><arxiv:journal_ref>J. Invented 1 (2024)</arxiv:journal_ref><arxiv:comment>",
        1,
    )
    first = works_from_feed(xml)[0]
    assert first.doi == "10.1000/invented.pub"
    assert first.venue == "J. Invented 1 (2024)"
