"""A DOI Crossref does not know is asked about at doi.org before it is called
broken.

arXiv, Zenodo and most data repositories register their DOIs with DataCite,
so a Crossref 404 alone said "does not resolve" for every arXiv reference in
a list. The records here are invented; no network is used.
"""

from __future__ import annotations

import httpx

from andamentum.references import check_references
from andamentum.references.cited import resolve_cited_work
from andamentum.references.http import ReferenceHttp

ARXIV_DOI = "10.48550/arXiv.2401.00001"
TITLE = "SaltLog: a benchmark for tidal sensor agents"

ENTRY = (
    f"Body [1].\n\nReferences\n1. Doe J, Roe A (2024) {TITLE}. arXiv. doi:{ARXIV_DOI}\n"
)


def _datacite(*, year: int = 2024, general: str = "Preprint", family: str = "Doe"):
    return {
        "data": {
            "attributes": {
                "doi": ARXIV_DOI.lower(),
                "titles": [{"title": TITLE}],
                "creators": [
                    {
                        "name": f"{family}, J",
                        "nameType": "Personal",
                        "familyName": family,
                    }
                ],
                "publicationYear": year,
                "dates": [{"date": f"{year}-01-03T00:00:00Z", "dateType": "Submitted"}],
                "types": {"resourceTypeGeneral": general},
            }
        }
    }


def _transport(*, agency: object, datacite: object = None, seen: list | None = None):
    """Crossref answers 404; doi.org answers *agency* (a body, or an int status
    for a failure); DataCite answers *datacite* (a body, or 404 when None)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        host = request.url.host
        if host == "doi.org":
            if isinstance(agency, int):
                return httpx.Response(agency, json={})
            return httpx.Response(200, json=agency)
        if host == "api.datacite.org":
            if datacite is None:
                return httpx.Response(404, json={})
            return httpx.Response(200, json=datacite)
        return httpx.Response(404, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _check(**kwargs):
    seen: list = []
    async with _transport(seen=seen, **kwargs) as client:
        report = await check_references(ENTRY, client=client)
    return report, seen


async def test_a_datacite_doi_is_found_and_reported_as_nothing() -> None:
    report, seen = await _check(
        agency=[{"DOI": ARXIV_DOI, "RA": "DataCite"}], datacite=_datacite()
    )
    assert [r.status for r in report.lookups] == ["found"]
    assert report.problems == []
    hosts = [r.url.host for r in seen]
    assert hosts == ["api.crossref.org", "doi.org", "api.datacite.org"]


async def test_a_preprint_year_is_not_held_against_the_entry() -> None:
    """A preprint's year says nothing about the entry's, as for Crossref
    reposts: the work is found and the year left unjudged."""
    report, _ = await _check(
        agency=[{"DOI": ARXIV_DOI, "RA": "DataCite"}], datacite=_datacite(year=2023)
    )
    assert [r.status for r in report.lookups] == ["found"]
    assert report.problems == []


async def test_a_datacite_doi_for_another_work_is_still_caught() -> None:
    other = _datacite()
    other["data"]["attributes"]["titles"] = [{"title": "Unrelated work on glaciers"}]
    report, _ = await _check(
        agency=[{"DOI": ARXIV_DOI, "RA": "DataCite"}], datacite=other
    )
    assert [p.kind for p in report.problems] == ["details_differ"]


async def test_a_doi_that_does_not_exist_is_still_major() -> None:
    report, _ = await _check(
        agency=[{"DOI": ARXIV_DOI, "status": "DOI does not exist"}]
    )
    assert [(p.kind, p.severity) for p in report.problems] == [
        ("doi_not_found", "major")
    ]


async def test_an_invalid_doi_does_not_resolve() -> None:
    report, _ = await _check(agency=[{"DOI": ARXIV_DOI, "status": "Invalid DOI"}])
    assert [p.kind for p in report.problems] == ["doi_not_found"]


async def test_another_agency_is_not_checked_and_not_reported() -> None:
    report, _ = await _check(agency=[{"DOI": ARXIV_DOI, "RA": "mEDRA"}])
    assert [r.status for r in report.lookups] == ["service_error"]
    assert "mEDRA" in report.lookups[0].detail
    assert report.problems == []


async def test_a_failed_agency_call_is_not_checked() -> None:
    report, _ = await _check(agency=503)
    assert [r.status for r in report.lookups] == ["service_error"]
    assert report.problems == []


async def test_datacite_listed_but_missing_is_not_checked() -> None:
    report, _ = await _check(
        agency=[{"DOI": ARXIV_DOI, "RA": "DataCite"}], datacite=None
    )
    assert [r.status for r in report.lookups] == ["service_error"]
    assert report.problems == []


async def test_the_cited_work_resolver_reads_datacite_too() -> None:
    client = _transport(
        agency=[{"DOI": ARXIV_DOI, "RA": "DataCite"}], datacite=_datacite()
    )
    async with ReferenceHttp(client=client) as http:
        result = await resolve_cited_work(http=http, doi=ARXIV_DOI, title=TITLE)
    absent = _transport(agency=[{"DOI": ARXIV_DOI, "status": "DOI does not exist"}])
    async with ReferenceHttp(client=absent) as http:
        gone = await resolve_cited_work(http=http, doi=ARXIV_DOI, title=TITLE)
    assert result.status == "found"
    assert result.match is not None and result.match.title == TITLE
    assert gone.status == "not_found"
