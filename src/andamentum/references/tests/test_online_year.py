"""A record with only an online date also accepts the next year as its print year.

Crossref often holds only the online date of a paper that later appears in an
issue, and lists print the issue year. The records here are invented; no
network is used.
"""

from __future__ import annotations

import httpx

from andamentum.references import check_references
from andamentum.references.lookup import _to_match

TITLE = "SaltLog: tidal sensor logging at scale"
DOI = "10.1234/saltlog.tidal"


def _item(**dates: int) -> dict:
    item: dict = {
        "title": [TITLE],
        "author": [{"family": "Doe"}],
        "type": "journal-article",
        "DOI": DOI,
    }
    for field, year in dates.items():
        item[field.replace("_", "-")] = {"date-parts": [[year]]}
    return item


def test_online_only_adds_the_next_year() -> None:
    years = _to_match(_item(published_online=2012)).years
    assert 2012 in years and 2013 in years


def test_online_and_print_add_nothing() -> None:
    years = _to_match(_item(published_online=2012, published_print=2013)).years
    assert 2014 not in years
    assert set(years) == {2012, 2013}


def test_print_only_is_unchanged() -> None:
    assert _to_match(_item(published_print=2013)).years == [2013]


def _entry(year: int) -> str:
    return (
        "Body [1].\n\nReferences\n"
        f"1. Doe J, Roe A ({year}) {TITLE}. Tidal Sens 4:10 doi:{DOI}\n"
    )


async def _problems(year: int) -> list[str]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": _item(published_online=2012)})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await check_references(_entry(year), client=client)
    return [p.kind for p in report.problems]


async def test_the_issue_year_after_the_online_year_is_not_a_difference() -> None:
    assert await _problems(2013) == []


async def test_the_online_year_itself_is_not_a_difference() -> None:
    assert await _problems(2012) == []


async def test_a_later_year_is_still_reported() -> None:
    assert await _problems(2015) == ["details_differ"]
