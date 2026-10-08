"""Origins: the site and the first author decide independence."""

from __future__ import annotations

import pytest

from andamentum.references import Work

from ..origin import independent_sources, origin_key, origin_label, registrable_domain
from ..types import Finding


def _work(
    work_id: str, *, authors: list[str] | None = None, url: str | None = None
) -> Work:
    return Work(
        work_id=work_id, title=f"Title {work_id}", authors=authors or [], url=url
    )


def _finding(number: int, work_id: str) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id="Q1",
        claim=f"claim number {number} about the cache",
        quote=f"verbatim quote number {number} from the source text",
        passage_id=f"p{number}",
        work_id=work_id,
    )


WORKS = {
    "W1": _work("W1", authors=["Jane Doe", "Ann Roe"]),
    "W2": _work("W2", authors=["Doe, Jane"]),  # same first author as W1
    "W3": _work("W3", authors=["Max Muster"]),
    "W4": _work("W4", url="https://www.news.example.com/a"),
    "W5": _work("W5", url="https://example.com/b"),  # same site as W4
}


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.example.com/x", "example.com"),
        ("https://news.example.com/x", "example.com"),
        ("https://example.com", "example.com"),
        ("https://www.bbc.co.uk/news", "bbc.co.uk"),
        ("https://a.b.csiro.edu.au/x", "csiro.edu.au"),
        ("not a url", ""),
    ],
)
def test_registrable_domain(url: str, domain: str) -> None:
    assert registrable_domain(url) == domain


def test_origin_prefers_the_first_authors_family_name() -> None:
    assert origin_key(WORKS["W1"]) == origin_key(WORKS["W2"]) == "author:doe"
    assert origin_key(WORKS["W3"]) == "author:muster"
    assert origin_key(WORKS["W4"]) == origin_key(WORKS["W5"]) == "site:example.com"
    assert origin_key(_work("W9")) == "work:W9"


def test_the_label_is_the_site_or_the_author() -> None:
    assert origin_label(WORKS["W4"]) == "example.com"
    assert origin_label(WORKS["W3"]) == "muster"
    assert origin_label(_work("W9")) == "unknown"


def test_independence_counts_origins_not_works() -> None:
    assert independent_sources([_finding(1, "W1"), _finding(2, "W2")], WORKS) == 1
    assert independent_sources([_finding(1, "W1"), _finding(2, "W3")], WORKS) == 2
    assert independent_sources([_finding(4, "W4"), _finding(5, "W5")], WORKS) == 1
    # A finding whose work is unknown is its own origin.
    assert independent_sources([_finding(8, "W404"), _finding(9, "W405")], WORKS) == 2
