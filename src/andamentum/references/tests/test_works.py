"""The Work record, identifier normalisation and merge rules. Pure functions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from andamentum.references import (
    LICENCE_NO_REDISTRIBUTE,
    Work,
    canonical_url,
    merge_works,
    normalise_arxiv_id,
    normalise_doi,
    normalise_pmcid,
    normalise_pmid,
    title_key,
    works_compatible,
)
from andamentum.references.works import normalise_openalex_id, normalise_s2_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10.1000/ABC.123", "10.1000/abc.123"),
        ("https://doi.org/10.1000/abc.123", "10.1000/abc.123"),
        ("http://dx.doi.org/10.1000/abc.123", "10.1000/abc.123"),
        ("doi: 10.1000/abc.123", "10.1000/abc.123"),
        ("10.1000/abc.123.", "10.1000/abc.123"),
        ("10.1000/abc%2F123", "10.1000/abc/123"),
        ("10.1016/S0140-6736(20)30183-5", "10.1016/s0140-6736(20)30183-5"),
        ("(10.1000/abc.123)", None),
        ("10.1000/abc.123)", "10.1000/abc.123"),
        ("not a doi", None),
        ("", None),
    ],
)
def test_normalise_doi(raw: str, expected: str | None) -> None:
    assert normalise_doi(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2101.00001", "2101.00001"),
        ("arXiv:2101.00001v3", "2101.00001"),
        ("https://arxiv.org/abs/2101.00001v2", "2101.00001"),
        ("https://www.arxiv.org/pdf/2101.00001v2.pdf", "2101.00001"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        ("arXiv:math.GT/0309136", "math.gt/0309136"),
        ("10.48550/arXiv.2101.00001", "2101.00001"),
        ("https://example.org/abs/2101.00001", None),
        ("21.0001", None),
    ],
)
def test_normalise_arxiv_id(raw: str, expected: str | None) -> None:
    assert normalise_arxiv_id(raw) == expected


def test_normalise_other_identifiers() -> None:
    assert normalise_pmid("PMID: 00123") == "123"
    assert normalise_pmid(123) == "123"
    assert normalise_pmid("12a") is None
    assert normalise_pmid("000") is None
    assert normalise_pmcid("pmc123") == "PMC123"
    assert normalise_pmcid("123") == "PMC123"
    assert normalise_pmcid("PMCX") is None
    assert normalise_openalex_id("https://openalex.org/w123") == "W123"
    assert normalise_openalex_id("A123") is None
    assert normalise_s2_id("A" * 40) == "a" * 40
    assert normalise_s2_id("abc") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "HTTPS://WWW.Example.ORG/Path/?b=2&a=1#frag",
            "https://example.org/Path?a=1&b=2",
        ),
        (
            "https://example.org/page/?utm_source=x&utm_medium=y&id=7",
            "https://example.org/page?id=7",
        ),
        ("https://example.org/page?fbclid=abc", "https://example.org/page"),
        ("https://example.org:443/page", "https://example.org/page"),
        ("http://example.org:8080/page", "http://example.org:8080/page"),
        ("https://user:pw@example.org/", "https://example.org"),
        ("https://example.org/", "https://example.org"),
        ("ftp://example.org/file", None),
        ("mailto:a@example.org", None),
        ("not a url", None),
    ],
)
def test_canonical_url(raw: str, expected: str | None) -> None:
    assert canonical_url(raw) == expected


def test_title_key_needs_a_year_and_a_specific_title() -> None:
    assert title_key("Introduction", 2020) is None
    assert title_key("A Study of Synthetic Widgets", None) is None
    assert title_key("A Study of  Synthetic Widgets!", 2020) == title_key(
        "a study of synthetic widgets", 2020
    )
    assert title_key("Étude des Widgets Synthétiques", 2020) == title_key(
        "Etude des widgets synthetiques", 2020
    )


def test_work_normalises_identifiers_on_construction() -> None:
    work = Work.model_validate(
        {
            "doi": "https://doi.org/10.1000/ABC",
            "pmid": 77,
            "arxiv_id": "arXiv:2101.00001v2",
        }
    )
    assert work.doi == "10.1000/abc"
    assert work.pmid == "77"
    assert work.arxiv_id == "2101.00001"


def test_work_treats_empty_identifiers_as_absent() -> None:
    work = Work(doi="", url="  ", oa_url="")
    assert work.doi is None and work.url is None and work.oa_url is None


def test_work_rejects_a_garbled_identifier() -> None:
    with pytest.raises(ValidationError, match="not a valid doi"):
        Work(doi="hello")
    with pytest.raises(ValidationError, match="not a valid url"):
        Work(url="ftp://example.org/x")


def test_work_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Work(unknown=1)  # type: ignore[call-arg]


def test_scholarly_urls_fill_missing_identifiers() -> None:
    assert Work(url="https://doi.org/10.1000/x1").doi == "10.1000/x1"
    assert Work(url="https://arxiv.org/abs/2101.00001v2").arxiv_id == "2101.00001"
    assert Work(url="https://pubmed.ncbi.nlm.nih.gov/12345/").pmid == "12345"
    assert (
        Work(url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC555/").pmcid == "PMC555"
    )
    assert Work(url="https://openalex.org/W42").openalex_id == "W42"
    # An identifier already given is not overwritten.
    assert (
        Work(doi="10.1000/own", url="https://doi.org/10.1000/other").doi
        == "10.1000/own"
    )


def test_identity_keys_are_in_merge_order_with_the_title_last() -> None:
    work = Work(
        doi="10.1000/x",
        pmid="5",
        openalex_id="W9",
        url="https://example.org/a",
        title="A Study of Synthetic Widgets",
        year=2020,
    )
    kinds = [kind for kind, _ in work.identity_keys()]
    assert kinds == ["doi", "pmid", "openalex", "url", "title"]
    assert [k for k, _ in work.hard_keys()] == ["doi", "pmid", "openalex", "url"]


def test_an_arxiv_doi_also_yields_the_arxiv_key() -> None:
    work = Work(doi="10.48550/arXiv.2101.00001")
    assert ("arxiv", "2101.00001") in work.identity_keys()


def test_works_compatible_only_when_no_shared_identifier_disagrees() -> None:
    assert works_compatible(Work(doi="10.1001/a"), Work(pmid="3"))
    assert works_compatible(Work(doi="10.1001/a", pmid="3"), Work(doi="10.1001/a"))
    assert not works_compatible(Work(doi="10.1001/a"), Work(doi="10.1001/b"))


def test_merge_fills_gaps_and_first_wins_on_conflict() -> None:
    first = Work(doi="10.1001/a", title="First title", year=2020, authors=["A One"])
    second = Work(
        doi="10.1001/b",
        pmid="9",
        title="Second title",
        year=2021,
        venue="Journal",
        authors=["A One", "B Two"],
        abstract="An abstract.",
        oa_url="https://example.org/oa.pdf",
    )
    merged = merge_works(first, second)
    assert merged.doi == "10.1001/a"
    assert merged.title == "First title"
    assert merged.year == 2020
    assert merged.pmid == "9"
    assert merged.venue == "Journal"
    assert merged.authors == ["A One", "B Two"]
    assert merged.abstract == "An abstract."
    assert merged.oa_url == "https://example.org/oa.pdf"
    assert first.pmid is None  # inputs are not mutated


def test_merge_keeps_the_more_informative_access_and_the_more_severe_retraction() -> (
    None
):
    a = Work(doi="10.1001/a", access="full_text", retracted="unknown")
    b = Work(doi="10.1001/a", access="abstract", retracted="retracted")
    merged = merge_works(a, b)
    assert merged.access == "full_text"
    assert merged.retracted == "retracted"
    assert merge_works(b, a).retracted == "retracted"
    assert (
        merge_works(Work(retracted="unknown"), Work(retracted="no")).retracted == "no"
    )


def test_merge_keeps_the_restrictive_licence_whichever_side_holds_it() -> None:
    open_work = Work(doi="10.1001/a", licence="cc0")
    closed = Work(doi="10.1001/a", licence=LICENCE_NO_REDISTRIBUTE)
    assert merge_works(open_work, closed).licence == LICENCE_NO_REDISTRIBUTE
    assert merge_works(closed, open_work).licence == LICENCE_NO_REDISTRIBUTE


def test_merge_keeps_the_redistributable_abstract_over_a_restricted_one() -> None:
    open_work = Work(doi="10.1001/a", abstract="Apollo cuts latency.", licence="")
    closed = Work(
        doi="10.1001/a",
        abstract="Apollo cuts median latency, a much longer restricted abstract.",
        licence=LICENCE_NO_REDISTRIBUTE,
    )
    for merged in (merge_works(open_work, closed), merge_works(closed, open_work)):
        assert merged.abstract == "Apollo cuts latency."
        assert merged.licence == "" and merged.redistributable


def test_merge_keeps_a_restricted_abstract_when_it_is_the_only_one() -> None:
    open_work = Work(doi="10.1001/a", licence="cc0")
    closed = Work(
        doi="10.1001/a", abstract="Restricted text.", licence=LICENCE_NO_REDISTRIBUTE
    )
    for merged in (merge_works(open_work, closed), merge_works(closed, open_work)):
        assert merged.abstract == "Restricted text."
        assert merged.licence == LICENCE_NO_REDISTRIBUTE


def test_a_restricted_record_alone_stays_restricted() -> None:
    closed = Work(doi="10.1001/a", abstract="x", licence=LICENCE_NO_REDISTRIBUTE)
    assert merge_works(closed, closed).licence == LICENCE_NO_REDISTRIBUTE


def test_to_csl() -> None:
    work = Work(
        work_id="w_1",
        doi="10.1000/x",
        title="A Study",
        authors=["Doe, Jane", "John Q Public", "Consortium"],
        year=2020,
        venue="Journal of Examples",
        abstract="Abstract text.",
    )
    item = work.to_csl()
    assert item["id"] == "w_1"
    assert item["type"] == "article-journal"
    assert item["author"] == [
        {"family": "Doe", "given": "Jane"},
        {"family": "Public", "given": "John Q"},
        {"literal": "Consortium"},
    ]
    assert item["issued"] == {"date-parts": [[2020]]}
    assert item["DOI"] == "10.1000/x"
    assert item["container-title"] == "Journal of Examples"
    assert item["abstract"] == "Abstract text."


def test_to_csl_omits_the_abstract_of_a_no_redistribute_work() -> None:
    work = Work(doi="10.1001/a", abstract="secret", licence=LICENCE_NO_REDISTRIBUTE)
    assert "abstract" not in work.to_csl()
    assert not work.redistributable


def test_a_restricted_abstract_is_never_in_a_dump() -> None:
    work = Work(doi="10.1001/a", abstract="secret", licence=LICENCE_NO_REDISTRIBUTE)
    assert work.model_dump()["abstract"] == ""
    assert "secret" not in work.model_dump_json()
    assert work.abstract == "secret"
    free = Work(doi="10.1001/b", abstract="open text")
    assert free.model_dump()["abstract"] == "open text"


def test_to_csl_types() -> None:
    assert Work(arxiv_id="2101.00001").to_csl()["type"] == "article"
    assert Work(url="https://example.org/a").to_csl()["type"] == "webpage"
