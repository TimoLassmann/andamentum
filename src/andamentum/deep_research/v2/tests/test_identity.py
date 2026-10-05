"""Merging the records of several sources into one candidate per work."""

from __future__ import annotations

from andamentum.deep_research.v2.identity import merge
from andamentum.deep_research.v2.types import Candidate
from andamentum.references import Work


def _cand(
    source: str,
    rank: int | None = None,
    snippet: str = "",
    sq: str = "s1",
    **fields: object,
) -> Candidate:
    return Candidate(
        work=Work(**fields),  # type: ignore[arg-type]
        sources=[source],
        snippet=snippet,
        queries=[f"q-{source}"],
        subquestion_ids=[sq],
        rank=rank,
    )


def test_the_same_doi_in_different_spellings_is_one_work() -> None:
    merged = merge(
        [
            _cand(
                "openalex", doi="10.1001/ABC", title="Alpha study of widgets", rank=3
            ),
            _cand(
                "europepmc",
                doi="https://doi.org/10.1001/abc",
                pmid="42",
                snippet="a longer snippet",
                rank=1,
            ),
        ]
    )
    assert len(merged) == 1
    only = merged[0]
    assert only.work.pmid == "42"
    assert only.work.title == "Alpha study of widgets"
    assert only.sources == ["openalex", "europepmc"]
    assert only.snippet == "a longer snippet"
    assert only.rank == 1
    assert only.queries == ["q-openalex", "q-europepmc"]


def test_a_scholarly_url_matches_the_record_with_that_identifier() -> None:
    merged = merge(
        [
            _cand("arxiv", arxiv_id="2101.00001v2"),
            _cand("web", url="https://arxiv.org/pdf/2101.00001v3.pdf"),
            _cand("datacite", doi="10.48550/arXiv.2101.00001"),
        ]
    )
    assert len(merged) == 1
    assert merged[0].sources == ["arxiv", "web", "datacite"]


def test_tracking_parameters_do_not_make_two_web_pages() -> None:
    merged = merge(
        [
            _cand("web", url="https://www.example.org/post/?utm_source=a"),
            _cand("web", url="https://example.org/post"),
            _cand("web", url="https://example.org/other"),
        ]
    )
    assert len(merged) == 2


def test_a_record_that_bridges_two_groups_joins_them() -> None:
    merged = merge(
        [
            _cand("a", doi="10.1001/x"),
            _cand("b", pmid="7"),
            _cand("c", doi="10.1001/x", pmid="7"),
            _cand("d", pmid="7", openalex_id="W5"),
        ]
    )
    assert len(merged) == 1
    work = merged[0].work
    assert (work.doi, work.pmid, work.openalex_id) == ("10.1001/x", "7", "W5")
    assert merged[0].sources == ["a", "c", "b", "d"]


def test_a_late_record_still_reaches_a_group_merged_away() -> None:
    merged = merge(
        [
            _cand("a", doi="10.1001/x"),
            _cand("b", pmid="7"),
            _cand("c", doi="10.1001/x", pmid="7"),
            _cand("d", pmid="7"),
        ]
    )
    assert len(merged) == 1


def test_same_title_and_year_merge_when_no_identifier_disagrees() -> None:
    merged = merge(
        [
            _cand("web", title="A Study of Synthetic Widgets", year=2020),
            _cand(
                "openalex",
                title="A study of synthetic widgets.",
                year=2020,
                doi="10.1001/w",
            ),
        ]
    )
    assert len(merged) == 1
    assert merged[0].work.doi == "10.1001/w"


def test_a_preprint_and_its_published_version_stay_two_works() -> None:
    merged = merge(
        [
            _cand(
                "biorxiv",
                title="A Study of Synthetic Widgets",
                year=2020,
                doi="10.1101/2020.01.01.000001",
            ),
            _cand(
                "openalex",
                title="A Study of Synthetic Widgets",
                year=2020,
                doi="10.1001/journal.1",
            ),
        ]
    )
    assert len(merged) == 2


def test_a_different_year_is_a_different_work() -> None:
    merged = merge(
        [
            _cand("a", title="A Study of Synthetic Widgets", year=2020),
            _cand("b", title="A Study of Synthetic Widgets", year=2021),
        ]
    )
    assert len(merged) == 2


def test_a_short_title_never_merges() -> None:
    assert (
        len(
            merge(
                [
                    _cand("a", title="Introduction", year=2020),
                    _cand("b", title="Introduction", year=2020),
                ]
            )
        )
        == 2
    )


def test_output_is_in_order_of_first_appearance() -> None:
    merged = merge(
        [
            _cand("a", doi="10.1001/one"),
            _cand("a", doi="10.1001/two"),
            _cand("b", doi="10.1001/one"),
            _cand("a", doi="10.1001/three"),
        ]
    )
    assert [c.work.doi for c in merged] == [
        "10.1001/one",
        "10.1001/two",
        "10.1001/three",
    ]


def test_provenance_lists_have_no_duplicates() -> None:
    merged = merge([_cand("a", doi="10.1001/x"), _cand("a", doi="10.1001/x")])
    assert merged[0].sources == ["a"]
    assert merged[0].subquestion_ids == ["s1"]


def test_records_with_no_identifier_stay_separate() -> None:
    assert len(merge([_cand("a"), _cand("b")])) == 2


def test_empty_input() -> None:
    assert merge([]) == []
