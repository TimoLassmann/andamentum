"""The benchmark's metrics, on tiny invented inputs."""

from __future__ import annotations

import pytest

from andamentum.deep_research.v2.bench.metrics import (
    Cost,
    extract_urls,
    mean_or_none,
    retrieval_score,
    total_cost,
    work_matches,
)
from andamentum.references import Work


def gold(title: str = "SaltLog and brine loss in Apollo kilns", **ids: str) -> Work:
    return Work(title=title, year=2021, **ids)  # type: ignore[arg-type]


# -- retrieval -----------------------------------------------------------------


def test_match_on_a_shared_doi_whatever_the_title() -> None:
    a = Work(doi="10.5555/x.1", title="A preprint title")
    assert work_matches(a, gold(doi="https://doi.org/10.5555/X.1"))


def test_match_on_equal_normalised_title_only_when_long_enough() -> None:
    assert work_matches(Work(title="SaltLog and Brine Loss in Apollo Kilns!"), gold())
    assert not work_matches(Work(title="Short"), gold(title="Short"))


def test_a_page_title_matches_only_when_titles_may_be_contained() -> None:
    page = Work(title="SaltLog and brine loss in Apollo kilns - Invented Archive")
    assert not work_matches(page, gold())
    assert work_matches(page, gold(), title_contained=True)


def test_a_different_doi_and_title_do_not_match() -> None:
    assert not work_matches(
        Work(doi="10.5555/other", title="Something else entirely about pottery"),
        gold(doi="10.5555/x.1"),
    )


def test_recall_at_k_counts_the_gold_found_within_k() -> None:
    wanted = [
        gold(doi="10.5555/g.1"),
        gold(title="Second gold paper about kilns", doi="10.5555/g.2"),
    ]
    retrieved = [
        Work(doi="10.5555/noise.1", title="Noise paper one about nothing"),
        Work(doi="10.5555/g.2", title="Second gold paper about kilns"),
        Work(doi="10.5555/noise.2", title="Noise paper two about nothing"),
        Work(doi="10.5555/g.1", title="SaltLog and brine loss in Apollo kilns"),
    ]
    score = retrieval_score(retrieved, wanted, ks=(1, 2, 3, 4))
    assert score.gold_ranks == [4, 2]
    assert score.recall_at == {1: 0.0, 2: 0.5, 3: 0.5, 4: 1.0}
    assert score.kept_n == 4
    assert score.kept_hits == 2
    assert score.kept_precision == 0.5


def test_nothing_retrieved_is_recall_zero_and_no_precision() -> None:
    score = retrieval_score([], [gold(doi="10.5555/g.1")], ks=(5,))
    assert score.recall_at == {5: 0.0}
    assert score.kept_precision is None


def test_a_query_without_gold_is_refused() -> None:
    with pytest.raises(ValueError, match="no gold"):
        retrieval_score([Work(title="A paper about kilns and salt")], [])


# -- links and cost ------------------------------------------------------------------


def test_extract_urls_trims_punctuation_and_reads_bare_dois() -> None:
    text = (
        "See https://example.org/a, and (https://example.org/b). Also doi 10.5555/Invented.1 "
        "and https://doi.org/10.5555/invented.1 again, https://example.org/a."
    )
    assert extract_urls(text) == [
        "https://example.org/a",
        "https://example.org/b",
        "https://doi.org/10.5555/invented.1",
    ]


def test_cost_totals_are_none_unless_every_run_reports_them() -> None:
    both = total_cost([Cost(wall_s=1.0, llm_calls=2), Cost(wall_s=2.0, llm_calls=3)])
    assert both.wall_s == 3.0 and both.llm_calls == 5
    mixed = total_cost([Cost(wall_s=1.0, llm_calls=2), Cost(wall_s=2.0)])
    assert mixed.llm_calls is None
    assert total_cost([]).llm_calls is None


def test_mean_or_none() -> None:
    assert mean_or_none([1.0, 2.0]) == 1.5
    assert mean_or_none([]) is None
