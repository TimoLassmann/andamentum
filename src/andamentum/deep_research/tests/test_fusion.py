"""Ranking the works of a run: reciprocal rank fusion and the result order."""

from __future__ import annotations

import pytest

from andamentum.references import Work

from ..fusion import (
    RRF_K,
    _competition_ranks,
    fused_scores,
    rank_works,
    rrf_scores,
    source_rank_score,
)
from ..types import Candidate


def _candidate(work_id: str, **ranks: int) -> Candidate:
    return Candidate(
        work=Work(work_id=work_id, title=work_id),
        sources=["alpha"],
        source_ranks=dict(ranks),
    )


CANDIDATES = {
    "W1": _candidate("W1", alpha=5),
    "W2": _candidate("W2", alpha=1),
    "W3": _candidate("W3", alpha=2, beta=1),
    "W4": _candidate("W4"),
    "W5": _candidate("W5", alpha=1),
}


def test_source_rank_score_sums_reciprocal_places() -> None:
    assert source_rank_score(CANDIDATES["W3"]) == pytest.approx(
        1 / (RRF_K + 2) + 1 / (RRF_K + 1)
    )
    assert source_rank_score(CANDIDATES["W4"]) == 0.0
    assert fused_scores(CANDIDATES)["W2"] == pytest.approx(1 / (RRF_K + 1))


def test_competition_ranks_share_the_better_place() -> None:
    assert _competition_ranks({"a": 3.0, "b": 3.0, "c": 1.0, "d": 2.0}) == {
        "a": 1,
        "b": 1,
        "d": 3,
        "c": 4,
    }


def test_rrf_scores_fuse_lists() -> None:
    fused = rrf_scores([{"a": 2.0, "b": 1.0}, {"b": 9.0}])
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert rrf_scores([]) == {}


def test_cited_first_then_chosen_then_the_rest() -> None:
    order = rank_works(CANDIDATES, cited=["W1", "W4"], chosen=["W2", "W1", "W3"])
    # Cited in citation order; chosen by fused score (W3 beats W2); then the rest.
    assert order == ["W1", "W4", "W3", "W2", "W5"]


def test_ties_keep_the_order_of_the_candidates_not_the_work_id() -> None:
    order = rank_works(CANDIDATES, cited=[], chosen=[])
    assert order == ["W3", "W2", "W5", "W1", "W4"]
    first = {"W9": _candidate("W9", alpha=1), "W1": _candidate("W1", alpha=1)}
    assert rank_works(first, cited=[], chosen=[]) == ["W9", "W1"]
    # works read keep the order they were read in
    assert rank_works(first, cited=[], chosen=["W1", "W9"]) == ["W1", "W9"]


def test_excluded_never_appear_and_unknown_ids_are_skipped() -> None:
    order = rank_works(
        CANDIDATES,
        cited=["W9", "W3", "W3"],
        chosen=["W2", "W8"],
        excluded={"W3", "W5"},
    )
    assert order == ["W2", "W1", "W4"]


def test_empty_inputs() -> None:
    assert rank_works({}, cited=["W1"], chosen=["W2"]) == []
    assert rank_works(CANDIDATES, cited=[], chosen=[], excluded=set(CANDIDATES)) == []
    assert fused_scores({}) == {}
