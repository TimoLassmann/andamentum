"""Ranking the works of a run: reciprocal rank fusion of the source ranks.

Package c implements this module (spec section 1.6) by copying the pure
functions ``source_rank_score``, ``_competition_ranks`` and ``rrf_scores`` from
the old ``strategies.py`` (``git show bbdb010:src/andamentum/deep_research/v2/strategies.py``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from .types import Candidate

RRF_K = 60


def fused_scores(candidates: Mapping[str, Candidate]) -> dict[str, float]:
    """Each candidate's reciprocal-rank-fusion score over its ``source_ranks``:
    the sum over its ranked lists of 1 / (``RRF_K`` + place)."""
    raise NotImplementedError("package c")


def rank_works(
    candidates: Mapping[str, Candidate],
    *,
    cited: Sequence[str],
    chosen: Sequence[str],
    excluded: Collection[str] = (),
) -> list[str]:
    """Work ids in the order a result ranks them: first the works behind cited
    findings (in order of first citation), then the other works that were chosen
    and read, then every remaining candidate; the second and third groups by
    fused score, best first, ties by work id. *excluded* never appear."""
    raise NotImplementedError("package c")
