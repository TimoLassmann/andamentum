"""Ranking the works of a run: reciprocal rank fusion of the source ranks.

The pure functions ``source_rank_score``, ``_competition_ranks`` and
``rrf_scores`` come from the old ``strategies.py`` (``git show
bbdb010:src/andamentum/deep_research/v2/strategies.py``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from .types import Candidate

# 1 / (RRF_K + rank). Large enough that one list's top place does not dominate.
RRF_K = 60


def source_rank_score(candidate: Candidate) -> float:
    """Reciprocal-rank fusion of the candidate's places in the lists it was found
    in: the sum of ``1 / (RRF_K + rank)`` over ``source_ranks`` (a source's
    search results, or a citation hop, each its own list)."""
    return sum(1.0 / (RRF_K + rank) for rank in candidate.source_ranks.values())


def _competition_ranks(values: Mapping[str, float]) -> dict[str, int]:
    """1-based ranks, highest value first; equal values share the better rank."""
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    previous: float | None = None
    current = 0
    for position, (key, value) in enumerate(ordered, start=1):
        if value != previous:
            current, previous = position, value
        ranks[key] = current
    return ranks


def rrf_scores(score_lists: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Reciprocal-rank fusion of several scored lists: each list ranks its keys
    by score (ties share a rank), and a key's fused score is the sum of
    ``1 / (RRF_K + rank)`` over the lists it is in."""
    fused: dict[str, float] = {}
    for scores in score_lists:
        for key, rank in _competition_ranks(scores).items():
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def fused_scores(candidates: Mapping[str, Candidate]) -> dict[str, float]:
    """Each candidate's reciprocal-rank-fusion score over its ``source_ranks``:
    the sum over its ranked lists of 1 / (``RRF_K`` + place)."""
    return {work_id: source_rank_score(c) for work_id, c in candidates.items()}


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
    fused score, best first. A tie keeps the order of *chosen* (the order the
    works were read) or of *candidates*: a work id is random per store, so it
    must not decide. *excluded* never appear."""
    scores = fused_scores(candidates)

    def usable(work_id: str) -> bool:
        return work_id in candidates and work_id not in excluded

    ordered = [w for w in dict.fromkeys(cited) if usable(w)]
    placed = set(ordered)

    def best_first(ids: Sequence[str]) -> list[str]:
        return sorted(ids, key=lambda w: -scores[w])  # stable: ties keep their order

    read = [w for w in dict.fromkeys(chosen) if usable(w) and w not in placed]
    ordered += best_first(read)
    placed |= set(read)
    ordered += best_first([w for w in candidates if usable(w) and w not in placed])
    return ordered
