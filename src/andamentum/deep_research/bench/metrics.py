"""The benchmark's metrics: pure functions over plain records, no I/O.

Retrieval (LitSearch): recall@k of the gold works and the precision of the kept
set. Cost: what one run spent. Each function states what it does with an empty
input, because an empty input is a result worth seeing, not a division by zero.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from andamentum.references import Work, normalise_doi
from andamentum.references.works import normalise_title

DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 50)
_MIN_TITLE_CHARS = 15  # as references.works: a shorter title identifies nothing

# -- retrieval ------------------------------------------------------------------


def work_matches(retrieved: Work, gold: Work) -> bool:
    """Is *retrieved* the gold work?

    True when the two share a hard identifier key (DOI, arXiv id, PMID, ...,
    canonical URL), or when the gold title (15+ characters once normalised)
    equals the retrieved title.
    """
    if set(retrieved.hard_keys()) & set(gold.hard_keys()):
        return True
    wanted = normalise_title(gold.title)
    if len(wanted) < _MIN_TITLE_CHARS:
        return False
    found = normalise_title(retrieved.title)
    return found == wanted


class RetrievalScore(BaseModel):
    """How one query fared: where each gold work was found and what was kept."""

    n_gold: int = Field(ge=0)
    gold_ranks: list[int | None] = Field(
        description="Per gold work, the 1-based rank of its best match; None if missed."
    )
    recall_at: dict[int, float]
    kept_n: int = Field(ge=0)
    kept_hits: int = Field(ge=0, description="Kept works that are a gold work.")
    kept_precision: float | None = Field(
        description="kept_hits / kept_n; None when nothing was kept."
    )


def retrieval_score(
    retrieved: Sequence[Work],
    gold: Sequence[Work],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalScore:
    """Recall@k and kept-set precision of one ranked result list.

    *retrieved* is the arm's result in rank order and is also its kept set.
    ``recall_at[k]`` is the share of gold works whose best match ranks within
    the first k. Raises ``ValueError`` for a query with no gold work, which
    has no recall.
    """
    if not gold:
        raise ValueError("a query with no gold work has no recall")
    ranks: list[int | None] = []
    for gold_work in gold:
        rank = next(
            (
                position
                for position, work in enumerate(retrieved, start=1)
                if work_matches(work, gold_work)
            ),
            None,
        )
        ranks.append(rank)
    hits = sum(
        1
        for work in retrieved
        if any(work_matches(work, gold_work) for gold_work in gold)
    )
    return RetrievalScore(
        n_gold=len(gold),
        gold_ranks=ranks,
        recall_at={
            k: sum(1 for r in ranks if r is not None and r <= k) / len(gold) for k in ks
        },
        kept_n=len(retrieved),
        kept_hits=hits,
        kept_precision=hits / len(retrieved) if retrieved else None,
    )


def mean_or_none(values: Sequence[float]) -> float | None:
    """The mean, or None for no values."""
    return sum(values) / len(values) if values else None


# -- links ----------------------------------------------------------------------------

_URL = re.compile(r"https?://[^\s<>\"'`\]]+")
_DOI_IN_TEXT = re.compile(r"\b10\.\d{4,9}/[^\s<>\"'`\]]+")
_TRAILING = ".,;:!?)}"


def extract_urls(text: str) -> list[str]:
    """The URLs and DOIs in *text* as URLs, first occurrence order, no repeats.

    A bare DOI becomes ``https://doi.org/<doi>``; trailing sentence
    punctuation is not part of a link.
    """
    found: list[str] = []
    for match in _URL.finditer(text):
        found.append(match.group().rstrip(_TRAILING))
    covered = " ".join(found)
    for match in _DOI_IN_TEXT.finditer(text):
        doi = normalise_doi(match.group().rstrip(_TRAILING))
        if doi is not None and doi not in covered.casefold():
            found.append(f"https://doi.org/{doi}")
    seen: set[str] = set()
    unique: list[str] = []
    for url in found:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


# -- cost ---------------------------------------------------------------------------


class Cost(BaseModel):
    """What one run cost. ``None`` means the arm does not count it, not zero."""

    wall_s: float = Field(ge=0)
    llm_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    decision_calls: int | None = None


def total_cost(costs: Sequence[Cost]) -> Cost:
    """Sum of costs; a count is None unless every cost reports it."""

    def total(values: list[int | None]) -> int | None:
        known = [v for v in values if v is not None]
        return sum(known) if len(known) == len(values) and values else None

    return Cost(
        wall_s=sum(c.wall_s for c in costs),
        llm_calls=total([c.llm_calls for c in costs]),
        input_tokens=total([c.input_tokens for c in costs]),
        output_tokens=total([c.output_tokens for c in costs]),
        decision_calls=total([c.decision_calls for c in costs]),
    )
