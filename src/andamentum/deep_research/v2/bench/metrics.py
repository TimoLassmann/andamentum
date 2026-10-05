"""The benchmark's metrics: pure functions over plain records, no I/O.

Retrieval (LitSearch): recall@k of the gold works and the precision of the kept
set. Screening: recall and precision of the kept set, and work saved at 95%
recall from the ranking. Web: answer correctness (``web_set``), claim-support
precision and URL integrity. Each function states what it does with an empty
input, because an empty input is a result worth seeing, not a division by zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, Field

from andamentum.references import Work, UrlStatus, normalise_doi
from andamentum.references.works import normalise_title

DEFAULT_KS: tuple[int, ...] = (5, 10, 20, 50)
TARGET_RECALL = 0.95
_MIN_TITLE_CHARS = 15  # as references.works: a shorter title identifies nothing

# -- retrieval ------------------------------------------------------------------


def work_matches(retrieved: Work, gold: Work, *, title_contained: bool = False) -> bool:
    """Is *retrieved* the gold work?

    True when the two share a hard identifier key (DOI, arXiv id, PMID, ...,
    canonical URL), or when the gold title (15+ characters once normalised)
    equals the retrieved title. With *title_contained* it is enough that the
    gold title occurs in the retrieved one, as a web page's title wraps the
    paper's (``"<title> - ACL Anthology"``): the web path of the legacy module
    knows pages, not works.
    """
    if set(retrieved.hard_keys()) & set(gold.hard_keys()):
        return True
    wanted = normalise_title(gold.title)
    if len(wanted) < _MIN_TITLE_CHARS:
        return False
    found = normalise_title(retrieved.title)
    if found == wanted:
        return True
    return title_contained and f" {wanted} " in f" {found} "


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
    title_contained: bool = False,
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
                if work_matches(work, gold_work, title_contained=title_contained)
            ),
            None,
        )
        ranks.append(rank)
    hits = sum(
        1
        for work in retrieved
        if any(
            work_matches(work, gold_work, title_contained=title_contained)
            for gold_work in gold
        )
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


# -- screening ------------------------------------------------------------------


class ScreeningScore(BaseModel):
    """How one screening arm fared on one review's records."""

    n: int = Field(ge=1)
    n_included: int = Field(ge=1)
    prevalence: float
    kept_n: int
    kept_recall: float = Field(description="Included records the arm kept / included.")
    kept_precision: float | None = Field(
        description="Included records among the kept; None when nothing was kept."
    )
    work_saved_kept: float = Field(description="1 - kept_n / n.")
    n_read_at_target: int = Field(
        description="Records read, in ranked order, to find TARGET_RECALL of the included."
    )
    work_saved_at_95: float = Field(
        description="1 - n_read_at_target / n: records never read at 95% recall."
    )
    wss_at_95: float = Field(
        description="work_saved_at_95 - 0.05: the classical WSS@95 (Cohen et al.)."
    )


def screening_score(
    scores: Sequence[float],
    kept: Sequence[bool],
    included: Sequence[bool],
    *,
    target_recall: float = TARGET_RECALL,
) -> ScreeningScore:
    """Recall, precision and work saved of one arm over one review's records.

    The three sequences run in the same record order. ``work_saved_at_95``
    ranks the records by score, highest first, and counts how many must be
    read to find 95% of the included ones; records with equal scores are read
    together (a tie is never resolved in the arm's favour). Raises
    ``ValueError`` when the lengths differ or no record is included.
    """
    n = len(scores)
    if not (n == len(kept) == len(included)):
        raise ValueError("scores, kept and included must be the same length")
    n_included = sum(included)
    if n_included == 0:
        raise ValueError("a review with no included record has no recall")
    needed = max(1, math.ceil(target_recall * n_included - 1e-9))
    order = sorted(range(n), key=lambda i: -scores[i])
    found = 0
    read = 0
    position = 0
    while position < n and found < needed:
        tied = [i for i in order[position:] if scores[i] == scores[order[position]]]
        found += sum(1 for i in tied if included[i])
        read += len(tied)
        position += len(tied)
    kept_n = sum(kept)
    kept_included = sum(1 for k, inc in zip(kept, included, strict=True) if k and inc)
    saved = 1 - read / n
    return ScreeningScore(
        n=n,
        n_included=n_included,
        prevalence=n_included / n,
        kept_n=kept_n,
        kept_recall=kept_included / n_included,
        kept_precision=kept_included / kept_n if kept_n else None,
        work_saved_kept=1 - kept_n / n,
        n_read_at_target=read,
        work_saved_at_95=saved,
        wss_at_95=saved - (1 - target_recall),
    )


# -- web: claim support and URL integrity ---------------------------------------------


class ClaimSupport(BaseModel):
    """How many of an answer's claims the evidence behind them supports.

    ``unsourced`` claims have nothing to check against (an uncited report
    sentence, a citation to a page that was never fetched). They are counted
    and reported, and kept out of the precision so that it measures only what
    a judge could check.
    """

    n_judged: int = Field(ge=0)
    n_supported: int = Field(ge=0)
    n_unsourced: int = Field(ge=0)
    precision: float | None = Field(description="None when no claim was judged.")


def claim_support(verdicts: Sequence[bool], *, n_unsourced: int = 0) -> ClaimSupport:
    """Precision of judged claims: supported / judged."""
    supported = sum(verdicts)
    return ClaimSupport(
        n_judged=len(verdicts),
        n_supported=supported,
        n_unsourced=n_unsourced,
        precision=supported / len(verdicts) if verdicts else None,
    )


_DEAD: frozenset[str] = frozenset({"not_found", "dns_failure", "server_error", "error"})
_URL = re.compile(r"https?://[^\s<>\"'`\]]+")
_DOI_IN_TEXT = re.compile(r"\b10\.\d{4,9}/[^\s<>\"'`\]]+")
_TRAILING = ".,;:!?)}"


class UrlIntegrity(BaseModel):
    """Do the links of an answer resolve?

    ``ok`` resolved; ``dead`` are 404/410, an unresolvable host, a server error
    or a failed connection; ``unverified`` could not be settled (a 4xx that is
    often a bot block, a timeout, a URL our own safety check refused).
    ``integrity`` is ok / (ok + dead): unverified links count neither way.
    """

    n: int = Field(ge=0)
    ok: int = Field(ge=0)
    dead: int = Field(ge=0)
    unverified: int = Field(ge=0)
    integrity: float | None = Field(description="None when no link could be settled.")


def url_integrity(statuses: Sequence[UrlStatus]) -> UrlIntegrity:
    """Sort probe results into resolved, dead and unverified."""
    ok = sum(1 for s in statuses if s == "ok")
    dead = sum(1 for s in statuses if s in _DEAD)
    return UrlIntegrity(
        n=len(statuses),
        ok=ok,
        dead=dead,
        unverified=len(statuses) - ok - dead,
        integrity=ok / (ok + dead) if ok + dead else None,
    )


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
