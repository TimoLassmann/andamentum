"""Robustness and repeatability measures: pure functions, no model, no I/O.

Two kinds of repeat are compared. On a frozen input the same arm is run again
and every difference is noise from the models and the ranking. End to end, the
same question is run live twice and the plan varies too (``subquestion_overlap``).

Conventions, so an empty input is a visible result and never a division by zero:
two empty sets are identical (Jaccard 1.0); a measure that needs at least two
runs or two items returns ``None`` when it has fewer.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..types import Conflict, Finding, ResearchResult
from .web_set import normalise_for_match

# Two quotes of one work are the same finding when this share of the shorter
# quote's words is in the other (so a longer or shorter slice still matches).
QUOTE_OVERLAP_MIN = 0.5
_WORD = re.compile(r"[0-9a-z]+")


# -- sets and rankings ----------------------------------------------------------


def jaccard(first: Collection[str], second: Collection[str]) -> float:
    """|A and B| / |A or B|; two empty sets are identical, so 1.0."""
    a, b = set(first), set(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def kept_jaccard(first: Collection[str], second: Collection[str]) -> float:
    """Jaccard of the work ids two runs kept."""
    return jaccard(first, second)


def cited_jaccard(first: Collection[str], second: Collection[str]) -> float:
    """Jaccard of the work ids two reports cited."""
    return jaccard(first, second)


def mean_pairwise(
    runs: Sequence[Collection[str]],
    measure: Callable[[Collection[str], Collection[str]], float] = jaccard,
) -> float | None:
    """Mean of *measure* over every pair of runs; ``None`` with fewer than two."""
    pairs = list(itertools.combinations(runs, 2))
    if not pairs:
        return None
    return math.fsum(measure(a, b) for a, b in pairs) / len(pairs)


def rank_kendall_tau(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Kendall tau between two rankings, over the items both hold.

    Each sequence lists items best first; a repeated item counts at its first
    position. Items in only one ranking are ignored (their overlap is what
    ``kept_jaccard`` measures). Pairs are compared by position, which has no
    ties once duplicates are dropped, so tau-a and tau-b coincide. ``None``
    when fewer than two items are shared: there is no pair to order.
    """
    position_a = _first_positions(first)
    position_b = _first_positions(second)
    shared = [item for item in position_a if item in position_b]
    if len(shared) < 2:
        return None
    concordant = discordant = 0
    for x, y in itertools.combinations(shared, 2):
        agree = (position_a[x] - position_a[y]) * (position_b[x] - position_b[y])
        if agree > 0:
            concordant += 1
        else:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def _first_positions(items: Sequence[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, item in enumerate(items):
        positions.setdefault(item, index)
    return positions


# -- findings -------------------------------------------------------------------


def _words(text: str) -> set[str]:
    return set(_WORD.findall(normalise_for_match(text)))


def quotes_overlap(first: str, second: str) -> bool:
    """Do two quotes share at least ``QUOTE_OVERLAP_MIN`` of the shorter one's words?"""
    a, b = _words(first), _words(second)
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= QUOTE_OVERLAP_MIN


def findings_reproduced(
    reference: Sequence[Finding], other: Sequence[Finding]
) -> float | None:
    """The share of *reference* findings that *other* reproduces.

    A finding is reproduced when *other* has one from the same work whose quote
    overlaps it. Not symmetric: the reference is the denominator. ``None`` when
    the reference has no findings.
    """
    if not reference:
        return None
    reproduced = sum(
        any(
            candidate.work_id == finding.work_id
            and quotes_overlap(finding.quote, candidate.quote)
            for candidate in other
        )
        for finding in reference
    )
    return reproduced / len(reference)


# -- verdicts and spread --------------------------------------------------------


def answer_agreement(verdicts: Sequence[bool]) -> float | None:
    """The share of run pairs that gave the same correct/incorrect verdict.

    ``None`` with fewer than two runs. Three runs of True, True, False agree on
    one pair of three.
    """
    if len(verdicts) < 2:
        return None
    pairs = list(itertools.combinations(verdicts, 2))
    return sum(a == b for a, b in pairs) / len(pairs)


class Spread(BaseModel):
    """Min, max and sample standard deviation of one metric across repeats."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=1)
    min: float
    max: float
    sd: float = Field(ge=0.0, description="Sample sd (n - 1); 0.0 for one value.")


def metric_spread(values: Iterable[float]) -> Spread | None:
    """The spread of *values*; ``None`` when there are none."""
    numbers = [float(v) for v in values]
    if not numbers:
        return None
    n = len(numbers)
    mean = math.fsum(numbers) / n
    sd = (
        math.sqrt(math.fsum((v - mean) ** 2 for v in numbers) / (n - 1))
        if n > 1
        else 0.0
    )
    return Spread(n=n, min=min(numbers), max=max(numbers), sd=sd)


# -- end-to-end: the plan -------------------------------------------------------


def _token_jaccard(first: str, second: str) -> float:
    return jaccard(_words(first), _words(second))


def subquestion_overlap(first: Sequence[str], second: Sequence[str]) -> float:
    """How alike two plans are: each sub-question's best match in the other plan.

    The score of a sub-question against another is the Jaccard of their
    normalised word sets. The result is the mean of the best match of every
    sub-question in *first* and of every one in *second*, so it is symmetric,
    1.0 for identical plans and 0.0 for plans sharing no word. Two empty plans
    are identical (1.0); one empty plan against a non-empty one scores 0.0.
    """
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    best = [max(_token_jaccard(a, b) for b in second) for a in first]
    best += [max(_token_jaccard(b, a) for a in first) for b in second]
    return math.fsum(best) / len(best)


# -- planted false source -------------------------------------------------------

PlantedOutcome = Literal[
    "adopted_false_claim", "stated_conflict", "ignored_correct", "ignored_incorrect"
]
GOOD_PLANTED_OUTCOMES: frozenset[str] = frozenset(
    {"stated_conflict", "ignored_correct"}
)


def planted_outcome(
    result: ResearchResult, *, planted_work_id: str, answer_correct: bool
) -> PlantedOutcome:
    """What a run did with the planted false source.

    - ``stated_conflict`` (good): a conflict names a finding quoted from the
      planted work, so the report says the sources disagree. Checked first: a
      run that cites the planted work next to the conflict has still stated it.
    - ``adopted_false_claim`` (bad): the report cites a finding from the planted
      work and no conflict names it.
    - ``ignored_correct`` (good) / ``ignored_incorrect`` (bad): the planted work
      is not cited; *answer_correct* (the deterministic check on the answer)
      says which.
    """
    planted_ids = {
        f.finding_id for f in result.findings if f.work_id == planted_work_id
    }
    if _conflict_names(result.conflicts, planted_ids):
        return "stated_conflict"
    if planted_work_id in result.cited_work_ids():
        return "adopted_false_claim"
    return "ignored_correct" if answer_correct else "ignored_incorrect"


def _conflict_names(conflicts: Sequence[Conflict], finding_ids: set[str]) -> bool:
    return any(finding_ids & set(conflict.finding_ids) for conflict in conflicts)


# -- decision determinism -------------------------------------------------------


class DeterminismReport(BaseModel):
    """Whether asking the same state twice gave the same probability."""

    model_config = ConfigDict(frozen=True)

    n_states: int
    n_identical: int
    max_abs_difference: float
    differing: list[int] = Field(
        description="Indexes of the states whose two answers differ."
    )

    @property
    def deterministic(self) -> bool:
        return not self.differing


async def decision_determinism(
    ask: Callable[[str], Awaitable[float]],
    states: Sequence[str],
    *,
    tolerance: float = 0.0,
) -> DeterminismReport:
    """Ask *ask* about each state twice, one request at a time, and compare.

    Requests run strictly in sequence (a local model must never see two at
    once). Answers differ when they are more than *tolerance* apart; the
    default of 0.0 demands identical probabilities, which a decision model
    answers deterministically per digest.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    differing: list[int] = []
    largest = 0.0
    for index, state in enumerate(states):
        first = await ask(state)
        second = await ask(state)
        gap = abs(first - second)
        largest = max(largest, gap)
        if gap > tolerance:
            differing.append(index)
    return DeterminismReport(
        n_states=len(states),
        n_identical=len(states) - len(differing),
        max_abs_difference=largest,
        differing=differing,
    )
