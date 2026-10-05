"""COVERAGE: is each sub-question supported by independent sources, and do any disagree?

Three parts, two of them pure code:

- ``compute_coverage``: per sub-question and per ``must`` criterion, the number
  of *independent* sources behind its findings.
- ``decide_stop``: the stop decision from the coverage, the discovery curve
  and the budget. A pure function.
- ``find_conflicts``: findings of one sub-question that contradict each other,
  judged over a capped set of pairs by the decision model (a yes/no question
  per pair) or, without one, by the language model.

Independence. Two findings are independent when their sources have different
*origins*. A work's origin is the family name of its first author when it has
authors, else the registrable domain of its URL, else the work itself. A
preprint and its journal version, or two papers by one first author, are one
origin; two pages of one website are one origin. This is deliberately
conservative: ``covered`` should not be claimed on one group's output counted
twice.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from andamentum.references import Work

from .ask import Asker, BudgetExhausted
from .types import (
    MIN_INDEPENDENT_SOURCES,
    Budget,
    CandidateScore,
    Conflict,
    Coverage,
    Criterion,
    Finding,
    SubQuestion,
)

# The discovery curve is flat when the last round kept fewer new candidates than this.
MIN_NEW_PER_ROUND = 3
# A sub-question is given up on (not covered, not worth more rounds) after at
# least this many rounds of searching.
MIN_ROUNDS_BEFORE_UNRESOLVABLE = 2
# Pairs of findings checked for contradiction, per sub-question, over the run.
MAX_PAIRS_PER_SUBQUESTION = 40
# Pairs put to the model per request round; a spent budget loses at most one round.
PAIR_CHUNK = 8
# A candidate satisfies a candidate-level criterion at or above this probability.
CRITERION_MET_P = 0.5

# Second-level labels under which a country code is the registrable part's parent
# (example.co.uk, example.com.au): not a public-suffix list, a cheap approximation.
_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or"})
_FAMILY_SKIP = frozenset({"jr", "sr", "ii", "iii", "iv"})
_WORD = re.compile(r"[^\W_]{3,}")


def registrable_domain(url: str) -> str:
    """The site a URL belongs to: ``www.news.example.com`` becomes ``example.com``."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _family_name(author: str) -> str:
    """The family name of an author string, 'Family, Given' or 'Given Family'."""
    author = author.strip()
    if "," in author:
        return author.split(",", 1)[0].strip().casefold()
    parts = [p for p in author.split() if p.casefold().rstrip(".") not in _FAMILY_SKIP]
    return parts[-1].casefold() if parts else ""


def origin_key(work: Work) -> str:
    """The independence key of a work: first author's family name, else the
    registrable domain of its URL, else the work itself."""
    if work.authors:
        family = _family_name(work.authors[0])
        if family:
            return f"author:{family}"
    if work.url:
        domain = registrable_domain(work.url)
        if domain:
            return f"site:{domain}"
    return f"work:{work.work_id}"


def independent_sources(findings: Sequence[Finding], works: Mapping[str, Work]) -> int:
    """How many distinct origins stand behind *findings*."""
    return len({_origin_of(f, works) for f in findings})


def _origin_of(finding: Finding, works: Mapping[str, Work]) -> str:
    work = works.get(finding.work_id)
    return origin_key(work) if work is not None else f"work:{finding.work_id}"


def compute_coverage(
    subquestions: Sequence[SubQuestion],
    *,
    criteria: Mapping[str, Sequence[Criterion]],
    findings: Sequence[Finding],
    works: Mapping[str, Work],
    candidate_scores: Sequence[CandidateScore] = (),
    rounds_done: int = 1,
    new_kept_last_round: Mapping[str, int] | None = None,
) -> list[Coverage]:
    """Coverage of each sub-question.

    ``covered`` means at least ``MIN_INDEPENDENT_SOURCES`` independent sources.
    ``per_criterion`` counts, for each ``must`` criterion, the independent
    sources that satisfy it: for a passage-level criterion, those with a finding
    that names it; for a candidate-level one, those whose candidate score has
    ``p >= CRITERION_MET_P`` for it. ``note`` names the ``must`` criteria that
    no independent source satisfies yet, which is where a further search
    should aim.

    ``unresolvable`` is set for a sub-question that is not covered after
    ``MIN_ROUNDS_BEFORE_UNRESOLVABLE`` rounds, when the last round kept fewer
    than ``MIN_NEW_PER_ROUND`` new candidates for it (``new_kept_last_round``
    maps a sub-question id to that count; absent means unknown, never given up).
    """
    met: dict[tuple[str, str], set[str]] = {}
    for score in candidate_scores:
        for criterion_id, p in score.criterion_p.items():
            if p >= CRITERION_MET_P:
                met.setdefault((score.subquestion_id, criterion_id), set()).add(
                    score.work_id
                )
    result: list[Coverage] = []
    for subquestion in subquestions:
        own = [f for f in findings if f.subquestion_id == subquestion.id]
        total = independent_sources(own, works)
        per_criterion: dict[str, int] = {}
        for criterion in criteria.get(subquestion.id, ()):
            if criterion.kind != "must":
                continue
            if criterion.applies_to == "passage":
                support = [f for f in own if criterion.id in f.criterion_ids]
            else:
                holders = met.get((subquestion.id, criterion.id), set())
                support = [f for f in own if f.work_id in holders]
            per_criterion[criterion.id] = independent_sources(support, works)
        covered = total >= MIN_INDEPENDENT_SOURCES
        new_kept = (new_kept_last_round or {}).get(subquestion.id)
        unresolvable = (
            not covered
            and new_kept is not None
            and rounds_done >= MIN_ROUNDS_BEFORE_UNRESOLVABLE
            and new_kept < MIN_NEW_PER_ROUND
        )
        gaps = [c for c, n in per_criterion.items() if n == 0]
        note = (
            f"no independent source yet for must criteria: {', '.join(gaps)}"
            if gaps and own
            else ""
        )
        result.append(
            Coverage(
                subquestion_id=subquestion.id,
                independent_sources=total,
                per_criterion=per_criterion,
                covered=covered,
                unresolvable=unresolvable,
                note=note,
            )
        )
    return result


@dataclass(frozen=True)
class StopDecision:
    """``stop`` is whether to end the search; ``reason`` is why (``''`` to go on).

    A budget reason has the form ``budget:<name>``.
    """

    stop: bool
    reason: str = ""


def decide_stop(
    coverage: Sequence[Coverage],
    *,
    round_number: int,
    new_kept_by_round: Sequence[int],
    budget: Budget,
    spent: str | None = None,
    min_new_per_round: int = MIN_NEW_PER_ROUND,
) -> StopDecision:
    """Whether the search ends after round *round_number* (1-based), and why.

    In order: every sub-question covered (``covered``); every one either
    covered or given up on (``unresolvable``); a bound the caller found spent
    (*spent*): ``budget:<name>``, ahead of the curve, because a round the
    budget cut short keeps few candidates for that reason; the discovery curve
    flat, the last round having kept fewer than *min_new_per_round* new
    candidates (``flat``; not before a second round, whose count is what is
    compared); ``max_rounds`` reached: ``budget:max_rounds``. Otherwise the
    search goes on.
    """
    if coverage and all(c.covered for c in coverage):
        return StopDecision(True, "covered")
    if coverage and all(c.covered or c.unresolvable for c in coverage):
        return StopDecision(True, "unresolvable")
    if spent:
        return StopDecision(True, f"budget:{spent}")
    if (
        round_number >= 2
        and new_kept_by_round
        and new_kept_by_round[-1] < min_new_per_round
    ):
        return StopDecision(True, "flat")
    if round_number >= budget.max_rounds:
        return StopDecision(True, "budget:max_rounds")
    return StopDecision(False)


# -- conflicts --------------------------------------------------------------

CONFLICT_QUESTION = (
    "Do the two findings contradict each other, so that they cannot both be true "
    "of the same thing? Answer no when they differ in subject, population, "
    "setting or time, when one only adds detail to the other, or when they merely "
    "differ in strength."
)

PairKey = tuple[str, str]


def pair_key(first: Finding, second: Finding) -> PairKey:
    """The ordered ``(id, id)`` key of a pair of findings (by finding number)."""
    a, b = sorted((first.finding_id, second.finding_id), key=lambda i: int(i[1:]))
    return a, b


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def _similarity(first: Finding, second: Finding) -> float:
    a, b = _terms(first.claim), _terms(second.claim)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _candidate_pairs(
    findings: Sequence[Finding], works: Mapping[str, Work]
) -> list[tuple[Finding, Finding]]:
    """Pairs within one sub-question from different origins, most alike first.

    Two findings from one origin cannot be a disagreement between sources.
    Likeness by shared claim words is a cheap proxy for "about the same thing",
    which a contradiction needs; it only orders the pairs the cap will cut.
    """
    pairs = [
        (a, b)
        for i, a in enumerate(findings)
        for b in findings[i + 1 :]
        if _origin_of(a, works) != _origin_of(b, works)
    ]
    pairs.sort(key=lambda p: (-_similarity(*p), pair_key(*p)))
    return pairs


def _pair_state(first: Finding, second: Finding) -> dict[str, object]:
    return {
        "finding_a": {"claim": first.claim, "quote": first.quote},
        "finding_b": {"claim": second.claim, "quote": second.quote},
    }


async def find_conflicts(
    findings: Sequence[Finding],
    *,
    works: Mapping[str, Work],
    asker: Asker,
    checked: dict[PairKey, Conflict | None],
    max_pairs: int = MAX_PAIRS_PER_SUBQUESTION,
) -> list[Conflict]:
    """Contradictions among *findings*, judged pair by pair.

    *checked* is the run's memory: every pair judged so far, with its conflict
    or ``None``. Only pairs not in it are put to the model, and at most
    *max_pairs* are ever judged per sub-question, so a growing findings table
    costs a bounded number of questions. A spent budget stops the judging and
    keeps what was found. Returns every conflict in *checked*, in pair order.
    """
    by_subquestion: dict[str, list[Finding]] = {}
    for finding in findings:
        by_subquestion.setdefault(finding.subquestion_id, []).append(finding)
    spent = False
    subquestion_of = {f.finding_id: f.subquestion_id for f in findings}
    for subquestion_id, own in by_subquestion.items():
        judged = sum(1 for a, _ in checked if subquestion_of.get(a) == subquestion_id)
        room = max_pairs - judged
        fresh = [p for p in _candidate_pairs(own, works) if pair_key(*p) not in checked]
        pairs = fresh[: max(room, 0)]
        by_decision = asker.decision_model is not None
        for start in range(0, len(pairs), PAIR_CHUNK):
            chunk = pairs[start : start + PAIR_CHUNK]
            try:
                probabilities = await asker.ask_each(
                    "conflict",
                    instruction=CONFLICT_QUESTION,
                    states=[_pair_state(a, b) for a, b in chunk],
                )
            except BudgetExhausted:
                spent = True
                break
            for (a, b), p in zip(chunk, probabilities):
                if p >= 0.5:
                    note = (
                        f"decision model, p={p:.2f}"
                        if by_decision
                        else "language model"
                    )
                    checked[pair_key(a, b)] = Conflict(
                        subquestion_id=subquestion_id,
                        finding_ids=list(pair_key(a, b)),
                        note=note,
                    )
                else:
                    checked[pair_key(a, b)] = None
        if spent:
            break
    conflicts = [c for c in checked.values() if c is not None]
    for conflict in conflicts:
        asker.store.add_run_item(
            asker.run_id, "conflict", "-".join(conflict.finding_ids), conflict
        )
    return conflicts


def record_coverage(asker: Asker, coverage: Sequence[Coverage]) -> None:
    """Write each sub-question's coverage to the run's items (replacing the last round's)."""
    for entry in coverage:
        asker.store.add_run_item(asker.run_id, "coverage", entry.subquestion_id, entry)
