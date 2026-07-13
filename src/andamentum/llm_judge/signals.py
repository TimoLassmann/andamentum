"""Pure math: verbalized distribution -> confidence/doubt/verdict, plus
canonicalisation, roll-up, and panel aggregation.

No I/O, no pydantic-ai, no dependency on any other andamentum sub-module —
this file is deliberately a leaf even within ``llm_judge``, so its
correctness can be verified with fast, deterministic unit tests (see
``tests/test_signals.py``). It mirrors
``andamentum.epistemic.judgment_signal`` (verbalized-distribution ->
argmax/confidence/entropy) and ``andamentum.epistemic.operations.integration``
(K-agreement: min-on-agreement, conservative aggregation) — the same
statistical shape, applied to a judge's verdict instead of an epistemic
claim's belief distribution.

DOUBT CAVEAT: every doubt/entropy value produced here is a SUSPICION-RAISER
ONLY. High doubt is a reason to escalate; low doubt never proves a verdict
correct. The signal is validated as informative on capable (~12B+/mini-class)
models and is noise on nano-class models — this module applies the same
formula uniformly regardless of model tier (no tier branching), so callers
reading doubt from a nano-class judge should weight it accordingly less.
"""

from __future__ import annotations

from collections import Counter
from math import isclose, isfinite, log

# ── Canonical orderings and tie-break tuples ────────────────────────────
#
# Tie-breaks are fixed and documented so aggregation is deterministic and
# reproducible — never resolved by iteration order, dict hashing, or any
# other incidental ordering.

SCORE_LABELS: tuple[str, str, str] = ("meets", "partial", "fails")
# Conservative: on a tie, prefer the MORE CAUTIOUS verdict.
SCORE_TIEBREAK: tuple[str, str, str] = ("fails", "partial", "meets")

COMPARE_LABELS: tuple[str, str, str] = ("a", "tie", "b")
# Conservative: on a tie, prefer 'tie' over declaring a winner.
COMPARE_TIEBREAK: tuple[str, str, str] = ("tie", "a", "b")

# The label a compare mode reports when it cannot pick a winner. A tie among
# judges is NOT broken toward 'a' or 'b': doing so would reintroduce, at the
# panel layer, exactly the position bias that running both presentation
# orders exists to cancel.
COMPARE_UNDECIDED = "tie"

# Masses within this absolute tolerance of each other count as TIED.
#
# Exact float equality (`v == best`) is the wrong test here. Masses arrive via
# normalize_three + mean_distributions, so a mathematically exact tie routinely
# lands on values like 0.465 vs 0.46499999999999997 — which `==` calls a
# winner. That silently bypasses the conservative tiebreak and hands the
# verdict to whichever side won a rounding error, i.e. resolves ties by
# incidental floating-point noise: precisely what this module's tie-break
# tuples exist to prevent. The tolerance is far above accumulated FP error for
# 3-element means and far below any belief difference a judge could mean.
TIE_TOLERANCE = 1e-9

# A doubt value at or above this threshold is, on its own, enough to flip
# needs_review — regardless of panel agreement. Named constant so it is
# one grep away from every predicate that uses it.
NEEDS_REVIEW_DOUBT_THRESHOLD = 0.5


# ── Verbalized-distribution primitives ──────────────────────────────────


def normalize_three(a: float, b: float, c: float) -> list[float]:
    """Normalise three non-negative belief points into a probability vector.

    Models are asked for integers summing to 100, but small models drift;
    normalise by the actual sum rather than assuming exactly 100.

    Raises:
        ValueError: if any value is non-finite or negative, or the total is
            not positive — a degenerate elicitation the caller should surface
            as a loud failure, never paper over with a uniform guess.
    """
    vals = [float(a), float(b), float(c)]
    # The finiteness check must come FIRST and be its own test. NaN defeats
    # every comparison it appears in (`nan < 0` and `nan <= 0` are both
    # False), so it slips through an ordering guard and propagates: a NaN
    # distribution makes normalized_entropy return 0.0 — reading as MAXIMALLY
    # CERTAIN — and compare_verdict return a silent 'tie'. A judge is allowed
    # to fail; it is not allowed to be confidently meaningless.
    if not all(isfinite(v) for v in vals):
        raise ValueError(f"belief points must be finite, got {vals}")
    if any(v < 0 for v in vals):
        raise ValueError(f"belief points must be non-negative, got {vals}")
    total = sum(vals)
    if total <= 0:
        raise ValueError("belief points must sum to a positive value, got all zero")
    return [v / total for v in vals]


def max_mass(dist: list[float]) -> float:
    """Probability mass on the highest-mass entry — the confidence signal."""
    return max(dist)


def normalized_entropy(dist: list[float]) -> float:
    """Normalised Shannon entropy in [0, 1] — the doubt signal.

    0 = one-hot (maximally sure), 1 = uniform (maximally unsure). See the
    module-level DOUBT CAVEAT: this is a suspicion-raiser only.
    """
    n = len(dist)
    if n <= 1:
        return 0.0
    h = -sum(p * log(p) for p in dist if p > 0.0)
    return h / log(n)


def argmax_label(
    dist: list[float], labels: tuple[str, ...], tiebreak: tuple[str, ...]
) -> str:
    """The label of the highest-mass entry in ``dist``, ties resolved by
    the first label in ``tiebreak`` among the tied indices.

    ``labels[i]`` is the label for ``dist[i]``; ``tiebreak`` need not be in
    the same order as ``labels`` — it is the preference order to consult
    when two or more entries are tied for the maximum. Ties are detected
    within :data:`TIE_TOLERANCE`, not by exact float equality — see that
    constant for why.
    """
    best = max(dist)
    tied_labels = {
        labels[i] for i, v in enumerate(dist) if isclose(v, best, abs_tol=TIE_TOLERANCE)
    }
    if len(tied_labels) == 1:
        return next(iter(tied_labels))
    for label in tiebreak:
        if label in tied_labels:
            return label
    # Unreachable if tiebreak is a permutation of labels, but fail loud
    # rather than silently returning an arbitrary label if it is not.
    raise ValueError(
        f"tiebreak {tiebreak!r} does not cover tied labels {tied_labels!r}"
    )


def mass_on(dist: list[float], labels: tuple[str, ...], label: str) -> float:
    """The probability mass ``dist`` puts on ``label``.

    This is what ``confidence`` must be built from. Reaching for
    :func:`max_mass` instead is only correct when the reported verdict IS the
    argmax — which is not true for compare, where an even a-vs-b split reports
    ``'tie'`` while the maximum mass sits on 'a' and 'b'. Publishing the
    argmax as confidence there would claim 0.45 confidence in a verdict
    holding 0.10 of the belief.
    """
    if label not in labels:
        raise ValueError(f"label {label!r} is not one of {labels!r}")
    return dist[labels.index(label)]


def to_percentages(dist: list[float]) -> list[int]:
    """Render a probability vector as integers that ALWAYS sum to exactly 100.

    Naive per-element ``round(p * 100)`` does not: a flat [1/3, 1/3, 1/3]
    rounds to [33, 33, 33] = 99, breaking the sum-to-100 contract that
    ``CriterionScore`` documents and that consumers read as percentages.

    This uses the largest-remainder (Hamilton) method: floor everything, then
    hand the leftover points to the entries with the largest discarded
    fractions. Ties in the remainder are broken by position, which is stable
    and deterministic — never by dict or set iteration order.
    """
    scaled = [p * 100 for p in dist]
    floors = [int(s) for s in scaled]
    leftover = 100 - sum(floors)
    if leftover:
        # Largest remainder first; position breaks ties deterministically.
        order = sorted(range(len(dist)), key=lambda i: (-(scaled[i] - floors[i]), i))
        for i in order[:leftover]:
            floors[i] += 1
    return floors


def mean_distributions(dists: list[list[float]]) -> list[float]:
    """Element-wise mean of already-normalized vectors, renormalised.

    Raises:
        ValueError: on an empty list or mismatched vector lengths.
    """
    if not dists:
        raise ValueError("mean_distributions requires at least one distribution")
    n = len(dists[0])
    if any(len(d) != n for d in dists):
        raise ValueError("all distributions must have the same length")
    means = [sum(d[i] for d in dists) / len(dists) for i in range(n)]
    total = sum(means)
    if total <= 0:
        raise ValueError("mean of distributions summed to zero")
    return [m / total for m in means]


# ── judge_score roll-up ─────────────────────────────────────────────────


def roll_up_score(
    criterion_dists: list[list[float]],
) -> tuple[list[float], str, float, float]:
    """Equal-weight roll-up of one judge's per-criterion distributions.

    Returns ``(overall_dist, verdict, confidence, doubt)``. The roll-up is
    an equal-weight mean across criteria (the ``Criterion`` schema has no
    weight field) — see ``ScoreResult`` docstring for the resulting
    dilution limitation.
    """
    overall_dist = mean_distributions(criterion_dists)
    verdict = argmax_label(overall_dist, SCORE_LABELS, SCORE_TIEBREAK)
    return (
        overall_dist,
        verdict,
        # Mass on the REPORTED verdict, not the argmax — they differ whenever
        # the conservative tiebreak moved the verdict off the raw maximum.
        mass_on(overall_dist, SCORE_LABELS, verdict),
        normalized_entropy(overall_dist),
    )


# ── judge_compare canonicalisation ──────────────────────────────────────


def canonicalize(raw: list[float], order: str) -> list[float]:
    """Map a raw ``[p1, ptie, p2]`` (as elicited, 'Response 1'/'Response 2')
    into the canonical ``[pa, ptie, pb]`` ordering.

    For order ``'AB'``, Response 1 = output_a, so the row is already
    canonical. For ``'BA'``, Response 1 = output_b, so the two ends swap.
    """
    p1, ptie, p2 = raw
    if order == "AB":
        return [p1, ptie, p2]
    if order == "BA":
        return [p2, ptie, p1]
    raise ValueError(f"order must be 'AB' or 'BA', got {order!r}")


def order_average(canon_ab: list[float], canon_ba: list[float]) -> list[float]:
    """Element-wise mean of the two canonicalised order histograms."""
    return mean_distributions([canon_ab, canon_ba])


def compare_verdict(dist: list[float]) -> str:
    """The winner label ('a'/'tie'/'b') for a canonical ``[pa, ptie, pb]``.

    Compare ties resolve TOWARD 'tie' — a stronger rule than the generic
    ``argmax_label`` tiebreak: ANY tie for the maximum mass (including an
    a/b tie whose own 'tie' bucket happens to carry less mass) is treated
    as undecided and reported as ``'tie'``, because a judge that split its
    belief evenly between 'a' and 'b' is, in effect, unable to pick a
    winner — declaring one anyway on a coin-flip would be less honest than
    calling it a tie. A unique (non-tied) maximum on 'a' or 'b' still wins
    outright.

    Ties are detected within :data:`TIE_TOLERANCE`, not by exact float
    equality: an even a/b split comes out of order-averaging as
    0.4944444444444445 vs 0.4944444444444444, which ``==`` calls a winner —
    letting a rounding error decide which response was better.
    """
    pa, ptie, pb = dist
    best = max(pa, ptie, pb)
    tied = sum(1 for v in (pa, ptie, pb) if isclose(v, best, abs_tol=TIE_TOLERANCE))
    if tied > 1:
        return COMPARE_UNDECIDED
    if isclose(pa, best, abs_tol=TIE_TOLERANCE):
        return "a"
    if isclose(pb, best, abs_tol=TIE_TOLERANCE):
        return "b"
    return COMPARE_UNDECIDED


def order_consistent(canon_ab: list[float], canon_ba: list[float]) -> bool:
    """Did the A-first and B-first orders agree on the winner?

    Computed as the argmax (with the same conservative tie-break) of each
    order's own canonical histogram — not the averaged one — so a flip
    between any pair of {'a', 'tie', 'b'} across orders is detected.
    """
    return compare_verdict(canon_ab) == compare_verdict(canon_ba)


# ── Panel aggregation (uniform rules — one code path for fast and panel) ─


def panel_majority(
    verdicts: list[str],
    tiebreak: tuple[str, ...],
    *,
    undecided: str | None = None,
) -> tuple[str, bool]:
    """Majority vote across judges' verdicts.

    Returns ``(majority_verdict, unanimous)``. On a tie for the most votes
    (including a 1-1-1 three-way split), resolution is deterministic — never
    by insertion order or dict iteration order:

    - ``undecided`` given (compare mode passes :data:`COMPARE_UNDECIDED`): ANY
      hung vote reports that label. This is load-bearing, not cosmetic.
      Consulting ``COMPARE_TIEBREAK`` here instead looks conservative but is
      not: with ``tied = {'a', 'b'}`` the preference order ``('tie','a','b')``
      skips 'tie' (it drew no votes) and returns **'a'** — handing every hung
      panel to whichever output the caller happened to pass FIRST. That is a
      systematic position bias, reintroduced at the panel layer by the very
      tuple meant to prevent it, and it would poison any preference labels
      harvested from split panels. A single judge with an even a/b mass split
      already reports 'tie' (see :func:`compare_verdict`); a hung panel is the
      same situation and must answer the same way.
    - ``undecided`` omitted (score mode): resolve via ``tiebreak``, whose
      labels are ordered most-cautious-first, so a hung score panel lands on
      the more cautious verdict rather than the more flattering one.
    """
    if not verdicts:
        raise ValueError("panel_majority requires at least one verdict")
    counts = Counter(verdicts)
    top = max(counts.values())
    tied = {label for label, c in counts.items() if c == top}
    if len(tied) == 1:
        majority = next(iter(tied))
    elif undecided is not None:
        majority = undecided
    else:
        majority = next(label for label in tiebreak if label in tied)
    unanimous = len(counts) == 1
    return majority, unanimous


def panel_confidence(masses_on_verdict: list[float]) -> float:
    """Panel confidence = MIN across judges of each judge's mass ON THE
    REPORTED VERDICT — conservative, never averaged up, on agreement OR
    disagreement (mirrors epistemic K-agreement).

    The argument must be each judge's belief in the verdict the panel is
    ACTUALLY REPORTING — not each judge's confidence in its own verdict.
    Those coincide on a unanimous panel and diverge badly on a split one:
    with judges (95,3,2), (95,3,2), (30,30,40) the majority is 'meets', but
    min-of-own-confidence returns 0.40 — the dissenter's belief in **fails**,
    published as the panel's confidence in **meets**. Feeding this function
    the mass on the majority label instead yields 0.30 (the least convinced
    judge's actual belief in 'meets'), which is both conservative and about
    the thing being reported. Unanimity still scores high, so the agreement
    gate's intended behaviour is preserved.
    """
    if not masses_on_verdict:
        raise ValueError("panel_confidence requires at least one judge")
    return min(masses_on_verdict)


def panel_doubt(doubts: list[float]) -> float:
    """Panel doubt = MAX across judges — a suspicion-raiser is never
    dampened by judges that happened to be more confident."""
    if not doubts:
        raise ValueError("panel_doubt requires at least one judge")
    return max(doubts)


# ── needs_review predicates (the ONE explicit trigger per mode) ─────────


def needs_review_score(unanimous: bool, doubt: float) -> bool:
    """``judge_score`` needs_review trigger.

    True when the panel is split (pass ``unanimous=True`` for fast mode,
    where there is only one judge and no split is possible) OR the doubt
    signal is at or above ``NEEDS_REVIEW_DOUBT_THRESHOLD``. The split
    trigger is load-bearing (structural, always reliable); the doubt
    trigger is a suspicion-raiser (see module docstring) — reliable on
    capable models, noisy on nano-class ones, but applied uniformly.
    """
    return (not unanimous) or (doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD)


def needs_review_compare(
    order_consistent_flag: bool, unanimous: bool, doubt: float
) -> bool:
    """``judge_compare`` needs_review trigger.

    True when the order-swap check disagreed (for a panel, pass
    ``all(judge order_consistent)``; for fast mode, the single judge's own
    order_consistent) OR the panel is split OR doubt is at or above
    ``NEEDS_REVIEW_DOUBT_THRESHOLD``. The order-flip and split triggers are
    load-bearing (structural); doubt is a suspicion-raiser only.
    """
    return (
        (not order_consistent_flag)
        or (not unanimous)
        or (doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD)
    )


__all__ = [
    "SCORE_LABELS",
    "SCORE_TIEBREAK",
    "COMPARE_LABELS",
    "COMPARE_TIEBREAK",
    "COMPARE_UNDECIDED",
    "TIE_TOLERANCE",
    "NEEDS_REVIEW_DOUBT_THRESHOLD",
    "normalize_three",
    "max_mass",
    "mass_on",
    "normalized_entropy",
    "argmax_label",
    "to_percentages",
    "mean_distributions",
    "roll_up_score",
    "canonicalize",
    "order_average",
    "compare_verdict",
    "order_consistent",
    "panel_majority",
    "panel_confidence",
    "panel_doubt",
    "needs_review_score",
    "needs_review_compare",
]
