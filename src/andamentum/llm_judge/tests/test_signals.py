"""Pure-math unit tests for ``andamentum.llm_judge.signals``.

No mocking, no network — these are the module's core correctness
guarantee: everything downstream (elicit.py, panel.py, __init__.py) is
plumbing around these deterministic functions.
"""

from __future__ import annotations

import math

import pytest

from andamentum.llm_judge import signals


# ── normalize_three ──────────────────────────────────────────────────────


def test_normalize_three_renormalises_drifted_sum():
    # Small models drift off exactly-100; normalise by the actual sum.
    dist = signals.normalize_three(60, 30, 20)  # sums to 110
    assert dist == pytest.approx([60 / 110, 30 / 110, 20 / 110])
    assert sum(dist) == pytest.approx(1.0)


def test_normalize_three_negative_raises():
    with pytest.raises(ValueError):
        signals.normalize_three(-5, 50, 55)


def test_normalize_three_zero_sum_raises():
    with pytest.raises(ValueError):
        signals.normalize_three(0, 0, 0)


# ── max_mass / normalized_entropy ────────────────────────────────────────


def test_max_mass_equals_confidence():
    dist = [0.7, 0.2, 0.1]
    assert signals.max_mass(dist) == 0.7


def test_normalized_entropy_one_hot_is_zero():
    dist = [1.0, 0.0, 0.0]
    assert signals.normalized_entropy(dist) == pytest.approx(0.0)


def test_normalized_entropy_uniform_is_one():
    dist = [1 / 3, 1 / 3, 1 / 3]
    assert signals.normalized_entropy(dist) == pytest.approx(1.0)


def test_normalized_entropy_between_extremes():
    dist = [0.5, 0.3, 0.2]
    h = signals.normalized_entropy(dist)
    assert 0.0 < h < 1.0
    assert not math.isnan(h)


# ── argmax_label / conservative tie-breaks ───────────────────────────────


def test_argmax_label_clear_winner():
    dist = [0.1, 0.2, 0.7]
    label = signals.argmax_label(dist, signals.SCORE_LABELS, signals.SCORE_TIEBREAK)
    assert label == "fails"  # SCORE_LABELS[2] = 'fails'


def test_argmax_label_score_tie_resolves_to_fails():
    # meets == fails, tied for the max -> conservative tiebreak prefers 'fails'.
    dist = [0.5, 0.0, 0.5]
    label = signals.argmax_label(dist, signals.SCORE_LABELS, signals.SCORE_TIEBREAK)
    assert label == "fails"


def test_argmax_label_three_way_score_tie_resolves_to_fails():
    dist = [1 / 3, 1 / 3, 1 / 3]
    label = signals.argmax_label(dist, signals.SCORE_LABELS, signals.SCORE_TIEBREAK)
    assert label == "fails"


def test_argmax_label_generic_tiebreak_picks_first_present_in_tied_set():
    # a == b, tied for the max, but 'tie' itself is NOT part of the tied
    # set (its own mass is 0) -> the generic tiebreak picks the first
    # preference-order label that IS among the tied labels ('a').
    # compare_verdict (below) applies the stronger, compare-specific "any
    # tie resolves to 'tie'" rule instead of this generic one.
    dist = [0.5, 0.0, 0.5]  # [a, tie, b]
    label = signals.argmax_label(dist, signals.COMPARE_LABELS, signals.COMPARE_TIEBREAK)
    assert label == "a"


def test_argmax_label_incomplete_tiebreak_raises():
    with pytest.raises(ValueError):
        signals.argmax_label([0.5, 0.5], ("x", "y"), ("z",))


# ── mean_distributions ────────────────────────────────────────────────────


def test_mean_distributions_averages_elementwise():
    dists = [[0.6, 0.3, 0.1], [0.4, 0.3, 0.3]]
    mean = signals.mean_distributions(dists)
    assert mean == pytest.approx([0.5, 0.3, 0.2])


def test_mean_distributions_empty_raises():
    with pytest.raises(ValueError):
        signals.mean_distributions([])


def test_mean_distributions_mismatched_length_raises():
    with pytest.raises(ValueError):
        signals.mean_distributions([[0.5, 0.5], [0.3, 0.3, 0.4]])


# ── roll_up_score ─────────────────────────────────────────────────────────


def test_roll_up_score_single_criterion_passes_through():
    dist = [0.1, 0.2, 0.7]
    overall, verdict, confidence, doubt = signals.roll_up_score([dist])
    assert overall == pytest.approx(dist)
    assert verdict == "fails"
    assert confidence == pytest.approx(0.7)
    assert doubt == pytest.approx(signals.normalized_entropy(dist))


def test_roll_up_score_dilutes_a_confident_fail():
    # One criterion is a confident FAIL (correctness); four others are
    # confident MEETS. Equal-weight roll-up dilutes the fail into an
    # overall 'meets' or 'partial' — the documented limitation.
    fail_dist = [0.0, 0.0, 1.0]  # confident fails
    meets_dists = [[1.0, 0.0, 0.0]] * 4  # confident meets, x4
    overall, verdict, _confidence, _doubt = signals.roll_up_score(
        [fail_dist] + meets_dists
    )
    assert overall == pytest.approx([0.8, 0.0, 0.2])
    assert verdict == "meets"  # the serious fault is diluted away


# ── canonicalize / order_average / compare_verdict / order_consistent ────


def test_canonicalize_ab_is_identity():
    raw = [70.0, 10.0, 20.0]
    assert signals.canonicalize(raw, "AB") == raw


def test_canonicalize_ba_swaps_ends():
    raw = [70.0, 10.0, 20.0]  # [p1, ptie, p2] with Response1=B, Response2=A
    assert signals.canonicalize(raw, "BA") == [20.0, 10.0, 70.0]


def test_canonicalize_invalid_order_raises():
    with pytest.raises(ValueError):
        signals.canonicalize([1.0, 1.0, 1.0], "XY")


def test_order_average_averages_canonical_rows():
    canon_ab = [0.6, 0.2, 0.2]
    canon_ba = [0.4, 0.2, 0.4]
    avg = signals.order_average(canon_ab, canon_ba)
    assert avg == pytest.approx([0.5, 0.2, 0.3])


def test_order_consistent_true_when_orders_agree():
    canon_ab = [0.7, 0.1, 0.2]  # argmax 'a'
    canon_ba = [0.6, 0.1, 0.3]  # argmax 'a'
    assert signals.order_consistent(canon_ab, canon_ba) is True


def test_order_consistent_false_on_flip():
    canon_ab = [0.7, 0.1, 0.2]  # argmax 'a'
    canon_ba = [0.1, 0.1, 0.8]  # argmax 'b'
    assert signals.order_consistent(canon_ab, canon_ba) is False


def test_order_consistent_false_on_flip_involving_tie():
    canon_ab = [0.4, 0.5, 0.1]  # argmax 'tie'
    canon_ba = [0.7, 0.1, 0.2]  # argmax 'a'
    assert signals.order_consistent(canon_ab, canon_ba) is False


def test_compare_verdict_a_b_tie_resolves_to_tie():
    # a and b tied for the max mass -> compare's own conservative rule
    # (distinct from the generic argmax_label tiebreak above) calls it
    # undecided rather than picking one on a coin-flip.
    assert signals.compare_verdict([0.5, 0.0, 0.5]) == "tie"


def test_compare_verdict_three_way_tie_resolves_to_tie():
    assert signals.compare_verdict([1 / 3, 1 / 3, 1 / 3]) == "tie"


def test_compare_verdict_unique_max_wins_outright():
    assert signals.compare_verdict([0.7, 0.1, 0.2]) == "a"
    assert signals.compare_verdict([0.1, 0.2, 0.7]) == "b"
    assert signals.compare_verdict([0.2, 0.6, 0.2]) == "tie"


# ── panel_majority / panel_confidence / panel_doubt ──────────────────────


def test_panel_majority_unanimous():
    majority, unanimous = signals.panel_majority(
        ["meets", "meets", "meets"], signals.SCORE_TIEBREAK
    )
    assert majority == "meets"
    assert unanimous is True


def test_panel_majority_clear_split():
    majority, unanimous = signals.panel_majority(
        ["meets", "meets", "partial"], signals.SCORE_TIEBREAK
    )
    assert majority == "meets"
    assert unanimous is False


def test_panel_majority_three_way_split_uses_conservative_tiebreak():
    majority, unanimous = signals.panel_majority(
        ["meets", "partial", "fails"], signals.SCORE_TIEBREAK
    )
    assert majority == "fails"  # 1-1-1: conservative tiebreak wins
    assert unanimous is False


def test_panel_majority_even_split_uses_conservative_tiebreak():
    majority, unanimous = signals.panel_majority(
        ["meets", "fails"], signals.SCORE_TIEBREAK
    )
    assert majority == "fails"  # 1-1: conservative tiebreak wins
    assert unanimous is False


def test_panel_confidence_is_min():
    assert signals.panel_confidence([0.9, 0.6, 0.8]) == 0.6


def test_panel_doubt_is_max():
    assert signals.panel_doubt([0.1, 0.4, 0.2]) == 0.4


# ── needs_review predicates ────────────────────────────────────────────


@pytest.mark.parametrize(
    "unanimous,doubt,expected",
    [
        (True, 0.1, False),  # agree, low doubt -> trust
        (True, 0.5, True),  # agree, doubt at threshold -> review
        (True, 0.9, True),  # agree, high doubt -> review
        (False, 0.0, True),  # split, even with zero doubt -> review
        (False, 0.9, True),  # split and high doubt -> review
    ],
)
def test_needs_review_score_truth_table(unanimous, doubt, expected):
    assert signals.needs_review_score(unanimous, doubt) is expected


@pytest.mark.parametrize(
    "order_ok,unanimous,doubt,expected",
    [
        (True, True, 0.1, False),  # everything fine
        (False, True, 0.0, True),  # order flip alone triggers review
        (True, False, 0.0, True),  # split alone triggers review
        (True, True, 0.5, True),  # doubt at threshold alone triggers review
        (False, False, 0.9, True),  # everything wrong
    ],
)
def test_needs_review_compare_truth_table(order_ok, unanimous, doubt, expected):
    assert signals.needs_review_compare(order_ok, unanimous, doubt) is expected


# ── to_percentages: the sum-to-100 contract ─────────────────────────────
#
# CriterionScore documents meets + partial + fails == 100. The panel path
# renders its pooled mean through to_percentages, so this contract is only
# as good as this function. Naive round(p * 100) broke it ([33, 33, 33] = 99).


def test_to_percentages_sums_to_100_on_the_flat_distribution():
    assert sum(signals.to_percentages([1 / 3, 1 / 3, 1 / 3])) == 100


def test_to_percentages_gives_the_leftover_point_to_the_largest_remainder():
    assert signals.to_percentages([1 / 3, 1 / 3, 1 / 3]) == [34, 33, 33]


def test_to_percentages_is_exact_when_no_rounding_is_needed():
    assert signals.to_percentages([0.8, 0.15, 0.05]) == [80, 15, 5]


def test_to_percentages_preserves_a_one_hot_distribution():
    assert signals.to_percentages([1.0, 0.0, 0.0]) == [100, 0, 0]


def test_to_percentages_never_reorders_the_argmax():
    """The rounded row must not claim a different winner than the float row."""
    pct = signals.to_percentages([0.34, 0.33, 0.33])
    assert pct.index(max(pct)) == 0


@pytest.mark.parametrize(
    "dist",
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.125, 0.125, 0.75],
        [0.005, 0.005, 0.99],
        [0.4999, 0.4999, 0.0002],
        [1 / 7, 2 / 7, 4 / 7],
        [0.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ],
)
def test_to_percentages_always_sums_to_exactly_100(dist):
    pct = signals.to_percentages(dist)
    assert sum(pct) == 100
    assert all(0 <= p <= 100 for p in pct)


# ═══════════════════════════════════════════════════════════════════════
# Regressions: the tie machinery
#
# Every one of these produced a WRONG-BUT-CONFIDENT verdict before the fix —
# the exact failure class that matters most for a component whose output may
# become a learning signal. The module header promises ties are never resolved
# "by iteration order, dict hashing, or any other incidental ordering"; these
# pin that promise.
# ═══════════════════════════════════════════════════════════════════════


def test_an_exact_tie_survives_floating_point_noise_and_hits_the_tiebreak():
    """meets = (0+93)/200 = 0.465, fails = (87+6)/200 = 0.465 — an exact tie,
    which SCORE_TIEBREAK says must resolve to the cautious 'fails'. The means
    come out as 0.465 vs 0.46499999999999997, so exact `==` called it 'meets'
    — the LEAST cautious label, chosen by a rounding error."""
    dist = signals.mean_distributions(
        [signals.normalize_three(0, 13, 87), signals.normalize_three(93, 1, 6)]
    )
    assert dist[0] != dist[2], "precondition: the tie is inexact in float"
    assert (
        signals.argmax_label(dist, signals.SCORE_LABELS, signals.SCORE_TIEBREAK)
        == "fails"
    )


def test_roll_up_score_resolves_a_float_noisy_tie_conservatively():
    _dist, verdict, _conf, _doubt = signals.roll_up_score(
        [signals.normalize_three(0, 13, 87), signals.normalize_three(93, 1, 6)]
    )
    assert verdict == "fails"


def test_compare_verdict_does_not_let_a_rounding_error_pick_the_winner():
    """An even a/b split order-averages to 0.4944…446 vs 0.4944…444. Exact
    equality declared 'a' the winner; neighbouring inputs flipped it to 'b'."""
    canon_ab = signals.canonicalize(signals.normalize_three(0, 2, 88), "AB")
    canon_ba = signals.canonicalize(signals.normalize_three(1, 0, 89), "BA")
    avg = signals.order_average(canon_ab, canon_ba)
    assert avg[0] != avg[2], "precondition: the tie is inexact in float"
    assert signals.compare_verdict(avg) == "tie"


def test_roll_up_confidence_is_the_mass_on_the_reported_verdict():
    """When the tiebreak moves the verdict off the raw maximum, confidence
    must follow the verdict, not the maximum."""
    dists = [signals.normalize_three(0, 13, 87), signals.normalize_three(93, 1, 6)]
    dist, verdict, confidence, _doubt = signals.roll_up_score(dists)
    assert verdict == "fails"
    assert confidence == pytest.approx(
        signals.mass_on(dist, signals.SCORE_LABELS, "fails")
    )


# ── panel_majority: no position bias on a hung compare panel ─────────────


@pytest.mark.parametrize(
    "votes",
    [
        ["a", "b"],
        ["b", "a"],
        ["a", "a", "b", "b"],
        ["b", "b", "a", "a"],
        ["b", "b", "a", "a", "tie"],
    ],
)
def test_a_hung_compare_panel_reports_tie_never_the_first_output(votes):
    """THE position-bias regression. COMPARE_TIEBREAK = ('tie','a','b'); with
    tied = {'a','b'} the old code skipped 'tie' (it drew no votes) and returned
    'a' — awarding every hung panel to whichever output was passed FIRST. The
    whole both-orders machinery exists to cancel exactly that bias."""
    majority, unanimous = signals.panel_majority(
        votes, signals.COMPARE_TIEBREAK, undecided=signals.COMPARE_UNDECIDED
    )
    assert majority == "tie"
    assert unanimous is False


def test_a_hung_compare_panel_is_symmetric_under_swapping_the_votes():
    """The verdict must not depend on the order the votes were collected."""
    forward = signals.panel_majority(
        ["a", "b"], signals.COMPARE_TIEBREAK, undecided=signals.COMPARE_UNDECIDED
    )
    backward = signals.panel_majority(
        ["b", "a"], signals.COMPARE_TIEBREAK, undecided=signals.COMPARE_UNDECIDED
    )
    assert forward == backward


def test_a_decided_compare_panel_still_names_a_winner():
    """The undecided rule must only fire on a genuine tie."""
    majority, unanimous = signals.panel_majority(
        ["a", "a", "b"], signals.COMPARE_TIEBREAK, undecided=signals.COMPARE_UNDECIDED
    )
    assert majority == "a"
    assert unanimous is False


def test_a_hung_score_panel_still_uses_the_cautious_tiebreak():
    """Score mode passes no `undecided`, so a hung panel lands on the more
    cautious label rather than the more flattering one."""
    majority, _ = signals.panel_majority(["meets", "fails"], signals.SCORE_TIEBREAK)
    assert majority == "fails"


def test_panel_majority_rejects_an_empty_vote_list():
    with pytest.raises(ValueError, match="at least one verdict"):
        signals.panel_majority([], signals.SCORE_TIEBREAK)


# ── mass_on / panel_confidence coherence ────────────────────────────────


def test_mass_on_returns_the_mass_of_the_named_label():
    assert signals.mass_on([0.45, 0.10, 0.45], signals.COMPARE_LABELS, "tie") == 0.10


def test_mass_on_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="not one of"):
        signals.mass_on([0.5, 0.3, 0.2], signals.SCORE_LABELS, "excellent")


def test_panel_confidence_rejects_an_empty_panel():
    with pytest.raises(ValueError, match="at least one judge"):
        signals.panel_confidence([])


def test_panel_doubt_rejects_an_empty_panel():
    with pytest.raises(ValueError, match="at least one judge"):
        signals.panel_doubt([])


# ── normalize_three: NaN/inf must fail LOUD, not read as certainty ───────


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_normalize_three_rejects_non_finite_values(bad):
    """NaN defeats every ordering guard (`nan < 0` and `nan <= 0` are both
    False), so it used to sail through. Downstream, a NaN distribution made
    normalized_entropy return 0.0 — reading as MAXIMALLY CERTAIN — and
    compare_verdict return a silent 'tie'. Confidently meaningless is the one
    thing a judge must never be."""
    with pytest.raises(ValueError, match="finite"):
        signals.normalize_three(bad, 50, 50)


# ── expectation: the continuous ranking signal ───────────────────────────


def test_expectation_score_weights_meets_partial_fails():
    """meets=1, partial=0.5, fails=0 — a scalar in [0, 1], higher = better.
    A one-hot 'meets' is 1.0; one-hot 'fails' is 0.0; one-hot 'partial' is
    0.5; a flat split lands at the midpoint."""
    W = signals.SCORE_EXPECTATION_WEIGHTS
    assert signals.expectation([1.0, 0.0, 0.0], W) == pytest.approx(1.0)
    assert signals.expectation([0.0, 0.0, 1.0], W) == pytest.approx(0.0)
    assert signals.expectation([0.0, 1.0, 0.0], W) == pytest.approx(0.5)
    # 0.6 meets + 0.1 partial + 0.3 fails = 0.6 + 0.05 = 0.65
    assert signals.expectation([0.6, 0.1, 0.3], W) == pytest.approx(0.65)


def test_expectation_compare_weights_a_tie_b():
    """a=1, tie=0.5, b=0 — E[preference for A]. Symmetric pairs mirror
    around 0.5, an even a/b split is exactly 0.5."""
    W = signals.COMPARE_EXPECTATION_WEIGHTS
    assert signals.expectation([1.0, 0.0, 0.0], W) == pytest.approx(1.0)  # all A
    assert signals.expectation([0.0, 0.0, 1.0], W) == pytest.approx(0.0)  # all B
    assert signals.expectation([0.5, 0.0, 0.5], W) == pytest.approx(0.5)  # even split
    assert signals.expectation([0.0, 1.0, 0.0], W) == pytest.approx(0.5)  # all tie
    # A leaning pair mirrors its swap around 0.5.
    lean_a = signals.expectation([0.7, 0.1, 0.2], W)
    lean_b = signals.expectation([0.2, 0.1, 0.7], W)
    assert lean_a == pytest.approx(1.0 - lean_b)


def test_expectation_stays_in_unit_interval_for_any_distribution():
    for dist in ([0.34, 0.33, 0.33], [0.9, 0.05, 0.05], [0.1, 0.1, 0.8]):
        for W in (signals.SCORE_EXPECTATION_WEIGHTS, signals.COMPARE_EXPECTATION_WEIGHTS):
            v = signals.expectation(dist, W)
            assert 0.0 <= v <= 1.0


def test_expectation_rejects_length_mismatch():
    """A dist/weights length mismatch would silently score against the wrong
    labels — fail loud instead."""
    with pytest.raises(ValueError, match="must match"):
        signals.expectation([0.5, 0.5], signals.SCORE_EXPECTATION_WEIGHTS)
