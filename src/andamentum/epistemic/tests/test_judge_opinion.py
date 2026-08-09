"""``JudgeOpinion`` — one record, one doubt scale, nothing stored twice.

Both second-opinion sites (``direction_opinion``, ``scrutiny_opinion``) write
this object. It is the place the doubt-scale rule is enforced by
construction, so its unit tests are about arithmetic identity and about what
it refuses to accept — not about either driver.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic import judgment_signal, thresholds
from andamentum.epistemic.entities.judge_opinion import OPINION_LABELS, JudgeOpinion
from andamentum.llm_judge import signals as judge_signals

_TRIPLES: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1 / 3, 1 / 3, 1 / 3],
    [0.95, 0.03, 0.02],
    [0.45, 0.45, 0.10],
    [0.6, 0.2, 0.2],
]


def _opinion(dist: list[float], **kw) -> JudgeOpinion:
    return JudgeOpinion(
        subject="does the evidence establish the claim?",
        reference=kw.pop("reference", "moderate"),
        distribution=dist,
        overall=kw.pop("overall", "meets"),
        expected_score=kw.pop("expected_score", 0.5),
        summary="stub",
        **kw,
    )


@pytest.mark.parametrize("dist", _TRIPLES)
def test_doubt_is_the_shared_entropy_primitive(dist: list[float]) -> None:
    """PINS the doubt-scale rule at the record itself.

    A future edit that stores llm_judge's number instead of deriving it, or
    that swaps in a different formula ("1 - mass on meets" is the tempting
    one), fails here — and it fails against BOTH implementations at once.
    """
    op = _opinion(dist)
    assert op.doubt == pytest.approx(judgment_signal.distribution_entropy(dist))
    assert op.doubt == pytest.approx(judge_signals.normalized_entropy(dist))


@pytest.mark.parametrize("dist", _TRIPLES)
def test_confidence_is_the_shared_max_mass_primitive(dist: list[float]) -> None:
    op = _opinion(dist)
    assert op.confidence == pytest.approx(judgment_signal.distribution_confidence(dist))
    assert op.confidence == pytest.approx(judge_signals.max_mass(dist))


def test_needs_review_uses_the_one_shared_cutoff_and_no_private_one() -> None:
    """PINS: epistemic must not grow a second doubt threshold behind this
    object. The cutoff is the one already pinned equal to llm_judge's."""
    cut = thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD
    assert cut == judge_signals.NEEDS_REVIEW_DOUBT_THRESHOLD

    uniform = _opinion([1 / 3, 1 / 3, 1 / 3])
    assert uniform.doubt >= cut
    assert uniform.needs_review is True

    one_hot = _opinion([1.0, 0.0, 0.0])
    assert one_hot.doubt < cut
    assert one_hot.needs_review is False


def test_a_confident_dissent_has_near_zero_doubt() -> None:
    """THE REASON ``dissents`` IS A SEPARATE FIELD, made concrete.

    A judge that flatly contradicts the primary reading is maximally sure,
    so its ENTROPY is ~0. Any design that routed disagreement through the
    doubt number would report "no doubt" about exactly the item most likely
    to be wrong.

    PINS: folding dissent into ``doubt`` fails here — the two numbers on
    this object genuinely measure different things.
    """
    op = _opinion([0.0, 0.02, 0.98], overall="fails", dissents=True)
    assert op.doubt < 0.2
    assert op.needs_review is False
    assert op.dissents is True


def test_derived_signals_are_not_stored_fields() -> None:
    """P3, pinned. Storing them would let a persisted record disagree with
    its own distribution after a round-trip."""
    dumped = _opinion([0.5, 0.3, 0.2]).model_dump()
    for derived in ("doubt", "confidence", "needs_review"):
        assert derived not in dumped


def test_a_populated_opinion_round_trips_through_its_own_dump() -> None:
    original = _opinion([0.7, 0.2, 0.1], reference="supports", dissents=False)
    assert JudgeOpinion(**original.model_dump()) == original


@pytest.mark.parametrize(
    "bad",
    [
        [0.5, 0.5],
        [0.3, 0.3, 0.2, 0.2],
        [float("nan"), 0.5, 0.5],
        [float("inf"), 0.0, 0.0],
        [-0.1, 0.6, 0.5],
        [0.0, 0.0, 0.0],
    ],
)
def test_a_malformed_distribution_is_unconstructable(bad: list[float]) -> None:
    """The same commitment ``normalize_distribution`` makes: a judge is
    allowed to fail; it is not allowed to be confidently meaningless.

    Without the finiteness guard a NaN vector scores entropy 0.0 — reading
    downstream as MAXIMALLY CERTAIN.
    """
    with pytest.raises(ValueError):
        _opinion(bad)


def test_the_label_ordering_is_named_and_matches_llm_judge() -> None:
    """PINS: a stored distribution is only readable if index 0 is known to
    mean 'meets'. If llm_judge reorders SCORE_LABELS, every persisted
    opinion silently changes meaning — so the two are asserted equal."""
    assert OPINION_LABELS == judge_signals.SCORE_LABELS
