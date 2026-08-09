"""ONE definition of doubt, machine-checked across two implementations.

``andamentum.epistemic.judgment_signal`` and ``andamentum.llm_judge.signals``
hold the same three verbalized-distribution primitives, and
``epistemic.thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD`` is deliberately
equal to ``llm_judge.signals.NEEDS_REVIEW_DOUBT_THRESHOLD``. They are kept in
step by THIS FILE rather than by an import — importing ``llm_judge.signals``
executes the package ``__init__`` → ``panel`` → ``elicit`` → ``pydantic_ai``,
and ``judgment_signal`` is imported by ``entities/evidence.py``, a documented
pure-stdlib leaf.

Convergence-by-contract only works if the contract is enforced. Every test
here fails the moment either implementation drifts.
"""

from __future__ import annotations

import inspect

import pytest

from andamentum.epistemic import judgment_signal, thresholds
from andamentum.llm_judge import signals as judge_signals

# A table that exercises every regime the two implementations could disagree
# in: one-hot, uniform, near-tie, two-of-three, all-mass-on-the-third.
_DISTRIBUTIONS: list[list[float]] = [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [1 / 3, 1 / 3, 1 / 3],
    [0.34, 0.33, 0.33],
    [0.5, 0.5, 0.0],
    [0.4999999999999999, 0.5, 0.0],
    [0.95, 0.03, 0.02],
    [0.6, 0.2, 0.2],
    [0.45, 0.45, 0.10],
]


@pytest.mark.parametrize("dist", _DISTRIBUTIONS)
def test_entropy_matches_llm_judge_over_table(dist: list[float]) -> None:
    """The doubt formula is ONE formula.

    PINS: a future edit to either ``distribution_entropy`` or
    ``normalized_entropy`` — a different log base, a different
    normalisation, a clamp — fails here rather than quietly giving the
    epistemic pipeline and the judge two different doubt scales.
    """
    assert judgment_signal.distribution_entropy(dist) == pytest.approx(
        judge_signals.normalized_entropy(dist)
    )


@pytest.mark.parametrize("dist", _DISTRIBUTIONS)
def test_confidence_matches_llm_judge_over_table(dist: list[float]) -> None:
    """One confidence definition (mass on the top class)."""
    assert judgment_signal.distribution_confidence(dist) == pytest.approx(
        judge_signals.max_mass(dist)
    )


@pytest.mark.parametrize(
    "triple",
    [(100, 0, 0), (60, 50, 30), (1, 1, 1), (34, 33, 33), (0, 0, 7), (2.5, 0.5, 7.0)],
)
def test_normalize_matches_llm_judge_over_valid_triples(triple) -> None:
    """One normalisation. Small models drift off sum-to-100; both modules
    must renormalise by the actual sum in exactly the same way."""
    a, b, c = triple
    assert judgment_signal.normalize_distribution(a, b, c) == pytest.approx(
        judge_signals.normalize_three(a, b, c)
    )


@pytest.mark.parametrize(
    "triple", [(float("nan"), 1, 1), (float("inf"), 1, 1), (1, 1, float("-inf"))]
)
def test_normalize_rejects_non_finite_like_llm_judge(triple) -> None:
    """Convergence on the STRICTER guard.

    Before this change ``normalize_distribution`` had no finiteness check:
    NaN defeats every comparison (``nan < 0`` and ``nan <= 0`` are both
    False), so it slipped through and produced ``[nan, nan, nan]`` — which
    ``distribution_entropy`` scores 0.0, i.e. reads downstream as MAXIMALLY
    CERTAIN. A judge is allowed to fail; it is not allowed to be
    confidently meaningless.

    PINS: reverting to the silent-NaN behaviour fails here.
    """
    a, b, c = triple
    with pytest.raises(ValueError, match="finite"):
        judgment_signal.normalize_distribution(a, b, c)
    with pytest.raises(ValueError, match="finite"):
        judge_signals.normalize_three(a, b, c)


@pytest.mark.parametrize("triple", [(-1, 1, 1), (0, 0, 0)])
def test_normalize_rejects_negative_and_all_zero_like_llm_judge(triple) -> None:
    """The shared degenerate-input contract: reject, never guess uniform."""
    a, b, c = triple
    with pytest.raises(ValueError):
        judgment_signal.normalize_distribution(a, b, c)
    with pytest.raises(ValueError):
        judge_signals.normalize_three(a, b, c)


def test_escalation_threshold_is_the_llm_judge_needs_review_threshold() -> None:
    """One doubt scale, one cutoff.

    PINS: epistemic must not grow a private doubt threshold. Tuning either
    side without the other fails here, which forces the tuner to decide
    whether the two really do measure different things (they do not).
    """
    assert (
        thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD
        == judge_signals.NEEDS_REVIEW_DOUBT_THRESHOLD
    )


def test_leaf_modules_do_not_import_llm_judge() -> None:
    """The no-import-at-the-leaf decision, pinned.

    ``import andamentum.llm_judge.signals`` executes the package
    ``__init__`` → ``panel`` → ``elicit`` → ``andamentum.core`` →
    ``pydantic_ai``. ``judgment_signal`` is imported by
    ``entities/evidence.py``; ``thresholds`` is imported by
    ``judgment_signal``. Both are documented pure-stdlib leaves, and a
    well-meaning "just import it instead of duplicating" refactor would
    drag an agent framework into a pydantic entity.

    PINS: that refactor fails loudly here instead of landing silently.
    """
    for module in (judgment_signal, thresholds):
        source = inspect.getsource(module)
        code_lines = [
            line
            for line in source.splitlines()
            if line.lstrip().startswith(("import ", "from "))
        ]
        offenders = [line for line in code_lines if "llm_judge" in line]
        assert not offenders, (
            f"{module.__name__} imports llm_judge: {offenders}. These two "
            "modules are deliberately kept import-free of it — see this "
            "test's docstring and the module docstring of judgment_signal."
        )


def test_argmax_is_deliberately_not_converged() -> None:
    """The ONE primitive that is deliberately NOT shared.

    ``llm_judge.signals.argmax_label`` detects ties within
    ``TIE_TOLERANCE``; ``judgment_signal.argmax_verdict`` uses exact
    ``max``. On a near-tie they disagree, and epistemic's tie-break is
    load-bearing (it feeds the ``in_scope`` cross-field validator, the
    supporting-source count and the counting posterior).

    PINS: the divergence is intentional and documented. If someone
    converges them "for consistency", this test tells them what they just
    changed — a routing behaviour, not a formatting one.
    """
    near_tie = [0.4999999999999999, 0.5, 0.0]
    # Same class ordering AND the same documented preference order
    # (supports > contradicts > no_bearing) that argmax_verdict's docstring
    # states. They still disagree, because argmax_verdict never SEES the tie:
    # exact `max` hands it to index 1 outright.
    epistemic_verdict = judgment_signal.argmax_verdict(near_tie)
    judge_verdict = judge_signals.argmax_label(
        near_tie,
        judgment_signal.JUDGMENT_CLASSES,
        judgment_signal.JUDGMENT_CLASSES,
    )
    assert epistemic_verdict == "contradicts"
    assert judge_verdict == "supports"
    assert judge_verdict != epistemic_verdict, (
        "The two argmax implementations are expected to disagree on a "
        "near-tie. If they now agree, argmax_verdict was converged onto "
        "argmax_label — re-read judgment_signal.argmax_verdict's docstring "
        "before accepting that."
    )
