"""Verbalized-confidence signal for evidence-claim judgments (Tier 0).

The ``epistemic_judge_evidence`` agent returns a 3-way *belief distribution*
over {supports, contradicts, no_bearing} rather than a bare categorical. This
module holds the canonical class ordering and the pure functions that turn that
distribution into the derived signals the system stores:

- the **verdict** (argmax — the load-bearing categorical, unchanged in meaning),
- the **confidence** (probability mass on the chosen verdict),
- the **entropy** (normalised Shannon entropy in [0, 1] — the validated
  "is this judgment wrong?" detector; higher = less sure),
- the **one-hot** meta-flag (the distribution is degenerate, so its entropy is
  uninformative — see ``thresholds.JUDGMENT_ONE_HOT_THRESHOLD``).

Why a verbalized distribution at all: an offline study (``experiments/
dirichlet_confidence``) found that the entropy of a single verbalized histogram
is as good an abstention signal as far more expensive multi-call methods, and a
local-model validation (gemma4:12b, gpt-oss:20b) confirmed it survives on the
models this system runs — provided the elicitation forces reasoning before the
numbers. See ``docs/tier0_design.md`` (worktree) for the full rationale.

This is a leaf module: pure stdlib, no dependencies on other epistemic
internals beyond the threshold constant. Both the agent output schema
(``agents/output_models.py``) and the ``Evidence`` entity import from here so
the maths lives in exactly one place.

Architecture: Layer 1 (framework-agnostic, pure functions).
"""

from __future__ import annotations

from math import log

from .thresholds import JUDGMENT_ONE_HOT_THRESHOLD

# Canonical ordering for the 3-way belief distribution. Every distribution list
# in the system (``Evidence.judgment_distribution``, the agent output) is
# ordered by this tuple, so index i always means JUDGMENT_CLASSES[i].
JUDGMENT_CLASSES: tuple[str, str, str] = ("supports", "contradicts", "no_bearing")


def normalize_distribution(
    supports: float, contradicts: float, no_bearing: float
) -> list[float]:
    """Normalise three non-negative belief points into a probability vector.

    Models are asked for integers summing to 100, but small models drift; we
    normalise by the actual sum rather than assuming exactly 100.

    Raises:
        ValueError: if any value is negative or the total is not positive
            (a degenerate output the caller should reject / retry, not paper
            over with a uniform guess).
    """
    vals = [float(supports), float(contradicts), float(no_bearing)]
    if any(v < 0 for v in vals):
        raise ValueError(f"belief points must be non-negative, got {vals}")
    total = sum(vals)
    if total <= 0:
        raise ValueError("belief points must sum to a positive value, got all zero")
    return [v / total for v in vals]


def argmax_verdict(distribution: list[float]) -> str:
    """The verdict = the highest-mass class. Ties resolve by ``JUDGMENT_CLASSES``
    order (supports > contradicts > no_bearing), so the result is deterministic."""
    best_i = max(range(len(distribution)), key=lambda i: distribution[i])
    return JUDGMENT_CLASSES[best_i]


def distribution_confidence(distribution: list[float]) -> float:
    """Probability mass on the chosen (argmax) class, in [0, 1]."""
    return max(distribution)


def distribution_entropy(distribution: list[float]) -> float:
    """Normalised Shannon entropy in [0, 1]. 0 = one-hot (maximally sure),
    1 = uniform (maximally unsure). The validated wrong-answer detector."""
    n = len(distribution)
    if n <= 1:
        return 0.0
    h = -sum(p * log(p) for p in distribution if p > 0.0)
    return h / log(n)


def distribution_is_one_hot(distribution: list[float]) -> bool:
    """True if the distribution is effectively one-hot (top class ≥
    ``JUDGMENT_ONE_HOT_THRESHOLD``), meaning its entropy is uninformative."""
    return max(distribution) >= JUDGMENT_ONE_HOT_THRESHOLD


def support_contradict_split(
    distribution: list[float] | None, support_judgment: str | None
) -> tuple[float, float]:
    """The (supporting, contradicting) probability mass of one judgment, in [0, 1].

    Prefers the verbalized distribution (Tier 0): ``distribution[0]`` supports,
    ``distribution[1]`` contradicts, ``distribution[2]`` (no_bearing) counts for
    neither. When no distribution was captured — adversarial counter-evidence or
    pre-Tier-0 data — it falls back to the hard one-vote-per-item from
    ``support_judgment``, so the two regimes coincide exactly in the limiting
    case (a one-hot distribution equals its own hard vote).

    This is the *unweighted* primitive. Callers apply their own corroboration
    (Reichenbach ``1 + log(cluster_size)``) and/or quality weighting on top; it
    lives here so the counting posterior (``confidence``) and the scrutiny
    weight (``scrutiny_weight``) share one soft-vote definition.
    """
    if distribution is not None and len(distribution) == 3:
        return distribution[0], distribution[1]
    if support_judgment == "supports":
        return 1.0, 0.0
    if support_judgment == "contradicts":
        return 0.0, 1.0
    return 0.0, 0.0


__all__ = [
    "JUDGMENT_CLASSES",
    "normalize_distribution",
    "argmax_verdict",
    "distribution_confidence",
    "distribution_entropy",
    "distribution_is_one_hot",
    "support_contradict_split",
]
