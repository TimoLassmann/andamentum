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

ONE DEFINITION OF DOUBT, TWO IMPLEMENTATIONS (deliberate).
``andamentum.llm_judge.signals`` holds the same three primitives —
``normalize_three`` / ``max_mass`` / ``normalized_entropy`` are
character-for-character equivalent to ``normalize_distribution`` /
``distribution_confidence`` / ``distribution_entropy`` here, and
``thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD`` is deliberately equal to
``llm_judge.signals.NEEDS_REVIEW_DOUBT_THRESHOLD``. The two are kept in step by
a DRIFT TEST (``tests/test_doubt_scale_single_source.py``), NOT by an import.
The import is not available: ``import andamentum.llm_judge.signals`` executes
the package ``__init__`` → ``panel`` → ``elicit`` → ``andamentum.core`` →
``pydantic_ai``, and this module is imported by ``entities/evidence.py``, a
documented pure-stdlib leaf. Converging by import would drag an agent framework
into a pydantic entity. If you are here to "just import it", read the drift test
first — it exists to make that refactor fail loudly rather than silently.

Architecture: Layer 1 (framework-agnostic, pure functions).
"""

from __future__ import annotations

from math import isfinite, log

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
        ValueError: if any value is non-finite or negative, or the total is
            not positive (a degenerate output the caller should reject /
            retry, not paper over with a uniform guess).
    """
    vals = [float(supports), float(contradicts), float(no_bearing)]
    # The finiteness check must come FIRST and be its own test. NaN defeats
    # every comparison it appears in (`nan < 0` and `nan <= 0` are both
    # False), so it slips through the ordering guard below and propagates: a
    # NaN distribution makes ``distribution_entropy`` return 0.0 — reading
    # downstream as MAXIMALLY CERTAIN. A judge is allowed to fail; it is not
    # allowed to be confidently meaningless. Same guard, same wording as
    # ``llm_judge.signals.normalize_three`` (see the module docstring's
    # co-definition rule).
    if not all(isfinite(v) for v in vals):
        raise ValueError(f"belief points must be finite, got {vals}")
    if any(v < 0 for v in vals):
        raise ValueError(f"belief points must be non-negative, got {vals}")
    total = sum(vals)
    if total <= 0:
        raise ValueError("belief points must sum to a positive value, got all zero")
    return [v / total for v in vals]


def argmax_verdict(distribution: list[float]) -> str:
    """The verdict = the highest-mass class. Ties resolve by ``JUDGMENT_CLASSES``
    order (supports > contradicts > no_bearing), so the result is deterministic.

    Deliberately NOT converged onto ``llm_judge.signals.argmax_label``, unlike
    the three doubt primitives above. That function detects ties within
    ``isclose(abs_tol=1e-9)``; this one uses exact ``max``. On a near-tie such
    as ``[0.4999999999999999, 0.5, 0.0]`` the two disagree ("contradicts" vs
    the tie-break's answer). The verdict tie-break here is load-bearing — it
    feeds ``EvidenceJudgmentOutput``'s ``in_scope`` cross-field validator,
    ``gates.count_supporting_sources`` and the counting posterior — so changing
    its near-tie behaviour is a routing change, not a cleanup. The doubt scale
    is shared; the categorical tie-break is not.
    """
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
