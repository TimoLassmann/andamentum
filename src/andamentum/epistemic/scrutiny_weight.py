"""Deterministic evidence-weight for scrutiny (replaces an LLM categorical).

Scrutiny asks: does a claim have enough evidential weight to proceed, and does
its evidence genuinely conflict? That question used to be answered by a free LLM
call (``epistemic_assess_evidence`` → strong / moderate / weak / conflicting).
Every input it needs already exists as structured data, so it is computed here:

- each eligible representative's **verbalized judgment distribution** (Tier 0)
  gives the soft supporting / contradicting split (``support_contradict_split``);
- **corroboration count** amplifies mass (``1 + log(cluster_size)`` — the same
  Reichenbach common-cause weighting the counting posterior uses);
- **quality score** scales mass (bibliometric when available, a neutral prior
  otherwise — absence of a citation score is not evidence of low quality).

The verdict is then a function of the resulting supporting / contradicting mass:

- **conflicting** — substantial mass on *both* directions (genuine disagreement);
- **weak** — too little dominant-direction mass to establish weight at all;
- **strong** / **moderate** — dominant-direction mass above / below the strong cut.

This deliberately does NOT judge *methodological* soundness (small n, wrong
population); that stays with the content-reading ``epistemic_identify_single_issue``
agent, whose blocking issues already downgrade the verdict in the scrutiny
operation. This module only weighs the directional evidence mass the judges
produced.

The classification cutoffs are named constants in ``thresholds`` and are
benchmark-pending (the *shape* — quality- and corroboration-weighted directional
mass — is the commitment; the exact cutoffs are tunable).

Architecture: Layer 1 (framework-agnostic, pure functions).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable, Protocol

from .judgment_signal import support_contradict_split
from .thresholds import (
    SCRUTINY_CONFLICT_MINORITY_FRACTION,
    SCRUTINY_DEFAULT_QUALITY,
    SCRUTINY_STRONG_MASS_THRESHOLD,
    SCRUTINY_WEAK_MASS_THRESHOLD,
)

# The four-way vocabulary preserved from the retired LLM output, so the
# downstream scrutiny combination (blocking-issue downgrade + verdict mapping)
# is unchanged.
EvidenceWeightLabel = str  # "strong" | "moderate" | "weak" | "conflicting"


class _EvidenceLike(Protocol):
    """The minimal evidence surface this module reads (duck-typed so tests can
    pass stubs and the real ``Evidence`` entity works unchanged)."""

    judgment_distribution: list[float] | None
    support_judgment: str | None
    quality_score: float | None
    corroboration_count: int


@dataclass(frozen=True)
class EvidenceWeight:
    """The computed weight plus the masses behind it (for logging / traceability)."""

    label: EvidenceWeightLabel
    supporting_mass: float
    contradicting_mass: float

    def justification(self) -> str:
        return (
            f"deterministic evidence weight={self.label} "
            f"(supporting_mass={self.supporting_mass:.2f}, "
            f"contradicting_mass={self.contradicting_mass:.2f})"
        )


def _item_masses(e: _EvidenceLike) -> tuple[float, float]:
    """Quality- and corroboration-weighted (supporting, contradicting) mass of
    one evidence item."""
    cluster_size = max(1, getattr(e, "corroboration_count", 1) or 1)
    corroboration_weight = 1.0 + log(cluster_size)

    quality = getattr(e, "quality_score", None)
    quality_eff = SCRUTINY_DEFAULT_QUALITY if quality is None else float(quality)

    p_supports, p_contradicts = support_contradict_split(
        getattr(e, "judgment_distribution", None),
        getattr(e, "support_judgment", None),
    )
    factor = corroboration_weight * quality_eff
    return factor * p_supports, factor * p_contradicts


def compute_evidence_weight(
    evidence_items: Iterable[_EvidenceLike],
) -> EvidenceWeight:
    """Deterministic strong / moderate / weak / conflicting weight for a claim.

    ``evidence_items`` should be the same eligible representative set scrutiny
    reasons over (extracted, not invalidated, not corroborative/deferred). The
    result's ``label`` feeds the existing deterministic combination in
    ``ScrutiniseClaimOperation`` (blocking-issue downgrade, then verdict map).
    """
    supporting = 0.0
    contradicting = 0.0
    for e in evidence_items:
        s, c = _item_masses(e)
        supporting += s
        contradicting += c

    total_directional = supporting + contradicting
    dominant = max(supporting, contradicting)
    minority = min(supporting, contradicting)

    # 1. Genuine disagreement: substantial mass on BOTH directions. Checked
    #    first because a claim can have large total mass yet no usable verdict
    #    if the evidence points both ways. The minority must be both a
    #    meaningful FRACTION of the directional mass and above an absolute
    #    floor (reuse the weak threshold) so a tiny wobble never reads as
    #    conflict.
    if (
        minority >= SCRUTINY_WEAK_MASS_THRESHOLD
        and total_directional > 0.0
        and (minority / total_directional) >= SCRUTINY_CONFLICT_MINORITY_FRACTION
    ):
        label: EvidenceWeightLabel = "conflicting"
    # 2. Too thin to establish weight in either direction.
    elif dominant < SCRUTINY_WEAK_MASS_THRESHOLD:
        label = "weak"
    # 3. Dominant direction with enough mass — strong above the cut, else moderate.
    elif dominant >= SCRUTINY_STRONG_MASS_THRESHOLD:
        label = "strong"
    else:
        label = "moderate"

    return EvidenceWeight(
        label=label,
        supporting_mass=supporting,
        contradicting_mass=contradicting,
    )


__all__ = [
    "EvidenceWeight",
    "EvidenceWeightLabel",
    "compute_evidence_weight",
]
