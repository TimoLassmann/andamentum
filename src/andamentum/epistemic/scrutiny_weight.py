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

The same pass also reports the **doubt** of the judgments that produced the
verdict — the directional-mass-weighted mean of each item's normalized
judgment entropy. The weight therefore answers two questions at once: which
way does the mass point, and how internally unsure were the judgments that
pointed it that way. Doubt never changes the label; it is a suspicion-raiser
routed through the lazy-escalation ``Demand`` loop (see
``thresholds.JUDGMENT_DOUBT_ESCALATION_THRESHOLD``).

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

from .judgment_signal import distribution_entropy, support_contradict_split
from .thresholds import (
    SCRUTINY_BORDERLINE_BAND,
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
    #: Directional-mass-weighted mean normalized entropy of the judgments that
    #: produced the masses above, in [0, 1] — a MEASUREMENT of how internally
    #: unsure those judgments were, not an instruction to any stage.
    #:
    #: ``None``, never 0.0, when no distribution-carrying item contributed
    #: directional mass. That distinction is load-bearing: 0.0 means "measured,
    #: and the judgments were certain", while ``None`` means "not measured".
    #: Coercing the latter to 0.0 would read as certainty the system never
    #: established; coercing it to 1.0 would escalate on every pre-Tier-0 and
    #: adversarial item. Only ``None`` can never escalate.
    doubt: float | None = None

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


def item_directional_mass(e: _EvidenceLike) -> float:
    """How much this one item moved the directional verdict, in mass units.

    The sum of the item's supporting and contradicting mass — exactly the
    weight ``compute_evidence_weight`` already gives it when accumulating
    doubt. Public because the direction-confirmation selector
    (``direction_opinion.select_confirmation_candidates``) ranks candidates
    by it: one definition of "how load-bearing was this item", two
    consumers, rather than a re-implementation that can drift.
    """
    s, c = _item_masses(e)
    return s + c


def is_borderline(weight: EvidenceWeight) -> bool:
    """Would this reading's LABEL flip under a small perturbation of the masses?

    Pure, free, and evaluated BEFORE any model call — it is what makes the
    scrutiny second opinion cost 0 on the common case. The label
    ``compute_evidence_weight`` produces is a step function of three
    cutoffs; a reading sitting within ``SCRUTINY_BORDERLINE_BAND`` (relative)
    of any of them is one whose label is an artefact of where the cutoff
    happens to sit rather than of the evidence, and is therefore worth one
    independent second opinion.

    Deliberately does NOT consider ``weight.doubt``: Gate 0 already escalates
    on high measured doubt, so paying a model call there would be pure
    duplication. This predicate asks a different question — is the
    DETERMINISTIC READING itself a coin flip.

    A reading with no directional mass at all is not borderline: there is no
    body of evidence to put to a judge.
    """
    total_directional = weight.supporting_mass + weight.contradicting_mass
    if total_directional <= 0.0:
        return False

    dominant = max(weight.supporting_mass, weight.contradicting_mass)
    minority = min(weight.supporting_mass, weight.contradicting_mass)

    def _near(value: float, cutoff: float) -> bool:
        return abs(value - cutoff) <= SCRUTINY_BORDERLINE_BAND * cutoff

    return (
        _near(dominant, SCRUTINY_WEAK_MASS_THRESHOLD)
        or _near(dominant, SCRUTINY_STRONG_MASS_THRESHOLD)
        or _near(minority / total_directional, SCRUTINY_CONFLICT_MINORITY_FRACTION)
    )


def compute_evidence_weight(
    evidence_items: Iterable[_EvidenceLike],
) -> EvidenceWeight:
    """Deterministic strong / moderate / weak / conflicting weight for a claim.

    ``evidence_items`` should be the same eligible representative set scrutiny
    reasons over (extracted, not invalidated, not corroborative/deferred). The
    result's ``label`` feeds the existing deterministic combination in
    ``ScrutiniseClaimOperation`` (blocking-issue downgrade, then verdict map).

    ``doubt`` is accumulated in the SAME pass, weighted by each item's own
    directional mass (``s + c``) — that is, by exactly the factor with which
    the item moved the verdict. An item that barely shifted the balance barely
    shifts its doubt. Items with no captured ``judgment_distribution``
    (adversarial counter-evidence, pre-Tier-0 data) contribute mass but NOT
    doubt: absence of a measured confidence is not evidence of low confidence,
    the same exemption ``GATE_MIN_JUDGMENT_CONFIDENCE`` already encodes.
    """
    supporting = 0.0
    contradicting = 0.0
    # Doubt accumulator, weighted by each item's own directional mass.
    doubt_weight = 0.0
    doubt_weighted_sum = 0.0
    for e in evidence_items:
        s, c = _item_masses(e)
        supporting += s
        contradicting += c

        dist = getattr(e, "judgment_distribution", None)
        if dist is None:
            continue
        item_directional = s + c
        if item_directional <= 0.0:
            # A no_bearing judgment moved neither direction, so it has no
            # say in how sure the DIRECTIONAL verdict's judgments were.
            continue
        doubt_weight += item_directional
        doubt_weighted_sum += item_directional * distribution_entropy(dist)

    doubt = doubt_weighted_sum / doubt_weight if doubt_weight > 0.0 else None

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
        doubt=doubt,
    )


__all__ = [
    "EvidenceWeight",
    "EvidenceWeightLabel",
    "compute_evidence_weight",
    "is_borderline",
    "item_directional_mass",
]
