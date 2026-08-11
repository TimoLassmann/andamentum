"""One independent judge's read on one proposition — the shared record.

Two places in the pipeline now ask ``llm_judge`` for a SECOND OPINION on a
judgment the system has already made:

* ``direction_opinion`` — per evidence item, during scrutiny: *does this
  source text really support (/contradict) the claim, as the primary judge
  said it did?* (``epistemic.direction_opinion``, stored on
  ``Evidence.direction_opinion``);
* ``scrutiny_opinion`` — per claim, when the deterministic evidence weight
  sits close to one of its cutoffs: *does this body of evidence establish
  the claim as stated?* (``epistemic.scrutiny_opinion``, stored on
  ``Claim.scrutiny_opinion``).

Both are the same object at two granularities: one criterion, one FAST
elicitation, a verbalized distribution over ``[meets, partial, fails]``, and
a comparison against the reading the opinion was taken about. So they share
ONE record type rather than two near-copies that would drift.

THE DOUBT-SCALE RULE, enforced here. ``doubt`` is normalized Shannon entropy
of the distribution, via ``judgment_signal.distribution_entropy`` — the same
formula, on the same scale, as every other ``*_doubt`` in the package.
DISAGREEMENT IS NOT DOUBT and is never folded into that number: a judge that
confidently contradicts the primary reading has entropy near ZERO, so an
entropy-shaped field would report "no doubt" about precisely the item most
likely to be wrong. Disagreement is therefore its own named boolean,
``dissents``, and Gate 0 reads the two independently.

Because each site uses EXACTLY ONE ``Criterion``, ``llm_judge``'s roll-up
distribution *is* that criterion's normalized triple, so
``distribution_entropy(distribution)`` is provably the same number as
``ScoreResult.doubt``. The one-doubt-scale rule therefore holds BY
CONSTRUCTION, without this module importing ``llm_judge`` — which it must
not, since it is embedded on ``Evidence``, ``Claim`` and ``EpistemicVerdict``,
all documented leaves (see ``judgment_signal``'s module docstring for the
leaf-import hazard, and ``tests/test_doubt_scale_single_source.py`` for the
scan that pins it).

``dissents`` is STORED rather than derived, deliberately. It is a
CROSS-OBJECT comparison against ``reference`` — the primary
``support_judgment`` at the evidence site, the deterministic weight label at
the claim site — computable only at the write site, where both halves are in
hand. ``doubt`` / ``confidence`` / ``needs_review`` stay derived properties
(P3: entity fields are data; a scheduling flag never lands on the entity).

Architecture: Layer 1 (framework-agnostic, pure data + pure functions).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..judgment_signal import distribution_confidence, distribution_entropy
from ..thresholds import JUDGMENT_DOUBT_ESCALATION_THRESHOLD

#: Canonical ordering of the belief vector, matching
#: ``llm_judge.signals.SCORE_LABELS``. Named here so a reader of a stored
#: opinion never has to guess which index is which.
OPINION_LABELS: tuple[str, str, str] = ("meets", "partial", "fails")


class JudgeOpinion(BaseModel):
    """One judge's verbalized read on one proposition, plus its dissent.

    Never a verdict. Nothing in the pipeline changes a label, a mass, a
    posterior or a stage because of this object: it is a MEASUREMENT that
    can, at most, buy the objective one more bounded round of inquiry via
    the existing lazy-escalation ``Demand`` (P7).
    """

    subject: str = Field(
        description=(
            "The proposition that was put to the judge, in human-readable "
            "form. Rendered for audit; never parsed."
        )
    )
    reference: str = Field(
        description=(
            "The existing reading this opinion was taken against — the "
            "primary support_judgment ('supports' / 'contradicts') at the "
            "evidence site, or the deterministic weight label ('strong' / "
            "'moderate' / 'weak' / 'conflicting') at the claim site. Without "
            "it, ``dissents`` is un-interpretable: dissent FROM WHAT."
        )
    )
    distribution: list[float] = Field(
        description=(
            "Normalized belief vector over OPINION_LABELS = "
            "[meets, partial, fails]."
        )
    )
    overall: Literal["meets", "partial", "fails"] = Field(
        description="The judge's label — the categorical to read, never a verdict."
    )
    expected_score: float = Field(
        description=(
            "llm_judge's continuous companion to ``overall``. RANKING ONLY: "
            "no decision in this package is routed on it (the repo's "
            "'label for decisions, scalar for ranking' rule)."
        )
    )
    summary: str = Field(
        default="", description="The judge's one-paragraph reasoning, for audit."
    )
    dissents: bool = Field(
        default=False,
        description=(
            "True when this opinion CONTRADICTS ``reference``. Stored, not "
            "derived: it is a cross-object comparison the write site "
            "performs while both halves are in hand — see the module "
            "docstring. A 'partial' opinion never dissents."
        ),
    )

    @field_validator("distribution")
    @classmethod
    def _validate_distribution(cls, v: list[float]) -> list[float]:
        """Reject a malformed vector at construction rather than letting it
        become a silently-meaningless doubt downstream.

        The same commitment ``judgment_signal.normalize_distribution`` makes:
        a judge is allowed to fail; it is not allowed to be confidently
        meaningless.
        """
        if len(v) != 3:
            raise ValueError(
                f"distribution must have exactly 3 entries "
                f"(OPINION_LABELS = {OPINION_LABELS}), got {len(v)}"
            )
        from math import isfinite

        if not all(isfinite(x) for x in v):
            raise ValueError(f"distribution entries must be finite, got {v}")
        if any(x < 0 for x in v):
            raise ValueError(f"distribution entries must be non-negative, got {v}")
        if sum(v) <= 0:
            raise ValueError("distribution must sum to a positive value")
        return v

    @property
    def doubt(self) -> float:
        """Normalised Shannon entropy of the belief vector, in [0, 1].

        THE one doubt scale — identical to ``Evidence.judgment_entropy``,
        ``Evidence.quality_doubt`` and ``Claim.scrutiny_doubt``, and (because
        there is exactly one criterion) numerically identical to
        ``llm_judge``'s own ``ScoreResult.doubt`` for the same triple.
        """
        return distribution_entropy(self.distribution)

    @property
    def confidence(self) -> float:
        """Belief mass on the judge's top label, in [0, 1]."""
        return distribution_confidence(self.distribution)

    @property
    def needs_review(self) -> bool:
        """The judge was unsure enough to warrant a second look.

        Uses ``JUDGMENT_DOUBT_ESCALATION_THRESHOLD``, which is pinned equal
        to ``llm_judge.signals.NEEDS_REVIEW_DOUBT_THRESHOLD`` — this module
        introduces no cutoff of its own.
        """
        return self.doubt >= JUDGMENT_DOUBT_ESCALATION_THRESHOLD


__all__ = ["OPINION_LABELS", "JudgeOpinion"]
