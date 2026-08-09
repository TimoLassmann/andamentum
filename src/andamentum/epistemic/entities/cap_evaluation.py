"""Typed record of the evidence reading taken when the cycle cap fired.

When the Scrutinise ↔ Resolve cycle cap fires for a claim, the system
stops spending LLM work on that claim. It does *not* stop knowing what
the evidence it already gathered said. ``ConcludeAtCycleCapOperation``
reads that evidence deterministically — the same
``scrutiny_weight.compute_evidence_weight`` engine scrutiny itself uses —
and records the reading here.

This is a **snapshot taken at cap-firing time**, exactly like
``Claim.persistent_concerns``: P3 data describing what was observed, not
a live field and not a scheduling signal. Nothing downstream reads it to
decide what work to do next; it exists so that a capped claim's terminal
outcome is auditable.

Its second job is to make "insufficient" honest. A capped claim whose
``cap_evaluation.verdict`` is ``None`` is the positive record that the
system *did* look at the evidence and the evidence *did not decide* —
as opposed to the pre-2026-08 behaviour, where "insufficient" could be a
pure side effect of the claim failing to climb the stage ladder.

Architecture: Layer 1 (framework-agnostic).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CapEvaluation(BaseModel):
    """What the claim's eligible evidence said at the moment the cap fired.

    The ``model_validator`` below makes an inconsistent record
    unconstructable, so no downstream reader has to defend against one:

    * ``verdict`` and ``confidence`` are both-or-neither — a direction
      without a magnitude (or the reverse) is not a reading;
    * a ``verdict`` requires ``weight_label`` in {strong, moderate} —
      that is exactly the set for which
      ``ScrutiniseClaimOperation._verdict_from_weight`` returns "pass",
      i.e. the system's existing standard for "this claim has evidential
      weight". ``conflicting`` (genuine disagreement) and ``weak`` (too
      thin) must never yield a direction;
    * ``wrote_live_verdict`` implies ``verdict is not None`` — the cap
      cannot have written a live verdict it never derived.
    """

    weight_label: Literal["strong", "moderate", "weak", "conflicting"] = Field(
        description=(
            "The deterministic evidence-weight label computed over the "
            "claim's eligible representative evidence at cap-firing time."
        )
    )
    supporting_mass: float = Field(
        description="Quality- and corroboration-weighted supporting mass."
    )
    contradicting_mass: float = Field(
        description="Quality- and corroboration-weighted contradicting mass."
    )
    n_evidence: int = Field(
        ge=0,
        description="How many eligible representative evidence items were read.",
    )
    verdict: Optional[Literal["supports", "contradicts"]] = Field(
        default=None,
        description=(
            "The direction the evidence reading supported, or None when the "
            "reading was not directional (conflicting or too thin). None is "
            "a positive record that the evidence did not decide."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised directional margin behind ``verdict``, ceilinged at "
            "``thresholds.MECHANICAL_VERDICT_MAX_CONFIDENCE``. The output-layer "
            "``CYCLE_CAP_CONFIDENCE_PENALTY`` is NOT pre-applied here — it is "
            "applied exactly once, downstream, in the posterior paths."
        ),
    )
    wrote_live_verdict: bool = Field(
        default=False,
        description=(
            "True when THIS cap firing wrote the claim's live "
            "``integrated_assessment`` / ``integrated_confidence`` / "
            "``integrated_reasoning``. It is the explicit ownership record "
            "``ConcludeAtCycleCapOperation`` needs in order to revise or "
            "clear only the verdict it authored: a cap firing must never "
            "clear a verdict certified by the IBE chain or by "
            "``PromoteAsRefutedOperation``. False means the reading above "
            "was recorded but some other writer owns (or nobody holds) the "
            "live verdict."
        ),
    )

    @model_validator(mode="after")
    def _verdict_and_confidence_are_coherent(self) -> "CapEvaluation":
        if (self.verdict is None) != (self.confidence is None):
            raise ValueError(
                "CapEvaluation.verdict and .confidence are both-or-neither: "
                f"got verdict={self.verdict!r}, confidence={self.confidence!r}"
            )
        if self.verdict is not None and self.weight_label not in (
            "strong",
            "moderate",
        ):
            raise ValueError(
                "CapEvaluation may only carry a verdict when weight_label is "
                f"'strong' or 'moderate'; got weight_label={self.weight_label!r} "
                f"with verdict={self.verdict!r}"
            )
        if self.wrote_live_verdict and self.verdict is None:
            raise ValueError(
                "CapEvaluation.wrote_live_verdict=True requires a verdict: "
                "the cap cannot have written a live verdict it never derived."
            )
        return self
