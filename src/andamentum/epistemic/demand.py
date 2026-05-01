"""Demand — uniform "what's missing?" object passed between layers.

Phase 0 of the lazy-escalation plan
(``docs/superpowers/plans/2026-05-02-lazy-escalation.md``). Each layer
of the epistemic pipeline can produce a ``Demand`` describing whether
its work was satisfied, and if not, what specifically is missing. The
graph routes demands to the layer that can satisfy them minimally.

Three flat fields chosen for small-LLM compatibility:

* ``needs_more``: the satisfaction signal itself (boolean).
* ``justification``: freeform reason — what evidence settled the
  question, or what specifically is still missing.
* ``target_hint``: optional freeform suggestion of where to look
  (provider type, evidence shape, claim aspect). Empty when the
  generator can't suggest a target.

The demand is intentionally NOT layer-specific. One uniform shape
travels everywhere; each consumer interprets the freeform fields in
its own context. This keeps small-LLM agents reliable (they fill
flat schemas best) and observability simple (a demand chain rendered
as text reads naturally).

This module has no dependencies on graph nodes, operations, or
entities — it's a leaf module that everything else can import.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Demand(BaseModel):
    """A demand describing whether a layer is satisfied with its work.

    Produced by satisfaction checks at any layer (synthesis, verification,
    scrutiny, evidence). Consumed by the next layer down to decide what
    minimal additional work would close the gap.

    Two helpers (``Demand.satisfied`` and ``Demand.needs``) are the
    canonical constructors for the two semantic uses; direct
    construction is also fine for cases that need to set all fields
    explicitly.
    """

    needs_more: bool = Field(
        description=(
            "False when the layer is satisfied (the work it did is "
            "sufficient for the demand it was asked to address). "
            "True when more work is needed — in which case "
            "``justification`` names the specific gap."
        )
    )
    justification: str = Field(
        description=(
            "Freeform reason. When ``needs_more`` is False: explains "
            "what evidence or finding settled the question. When True: "
            "names the specific gap that's still missing — concrete "
            "enough that the consuming layer can choose targeted work."
        )
    )
    target_hint: str = Field(
        default="",
        description=(
            "Optional freeform hint about where to look or what shape "
            "of evidence would close the gap. Examples: 'try a "
            "clinical-trial registry', 'mechanistic literature', "
            "'a different population subgroup'. Empty when the "
            "generator can't suggest a target — the consuming layer "
            "then decides on its own."
        ),
    )

    @classmethod
    def satisfied(cls, justification: str = "") -> "Demand":
        """Construct a satisfied demand. The justification is optional
        but recommended — explaining what made the layer satisfied is
        load-bearing for observability (later passes can audit *why*
        the system stopped here).
        """
        return cls(needs_more=False, justification=justification)

    @classmethod
    def needs(cls, justification: str, target_hint: str = "") -> "Demand":
        """Construct an unsatisfied demand naming a specific gap.

        ``justification`` should be concrete enough that the consuming
        layer can pick targeted work — not vague phrases like "more
        evidence" but specific gaps like "no RCT-level mortality
        outcomes are present in the current evidence pool".
        """
        return cls(
            needs_more=True,
            justification=justification,
            target_hint=target_hint,
        )
