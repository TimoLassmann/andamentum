"""Typed model for the decomposition stored on Objective.

Phase 6 of the Move-3 plan. Replaces the previous
``Objective.decomposition: dict[str, Any]`` with a proper Pydantic
model so consumers can read fields by name rather than via
``dict.get(...)``. The previous bug shape — Bug C in the post-audit-2
fix queue — was exactly the divergent-lookup-on-untyped-dict failure
mode this typing prevents.

The model lives in entities/ rather than agents/output_models.py
because the decomposition is data on the entity, not just an LLM
output schema. ``agents/output_models.py``'s ``QuestionDecomposition``
is now a thin alias for the agent-output role; both refer to the same
underlying shape defined here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SubInvestigation(BaseModel):
    """One sub-investigation in a top-down question decomposition.

    Each sub-investigation is the form of a checkable claim — testable
    for verificatory questions, characterizable for exploratory questions,
    or condition-like for predictive questions. The pipeline treats each
    one the same way (seed_claim flow), so the uniform schema is correct
    even when the question-type semantics differ.
    """

    id: str = Field(
        description=("Stable identifier for this sub-investigation: 'A', 'B', 'C', ...")
    )
    seed_claim: str = Field(
        description=(
            "The sub-investigation expressed as a testable / characterizable "
            "claim. For verificatory questions, this is a falsifiable sub-claim "
            "whose truth would partially settle the original question. For "
            "exploratory questions, this is a facet-claim. For predictive "
            "questions, this is a condition-claim. The pipeline runs the same "
            "machinery regardless."
        )
    )
    rationale: str = Field(
        description=(
            "One sentence: why this sub-investigation is load-bearing for "
            "the original question."
        )
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Relative importance of this sub-investigation to the question's "
            "answer, on a 0-10 scale. Used by the WEIGHTED_AND combination "
            "rule (weighted mean of child posteriors). For AND / OR / UNION "
            "the weight is ignored."
        ),
    )


class Decomposition(BaseModel):
    """Top-down decomposition of a research question into
    sub-investigations.

    Stored on ``Objective.decomposition`` after
    ``DecomposeQuestionOperation`` runs. It carries the question's
    structure only — the aggregate outcome lives on
    ``EpistemicGraphState.verdict`` / ``Snapshot.verdict`` as a typed
    ``EpistemicVerdict``, built fresh from the live DB on every pass
    through ``CombineClaimVerdicts``. (The previous
    ``combined_verdict: CombinedVerdictData`` field made the aggregate
    reachable only for decomposed runs, so verify-mode and
    open-research runs produced no structured verdict at all.)

    A good decomposition has 2-5 sub-investigations that are:
    - Load-bearing (each one's outcome materially affects the answer)
    - Roughly orthogonal (investigating one doesn't trivialize another)
    - Cover the question's scope
    """

    sub_investigations: list[SubInvestigation] = Field(
        default_factory=list,
        description=(
            "2-5 sub-investigations under normal use; an empty list is "
            "permitted to support the degenerate-decomposition fallback "
            "in CreateClaims (graph node)."
        ),
        max_length=5,
    )
    combination_rule: Optional[Literal["AND", "OR", "WEIGHTED_AND", "UNION"]] = Field(
        default=None,
        description=(
            "How sub-investigation outcomes combine into the question's "
            "answer. AND: all must support. OR: any one supports. "
            "WEIGHTED_AND: each contributes by importance. UNION: each "
            "contributes a piece of the answer (typical for exploratory "
            "questions where there's no single verdict). "
            "None when the decomposition was constructed without a rule "
            "(e.g. partial state or open-research handoff). "
            "``resolve_combination_rule`` (graph/combination.py) treats "
            "None as a real missing-rule signal."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "1-2 sentences explaining why this decomposition captures the "
            "question's load-bearing structure."
        ),
    )
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Decomposition | None":
        """Round-trip helper: build from a dict (e.g. legacy metadata).

        Returns ``None`` for ``None`` input to make consumer code
        ergonomic on optional fields.
        """
        if data is None:
            return None
        return cls.model_validate(data)
