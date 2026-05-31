"""Typed model for testable predictions stored on Claim.

Phase 6 of the Move-3 plan (deferred). Replaces the previous
``Claim.predictions: list[dict[str, Any]]`` with a typed list of
``Prediction`` so consumers (gates, render, audit) access fields by
attribute rather than via ``dict.get(...)``.

The shape mirrors what ``GeneratePredictionOperation`` builds from
the four-stage prediction pipeline (identify_testable_aspect →
specify_prediction → define_falsification → classify_prediction).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """A single testable prediction derived from a robust Claim.

    Predictions are produced by ``GeneratePredictionOperation`` for
    Claims at the ROBUST stage (Lakatos: progressive research
    programmes make novel predictions). Each prediction has a
    success criterion (what would confirm it), a failure criterion
    (what would falsify it), and a time horizon for evaluation.
    """

    statement: str = Field(description="The prediction itself, in declarative form.")
    type: str = Field(
        description=(
            "Prediction type from epistemic_classify_prediction "
            "(e.g. 'novel', 'confirmatory', 'risky')."
        )
    )
    specificity: float = Field(
        description=(
            "How specific the prediction is, scored 0.0-1.0 by "
            "epistemic_classify_prediction."
        )
    )
    success_criteria: str = Field(
        description="What observation would count as the prediction holding."
    )
    failure_criteria: str = Field(
        description=(
            "What observation would falsify the prediction. The gate at "
            "gates.py checks this is non-empty when "
            "``requires_falsification_criteria`` is set on the routing profile."
        )
    )
    time_horizon: str = Field(
        description=(
            "When the prediction would be evaluable "
            "(e.g. '6 months', '1 year', 'within next major release')."
        )
    )
    conditions: str = Field(
        default="",
        description="Conditions under which the prediction applies.",
    )
    measurability: str = Field(
        default="",
        description=(
            "How the success / failure observation could be measured "
            "(method, instrument, dataset)."
        ),
    )
    observation_type: str = Field(
        default="",
        description=(
            "Type of observation that would test the prediction "
            "(e.g. 'experimental', 'observational', 'computational')."
        ),
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        """Round-trip helper: build from a legacy dict (e.g. metadata)."""
        return cls.model_validate(data)
