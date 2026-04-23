"""Tests proving that agent output models give pydantic-ai proper enum
constraints in their JSON schemas — so strict-structured-outputs mode on
GPT-5 series rejects invalid enum values at the API, not at our brittle
boundary coercion.

Each test asserts two things for an enum-like field:

1. The field's JSON schema node has an ``enum`` key listing the valid values
   (either inline or via a ``$defs/<enum>`` reference that contains ``enum``).
2. Constructing the model with an out-of-vocabulary value raises
   ``pydantic.ValidationError``.

The set of tests grows as Layer-2 conversions land. Together they are a
regression barrier against anyone re-introducing the bare-``str`` pattern.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError


def _field_allowed_values(schema: dict[str, Any], field: str) -> list[str]:
    """Return the set of allowed values for *field* in a pydantic JSON schema.

    Handles both inline ``enum`` and ``$ref`` into ``$defs``. Returns empty
    list when no enum is expressed (i.e., the field is a bare string).
    """
    prop = schema["properties"][field]
    if "enum" in prop:
        return list(prop["enum"])
    ref = prop.get("$ref")
    if ref and ref.startswith("#/$defs/"):
        def_name = ref.split("/", 3)[-1]
        defn = schema.get("$defs", {}).get(def_name, {})
        if "enum" in defn:
            return list(defn["enum"])
    return []


class TestClassifyEvidenceDomainEnumConstraints:
    def test_method_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        values = _field_allowed_values(
            ClassifyEvidenceDomainOutput.model_json_schema(), "method_type"
        )
        assert set(values) == {
            "experimental",
            "observational",
            "computational",
            "theoretical",
        }, values

    def test_data_source_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        values = _field_allowed_values(
            ClassifyEvidenceDomainOutput.model_json_schema(), "data_source"
        )
        assert set(values) == {"primary", "secondary", "synthetic", "meta"}, values

    def test_temporal_approach_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        values = _field_allowed_values(
            ClassifyEvidenceDomainOutput.model_json_schema(), "temporal_approach"
        )
        assert set(values) == {
            "cross_sectional",
            "longitudinal",
            "retrospective",
            "prospective",
        }, values

    def test_causal_role_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        values = _field_allowed_values(
            ClassifyEvidenceDomainOutput.model_json_schema(), "causal_role"
        )
        assert set(values) == {
            "mechanistic",
            "phenomenological",
            "interventional",
            "predictive",
        }, values


class TestClassifyEvidenceDomainRejectsInvalidValues:
    def _valid_kwargs(self) -> dict[str, Any]:
        return {
            "method_type": "experimental",
            "data_source": "primary",
            "temporal_approach": "cross_sectional",
            "causal_role": "mechanistic",
            "confidence": 0.8,
            "justification": "x",
        }

    def test_rejects_meta_analytic_hallucination(self) -> None:
        # The exact bug from the homeopathy probe run: LLM emitted
        # "meta_analytic" but the DataSourceType enum only has "meta".
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        kwargs = self._valid_kwargs()
        kwargs["data_source"] = "meta_analytic"
        with pytest.raises(ValidationError):
            ClassifyEvidenceDomainOutput(**kwargs)

    def test_rejects_invalid_method_type(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        kwargs = self._valid_kwargs()
        kwargs["method_type"] = "qualitative"  # not in the 4-value set
        with pytest.raises(ValidationError):
            ClassifyEvidenceDomainOutput(**kwargs)

    def test_accepts_valid_values(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            ClassifyEvidenceDomainOutput,
        )

        out = ClassifyEvidenceDomainOutput(**self._valid_kwargs())
        # Values come back as the enum value strings (StrEnum behaviour).
        assert out.method_type == "experimental"
        assert out.data_source == "primary"


class TestClassifyQuestionEnumConstraints:
    def test_question_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyQuestionOutput

        values = _field_allowed_values(
            ClassifyQuestionOutput.model_json_schema(), "question_type"
        )
        assert set(values) == {
            "verificatory",
            "explanatory",
            "exploratory",
            "comparative",
            "predictive",
            "compositional",
            "normative",
        }, values

    def test_rejects_invalid_question_type(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyQuestionOutput

        with pytest.raises(ValidationError):
            ClassifyQuestionOutput(question_type="speculative", reasoning="x")  # type: ignore[arg-type]


class TestClassifyPredictionEnumConstraints:
    def _valid_kwargs(self) -> dict[str, Any]:
        return {
            "prediction_type": "quantitative",
            "specificity": 0.8,
            "success_criteria": "x",
            "failure_criteria": "y",
            "time_horizon": "short_term",
            "justification": "z",
        }

    def test_prediction_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyPredictionOutput

        values = _field_allowed_values(
            ClassifyPredictionOutput.model_json_schema(), "prediction_type"
        )
        assert set(values) == {
            "quantitative",
            "qualitative",
            "conditional",
            "temporal",
            "binary",
        }, values

    def test_time_horizon_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyPredictionOutput

        values = _field_allowed_values(
            ClassifyPredictionOutput.model_json_schema(), "time_horizon"
        )
        assert set(values) == {
            "immediate",
            "short_term",
            "medium_term",
            "long_term",
            "indefinite",
        }, values

    def test_rejects_invalid_prediction_type(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyPredictionOutput

        kwargs = self._valid_kwargs()
        kwargs["prediction_type"] = "probabilistic"
        with pytest.raises(ValidationError):
            ClassifyPredictionOutput(**kwargs)

    def test_rejects_invalid_time_horizon(self) -> None:
        from andamentum.epistemic.agents.output_models import ClassifyPredictionOutput

        kwargs = self._valid_kwargs()
        kwargs["time_horizon"] = "eventually"
        with pytest.raises(ValidationError):
            ClassifyPredictionOutput(**kwargs)
