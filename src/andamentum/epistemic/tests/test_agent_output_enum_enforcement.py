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


class TestVerdictLikeLiteralConstraints:
    """Cluster of verdict-like Literal[...] fields (evidence_weight,
    deductive_soundness, recommendation, validity, soundness, direction).
    Each was bare ``str`` with values in the description — now a Literal."""

    def test_evidence_weight_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import AssessEvidenceOutput

        values = _field_allowed_values(
            AssessEvidenceOutput.model_json_schema(), "evidence_weight"
        )
        assert set(values) == {"strong", "moderate", "weak", "conflicting"}

    def test_deductive_soundness_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            DeductiveValidationOutput,
        )

        values = _field_allowed_values(
            DeductiveValidationOutput.model_json_schema(), "deductive_soundness"
        )
        assert set(values) == {"sound", "questionable", "unsound"}

    def test_recommendation_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            DeductiveValidationOutput,
        )

        values = _field_allowed_values(
            DeductiveValidationOutput.model_json_schema(), "recommendation"
        )
        assert set(values) == {"promote", "hold", "demote"}

    def test_validity_and_soundness_have_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import AnalyzeArgumentOutput

        schema = AnalyzeArgumentOutput.model_json_schema()
        assert set(_field_allowed_values(schema, "validity")) == {
            "valid",
            "invalid",
            "indeterminate",
        }
        assert set(_field_allowed_values(schema, "soundness")) == {
            "sound",
            "unsound",
            "questionable",
        }

    def test_direction_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import DraftClaimOutput

        values = _field_allowed_values(
            DraftClaimOutput.model_json_schema(), "direction"
        )
        assert set(values) == {"supports", "undermines", "neutral"}

    def test_rejects_invalid_evidence_weight(self) -> None:
        from andamentum.epistemic.agents.output_models import AssessEvidenceOutput

        with pytest.raises(ValidationError):
            AssessEvidenceOutput(
                claim_id="c1",
                evidence_weight="overwhelming",  # type: ignore[arg-type]
                confidence_estimate=0.9,
                justification="x",
            )


class TestObservationMeasurabilityAmbiguityConstraints:
    def test_observation_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            IdentifyTestableAspectOutput,
        )

        values = _field_allowed_values(
            IdentifyTestableAspectOutput.model_json_schema(), "observation_type"
        )
        assert set(values) == {"quantitative", "qualitative", "binary"}

    def test_measurability_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import SpecifyPredictionOutput

        values = _field_allowed_values(
            SpecifyPredictionOutput.model_json_schema(), "measurability"
        )
        assert set(values) == {"quantitative", "qualitative", "binary"}

    def test_ambiguity_level_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import ClarifyQuestionOutput

        values = _field_allowed_values(
            ClarifyQuestionOutput.model_json_schema(), "ambiguity_level"
        )
        assert set(values) == {"clear", "moderate", "high"}

    def test_rejects_invalid_observation_type(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            IdentifyTestableAspectOutput,
        )

        with pytest.raises(ValidationError):
            IdentifyTestableAspectOutput(
                testable_dimension="x",
                observation_type="categorical",  # type: ignore[arg-type]
            )


class TestComplexVocabConstraints:
    def test_extract_evidence_source_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import ExtractEvidenceOutput

        values = _field_allowed_values(
            ExtractEvidenceOutput.model_json_schema(), "source_type"
        )
        assert set(values) == {
            "paper",
            "dataset",
            "note",
            "conversation",
            "webpage",
            "book",
            "report",
        }

    def test_identify_single_issue_type_has_enum(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            IdentifySingleIssueOutput,
        )

        values = _field_allowed_values(
            IdentifySingleIssueOutput.model_json_schema(), "issue_type"
        )
        # Includes "" for the has_issue=False case, plus the 9 UncertaintyType
        # values the LLM may pick + the "evidence_corrupted" sentinel that
        # triggers evidence invalidation rather than uncertainty creation.
        assert set(values) == {
            "",
            "evidence_corrupted",
            "unknown",
            "contradiction",
            "evidence_gap",
            "risk",
            "assumption",
            "scope_difference",
            "methodological_variation",
            "definitional_variation",
            "perspectival",
        }

    def test_evaluate_counterargument_category_has_enum(self) -> None:
        # This field was bare str with a description vocabulary that drifted
        # from the CriticismCategory enum downstream code actually consumes
        # (primitives.py coerces via CriticismCategory(...)). Align the
        # LLM's vocabulary with the enum so strict-structured-outputs mode
        # cannot produce ValueErrors at the boundary.
        from andamentum.epistemic.agents.output_models import (
            EvaluateCounterargumentOutput,
        )

        values = _field_allowed_values(
            EvaluateCounterargumentOutput.model_json_schema(), "category"
        )
        assert set(values) == {
            "methodological",
            "statistical",
            "replication_failure",
            "confounding",
            "generalization",
            "interpretation",
            "theoretical",
            "fringe",
            "ad_hominem",
        }

    def test_rejects_drift_values_on_category(self) -> None:
        from andamentum.epistemic.agents.output_models import (
            EvaluateCounterargumentOutput,
        )

        # These were in the previous prompt/description but NOT in the
        # CriticismCategory enum — the exact drift that used to cause
        # ValueErrors at primitives.py::AdversarialFinding construction.
        for bad in (
            "empirical",
            "logical",
            "scope",
            "alternative_explanation",
            "ethical",
        ):
            with pytest.raises(ValidationError):
                EvaluateCounterargumentOutput(
                    relevance=0.5,
                    specificity=0.5,
                    evidence_backed=0.5,
                    source_credibility=0.5,
                    category=bad,  # type: ignore[arg-type]
                    justification="x",
                )


# ─────────────────────────────────────────────────────────────────────────
# Regression barrier — fire if anyone reverts a field back to bare `str`
# ─────────────────────────────────────────────────────────────────────────

# (model class name, field name) pairs that MUST carry a JSON-schema
# enum constraint. If you intentionally remove one, delete its row here;
# if you add a new enum-like field to an agent output model, add its row.
EXPECTED_ENUM_FIELDS: list[tuple[str, str]] = [
    ("ClarifyQuestionOutput", "ambiguity_level"),
    ("ClassifyQuestionOutput", "question_type"),
    ("ExtractEvidenceOutput", "source_type"),
    ("AssessEvidenceOutput", "evidence_weight"),
    ("IdentifySingleIssueOutput", "issue_type"),
    ("DeductiveValidationOutput", "deductive_soundness"),
    ("DeductiveValidationOutput", "recommendation"),
    ("AnalyzeArgumentOutput", "validity"),
    ("AnalyzeArgumentOutput", "soundness"),
    ("IdentifyTestableAspectOutput", "observation_type"),
    ("SpecifyPredictionOutput", "measurability"),
    ("EvaluateCounterargumentOutput", "category"),
    ("ClassifyEvidenceDomainOutput", "method_type"),
    ("ClassifyEvidenceDomainOutput", "data_source"),
    ("ClassifyEvidenceDomainOutput", "temporal_approach"),
    ("ClassifyEvidenceDomainOutput", "causal_role"),
    ("ClassifyPredictionOutput", "prediction_type"),
    ("ClassifyPredictionOutput", "time_horizon"),
    ("DraftClaimOutput", "direction"),
]


class TestAgentOutputEnumManifest:
    """All enum-constrained fields across the agent output models — one
    regression-barrier test that iterates the manifest.

    If a new controlled-vocabulary field is added to any output model,
    append its (model, field) pair to ``EXPECTED_ENUM_FIELDS`` above.
    """

    @pytest.mark.parametrize("model_name,field_name", EXPECTED_ENUM_FIELDS)
    def test_field_has_enum_constraint(self, model_name: str, field_name: str) -> None:
        from andamentum.epistemic.agents import output_models

        model_cls = getattr(output_models, model_name)
        schema = model_cls.model_json_schema()
        values = _field_allowed_values(schema, field_name)
        assert values, (
            f"{model_name}.{field_name} has no enum constraint in its JSON "
            f"schema — did you revert it to bare str? pydantic-ai can only "
            f"enforce the vocabulary via strict structured outputs when the "
            f"schema carries an `enum` node."
        )
