"""Tests for agent adapters — verify every adapter correctly transforms agent output.

These tests catch the exact class of bug that caused silent failures:
field name mismatches between agent manifests and adapter code.
Each adapter is tested with the canned responses from conftest._FAKE_DEFAULTS
AND with edge-case inputs (missing fields, empty lists, unexpected types).
"""

import pytest
from types import SimpleNamespace

from ..adapters import (
    ADAPTERS,
    adapt_write_answer,
    adapt_extract,
    adapt_deductive,
    adapt_computational,
    adapt_clarify,
    adapt_conceptual_analysis,
    adapt_resolve_uncertainty,
    adapt_investigate_claim,
    adapt_analyze_argument,
    adapt_record_decision,
    adapt_evaluate_counterargument,
    adapt_classify_evidence_domain,
    adapt_classify_prediction,
    adapt_assess_evidence_quality,
    adapt_agent_output,
    get_adapter,
    WriteAnswerResult,
    ExtractResult,
    DeductiveResult,
    ComputationalResult,
    ClarifyResult,
    ConceptualAnalysisResult,
    ResolveUncertaintyResult,
    InvestigateClaimResult,
    AnalyzeArgumentResult,
    RecordDecisionResult,
    EvaluateCounterargumentResult,
    ClassifyEvidenceDomainResult,
    ClassifyPredictionResult,
    AssessEvidenceQualityResult,
)


def ns(**kwargs):
    """Create a SimpleNamespace (mimics agent output with attribute access)."""
    return SimpleNamespace(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Registry completeness
# ──────────────────────────────────────────────────────────────────────────────


class TestAdapterRegistry:
    def test_all_adapters_registered(self):
        """Every adapter function should be in the ADAPTERS dict."""
        expected = {
            "epistemic_write_answer",
            "epistemic_extract_evidence",
            "epistemic_deductive_validation",
            "epistemic_verify_computationally",
            "epistemic_clarify_question",
            "epistemic_conceptual_analysis",
            "epistemic_resolve_uncertainty",
            "epistemic_investigate_claim",
            "epistemic_analyze_argument",
            "epistemic_record_decision",
            "epistemic_generate_counterquery",
            "epistemic_evaluate_counterargument",
            "epistemic_classify_evidence_domain",
            "epistemic_check_pairwise_independence",
            "epistemic_classify_prediction",
            "epistemic_identify_testable_aspect",
            "epistemic_specify_prediction",
            "epistemic_define_falsification",
            "epistemic_assess_evidence_quality",
            "epistemic_classify_question",
            "epistemic_contrastive_evaluation",
            "epistemic_cross_claim_consistency",
            "epistemic_formulate_query",
            "epistemic_extract_assertion",
            "epistemic_draft_claim",
            "epistemic_identify_single_issue",
        }
        assert set(ADAPTERS.keys()) == expected

    def test_get_adapter_returns_callable(self):
        for name in ADAPTERS:
            assert callable(get_adapter(name))

    def test_get_adapter_returns_none_for_unknown(self):
        assert get_adapter("nonexistent_agent") is None

    def test_adapt_agent_output_passthrough_for_unknown(self):
        raw = ns(foo="bar")
        result = adapt_agent_output("nonexistent_agent", raw)
        assert result is raw


# ──────────────────────────────────────────────────────────────────────────────
# Write answer adapter
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptWriteAnswer:
    def test_basic(self):
        raw = ns(title="My Title", answer="Some answer")
        result = adapt_write_answer(raw)
        assert isinstance(result, WriteAnswerResult)
        assert result.title == "My Title"
        assert result.answer == "Some answer"

    def test_missing_field_raises(self):
        raw = ns(title="T")
        with pytest.raises(AttributeError):
            adapt_write_answer(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Extract evidence adapter
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptExtract:
    def test_joins_quotes(self):
        raw = ns(
            relevant_quotes=["Quote 1", "Quote 2"],
            limitations=["Small sample"],
            experimental_context="Lab",
        )
        result = adapt_extract(raw)
        assert isinstance(result, ExtractResult)
        assert "Quote 1" in result.content
        assert "Quote 2" in result.content
        assert result.limitations == ["Small sample"]

    def test_empty_quotes(self):
        raw = ns(relevant_quotes=[], limitations=[], experimental_context="None")
        result = adapt_extract(raw)
        assert result.content == ""

    def test_none_quotes_returns_empty(self):
        """None quotes are handled gracefully (empty string content)."""
        raw = ns(relevant_quotes=None, limitations=[], experimental_context="")
        result = adapt_extract(raw)
        assert result.content == ""


# ──────────────────────────────────────────────────────────────────────────────
# Deductive validation adapter
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptDeductive:
    def test_passes(self):
        raw = ns(passes_deductive_validation=True, issues_found=[], issue_types=[])
        result = adapt_deductive(raw)
        assert isinstance(result, DeductiveResult)
        assert result.passes_deductive_validation is True

    def test_fails(self):
        raw = ns(
            passes_deductive_validation=False,
            issues_found=["Non sequitur"],
            issue_types=["logical"],
        )
        result = adapt_deductive(raw)
        assert result.passes_deductive_validation is False
        assert result.issues_found == ["Non sequitur"]

    def test_missing_issue_types_raises(self):
        """Missing issue_types should raise AttributeError, not silently default."""
        raw = ns(passes_deductive_validation=True, issues_found=[])
        with pytest.raises(AttributeError):
            adapt_deductive(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Computational verification adapter
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptComputational:
    def test_basic(self):
        raw = ns(
            verification_code="assert 1 + 1 == 2",
            packages_required=[],
            expected_behavior="Passes",
            test_description="Basic math",
        )
        result = adapt_computational(raw)
        assert isinstance(result, ComputationalResult)
        assert "assert" in result.verification_code


# ──────────────────────────────────────────────────────────────────────────────
# Preplanning adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptClarify:
    def test_basic(self):
        raw = ns(
            clarified_question="Refined Q", key_terms=["term1"], reasoning="Because"
        )
        result = adapt_clarify(raw)
        assert isinstance(result, ClarifyResult)
        assert result.clarified_question == "Refined Q"


class TestAdaptConceptualAnalysis:
    def test_basic(self):
        raw = ns(
            terms=["A"],
            definitions=["Def A"],
            assumptions=["Assumed"],
            context_summary="Context",
        )
        result = adapt_conceptual_analysis(raw)
        assert isinstance(result, ConceptualAnalysisResult)
        assert result.terms == ["A"]


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty and investigation adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptResolveUncertainty:
    def test_resolved(self):
        raw = ns(resolution="Fixed", can_resolve=True, remaining_concerns=[])
        result = adapt_resolve_uncertainty(raw)
        assert isinstance(result, ResolveUncertaintyResult)
        assert result.can_resolve is True

    def test_unresolved(self):
        raw = ns(
            resolution="Tried", can_resolve=False, remaining_concerns=["Still unclear"]
        )
        result = adapt_resolve_uncertainty(raw)
        assert result.can_resolve is False
        assert result.remaining_concerns == ["Still unclear"]


class TestAdaptInvestigateClaim:
    def test_basic(self):
        raw = ns(evidence_queries=["query1"], reasoning="Need more data")
        result = adapt_investigate_claim(raw)
        assert isinstance(result, InvestigateClaimResult)
        assert result.evidence_queries == ["query1"]


# ──────────────────────────────────────────────────────────────────────────────
# Analysis and prediction adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptAnalyzeArgument:
    def test_basic(self):
        raw = ns(
            premises=["P1"],
            conclusion="C",
            validity="valid",
            soundness="sound",
            fallacies=[],
        )
        result = adapt_analyze_argument(raw)
        assert isinstance(result, AnalyzeArgumentResult)
        assert result.validity == "valid"


class TestAdaptRecordDecision:
    def test_basic(self):
        raw = ns(statement="Decided X", justification="Because evidence")
        result = adapt_record_decision(raw)
        assert isinstance(result, RecordDecisionResult)
        assert result.statement == "Decided X"


# ──────────────────────────────────────────────────────────────────────────────
# Quality assessment adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestAdaptEvaluateCounterargument:
    def test_basic(self):
        raw = ns(
            relevance=0.8,
            specificity=0.7,
            evidence_backed=0.6,
            source_credibility=0.5,
            category="logical",
            justification="Strong",
        )
        result = adapt_evaluate_counterargument(raw)
        assert isinstance(result, EvaluateCounterargumentResult)
        assert result.relevance == 0.8

    def test_missing_field_raises(self):
        raw = ns(relevance=0.8)
        with pytest.raises(AttributeError):
            adapt_evaluate_counterargument(raw)


class TestAdaptClassifyEvidenceDomain:
    def test_basic(self):
        raw = ns(
            method_type="experimental",
            data_source="primary",
            temporal_approach="longitudinal",
            causal_role="cause",
            confidence=0.9,
            justification="Lab study",
        )
        result = adapt_classify_evidence_domain(raw)
        assert isinstance(result, ClassifyEvidenceDomainResult)
        assert result.method_type == "experimental"


class TestAdaptClassifyPrediction:
    def test_basic(self):
        raw = ns(
            prediction_type="empirical",
            specificity=0.8,
            success_criteria="Score > 80",
            failure_criteria="Score < 60",
            time_horizon="6 months",
            justification="Testable",
        )
        result = adapt_classify_prediction(raw)
        assert isinstance(result, ClassifyPredictionResult)
        assert result.prediction_type == "empirical"


class TestAdaptAssessEvidenceQuality:
    def test_basic(self):
        raw = ns(
            source_credibility=0.7,
            relevance=0.8,
            specificity=0.6,
            recency_appropriate=0.7,
            justification="Good",
        )
        result = adapt_assess_evidence_quality(raw)
        assert isinstance(result, AssessEvidenceQualityResult)
        assert result.relevance == 0.8

    def test_missing_field_raises(self):
        raw = ns(source_credibility=0.7)
        with pytest.raises(AttributeError):
            adapt_assess_evidence_quality(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-check: conftest._FAKE_DEFAULTS through adapters
# ──────────────────────────────────────────────────────────────────────────────


class TestFakeDefaultsThroughAdapters:
    """Verify that every canned response in conftest._FAKE_DEFAULTS passes
    through its adapter without error. This catches field name drift between
    conftest and the actual adapters."""

    def test_all_fake_defaults_through_adapters(self):
        import sys
        import pathlib

        # Ensure conftest is importable
        test_dir = str(pathlib.Path(__file__).parent)
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from conftest import _FAKE_DEFAULTS, _to_namespace  # type: ignore[import-not-found]

        for agent_name, raw_dict in _FAKE_DEFAULTS.items():
            adapter = get_adapter(agent_name)
            if adapter is None:
                continue
            raw = _to_namespace(raw_dict)
            # Should not raise
            result = adapter(raw)
            assert result is not None, f"Adapter for {agent_name} returned None"
