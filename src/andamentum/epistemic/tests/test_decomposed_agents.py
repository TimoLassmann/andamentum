"""Tests for decomposed agent operations (Phase 3)."""

from types import SimpleNamespace

from ..agents import get_agent
from ..agents.output_models import (
    GenerateCounterqueryOutput,
    CheckPairwiseIndependenceOutput,
    IdentifyTestableAspectOutput,
    SpecifyPredictionOutput,
    DefineFalsificationOutput,
)
from ..adapters import adapt_agent_output


class TestGenerateCounterqueryAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_generate_counterquery")
        assert agent.name == "epistemic_generate_counterquery"

    def test_output_model_fields(self):
        fields = GenerateCounterqueryOutput.model_fields
        assert "query" in fields
        assert "framing" in fields
        assert len(fields) == 2

    def test_adapter(self):
        raw = SimpleNamespace(
            query="  failed replication test  ", framing="  replication_failures  "
        )
        result = adapt_agent_output("epistemic_generate_counterquery", raw)
        assert result.query == "failed replication test"
        assert result.framing == "replication_failures"


class TestCheckPairwiseIndependenceAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_check_pairwise_independence")
        assert agent.name == "epistemic_check_pairwise_independence"

    def test_output_model_fields(self):
        fields = CheckPairwiseIndependenceOutput.model_fields
        assert "independent" in fields
        assert "rationale" in fields
        assert len(fields) == 2

    def test_adapter_independent(self):
        raw = SimpleNamespace(independent=True, rationale="Different methods")
        result = adapt_agent_output("epistemic_check_pairwise_independence", raw)
        assert result.independent is True

    def test_adapter_not_independent(self):
        raw = SimpleNamespace(independent=False, rationale="Same lab, same method")
        result = adapt_agent_output("epistemic_check_pairwise_independence", raw)
        assert result.independent is False


# ──────────────────────────────────────────────────────────────────────────────
# Prediction decomposition agents (identify → specify → falsify → classify)
# ──────────────────────────────────────────────────────────────────────────────


class TestIdentifyTestableAspectAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_identify_testable_aspect")
        assert agent.name == "epistemic_identify_testable_aspect"

    def test_output_model_fields(self):
        fields = IdentifyTestableAspectOutput.model_fields
        assert "testable_dimension" in fields
        assert "observation_type" in fields
        assert len(fields) == 2

    def test_pydantic_model_rejects_out_of_vocab_observation_type(self):
        # observation_type is a Literal — pydantic rejects case/whitespace
        # drift at model-construction time. The old adapter-level
        # .strip().lower() that these tests exercised is therefore dead.
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            IdentifyTestableAspectOutput(
                testable_dimension="x",
                observation_type="  QUANTITATIVE  ",  # type: ignore[arg-type]
            )
        with pytest.raises(ValidationError):
            IdentifyTestableAspectOutput(
                testable_dimension="x",
                observation_type="  Binary  ",  # type: ignore[arg-type]
            )

    def test_adapter_passes_valid_observation_type_unchanged(self):
        raw = SimpleNamespace(
            testable_dimension="BP decrease", observation_type="binary"
        )
        result = adapt_agent_output("epistemic_identify_testable_aspect", raw)
        assert result.observation_type == "binary"
        assert result.testable_dimension == "BP decrease"


class TestSpecifyPredictionAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_specify_prediction")
        assert agent.name == "epistemic_specify_prediction"

    def test_output_model_fields(self):
        fields = SpecifyPredictionOutput.model_fields
        assert "expected_observation" in fields
        assert "conditions" in fields
        assert "timeframe" in fields
        assert "measurability" in fields
        assert len(fields) == 4

    def test_adapter(self):
        raw = SimpleNamespace(
            expected_observation="test",
            conditions="test",
            timeframe="3 months",
            measurability="  QUANTITATIVE  ",
        )
        result = adapt_agent_output("epistemic_specify_prediction", raw)
        assert result.measurability == "quantitative"

    def test_adapter_preserves_content(self):
        raw = SimpleNamespace(
            expected_observation="Systolic BP decreases",
            conditions="In adults",
            timeframe="6 months",
            measurability="qualitative",
        )
        result = adapt_agent_output("epistemic_specify_prediction", raw)
        assert result.expected_observation == "Systolic BP decreases"
        assert result.conditions == "In adults"
        assert result.timeframe == "6 months"


class TestDefineFalsificationAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_define_falsification")
        assert agent.name == "epistemic_define_falsification"

    def test_output_model_fields(self):
        fields = DefineFalsificationOutput.model_fields
        assert "falsification_criterion" in fields
        assert len(fields) == 1

    def test_adapter(self):
        raw = SimpleNamespace(falsification_criterion="No change observed")
        result = adapt_agent_output("epistemic_define_falsification", raw)
        assert result.falsification_criterion == "No change observed"


# ──────────────────────────────────────────────────────────────────────────────
# Claim proposal decomposition: extract_assertion + cluster + draft_claim
# ──────────────────────────────────────────────────────────────────────────────

from ..similarity import group_by_similarity as cluster_by_similarity  # noqa: E402
from ..agents.output_models import ExtractAssertionOutput, DraftClaimOutput  # noqa: E402


class TestEmbeddingsClustering:
    def test_empty_input(self):
        assert cluster_by_similarity([], 0.75) == []

    def test_single_input(self):
        assert cluster_by_similarity([[1.0, 0.0]], 0.75) == [[0]]

    def test_identical_vectors_cluster_together(self):
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        clusters = cluster_by_similarity(embeddings, threshold=0.99)
        # First two identical, third different
        assert any(0 in c and 1 in c for c in clusters)

    def test_orthogonal_vectors_separate(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        clusters = cluster_by_similarity(embeddings, threshold=0.5)
        assert len(clusters) == 2

    def test_all_indices_covered(self):
        embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        clusters = cluster_by_similarity(embeddings, threshold=0.8)
        all_indices = sorted([i for c in clusters for i in c])
        assert all_indices == [0, 1, 2, 3]

    def test_zero_vectors_separate(self):
        """Zero vectors should not cluster with anything (cosine_sim returns 0.0)."""
        embeddings = [[0.0, 0.0], [1.0, 0.0]]
        clusters = cluster_by_similarity(embeddings, threshold=0.5)
        assert len(clusters) == 2

    def test_low_threshold_clusters_all(self):
        """Very low threshold should cluster nearly everything together."""
        embeddings = [[1.0, 0.0], [0.7, 0.3], [0.5, 0.5]]
        clusters = cluster_by_similarity(embeddings, threshold=0.1)
        # With threshold 0.1, all should cluster together since all have positive components
        assert len(clusters) <= 2


class TestExtractAssertionAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_extract_assertion")
        assert agent.name == "epistemic_extract_assertion"

    def test_output_model_fields(self):
        fields = ExtractAssertionOutput.model_fields
        assert "assertion" in fields
        assert len(fields) == 1

    def test_adapter(self):
        raw = SimpleNamespace(assertion="  Coffee reduces diabetes risk  ")
        result = adapt_agent_output("epistemic_extract_assertion", raw)
        assert result.assertion == "Coffee reduces diabetes risk"


class TestDraftClaimAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_draft_claim")
        assert agent.name == "epistemic_draft_claim"

    def test_output_model_fields(self):
        fields = DraftClaimOutput.model_fields
        assert "statement" in fields
        assert "scope" in fields
        assert "direction" in fields
        assert len(fields) == 3

    def test_pydantic_model_rejects_out_of_vocab_direction(self):
        # `direction` is now a Literal on DraftClaimOutput, so pydantic
        # rejects case/whitespace drift at model-construction time. The
        # previous adapter-level `.strip().lower()` is therefore dead.
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DraftClaimOutput(statement="x", scope="y", direction="  SUPPORTS  ")  # type: ignore[arg-type]

    def test_adapter_strips_whitespace_on_free_text(self):
        raw = SimpleNamespace(
            statement="  claim  ", scope="  wide  ", direction="neutral"
        )
        result = adapt_agent_output("epistemic_draft_claim", raw)
        assert result.statement == "claim"
        assert result.scope == "wide"
        assert result.direction == "neutral"
