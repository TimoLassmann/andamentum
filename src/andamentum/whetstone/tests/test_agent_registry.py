"""Verify every whetstone agent is registered at import time."""

import pytest

from andamentum.whetstone.agents import AGENT_REGISTRY, get_agent

EXPECTED = {
    # editing
    "unified_editor",
    "grammar_specialist",
    "academic_writing_specialist",
    "polish_specialist",
    # review
    "clarity_accessibility_reviewer",
    "core_scientific_merit_reviewer",
    "methodology_reviewer",
    "results_interpretation_reviewer",
    # synthesis
    "document_review_synthesizer",
    "review_synthesizer",
    "results_formatter",
    # multi-expert
    "keyword_extractor",
    "expert_generator",
    "expert_reviewer",
    # custom
    "custom_document_reviewer",
    "schema_generator",
    # consistency
    "consistency_reviewer",
}


def test_all_agents_registered():
    missing = EXPECTED - set(AGENT_REGISTRY)
    assert not missing, f"Missing agents: {sorted(missing)}"


def test_get_agent_returns_definition():
    defn = get_agent("unified_editor")
    assert defn.name == "unified_editor"
    assert defn.prompt  # non-empty


def test_get_agent_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("nonexistent_agent")


def test_custom_reviewer_has_dynamic_output():
    defn = get_agent("custom_document_reviewer")
    assert defn.output_model is None  # signals dynamic schema at runtime


def test_consistency_reviewer_registered():
    from andamentum.whetstone.agents import AGENT_REGISTRY
    from andamentum.whetstone.agents.output_models import ConsistencyReviewOutput

    assert "consistency_reviewer" in AGENT_REGISTRY
    defn = AGENT_REGISTRY["consistency_reviewer"]
    assert defn.output_model is ConsistencyReviewOutput
