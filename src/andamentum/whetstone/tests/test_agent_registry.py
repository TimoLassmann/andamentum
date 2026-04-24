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
    # checklist
    "checklist_item_evaluator",
    "journal_guidelines_extractor",
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


def test_checklist_agents_registered():
    from andamentum.whetstone.agents import AGENT_REGISTRY

    assert "checklist_item_evaluator" in AGENT_REGISTRY
    assert "journal_guidelines_extractor" in AGENT_REGISTRY


def test_baseline_checks_shape():
    from andamentum.whetstone.agents.checklist import BASELINE_CHECKS

    assert len(BASELINE_CHECKS) >= 10
    for check in BASELINE_CHECKS:
        assert check.name
        assert check.category
        if check.kind == "deterministic":
            assert check.scanner is not None
            assert check.prompt_hint is None
        else:
            assert check.prompt_hint is not None
            assert check.scanner is None


def test_baseline_scanners_exist():
    """Every deterministic BASELINE_CHECK must point to a real scanner function."""
    from andamentum.whetstone.agents.checklist import BASELINE_CHECKS
    from andamentum.whetstone import checklist_scanners

    for check in BASELINE_CHECKS:
        if check.kind == "deterministic":
            assert check.scanner is not None
            assert hasattr(checklist_scanners, check.scanner), (
                f"BASELINE_CHECK '{check.name}' references missing scanner '{check.scanner}'"
            )
