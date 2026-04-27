"""Tests for the reflection agent — senior reviewer that proposes
investigation tasks for the bounded loop.
"""

from __future__ import annotations

from andamentum.whetstone.v2.agents import (
    REFLECTION_AGENT,
    ReflectionOutput,
    ReflectionTask,
    get_agent,
)


def test_reflection_agent_registered() -> None:
    agent = get_agent("reflection")
    assert agent.name == "reflection"
    assert agent.output_model is ReflectionOutput


def test_reflection_module_constant_matches_registry() -> None:
    assert REFLECTION_AGENT.name == "reflection"


def test_reflection_task_minimal_construction() -> None:
    t = ReflectionTask(
        description="Verify section 4's claim against section 9.",
        section_ids=["sec_004", "sec_009"],
    )
    assert t.related_note_ids == []


def test_reflection_task_with_related_notes() -> None:
    t = ReflectionTask(
        description="Consolidate evidence concerns about the methodology.",
        section_ids=["sec_002"],
        related_note_ids=["abc123", "def456"],
    )
    assert len(t.related_note_ids) == 2


def test_reflection_output_default_empty_list() -> None:
    out = ReflectionOutput()
    assert out.tasks == []


def test_reflection_prompt_forbids_pattern_categories() -> None:
    """The prompt must explicitly let the model decide what shapes matter,
    not enumerate categories. Sanity-check key phrasings."""
    prompt = REFLECTION_AGENT.prompt
    # Prompt should NOT enumerate scan kinds in a closed taxonomy.
    # It should explicitly tell the model "you decide".
    assert "You decide" in prompt or "your own judgement" in prompt.lower()
    # And the hard rules — these MUST be there.
    assert "section ids" in prompt.lower()
    assert "do not duplicate" in prompt.lower() or "not duplicate" in prompt.lower()
    assert "empty list" in prompt.lower()


def test_reflection_prompt_specifies_round_cap_idea() -> None:
    """The prompt should be aware that this runs in rounds (not exhaustive)."""
    prompt = REFLECTION_AGENT.prompt
    # The phrase "earlier rounds" or "prior" must appear since the prompt
    # is told its tasks accumulate across rounds.
    assert (
        "rounds" in prompt.lower() or "earlier" in prompt.lower()
    ), "prompt should reference loop rounds"
