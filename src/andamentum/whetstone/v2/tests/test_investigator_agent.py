"""Tests for the investigator agent — the focused per-task LLM call in
the reflection loop.
"""

from __future__ import annotations

import pytest

from andamentum.whetstone.v2.agents import (
    INVESTIGATOR_AGENT,
    InvestigatorOutput,
    NewNote,
    NoteUpdate,
    get_agent,
)


def test_investigator_agent_registered() -> None:
    agent = get_agent("investigator")
    assert agent.name == "investigator"
    assert agent.output_model is InvestigatorOutput


def test_investigator_module_constant_matches_registry() -> None:
    assert INVESTIGATOR_AGENT.name == "investigator"


# ── NoteUpdate ─────────────────────────────────────────────────────────


def test_note_update_keep_minimal() -> None:
    upd = NoteUpdate(note_id="abc123", action="keep")
    assert upd.refined_title == ""  # not used for "keep"


def test_note_update_drop_minimal() -> None:
    upd = NoteUpdate(note_id="abc123", action="drop")
    assert upd.action == "drop"


def test_note_update_refine_carries_full_payload() -> None:
    upd = NoteUpdate(
        note_id="abc123",
        action="refine",
        refined_title="more accurate title",
        refined_severity="major",
        refined_confidence="high",
        refined_rationale="The actual issue is X, not Y as previously claimed.",
        refined_quote_text="some verbatim span",
        refined_quote_section_id="sec_004",
    )
    assert upd.refined_severity == "major"
    assert upd.refined_quote_section_id == "sec_004"


def test_note_update_invalid_action_rejected() -> None:
    with pytest.raises(ValueError):
        NoteUpdate(note_id="x", action="merge")  # type: ignore[arg-type]


# ── NewNote ────────────────────────────────────────────────────────────


def test_new_note_requires_quote() -> None:
    """quote_text is a required field — without it the model can't anchor."""
    with pytest.raises(ValueError):
        NewNote(  # type: ignore[call-arg]
            title="x",
            severity="moderate",
            confidence="medium",
            rationale="...",
            quote_section_id="sec_001",
        )


def test_new_note_requires_quote_section() -> None:
    """quote_section_id is a required field."""
    with pytest.raises(ValueError):
        NewNote(  # type: ignore[call-arg]
            title="x",
            severity="moderate",
            confidence="medium",
            rationale="...",
            quote_text="some text",
        )


def test_new_note_minimal_construction_ok() -> None:
    n = NewNote(
        title="x",
        severity="moderate",
        confidence="medium",
        rationale="...",
        quote_text="some text",
        quote_section_id="sec_001",
    )
    assert n.category == ""


# ── InvestigatorOutput ─────────────────────────────────────────────────


def test_investigator_output_default_empty() -> None:
    out = InvestigatorOutput()
    assert out.updates == []
    assert out.new_notes == []


def test_investigator_output_can_carry_both() -> None:
    out = InvestigatorOutput(
        updates=[NoteUpdate(note_id="a", action="keep")],
        new_notes=[
            NewNote(
                title="x",
                severity="minor",
                confidence="low",
                rationale="...",
                quote_text="t",
                quote_section_id="sec_001",
            )
        ],
    )
    assert len(out.updates) == 1
    assert len(out.new_notes) == 1


# ── Prompt sanity ──────────────────────────────────────────────────────


def test_prompt_emphasises_source_grounding() -> None:
    prompt = INVESTIGATOR_AGENT.prompt
    assert "verbatim" in prompt.lower()
    assert "source" in prompt.lower()
    # Must explicitly call out "do not analyse the analysis" rule.
    assert "do not analyse" in prompt.lower() or "not analyse" in prompt.lower()


def test_prompt_describes_each_action() -> None:
    prompt = INVESTIGATOR_AGENT.prompt
    assert "keep" in prompt.lower()
    assert "refine" in prompt.lower()
    assert "drop" in prompt.lower()


def test_prompt_describes_merge_via_drop_plus_new() -> None:
    """Merging is expressed as drop + new note, NOT a special action."""
    prompt = INVESTIGATOR_AGENT.prompt
    assert "merge" in prompt.lower()
    # Specifically must instruct the model to drop+add for merging.
    assert "drop them" in prompt.lower() or "drop them all" in prompt.lower()
