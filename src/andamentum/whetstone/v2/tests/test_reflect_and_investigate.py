"""Unit tests for the bounded reflect–investigate loop.

Tests focus on the controller logic in ``reflect_and_investigate.py``:
loop termination, anchor checks, stale-id handling, fed-section
restriction. Every reflection / investigation call is stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from andamentum.whetstone.v2.agents import (
    InvestigatorOutput,
    NewNote,
    NoteUpdate,
    ReflectionOutput,
    ReflectionTask,
)
from andamentum.whetstone.v2.deps import ReviewDeps
from andamentum.whetstone.v2.nodes.reflect_and_investigate import (
    _apply_investigator_result,
    _run_reflection,
    _run_investigation,
)
from andamentum.whetstone.v2.schemas import Finding, Quote, SectionCard
from andamentum.whetstone.v2.state import ReviewState
from andamentum.whetstone.v2.structural.types import SectionRef


# ── Builders ────────────────────────────────────────────────────────────


def _section(
    *,
    id: str = "sec_001",
    title: str = "Methods",
    text: str = "The quick brown fox jumps over the lazy dog.",
) -> SectionRef:
    return SectionRef(
        id=id, title=title, text=text, char_start=0, char_end=len(text)
    )


def _finding(
    *,
    id: str = "abc12345",
    section_id: str = "sec_001",
    title: str = "argument doesn't follow",
    severity: str = "moderate",
    confidence: str = "medium",
    rationale: str = "rationale text",
    quote_text: str = "",
    perspective: str = "rigorous",
    category: str = "evidence",
) -> Finding:
    quotes = [Quote(section_id=section_id, char_start=0, char_end=len(quote_text), text=quote_text)] if quote_text else []
    return Finding(
        id=id,
        title=title,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        rationale=rationale,
        quotes=quotes,
        sections_involved=[section_id],
        source="investigate",
        perspective=perspective,
        category=category,
    )


def _state(
    *,
    sections: list[SectionRef] | None = None,
    findings: list[Finding] | None = None,
    round_cap: int = 3,
) -> ReviewState:
    state = ReviewState(source="dummy")
    state.sections = sections or [_section()]
    state.findings = findings or []
    state.reflection_round_cap = round_cap
    state.document_map = [
        SectionCard(section_id=s.id, title=s.title, one_line_gist=s.title)
        for s in state.sections
    ]
    return state


@dataclass
class _FakeRunResult:
    output: Any


class _FakeAgent:
    def __init__(self, output: Any):
        self.output = output
        self.last_prompt: str | None = None

    async def run(self, prompt: str) -> _FakeRunResult:
        self.last_prompt = prompt
        return _FakeRunResult(output=self.output)


def _deps() -> ReviewDeps:
    return ReviewDeps(model="stub-model")


# ── _apply_investigator_result: the anchor discipline ──────────────────


def test_apply_keep_action_leaves_note_unchanged() -> None:
    section = _section()
    sections_by_id = {section.id: section}
    note = _finding()
    notes_by_id: dict[str, Finding] = {note.id: note}

    result = InvestigatorOutput(
        updates=[NoteUpdate(note_id=note.id, action="keep")],
    )
    task = ReflectionTask(
        description="x", section_ids=[section.id], related_note_ids=[note.id]
    )
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert notes_by_id[note.id] is note  # same object


def test_apply_drop_action_removes_note() -> None:
    section = _section()
    sections_by_id = {section.id: section}
    note = _finding()
    notes_by_id: dict[str, Finding] = {note.id: note}

    result = InvestigatorOutput(
        updates=[NoteUpdate(note_id=note.id, action="drop")],
    )
    task = ReflectionTask(
        description="x", section_ids=[section.id], related_note_ids=[note.id]
    )
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert note.id not in notes_by_id


def test_apply_refine_with_anchored_quote_replaces_note() -> None:
    section = _section(text="The quick brown fox jumps.")
    sections_by_id = {section.id: section}
    note = _finding()
    notes_by_id: dict[str, Finding] = {note.id: note}

    result = InvestigatorOutput(
        updates=[NoteUpdate(
            note_id=note.id,
            action="refine",
            refined_title="more accurate title",
            refined_severity="major",
            refined_confidence="high",
            refined_rationale="The actual issue is X.",
            refined_quote_text="quick brown fox",
            refined_quote_section_id=section.id,
        )],
    )
    task = ReflectionTask(
        description="x", section_ids=[section.id], related_note_ids=[note.id]
    )
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    refined = notes_by_id[note.id]
    assert refined.title == "more accurate title"
    assert refined.severity == "major"
    assert refined.confidence == "high"
    assert refined.rationale == "The actual issue is X."
    assert len(refined.quotes) == 1
    assert refined.quotes[0].text == "quick brown fox"


def test_apply_refine_with_fabricated_quote_keeps_original() -> None:
    """If the refined quote can't be anchored, the refinement is rejected."""
    section = _section(text="alpha beta gamma")
    sections_by_id = {section.id: section}
    note = _finding()
    notes_by_id: dict[str, Finding] = {note.id: note}

    result = InvestigatorOutput(
        updates=[NoteUpdate(
            note_id=note.id,
            action="refine",
            refined_title="never lands",
            refined_rationale="...",
            refined_quote_text="this text isn't anywhere in the section",
            refined_quote_section_id=section.id,
        )],
    )
    task = ReflectionTask(
        description="x", section_ids=[section.id], related_note_ids=[note.id]
    )
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert notes_by_id[note.id] is note  # unchanged
    assert notes_by_id[note.id].title != "never lands"


def test_apply_refine_with_unfed_section_keeps_original() -> None:
    """A refinement quoting a section the task didn't feed is rejected."""
    section_a = _section(id="sec_001", text="alpha beta gamma")
    section_b = _section(id="sec_002", text="delta epsilon zeta")
    sections_by_id = {section_a.id: section_a, section_b.id: section_b}
    note = _finding(section_id=section_a.id)
    notes_by_id: dict[str, Finding] = {note.id: note}

    result = InvestigatorOutput(
        updates=[NoteUpdate(
            note_id=note.id,
            action="refine",
            refined_title="never lands",
            refined_rationale="...",
            refined_quote_text="delta",  # really exists, but in sec_002
            refined_quote_section_id=section_b.id,
        )],
    )
    # Task only fed sec_001 → refinement claiming sec_002 must be rejected.
    task = ReflectionTask(
        description="x",
        section_ids=[section_a.id],
        related_note_ids=[note.id],
    )
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert notes_by_id[note.id] is note


def test_apply_stale_note_id_silently_ignored() -> None:
    section = _section()
    sections_by_id = {section.id: section}
    notes_by_id: dict[str, Finding] = {}

    result = InvestigatorOutput(
        updates=[NoteUpdate(note_id="ghost-id", action="drop")],
    )
    task = ReflectionTask(
        description="x", section_ids=[section.id], related_note_ids=["ghost-id"]
    )
    # Should not raise.
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)
    assert notes_by_id == {}


def test_apply_new_note_with_anchored_quote_added() -> None:
    section = _section(text="The quick brown fox.")
    sections_by_id = {section.id: section}
    notes_by_id: dict[str, Finding] = {}

    result = InvestigatorOutput(
        new_notes=[NewNote(
            title="newly raised",
            severity="major",
            confidence="high",
            rationale="...",
            quote_text="quick brown fox",
            quote_section_id=section.id,
            category="evidence",
        )],
    )
    task = ReflectionTask(description="x", section_ids=[section.id])
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert len(notes_by_id) == 1
    new = next(iter(notes_by_id.values()))
    assert new.title == "newly raised"
    assert new.perspective == "reflection"
    assert new.quotes[0].text == "quick brown fox"


def test_apply_new_note_with_fabricated_quote_dropped() -> None:
    section = _section(text="alpha beta gamma")
    sections_by_id = {section.id: section}
    notes_by_id: dict[str, Finding] = {}

    result = InvestigatorOutput(
        new_notes=[NewNote(
            title="fabricated quote claim",
            severity="major",
            confidence="high",
            rationale="...",
            quote_text="this text is not in the section",
            quote_section_id=section.id,
        )],
    )
    task = ReflectionTask(description="x", section_ids=[section.id])
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert notes_by_id == {}


def test_apply_new_note_naming_unfed_section_dropped() -> None:
    section_a = _section(id="sec_001", text="alpha beta gamma")
    section_b = _section(id="sec_002", text="delta epsilon zeta")
    sections_by_id = {section_a.id: section_a, section_b.id: section_b}
    notes_by_id: dict[str, Finding] = {}

    result = InvestigatorOutput(
        new_notes=[NewNote(
            title="should be rejected",
            severity="moderate",
            confidence="medium",
            rationale="...",
            quote_text="delta",
            quote_section_id=section_b.id,  # NOT in task.section_ids
        )],
    )
    task = ReflectionTask(description="x", section_ids=[section_a.id])
    _apply_investigator_result(result, task, sections_by_id, notes_by_id)

    assert notes_by_id == {}


# ── Reflection prompt assembly ─────────────────────────────────────────


async def test_run_reflection_prompt_includes_pool_and_history() -> None:
    section = _section(id="sec_001", title="Methods")
    note = _finding(id="abcdef01", section_id="sec_001", title="claim mismatch")
    state = _state(sections=[section], findings=[note])
    state.prior_task_descriptions = ["already-run thing"]
    state.reflection_round = 2

    fake = _FakeAgent(ReflectionOutput(tasks=[]))
    notes_by_id: dict[str, Finding] = {note.id: note}

    with patch(
        "andamentum.whetstone.v2.nodes.reflect_and_investigate.build_pydantic_ai_agent",
        return_value=fake,
    ):
        tasks = await _run_reflection(_deps(), state, notes_by_id)

    assert tasks == []
    assert fake.last_prompt is not None
    # Document map present
    assert "sec_001" in fake.last_prompt
    assert "Methods" in fake.last_prompt
    # Note id and title present
    assert "abcdef01" in fake.last_prompt
    assert "claim mismatch" in fake.last_prompt
    # Prior task included
    assert "already-run thing" in fake.last_prompt
    # Round number visible
    assert "round 2" in fake.last_prompt


# ── Investigation prompt assembly ──────────────────────────────────────


async def test_run_investigation_prompt_includes_section_text_and_notes() -> None:
    section = _section(id="sec_007", text="The quick brown fox.")
    note = _finding(id="zz9", section_id="sec_007", title="potential issue")
    sections_by_id = {section.id: section}
    notes_by_id = {note.id: note}

    fake = _FakeAgent(InvestigatorOutput())
    task = ReflectionTask(
        description="Verify whether the section actually claims X.",
        section_ids=["sec_007"],
        related_note_ids=["zz9"],
    )

    with patch(
        "andamentum.whetstone.v2.nodes.reflect_and_investigate.build_pydantic_ai_agent",
        return_value=fake,
    ):
        await _run_investigation(_deps(), task, sections_by_id, notes_by_id)

    p = fake.last_prompt or ""
    assert "Verify whether" in p
    assert "The quick brown fox." in p
    assert "zz9" in p
    assert "potential issue" in p


async def test_run_investigation_prompt_handles_missing_sections_gracefully() -> None:
    """If task names sections we don't have, prompt notes the issue
    rather than crashing."""
    sections_by_id: dict[str, SectionRef] = {}
    notes_by_id: dict[str, Finding] = {}
    fake = _FakeAgent(InvestigatorOutput())
    task = ReflectionTask(description="x", section_ids=["ghost"])

    with patch(
        "andamentum.whetstone.v2.nodes.reflect_and_investigate.build_pydantic_ai_agent",
        return_value=fake,
    ):
        await _run_investigation(_deps(), task, sections_by_id, notes_by_id)

    assert fake.last_prompt is not None
    assert "no valid sections" in fake.last_prompt.lower()
