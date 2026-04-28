"""Tests for the EditSections node + editor agent.

Patches every LLM-calling agent in the new pipeline so we exercise the
EditSections flow end-to-end without real LLM calls. Verifies:
  • editor=False (default) → EditSections is a pass-through, no LLM call
  • editor=True → editor_agent runs once per section, edits accumulate
  • Each Edit's original_text is anchored to a real char span via find_anchor
  • Edits the agent fabricates (text not in section) are dropped silently
"""

from dataclasses import dataclass
from typing import Any

import pytest

from andamentum.whetstone import review_document
from andamentum.whetstone.agents import (
    AuthorQuestionOutput,
    ChallengeVerdict,
    EditorOutput,
    EditProposal,
    LensReadOutput,
    ReflectionOutput,
    ReviewSummary,
)


PAPER = """## 1 Introduction

It is generally the case that approaches to this problem have been varied.
We had N=50 participants in our study.

## 2 Methods

The methodology employed in this paper is broadly conventional.
"""


@dataclass
class _FakeRunResult:
    output: Any


class _FakeAgent:
    def __init__(self, output: Any):
        self.output = output

    async def run(self, prompt: str) -> _FakeRunResult:
        return _FakeRunResult(output=self.output)


@pytest.fixture
def patched_agents(monkeypatch):
    canned: dict[str, Any] = {}

    def fake_build(name: str, model: Any) -> _FakeAgent:
        # Lenses share output schema — match by lens.<name> prefix.
        if name.startswith("lens.") and "lens" in canned:
            return _FakeAgent(output=canned["lens"])
        if name not in canned:
            raise AssertionError(f"agent {name!r} called with no canned output")
        return _FakeAgent(output=canned[name])

    import andamentum.whetstone.agents as agents_mod
    import andamentum.whetstone.nodes.author_questions as aq_mod
    import andamentum.whetstone.nodes.challenge as ch_mod
    import andamentum.whetstone.nodes.critical_read as cr_mod
    import andamentum.whetstone.nodes.edit_sections as es_mod
    import andamentum.whetstone.nodes.reflect_and_investigate as ri_mod
    import andamentum.whetstone.nodes.synthesise as sy_mod

    for mod in (agents_mod, aq_mod, ch_mod, cr_mod, es_mod, ri_mod, sy_mod):
        monkeypatch.setattr(
            mod, "build_pydantic_ai_agent", fake_build, raising=True
        )
    return canned


def _empty_pipeline(canned: dict[str, Any]) -> None:
    """Set the non-editor agents to no-op outputs.

    The lens emits no issues; the reflection loop returns nothing; the
    challenge agent never gets called (no findings to challenge); the
    author-question agent stays a no-op; synthesise gets canned prose.
    """
    canned["lens"] = LensReadOutput(issues=[])
    canned["reflection"] = ReflectionOutput(tasks=[])
    canned["challenge"] = ChallengeVerdict(verdict="stand", reason="n/a")
    canned["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    canned["synthesise"] = ReviewSummary(
        executive_summary="ok",
        must_fix_summary="No must-fix findings.",
        should_fix_summary="No should-fix findings.",
        consider_summary="No consider findings.",
    )


# ── Editor disabled by default ─────────────────────────────────────────


async def test_editor_disabled_by_default_emits_no_edits(patched_agents):
    _empty_pipeline(patched_agents)
    # Note: no canned editor output. If EditSections invoked it we'd crash.
    result = await review_document(PAPER, model="fake:test")
    assert result.edits == []
    assert result.metrics.edits_count == 0


# ── Editor enabled: edits accumulate from each section ────────────────


async def test_editor_enabled_runs_per_section_and_anchors_edits(patched_agents):
    _empty_pipeline(patched_agents)
    # The editor agent emits one edit per section (real call sites), with
    # original_text taken verbatim from the section. find_anchor inside
    # EditSections will then locate it.
    patched_agents["editor"] = EditorOutput(
        edits=[
            EditProposal(
                title="Tighten",
                rationale="Wordier than necessary.",
                severity="minor",
                confidence="high",
                original_text="approaches to this problem",
                new_text="approaches",
            ),
        ],
    )
    result = await review_document(
        PAPER,
        model="fake:test",
        editor=True,
        editor_criteria=["concision"],
    )
    # The editor sees each section in turn. Only section 1 contains the
    # phrase "approaches to this problem", so exactly 1 Edit is anchored.
    assert len(result.edits) == 1
    edit = result.edits[0]
    assert edit.title == "Tighten"
    assert edit.section_id == "sec_001"
    assert edit.original_text == "approaches to this problem"
    assert edit.new_text == "approaches"
    assert 0 <= edit.char_start < edit.char_end


async def test_editor_drops_edits_with_unfindable_original_text(patched_agents):
    """Fabricated quotes (not in source) are dropped silently."""
    _empty_pipeline(patched_agents)
    patched_agents["editor"] = EditorOutput(
        edits=[
            EditProposal(
                title="Fake",
                rationale="r",
                severity="minor",
                confidence="medium",
                original_text="this exact phrase does not appear anywhere in the paper",
                new_text="x",
            ),
        ],
    )
    result = await review_document(PAPER, model="fake:test", editor=True)
    assert result.edits == []  # all edits dropped because anchor failed


async def test_editor_metric_count_matches_emitted_edits(patched_agents):
    _empty_pipeline(patched_agents)
    patched_agents["editor"] = EditorOutput(
        edits=[
            EditProposal(
                title="t",
                rationale="r",
                severity="minor",
                confidence="medium",
                original_text="N=50 participants",
                new_text="50 participants",
            ),
        ],
    )
    result = await review_document(PAPER, model="fake:test", editor=True)
    assert result.metrics.edits_count == len(result.edits) >= 1
