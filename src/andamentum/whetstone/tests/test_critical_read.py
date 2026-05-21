"""Tests for the CriticalRead node — section × lens parallel reads.

The agent is stubbed at the pydantic-ai layer via patching the
``build_pydantic_ai_agent`` factory. The tests exercise the node's
controller logic: dispatch, anchoring, finding shape, error tolerance.

Tests do NOT exercise the full graph (that requires
ReflectAndInvestigate, which lands in Task 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from andamentum.whetstone.agents.lens import LensIssueProposal, LensReadOutput
from andamentum.whetstone.deps import ReviewDeps
from andamentum.whetstone.nodes.critical_read import (
    _build_document_context,
    _run_lens,
)
from andamentum.whetstone.schemas import SectionCard
from andamentum.whetstone.structural.types import SectionRef


@dataclass
class _FakeAgentResult:
    output: LensReadOutput


class _FakeAgent:
    """Minimal stand-in for a pydantic-ai Agent. Returns a fixed output."""

    def __init__(self, output: LensReadOutput):
        self._output = output
        self.last_prompt: str | None = None

    async def run(self, prompt: str):
        self.last_prompt = prompt
        return _FakeAgentResult(output=self._output)


def _section(*, id: str = "sec_001", title: str = "Methods", text: str | None = None) -> SectionRef:
    return SectionRef(
        id=id,
        title=title,
        text=text or "The quick brown fox jumps over the lazy dog.",
        char_start=0,
        char_end=len(text or "The quick brown fox jumps over the lazy dog."),
    )


def _proposal(**overrides) -> LensIssueProposal:
    defaults = dict(
        title="argument doesn't follow",
        severity="moderate",
        confidence="medium",
        rationale="The conclusion outpaces the data shown.",
        quote_text="",
        category="",
    )
    defaults.update(overrides)
    return LensIssueProposal(**defaults)  # type: ignore[arg-type]


# ── _run_lens — anchoring + Finding shape ───────────────────────────────


async def test_run_lens_returns_finding_with_lens_perspective() -> None:
    section = _section()
    output = LensReadOutput(issues=[_proposal()])
    fake = _FakeAgent(output)

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        findings = await _run_lens(deps=_deps(), section=section, lens="rigorous")

    assert len(findings) == 1
    f = findings[0]
    assert f.perspective == "rigorous"
    assert f.sections_involved == [section.id]
    assert f.source == "investigate"
    assert f.title == "argument doesn't follow"


async def test_run_lens_anchors_verbatim_quote() -> None:
    section = _section(text="The quick brown fox jumps over the lazy dog.")
    output = LensReadOutput(issues=[_proposal(quote_text="quick brown fox")])
    fake = _FakeAgent(output)

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        findings = await _run_lens(deps=_deps(), section=section, lens="writer")

    f = findings[0]
    assert len(f.quotes) == 1
    assert f.quotes[0].text == "quick brown fox"
    assert f.quotes[0].section_id == section.id


async def test_run_lens_drops_fabricated_quote_but_keeps_finding() -> None:
    """A fabricated quote leaves the finding intact but with no quote attached."""
    section = _section(text="alpha beta gamma")
    output = LensReadOutput(
        issues=[_proposal(quote_text="this text is not in the section at all")]
    )
    fake = _FakeAgent(output)

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        findings = await _run_lens(deps=_deps(), section=section, lens="rigorous")

    assert len(findings) == 1
    assert findings[0].quotes == []


async def test_run_lens_passes_through_category() -> None:
    section = _section()
    output = LensReadOutput(issues=[_proposal(category="evidence")])
    fake = _FakeAgent(output)

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        findings = await _run_lens(deps=_deps(), section=section, lens="rigorous")

    assert findings[0].category == "evidence"


async def test_run_lens_empty_output_returns_empty_list() -> None:
    section = _section()
    fake = _FakeAgent(LensReadOutput(issues=[]))

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        findings = await _run_lens(deps=_deps(), section=section, lens="rigorous")

    assert findings == []


async def test_run_lens_includes_section_title_and_id_in_prompt() -> None:
    section = _section(id="sec_007", title="Methods")
    fake = _FakeAgent(LensReadOutput(issues=[]))

    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        await _run_lens(deps=_deps(), section=section, lens="methodology")

    assert fake.last_prompt is not None
    assert "sec_007" in fake.last_prompt
    assert "Methods" in fake.last_prompt
    # And the prompt names the lens persona, so the system prompt aligns.
    assert "methodology reviewer" in fake.last_prompt.lower()


# ── document-context preamble ───────────────────────────────────────────


def _doc_map() -> list[SectionCard]:
    return [
        SectionCard(section_id="sec_001", title="Introduction", one_line_gist="We study X."),
        SectionCard(section_id="sec_002", title="Methods", one_line_gist="We did Y."),
        SectionCard(section_id="sec_003", title="Results", one_line_gist="We found Z."),
    ]


def test_build_document_context_lists_all_sections_and_marks_current() -> None:
    ctx = _build_document_context(_doc_map(), "sec_002")
    # Position + total
    assert "section 2 of 3" in ctx
    # Every section title appears (table of contents)
    assert "Introduction" in ctx and "Methods" in ctx and "Results" in ctx
    # The current section is marked
    assert "▶ Methods" in ctx
    # Scope discipline instruction present
    assert "missing from the document" in ctx.lower()


def test_build_document_context_empty_when_no_map() -> None:
    assert _build_document_context([], "sec_001") == ""


async def test_run_lens_injects_document_context_into_prompt() -> None:
    section = _section(id="sec_002", title="Methods")
    fake = _FakeAgent(LensReadOutput(issues=[]))
    ctx = _build_document_context(_doc_map(), "sec_002")
    with patch(
        "andamentum.whetstone.nodes.critical_read.build_pydantic_ai_agent",
        return_value=fake,
    ):
        await _run_lens(deps=_deps(), section=section, lens="rigorous", doc_context=ctx)
    assert fake.last_prompt is not None
    assert "section 2 of 3" in fake.last_prompt
    assert "missing from the document" in fake.last_prompt.lower()


# ── Plumbing ────────────────────────────────────────────────────────────


def _deps() -> ReviewDeps:
    """Minimal ReviewDeps for tests — only `model` is consulted."""
    return ReviewDeps(model="stub-model")
