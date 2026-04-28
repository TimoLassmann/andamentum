"""Tests for the consistency lens (Step 5 — multi-section reading).

The consistency lens is the first v2 lens whose job is inherently
cross-section: it sees the WHOLE document at once, picks issues that
span multiple sections, and the controller anchors quotes by searching
across every section rather than just one.
"""

from __future__ import annotations

from unittest import mock

from andamentum.whetstone.v2.agents.lens_prompts import (
    LENS_MULTI_SECTION,
    LENS_PROMPTS,
)
from andamentum.whetstone.v2.structural.types import SectionRef


def _section(id_: str, title: str, text: str) -> SectionRef:
    return SectionRef(id=id_, title=title, text=text, char_start=0, char_end=len(text))


# ── Registry ───────────────────────────────────────────────────────────


def test_consistency_registered_as_lens():
    assert "consistency" in LENS_PROMPTS


def test_consistency_marked_multi_section():
    assert LENS_MULTI_SECTION["consistency"] is True


def test_other_lenses_default_to_per_section():
    for name in ("rigorous", "writer", "methodology", "statistician"):
        assert LENS_MULTI_SECTION.get(name, False) is False


def test_consistency_prompt_mentions_cross_section_focus():
    prompt = LENS_PROMPTS["consistency"]
    # Prompt should hammer the cross-section discipline because the
    # default-trailer single-section discipline doesn't apply.
    assert "WHOLE document" in prompt
    assert "2+ sections" in prompt
    # Prompt should warn against duplicating deterministic-substrate work
    assert "deterministic" in prompt.lower() or "scanner" in prompt.lower()


# ── Routing through CriticalRead ───────────────────────────────────────


async def test_critical_read_routes_consistency_through_multi_section_runner():
    """The CriticalRead node makes ONE call for the consistency lens
    (across the whole doc) and N calls for a per-section lens."""

    from andamentum.whetstone.v2 import nodes
    from andamentum.whetstone.v2.deps import ReviewDeps
    from andamentum.whetstone.v2.nodes.critical_read import CriticalRead
    from andamentum.whetstone.v2.schemas import Finding
    from andamentum.whetstone.v2.state import ReviewState

    sections = [
        _section("sec_001", "Abstract", "We had n=50 participants."),
        _section("sec_002", "Methods", "We had n=48 participants."),
    ]

    # Track which runner gets called and how many times.
    per_section_calls: list[tuple[str, str]] = []
    multi_section_calls: list[str] = []

    async def fake_run_lens(deps, section, lens):  # noqa: ARG001
        per_section_calls.append((section.id, lens))
        return []

    async def fake_run_multi(deps, sections, lens):  # noqa: ARG001
        multi_section_calls.append(lens)
        return [
            Finding(
                title="N disagreement across abstract/methods",
                severity="major",
                confidence="high",
                rationale="Abstract says n=50; methods says n=48.",
                quotes=[],
                sections_involved=["sec_001", "sec_002"],
                source="investigate",
                perspective=lens,
                category="consistency",
            )
        ]

    state = ReviewState(
        source="(test)",
        sections=sections,
        perspectives=["consistency", "rigorous"],
    )
    deps = ReviewDeps(model="fake:test")

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    cr_mod = nodes.critical_read
    with mock.patch.object(cr_mod, "_run_lens", fake_run_lens), mock.patch.object(
        cr_mod, "_run_multi_section_lens", fake_run_multi
    ):
        await CriticalRead().run(ctx)  # type: ignore[arg-type]

    # Per-section lens: 2 sections × 1 lens (rigorous) = 2 calls
    assert len(per_section_calls) == 2
    assert all(call[1] == "rigorous" for call in per_section_calls)

    # Multi-section lens: 1 call total for consistency
    assert multi_section_calls == ["consistency"]

    # The cross-section finding made it into the pool
    consistency_findings = [
        f for f in state.findings if f.perspective == "consistency"
    ]
    assert len(consistency_findings) == 1
    assert consistency_findings[0].category == "consistency"


async def test_multi_section_lens_anchors_quote_across_sections():
    """When the lens emits a quote_text that lives in section B, the
    Finding's anchored quote points at section B (not section A)."""
    from andamentum.whetstone.v2.deps import ReviewDeps
    from andamentum.whetstone.v2.nodes.critical_read import _run_multi_section_lens
    from andamentum.whetstone.v2.agents.lens import LensIssueProposal, LensReadOutput

    sections = [
        _section("sec_001", "Abstract", "Completely different prose without keywords."),
        _section(
            "sec_002",
            "Methods",
            "The xyzzy42 sentinel phrase appears here exactly once.",
        ),
    ]

    fake_output = LensReadOutput(
        issues=[
            LensIssueProposal(
                title="x",
                severity="moderate",
                confidence="medium",
                rationale="span lives in section B",
                quote_text="The xyzzy42 sentinel phrase",
                category="consistency",
            )
        ]
    )

    class FakeAgent:
        async def run(self, _prompt):
            class _R:
                output = fake_output

            return _R()

    from andamentum.whetstone.v2.nodes import critical_read as cr_mod

    with mock.patch.object(
        cr_mod, "build_pydantic_ai_agent", lambda *_a, **_k: FakeAgent()
    ):
        findings = await _run_multi_section_lens(
            ReviewDeps(model="fake:test"), sections, "consistency"
        )

    assert len(findings) == 1
    f = findings[0]
    assert f.sections_involved == ["sec_002"]
    assert len(f.quotes) == 1
    assert f.quotes[0].section_id == "sec_002"


async def test_multi_section_lens_drops_unanchored_quote_but_keeps_finding():
    """If the quote_text doesn't appear in ANY section, the finding still
    surfaces (with empty quotes/sections_involved)."""
    from andamentum.whetstone.v2.deps import ReviewDeps
    from andamentum.whetstone.v2.nodes.critical_read import _run_multi_section_lens
    from andamentum.whetstone.v2.agents.lens import LensIssueProposal, LensReadOutput

    sections = [
        _section("sec_001", "Abstract", "Real text A."),
        _section("sec_002", "Methods", "Real text B."),
    ]

    fake_output = LensReadOutput(
        issues=[
            LensIssueProposal(
                title="hallucinated-quote case",
                severity="moderate",
                confidence="low",
                rationale="The lens rationale stands even if the quote doesn't anchor.",
                quote_text="this exact phrase appears nowhere",
                category="consistency",
            )
        ]
    )

    class FakeAgent:
        async def run(self, _prompt):
            class _R:
                output = fake_output

            return _R()

    from andamentum.whetstone.v2.nodes import critical_read as cr_mod

    with mock.patch.object(
        cr_mod, "build_pydantic_ai_agent", lambda *_a, **_k: FakeAgent()
    ):
        findings = await _run_multi_section_lens(
            ReviewDeps(model="fake:test"), sections, "consistency"
        )

    assert len(findings) == 1
    f = findings[0]
    assert f.quotes == []
    assert f.sections_involved == []
    assert f.title == "hallucinated-quote case"


async def test_multi_section_lens_defaults_category_to_consistency():
    """When the lens omits a category, the runner fills in 'consistency'
    so downstream renderers can group these findings cleanly."""
    from andamentum.whetstone.v2.deps import ReviewDeps
    from andamentum.whetstone.v2.nodes.critical_read import _run_multi_section_lens
    from andamentum.whetstone.v2.agents.lens import LensIssueProposal, LensReadOutput

    sections = [_section("sec_001", "x", "Some text.")]

    fake_output = LensReadOutput(
        issues=[
            LensIssueProposal(
                title="t",
                severity="minor",
                confidence="low",
                rationale="r",
                quote_text="",
                category="",  # lens leaves it blank
            )
        ]
    )

    class FakeAgent:
        async def run(self, _prompt):
            class _R:
                output = fake_output

            return _R()

    from andamentum.whetstone.v2.nodes import critical_read as cr_mod

    with mock.patch.object(
        cr_mod, "build_pydantic_ai_agent", lambda *_a, **_k: FakeAgent()
    ):
        findings = await _run_multi_section_lens(
            ReviewDeps(model="fake:test"), sections, "consistency"
        )

    assert findings[0].category == "consistency"
