"""Tests for the claim-evidence anchoring lens (Step 9).

Three surfaces:

1. Section-kind classifier maps headings to a coarse IMRAD enum.
2. Lens registration: the claim_evidence lens is per-section and
   targets only Abstract / Results / Discussion / Conclusion.
3. CriticalRead's section-targeting filter routes lenses with target
   restrictions to only the matching sections.
"""

from __future__ import annotations

from unittest import mock

from andamentum.whetstone.agents.lens_prompts import (
    LENS_MULTI_SECTION,
    LENS_PROMPTS,
    LENS_TARGET_SECTIONS,
)
from andamentum.whetstone.agents.section_kinds import classify_section_kind
from andamentum.whetstone.structural.types import SectionRef


def _section(id_: str, title: str) -> SectionRef:
    text = f"## {title}\n\nbody"
    return SectionRef(id=id_, title=title, text=text, char_start=0, char_end=len(text))


# ── Section-kind classifier ────────────────────────────────────────────


def test_classify_abstract():
    assert classify_section_kind("Abstract") == "abstract"
    assert classify_section_kind("# Abstract") == "abstract"
    assert classify_section_kind("ABSTRACT") == "abstract"
    assert classify_section_kind("Summary") == "abstract"


def test_classify_introduction():
    assert classify_section_kind("Introduction") == "introduction"
    assert classify_section_kind("1. Introduction") == "introduction"
    assert classify_section_kind("Background") == "introduction"
    assert classify_section_kind("Motivation") == "introduction"


def test_classify_methods():
    for title in (
        "Methods",
        "Method",
        "Materials and Methods",
        "Experimental",
        "Methodology",
        "Procedure",
        "2. Methods",
    ):
        assert classify_section_kind(title) == "methods", title


def test_classify_results():
    assert classify_section_kind("Results") == "results"
    assert classify_section_kind("Findings") == "results"
    assert classify_section_kind("Observations") == "results"
    assert classify_section_kind("3. Results") == "results"


def test_classify_discussion():
    assert classify_section_kind("Discussion") == "discussion"
    assert classify_section_kind("Interpretation") == "discussion"


def test_classify_conclusion():
    assert classify_section_kind("Conclusion") == "conclusion"
    assert classify_section_kind("Conclusions") == "conclusion"
    assert classify_section_kind("Concluding remarks") == "conclusion"


def test_classify_references():
    assert classify_section_kind("References") == "references"
    assert classify_section_kind("Bibliography") == "references"
    assert classify_section_kind("Works Cited") == "references"


def test_classify_other_for_unrecognised():
    assert classify_section_kind("Acknowledgements") == "other"
    assert classify_section_kind("Conflict of interest") == "other"
    assert classify_section_kind("Author contributions") == "other"
    assert classify_section_kind("Untitled") == "other"


def test_classify_handles_numeric_prefixes():
    assert classify_section_kind("4.1 Methods of statistical analysis") == "methods"
    assert classify_section_kind("2 Background and prior work") == "introduction"


# ── Lens registration ──────────────────────────────────────────────────


def test_claim_evidence_registered_as_lens():
    assert "claim_evidence" in LENS_PROMPTS


def test_claim_evidence_is_per_section_not_multi():
    assert LENS_MULTI_SECTION.get("claim_evidence", False) is False


def test_claim_evidence_targets_specific_sections():
    targets = LENS_TARGET_SECTIONS["claim_evidence"]
    assert "abstract" in targets
    assert "results" in targets
    assert "discussion" in targets
    assert "conclusion" in targets
    # And must NOT include sections where the lens has nothing to say:
    assert "methods" not in targets
    assert "references" not in targets
    assert "introduction" not in targets


def test_claim_evidence_prompt_mentions_anchoring():
    prompt = LENS_PROMPTS["claim_evidence"]
    assert "anchor" in prompt.lower()
    assert "figure" in prompt.lower() or "table" in prompt.lower()


def test_claim_evidence_prompt_carves_out_unrelated_claims():
    """Prompt should explicitly say methodological/background claims
    are out of scope so the lens doesn't spam those sections."""
    prompt = LENS_PROMPTS["claim_evidence"].lower()
    assert "method" in prompt or "background" in prompt


def test_other_lenses_have_no_target_restriction():
    """The default lenses run everywhere; only claim_evidence is
    section-restricted today."""
    assert "rigorous" not in LENS_TARGET_SECTIONS
    assert "writer" not in LENS_TARGET_SECTIONS
    assert "methodology" not in LENS_TARGET_SECTIONS
    assert "statistician" not in LENS_TARGET_SECTIONS
    assert "overclaim" not in LENS_TARGET_SECTIONS


# ── CriticalRead section-target filter ─────────────────────────────────


async def test_critical_read_skips_off_target_sections_for_targeted_lens():
    """When claim_evidence is in the perspective list, it runs only
    against Results/Discussion/Abstract/Conclusion sections — not
    Methods or References."""
    from andamentum.whetstone import nodes
    from andamentum.whetstone.deps import ReviewDeps
    from andamentum.whetstone.nodes.critical_read import CriticalRead
    from andamentum.whetstone.state import ReviewState

    sections = [
        _section("sec_001", "Abstract"),
        _section("sec_002", "Methods"),
        _section("sec_003", "Results"),
        _section("sec_004", "References"),
    ]

    per_section_calls: list[tuple[str, str]] = []

    async def fake_run_lens(deps, section, lens):  # noqa: ARG001
        per_section_calls.append((section.id, lens))
        return []

    state = ReviewState(
        source="(test)",
        sections=sections,
        perspectives=["claim_evidence"],
    )
    deps = ReviewDeps(model="fake:test")

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    cr_mod = nodes.critical_read
    with mock.patch.object(cr_mod, "_run_lens", fake_run_lens):
        await CriticalRead().run(ctx)  # type: ignore[arg-type]

    # Only Abstract and Results should have been visited; Methods and
    # References are off-target for claim_evidence.
    visited = {section_id for section_id, _ in per_section_calls}
    assert visited == {"sec_001", "sec_003"}


async def test_critical_read_runs_all_sections_for_unrestricted_lens():
    """A lens with no target restriction runs against every section."""
    from andamentum.whetstone import nodes
    from andamentum.whetstone.deps import ReviewDeps
    from andamentum.whetstone.nodes.critical_read import CriticalRead
    from andamentum.whetstone.state import ReviewState

    sections = [
        _section("sec_001", "Abstract"),
        _section("sec_002", "Methods"),
        _section("sec_003", "Results"),
        _section("sec_004", "References"),
    ]

    per_section_calls: list[tuple[str, str]] = []

    async def fake_run_lens(deps, section, lens):  # noqa: ARG001
        per_section_calls.append((section.id, lens))
        return []

    state = ReviewState(
        source="(test)",
        sections=sections,
        perspectives=["rigorous"],
    )
    deps = ReviewDeps(model="fake:test")

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    cr_mod = nodes.critical_read
    with mock.patch.object(cr_mod, "_run_lens", fake_run_lens):
        await CriticalRead().run(ctx)  # type: ignore[arg-type]

    visited = {section_id for section_id, _ in per_section_calls}
    assert visited == {"sec_001", "sec_002", "sec_003", "sec_004"}
