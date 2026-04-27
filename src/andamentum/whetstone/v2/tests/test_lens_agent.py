"""Tests for the lens agent definitions and registry integration."""

from __future__ import annotations

import pytest

from andamentum.whetstone.v2.agents import (
    LensIssueProposal,
    LensReadOutput,
    build_lens_agent_definition,
    get_agent,
    list_available_lenses,
)


# ── Builder ─────────────────────────────────────────────────────────────


def test_list_available_lenses_includes_canonical_set() -> None:
    available = list_available_lenses()
    # The four ports from v1. Order is stable (sorted).
    assert "rigorous" in available
    assert "writer" in available
    assert "methodology" in available
    assert "statistician" in available


def test_known_lenses_build() -> None:
    for name in list_available_lenses():
        agent = build_lens_agent_definition(name)
        assert agent.name == f"lens.{name}"
        assert agent.prompt, f"lens {name!r} has empty prompt"
        assert agent.output_model is LensReadOutput


def test_unknown_lens_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="unknown lens"):
        build_lens_agent_definition("not-a-real-lens")


# ── Registry integration ────────────────────────────────────────────────


def test_lens_agents_are_in_the_registry() -> None:
    """Every lens registered at module import should be retrievable."""
    for name in list_available_lenses():
        agent = get_agent(f"lens.{name}")
        assert agent.name == f"lens.{name}"


# ── Output schema sanity ────────────────────────────────────────────────


def test_lens_issue_proposal_minimal_construction() -> None:
    p = LensIssueProposal(
        title="argument doesn't follow",
        severity="moderate",
        confidence="high",
        rationale="Section 4 concludes X but the only evidence shown is for Y.",
    )
    assert p.quote_text == ""
    assert p.category == ""


def test_lens_issue_proposal_with_optional_fields() -> None:
    p = LensIssueProposal(
        title="x",
        severity="major",
        confidence="medium",
        rationale="...",
        quote_text="some verbatim chunk",
        category="evidence",
    )
    assert p.quote_text == "some verbatim chunk"
    assert p.category == "evidence"


def test_lens_read_output_default_empty() -> None:
    out = LensReadOutput()
    assert out.issues == []


def test_output_trailer_in_every_lens_prompt() -> None:
    """Sanity check: the universal trailer is appended to every persona."""
    for name in list_available_lenses():
        agent = build_lens_agent_definition(name)
        assert "Output instructions" in agent.prompt, f"lens {name!r} missing trailer"
        assert "VERBATIM span" in agent.prompt, f"lens {name!r} missing quote rule"
