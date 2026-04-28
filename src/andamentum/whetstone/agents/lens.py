"""Lens agent — one configured reviewer personality reading one section.

A lens is one pydantic-ai agent with a tailored system prompt. The
underlying model is the one passed via ``--model``; only the prompt
changes per lens. Multiple lenses can run against the same section in
parallel — each produces its own list of issues from its own viewpoint.

Output schema is intentionally flat (six fields) so small local models
fill it reliably. The ``CriticalRead`` node converts each
``LensIssueProposal`` into a fully-formed ``Finding`` (anchoring the
quote, attaching the lens name, etc.).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition
from .lens_prompts import LENS_PROMPTS


# ── Output schema ───────────────────────────────────────────────────────


class LensIssueProposal(BaseModel):
    """One issue raised by a lens about the section it just read.

    Six flat fields — no nested structures, no lens-specific extras.
    Small models reliably fill schemas of this shape.
    """

    title: str = Field(description="≤80 chars, like a commit message")
    severity: Literal["minor", "moderate", "major"] = Field(
        description=(
            "How serious is this issue? "
            "minor = cosmetic / nice-to-have. "
            "moderate = real but local issue. "
            "major = load-bearing — undermines a section's claim or "
            "argument."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "How sure are you? "
            "low = judgement call, could go either way. "
            "medium = clearly an issue but reasonable people might disagree. "
            "high = unambiguous — verifiable from the section text."
        ),
    )
    rationale: str = Field(
        description=(
            "Explain what the issue is and why it matters. "
            "Maximum 3 sentences."
        ),
    )
    quote_text: str = Field(
        default="",
        description="One verbatim span from the section text (≤200 chars). Optional.",
    )
    category: str = Field(
        default="",
        description=(
            "Short tag picked from: evidence, methodology, argument-flow, "
            "framing, consistency, data-quality, scope. Optional."
        ),
    )


class LensReadOutput(BaseModel):
    """The lens's full output for one section read."""

    issues: list[LensIssueProposal] = Field(
        default_factory=list,
        description=(
            "0–3 issues for this section. Quality over quantity — only "
            "include issues that are concrete and load-bearing."
        ),
    )


# ── Builder ─────────────────────────────────────────────────────────────


def build_lens_agent_definition(lens_name: str) -> AgentDefinition:
    """Construct an ``AgentDefinition`` for the named lens.

    Raises ``ValueError`` if the lens name isn't registered. Use
    ``list_available_lenses()`` to see what's available.
    """
    if lens_name not in LENS_PROMPTS:
        raise ValueError(
            f"unknown lens: {lens_name!r}. "
            f"Available: {', '.join(sorted(LENS_PROMPTS))}"
        )
    return AgentDefinition(
        name=f"lens.{lens_name}",
        prompt=LENS_PROMPTS[lens_name],
        output_model=LensReadOutput,
        retries=2,
        output_retries=2,
    )


def list_available_lenses() -> list[str]:
    """All lens names this build supports, sorted."""
    return sorted(LENS_PROMPTS)
