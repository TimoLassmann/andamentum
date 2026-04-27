"""Investigate agent: resolve one hypothesis using 1–3 sections of context.

The agent emits one of three decisions via a flat schema (no tagged
unions — small models handle them poorly). The orchestrator then maps
the decision to one of (Finding, Unfounded, sub-hypotheses).

Quote anchoring is best-effort: the agent emits verbatim quote strings,
and ``_locate_quotes`` (in nodes/investigate.py) finds them via
``chunker.validation.find_anchor`` against the cited sections. Quotes
that can't be located are dropped — the rest of the finding survives.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

INVESTIGATE_PROMPT = """You are investigating ONE hypothesis about a document.

You have:
  • the hypothesis text
  • the document's section map (titles + one-line gists)
  • the FULL TEXT of 1–3 sections most relevant to the hypothesis

Your job: decide what's true and emit ONE of three outcomes.

DECISIONS:

  decision = "finding"
    → You found a real issue. Fill in finding_title, finding_severity
      (minor/moderate/major), finding_confidence (low/medium/high),
      finding_rationale (2–3 sentences), and finding_quotes (1–4 verbatim
      passages from the section text, EACH copied EXACTLY — no rewrites).
      finding_sections lists the section_ids the issue concerns.

  decision = "unfounded"
    → You looked, and the hypothesis is not supported by the document.
      Fill in unfounded_reason (one sentence: what you found instead).

  decision = "needs_subhypotheses"
    → The hypothesis is too broad to answer directly. Break it into 2–4
      narrower sub-questions in sub_hypothesis_texts. Each must be more
      specific than the original.

CRITICAL RULES:
  • finding_quotes must be VERBATIM — copy exactly, do not paraphrase.
  • Default to "unfounded" or "low" confidence when uncertain. The
    Challenge phase will catch over-confident findings, but it cannot
    rescue under-confident ones.
  • Only set decision = "finding" if you have at least one verbatim quote
    that supports the issue.

Return an InvestigationOutput with the appropriate fields filled in. The
unused fields will be ignored."""


class InvestigationOutput(BaseModel):
    """investigate_agent's flat output. Only the fields for the chosen
    `decision` are read by the orchestrator; the others can be empty."""

    decision: Literal["finding", "unfounded", "needs_subhypotheses"] = Field(
        description="What you concluded about the hypothesis."
    )

    # ── decision == "finding" ────────────────────────────────────────
    finding_title: str = ""
    finding_severity: Literal["minor", "moderate", "major"] = "minor"
    finding_confidence: Literal["low", "medium", "high"] = "low"
    finding_rationale: str = ""
    finding_quotes: list[str] = Field(default_factory=list)
    finding_sections: list[str] = Field(default_factory=list)

    # ── decision == "unfounded" ──────────────────────────────────────
    unfounded_reason: str = ""

    # ── decision == "needs_subhypotheses" ────────────────────────────
    sub_hypothesis_texts: list[str] = Field(default_factory=list)


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="investigate",
        prompt=INVESTIGATE_PROMPT,
        output_model=InvestigationOutput,
        retries=2,
        output_retries=2,
    )


INVESTIGATE_AGENT = _build()
