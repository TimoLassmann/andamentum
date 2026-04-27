"""Synthesise agent: write the final review prose.

Reads the surviving findings (post-Challenge) plus the DocumentMap and
produces a small, flat report. Output is intentionally plain prose
fields — not nested structures — so small models reliably fill them.
"""

from __future__ import annotations


from pydantic import BaseModel, Field

from ._definition import AgentDefinition

SYNTHESISE_PROMPT = """You are writing the final review of a document.

You have:
  • the document map (titles + one-line gists per section)
  • the list of FINDINGS the investigators produced (post-challenge)
  • each finding has severity (minor/moderate/major) and confidence

Your job is to produce a ReviewSummary with FOUR fields:

  • executive_summary (2 paragraphs):
      A reader-facing summary of the document's strengths and weaknesses,
      grounded in the findings. Lead with the most important issues.
      Be honest — if the findings are mostly minor, say so.

  • major_findings_summary (1 paragraph):
      Walk through the major findings in priority order. Reference them
      by their titles. Note section_ids in parentheses.

  • moderate_findings_summary (1 paragraph):
      Same for moderate findings. Group thematically if helpful.

  • minor_findings_summary (1 paragraph):
      Same for minor findings. May be terse.

If a category has zero findings, say so in one sentence ("No major
findings."). Don't invent new findings — only summarise what's been
investigated."""


class ReviewSummary(BaseModel):
    """synthesise_agent's flat output. All fields are plain prose."""

    executive_summary: str = Field(
        description="2 paragraphs of reader-facing summary."
    )
    major_findings_summary: str = Field(
        default="No major findings.",
        description="1 paragraph walking the major findings.",
    )
    moderate_findings_summary: str = Field(
        default="No moderate findings.",
        description="1 paragraph for moderate findings.",
    )
    minor_findings_summary: str = Field(
        default="No minor findings.",
        description="1 paragraph for minor findings.",
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="synthesise",
        prompt=SYNTHESISE_PROMPT,
        output_model=ReviewSummary,
        retries=2,
        output_retries=2,
    )


SYNTHESISE_AGENT = _build()
