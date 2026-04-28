"""Synthesise agent: write the final review prose.

Reads the surviving findings (post-Challenge) plus the DocumentMap and
produces a small, flat report. Output is intentionally plain prose
fields — not nested structures — so small models reliably fill them.

v2 buckets findings by ``priority`` (MUST FIX / SHOULD FIX / CONSIDER)
rather than the raw severity enum. Priority is derived from severity by
default (see ``schemas.Finding`` validator) but can be overridden by
reflection or other downstream steps.
"""

from __future__ import annotations


from pydantic import BaseModel, Field

from ._definition import AgentDefinition

SYNTHESISE_PROMPT = """You are writing the final review of a document.

You have:
  • the document map (titles + one-line gists per section)
  • the list of FINDINGS the investigators produced (post-challenge)
  • each finding has severity (minor/moderate/major), confidence
    (low/medium/high), and priority (must_fix/should_fix/consider)

Your job is to produce a ReviewSummary with FOUR fields, organised by
PRIORITY (which is what the author needs to act on) rather than
severity (which is the inherent seriousness of the issue):

  • executive_summary (2 paragraphs):
      A reader-facing summary of the document's strengths and weaknesses,
      grounded in the findings. Lead with the most important issues.
      Be honest — if the findings are mostly minor, say so.

  • must_fix_summary (1 paragraph):
      Walk through the MUST FIX findings in priority order. These are
      issues the author must address before submission. Reference each
      by its title and section_ids in parentheses. If there are none,
      say "No must-fix findings — the manuscript is submission-ready
      on the basics."

  • should_fix_summary (1 paragraph):
      Same for SHOULD FIX findings. These are improvements the author
      should make if time permits. Group thematically if helpful.

  • consider_summary (1 paragraph):
      Same for CONSIDER findings. These are suggestions and minor
      polish items. May be terse — bullet-style narrative is fine.

If a category has zero findings, say so in one sentence. Don't invent
new findings — only summarise what's been investigated. Don't moralise
or pad — the author wants the signal, not platitudes."""


class ReviewSummary(BaseModel):
    """synthesise_agent's flat output. All fields are plain prose."""

    executive_summary: str = Field(
        description="2 paragraphs of reader-facing summary."
    )
    must_fix_summary: str = Field(
        default="No must-fix findings.",
        description=(
            "1 paragraph walking the must-fix findings — issues the "
            "author must address before submission."
        ),
    )
    should_fix_summary: str = Field(
        default="No should-fix findings.",
        description=(
            "1 paragraph for should-fix findings — improvements to make "
            "if time permits."
        ),
    )
    consider_summary: str = Field(
        default="No consider findings.",
        description=(
            "1 paragraph for consider findings — suggestions and minor "
            "polish items."
        ),
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
