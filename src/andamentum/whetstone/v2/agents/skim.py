"""Skim agent: read DocumentMap + abstract + conclusion → emit hypotheses.

Output schema is intentionally tiny so small local models fill it
reliably:
  • SkimSection: just (section_id, one_line_gist) — the agent enriches
    each section's gist beyond the deterministic first-sentence default.
  • SkimHypothesis: (text, priority, relevant_section_ids).

The prompt avoids any branching logic — the agent does ONE thing: read
the inputs, emit two flat lists.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

SKIM_PROMPT = """You are reading a document's structural skeleton (sections, abstract, conclusion).

Your job has two parts:

1. ENRICH each section's one-line gist. The user will see this in their
   review interface. Each gist must be at most 25 words and capture WHAT
   THE SECTION DOES (e.g., "Introduces POET, the paired open-ended
   trailblazer algorithm, and contrasts it with prior open-ended methods").

2. EMIT 5–15 HYPOTHESES — concrete, testable questions to investigate
   later. Each hypothesis should be specific enough that an investigator
   reading 1–3 sections could answer it. Examples of GOOD hypotheses:
     • "Does Section 4 actually demonstrate the open-endedness claimed in the abstract?"
     • "Is the methodology in Section 3 consistent with the experimental setup in Section 4?"
     • "Does the conclusion address the research questions posed in the introduction?"
   Examples of BAD hypotheses:
     • "Is the paper good?" (too vague)
     • "Read the entire paper carefully." (not testable)

For each hypothesis, supply:
  • a short text question
  • priority: "high" (core to the paper's claims), "medium" (notable),
    or "low" (peripheral)
  • relevant_section_ids: the section_ids the investigator should read
    to answer the hypothesis (typically 1–3, never more than 5)

Output a SkimOutput with `enriched_sections` (one per section_id you
were given) and `hypotheses` (5–15)."""


class SkimSection(BaseModel):
    """One section's enriched gist (replaces the deterministic first-sentence)."""

    section_id: str
    one_line_gist: str = Field(
        description="What this section DOES, in ≤25 words. Not a summary — a function."
    )


class SkimHypothesis(BaseModel):
    """One question to investigate. Goes onto the InvestigateLoop queue."""

    text: str = Field(description="The question to investigate, one sentence.")
    priority: Literal["low", "medium", "high"] = "medium"
    relevant_section_ids: list[str] = Field(
        description="Sections an investigator should read to resolve this. 1–5 ids."
    )


class SkimOutput(BaseModel):
    """skim_agent's flat output: enriched map + initial hypotheses queue."""

    enriched_sections: list[SkimSection] = Field(default_factory=list)
    hypotheses: list[SkimHypothesis] = Field(default_factory=list)


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="skim",
        prompt=SKIM_PROMPT,
        output_model=SkimOutput,
        retries=2,
        output_retries=2,
    )


SKIM_AGENT = _build()
