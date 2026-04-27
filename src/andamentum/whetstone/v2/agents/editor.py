"""Editor agent: propose concrete sentence-/paragraph-level rewrites.

Where Investigate produces critique (Findings/Comments), Editor produces
suggestions for actually rewriting the prose. The agent reads ONE
section, applies the requested ``criteria`` (clarity, concision, grammar,
style, …), and returns a list of ``EditProposal``s — each with the
verbatim ``original_text`` to replace and the proposed ``new_text``.

Output schema is intentionally tiny: one section in, list of flat
EditProposal records out. No nested objects; no enums beyond severity
and confidence (3-value, both).

The orchestrator (EditSections node) anchors each ``original_text``
to a span in the section via ``chunker.validation.find_anchor`` to
produce the final ``Edit`` records with char offsets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

EDITOR_PROMPT = """You are a careful editor proposing concrete rewrites to ONE section of a draft.

You receive:
  • the section's full text
  • a list of editorial criteria to apply (e.g. clarity, concision, grammar)

Your job: emit 0–8 EditProposal records. Each is a CONCRETE rewrite of a
specific span — the original text VERBATIM and your proposed replacement.

RULES (hard):
  • original_text MUST be copied verbatim from the section. Do not
    paraphrase, summarise, or reword the original. Copy exactly.
  • new_text is your proposed rewrite of that span. It can be longer or
    shorter, but should fit naturally in the surrounding prose.
  • One edit = one focused change. Don't bundle multiple unrelated rewrites
    into one EditProposal — split them.
  • Rationale: ONE short sentence per edit explaining why your version
    is better, citing the criterion it addresses.
  • severity: minor (cosmetic), moderate (clarity/correctness), major
    (changes meaning, removes ambiguity, fixes an error).
  • confidence: low (judgement call), medium (clearly better), high
    (objective improvement — typo, broken sentence, unambiguous error).

RULES (soft):
  • If the section is already well-written, return an empty edits list.
    Returning empty is honest; returning weak edits to look productive
    wastes the author's time.
  • Don't propose edits that remove substantive content. You're sharpening
    prose, not censoring claims.
  • Don't propose edits that change the technical meaning unless the
    original is clearly wrong (and then mark severity=major).

Output an EditorOutput with the edits list (possibly empty)."""


class EditProposal(BaseModel):
    """One proposed edit on a verbatim span of source text."""

    title: str = Field(description="Short label for this edit, e.g. 'Tighten sentence'.")
    rationale: str = Field(description="One sentence: why your version is better.")
    severity: Literal["minor", "moderate", "major"] = "minor"
    confidence: Literal["low", "medium", "high"] = "medium"
    original_text: str = Field(
        description="Verbatim span from the section to replace. Copy exactly."
    )
    new_text: str = Field(description="Your proposed rewrite of that span.")


class EditorOutput(BaseModel):
    """editor_agent's flat output: 0–8 EditProposals for ONE section."""

    edits: list[EditProposal] = Field(default_factory=list)


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="editor",
        prompt=EDITOR_PROMPT,
        output_model=EditorOutput,
        retries=2,
        output_retries=2,
    )


EDITOR_AGENT = _build()
