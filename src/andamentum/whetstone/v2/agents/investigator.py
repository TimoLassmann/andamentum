"""Investigator agent — re-reads source for one reflection task.

One LLM call per task. Receives:
  • the senior reviewer's task description (plain text),
  • the full original text of the named section(s),
  • the current notes the task is about (presented as observations to
    verify, NOT as facts).

Returns:
  • per-note ``NoteUpdate``s (action: keep / refine / drop),
  • optional ``NewNote``s (issues the investigator is raising).

Every quote — refined or new — must be verbatim from one of the section
texts that was fed in. The controller (``ReflectAndInvestigate``)
performs an anchor check after the call returns; quotes that don't
match are silently dropped (refinements rejected, new notes discarded).

The investigator never sees:
  • prior conclusions from earlier rounds,
  • the senior reviewer's reflection prompt,
  • other investigators' outputs.

It only sees source text and current note state. This is the discipline
that prevents the loop from drifting into AI-talking-about-AI.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ._definition import AgentDefinition


# ── Output schema ───────────────────────────────────────────────────────


class NoteUpdate(BaseModel):
    """Outcome for one note that was passed into the investigator.

    ``action == "keep"`` is the no-op; the note stays as written.
    ``action == "refine"`` requires every refined_* field to be filled,
        and ``refined_quote_text`` MUST be verbatim from a fed section.
    ``action == "drop"`` removes the note from the pool.
    """

    note_id: str
    action: Literal["keep", "refine", "drop"]

    # Required iff action == "refine":
    refined_title: str = ""
    refined_severity: Optional[Literal["minor", "moderate", "major"]] = None
    refined_confidence: Optional[Literal["low", "medium", "high"]] = None
    refined_rationale: str = Field(
        default="",
        description=(
            "Rewritten rationale for the refined note. "
            "Maximum 3 sentences."
        ),
    )
    refined_quote_text: str = ""
    refined_quote_section_id: str = ""


class NewNote(BaseModel):
    """An issue the investigator is raising that wasn't in the input pool.

    Every field is required. ``quote_text`` MUST be verbatim from one of
    the fed sections; ``quote_section_id`` says which fed section the
    quote came from. Anchor verification rejects fabricated quotes.
    """

    title: str
    severity: Literal["minor", "moderate", "major"]
    confidence: Literal["low", "medium", "high"]
    rationale: str = Field(
        description=(
            "Explain what the issue is and why it matters. "
            "Maximum 3 sentences."
        ),
    )
    quote_text: str
    quote_section_id: str
    category: str = ""


class InvestigatorOutput(BaseModel):
    """Outcomes for the input notes plus any newly raised notes."""

    updates: list[NoteUpdate] = Field(default_factory=list)
    new_notes: list[NewNote] = Field(default_factory=list)


# ── System prompt ───────────────────────────────────────────────────────


_INVESTIGATOR_PROMPT = """\
# Investigator

A senior reviewer has handed you ONE specific task. Your job:

  1. Read the section text(s) provided below in full. They are the
     ONLY evidence available to you.
  2. Look at the existing notes you've been given. They are starting
     observations written by junior reviewers — NOT facts. Verify
     each against the source text.
  3. Decide outcomes.

## Outcomes for existing notes

For each note you were given, return one ``NoteUpdate`` with one of:

  • **keep** — the note stands as written. Nothing more to fill in.
  • **refine** — the note is partly right; rewrite it more accurately.
    You MUST fill in: ``refined_title``, ``refined_severity``,
    ``refined_confidence``, ``refined_rationale``, ``refined_quote_text``
    (VERBATIM from a fed section), and ``refined_quote_section_id``.
  • **drop** — the note is wrong on careful re-reading. Remove it.

## New notes

You may also raise NEW notes for issues you noticed that the juniors
missed. Every new note MUST include:
  • title, severity, confidence, rationale (as for any note),
  • ``quote_text`` — VERBATIM from one of the section texts below,
  • ``quote_section_id`` — which fed section the quote is from,
  • optionally a category tag.

## Hard rules

1. Every quote — refined or new — MUST appear verbatim in one of the
   section texts shown to you below. Anything you fabricate will be
   silently dropped, and any refinement with a fabricated quote will be
   rejected entirely (the original note stays unchanged).

2. Do NOT analyse the analysis. Your only evidence is the source text,
   not the existing notes. The notes tell you what the juniors thought;
   the source tells you what's true.

3. To merge several notes into one stronger note: drop them all and
   add ONE new note that summarises them. There is no special "merge"
   action.

4. If the task description asks for something the section text doesn't
   support, you may return an empty ``updates`` list and an empty
   ``new_notes`` list. Doing nothing is a valid outcome.

5. Be specific. Vague ratings or hand-waving rationales help nobody.
"""


# ── Builder + module-level definition ───────────────────────────────────


def build_investigator_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="investigator",
        prompt=_INVESTIGATOR_PROMPT,
        output_model=InvestigatorOutput,
        retries=2,
    )


INVESTIGATOR_AGENT = build_investigator_agent_definition()
