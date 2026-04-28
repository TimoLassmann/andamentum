"""Journal-guidelines → checkable-items agent (guidelines mode).

Single LLM call. Reads a journal's free-text author guidelines and
produces 10-30 short, actionable rule names. Each name becomes one
call to ``guideline_item_evaluator`` in the next phase.

Ports v1's ``journal_guidelines_extractor`` prompt with the v2 hygiene
pass: the output schema is named ``ExtractedItemsList`` (was
``ExtractedChecklistNames``) and is bounded to 10-30 items.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition

EXTRACT_CHECKABLE_ITEMS_PROMPT = """# Journal guidelines extractor

Read the journal author guidelines below and extract every rule an
author should verify before submission, one rule per item.

# Rules

- 10–30 items total. Aim for completeness on the actionable rules
  without padding the list with restatements of the same constraint.
- Skip general editorial prose ("We welcome submissions...", "Authors
  are advised to ensure their work is original..."). Keep only
  actionable, *checkable* rules — things you can answer pass / fail /
  unclear for a specific manuscript.
- Phrase each item as a short declarative requirement that names the
  thing being checked. Examples of well-formed items:
    • "Abstract ≤ 250 words"
    • "Figures supplied in vector format"
    • "Data availability statement present"
    • "Author contributions section included"
    • "References formatted in Vancouver style"
- One concept per item. Do NOT bundle (e.g. don't write "abstract is
  short and structured" — that's two items).
- If the guidelines are silent on a topic, omit it. Don't invent
  rules that aren't in the source.

# Output

Return an ``ExtractedItemsList`` with the rule names as a flat list
of strings.
"""


class ExtractedItemsList(BaseModel):
    """Flat output for the extract_checkable_items agent."""

    items: list[str] = Field(
        description=(
            "10-30 short, declarative rule names extracted from the "
            "journal author guidelines. Each name should be checkable "
            "against a manuscript with a pass/fail/unclear verdict."
        ),
    )


def _build() -> AgentDefinition:
    return AgentDefinition(
        name="extract_checkable_items",
        prompt=EXTRACT_CHECKABLE_ITEMS_PROMPT,
        output_model=ExtractedItemsList,
        retries=2,
        output_retries=2,
    )


EXTRACT_CHECKABLE_ITEMS_AGENT = _build()
