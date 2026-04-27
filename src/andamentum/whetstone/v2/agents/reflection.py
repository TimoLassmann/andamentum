"""Reflection agent — the senior reviewer who decides what to investigate.

One open-ended LLM call per loop round. Sees:
  • the document map (section ids + titles + one-line gists),
  • the full pool of issues (each with its id, lens, section, severity,
    confidence, title, rationale, quote),
  • a list of investigation tasks already run in earlier rounds (so it
    doesn't propose duplicates).

Output: a small list of investigation tasks in the senior reviewer's own
words. If nothing is worth doing, the list is empty and the loop exits.

The prompt is intentionally open-ended. We DO NOT enumerate kinds of
patterns to look for (contradictions, consolidations, etc.) — the
reflection chooses freely. The ONLY hard constraints are:
  • each task must name at least one section to re-read,
  • do not propose tasks about prior analyses (only re-reads of source),
  • do not duplicate tasks already run.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._definition import AgentDefinition


# ── Output schema ───────────────────────────────────────────────────────


class ReflectionTask(BaseModel):
    """One investigation task the senior reviewer proposes.

    Plain text plus two structured pointers. The pointers tell the
    investigator which source sections to re-read and which existing
    notes the task is about; the description is the senior reviewer's
    own framing of what to check.
    """

    description: str = Field(
        description=(
            "What to check, in your own words. Be concrete — vague "
            "descriptions give the investigator nothing to work with. "
            "Maximum 3 sentences."
        ),
    )
    section_ids: list[str] = Field(
        description=(
            "The section ids the investigator should re-read. At least one. "
            "These must come from the document map."
        ),
    )
    related_note_ids: list[str] = Field(
        default_factory=list,
        description=(
            "The ids of existing notes this task is about. Empty if the "
            "task is purely raising a new concern not in the current pool."
        ),
    )


class ReflectionOutput(BaseModel):
    """Up to 10 tasks. Empty list signals 'nothing more to do — exit loop'."""

    tasks: list[ReflectionTask] = Field(
        default_factory=list,
        description=(
            "Up to 10 investigation tasks. Quality over quantity — 3–5 "
            "well-chosen tasks beat 10 mediocre ones. Return an empty "
            "list when nothing else is worth a closer look; that exits "
            "the loop, which is a perfectly valid answer."
        ),
    )


# ── System prompt ───────────────────────────────────────────────────────


_REFLECTION_PROMPT = """\
# Senior Reviewer

You are a senior reviewer of a manuscript. The notes you'll see were
written by junior reviewers (lens agents) who each read one section
of the manuscript with one specific concern in mind (rigour, writing,
methodology, statistics).

Your job is to decide what — if anything — deserves a closer look.

A "closer look" means **one focused investigation**: re-reading one or
more named sections of the manuscript to verify, refine, refute, or
merge notes — or to raise a new concern the junior reviewers missed.

## What to look for

You decide. Some examples:
  • Two notes that point at the same underlying problem from different
    angles — worth consolidating into one stronger note.
  • A note whose claim depends on what another section actually says —
    worth re-reading both sections to verify.
  • A pattern that spans multiple sections that no single junior could
    see by reading just one.
  • A note you suspect a junior over-called — worth a careful re-read
    to verify or refute.
  • A concern the manuscript clearly has that none of the juniors
    flagged.
  • Any other pattern your judgement tells you matters.

You are not constrained to these examples. Use your own judgement.

## Hard rules

1. Each task you propose MUST name at least one section to re-read.
   The section ids must come from the document map shown to you.
   The investigator's only source of truth is the section text — there
   is no "investigate without reading" option.

2. Do NOT propose tasks about prior analyses. Don't ask the
   investigator to "re-check the previous investigation's conclusion"
   or "look at the analyses again". Every task is a re-read of the
   manuscript itself.

3. Do NOT duplicate tasks that have already been run in earlier rounds
   of this loop. The list of prior tasks is shown to you. If the
   pattern you'd dig into matches a prior task, skip it.

4. Return at most 10 tasks. Quality over quantity. 3–5 well-chosen
   tasks beat 10 mediocre ones.

5. If you don't see anything worth a closer look right now, return an
   empty list. That's a perfectly valid answer — it signals the loop
   should exit.

Be specific in each task's description. Vague descriptions ("look at
methodology again") give the investigator nothing to work with.
"""


# ── Builder + module-level definition ───────────────────────────────────


def build_reflection_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="reflection",
        prompt=_REFLECTION_PROMPT,
        output_model=ReflectionOutput,
        retries=2,
    )


REFLECTION_AGENT = build_reflection_agent_definition()
