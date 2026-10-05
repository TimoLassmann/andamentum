"""Drafting one section at a time from its plan and the facts it cites."""

from __future__ import annotations

from pydantic import BaseModel

from ._call import call_model
from .genres import Genre
from .log import ActionLog
from .schemas import Brief, Fact, Outline, SectionPlan

MARKER_RULES = """\
Facts and markers. Each fact has an id such as f12. Right after any sentence \
that relies on a fact, write its marker as [[f12]] (several: [[f3]][[f7]]). Use \
only facts listed for this section. Never state a specific (number, name, date, \
claim) that no listed fact supports: write [TO FILL: what is needed] instead. \
Never type a citation yourself ("Smith (2020)", "(Smith, 2020)", "[3]"): a marker \
is the only way to cite, and the citation is made from it; a work named in the \
text without a marker is a defect. \
Do not write headings, image syntax or a bibliography; the section title and \
references are added elsewhere. Plain paragraphs, "- " bullets, "1. " numbered \
lists and pipe tables are allowed."""

DRAFT_INSTRUCTIONS = (
    """\
You write one section of a document, in finished prose for the stated audience. \
Stay inside the section's purpose and key points, aim for the target word count, \
and do not repeat what earlier sections already said.

"""
    + MARKER_RULES
)

REVISE_INSTRUCTIONS = (
    """\
You revise one section of a document. Rewrite the whole section so that every \
listed issue is resolved. Keep what already works; change only what the issues \
need. Keep the section's purpose and key points.

"""
    + MARKER_RULES
)


class SectionBody(BaseModel):
    body: str


def facts_for(plan: SectionPlan, facts: list[Fact]) -> list[Fact]:
    wanted = set(plan.fact_ids)
    return [f for f in facts if f.id in wanted]


def facts_block(section_facts: list[Fact]) -> str:
    if not section_facts:
        return "FACTS FOR THIS SECTION: none. Write [TO FILL: ...] for any specific."
    rows = [f'{f.id}: {f.statement}  (quote: "{f.quote}")' for f in section_facts]
    return "FACTS FOR THIS SECTION:\n" + "\n".join(rows)


def plan_block(plan: SectionPlan) -> str:
    points = "\n".join(f"- {p}" for p in plan.key_points) or "- (none listed)"
    return (
        f"SECTION: {plan.title}\n"
        f"PURPOSE: {plan.purpose}\n"
        f"KEY POINTS:\n{points}\n"
        f"TARGET WORDS: {plan.target_words}"
    )


def brief_block(brief: Brief, genre: Genre) -> str:
    return (
        f"DOCUMENT: {brief.title} ({genre.name})\n"
        f"AUDIENCE: {brief.audience}\n"
        f"PURPOSE: {brief.purpose}\n"
        f"CONSTRAINTS: {'; '.join(brief.constraints) or 'none'}\n"
        f"STYLE: {genre.style_notes}\n"
        f"GUIDELINES: {brief.guidelines or 'none'}"
    )


async def draft_section(
    plan: SectionPlan,
    *,
    brief: Brief,
    genre: Genre,
    outline: Outline,
    facts: list[Fact],
    previous_summaries: list[str],
    model: str,
    log: ActionLog | None = None,
) -> str:
    """Return the section body as markdown, with ``[[fN]]`` markers."""
    order = "\n".join(
        f"{i}. {s.title}" for i, s in enumerate(outline.sections, start=1)
    )
    earlier = (
        "\n".join(f"- {s}" for s in previous_summaries)
        if previous_summaries
        else "(this is the first section)"
    )
    message = "\n\n".join(
        [
            brief_block(brief, genre),
            f"OUTLINE:\n{order}",
            f"EARLIER SECTIONS (summaries):\n{earlier}",
            plan_block(plan),
            facts_block(facts_for(plan, facts)),
        ]
    )
    out = await call_model(
        model=model,
        instructions=DRAFT_INSTRUCTIONS,
        output_type=SectionBody,
        user_message=message,
        node="draft",
        log=log,
    )
    return out.body.strip()
