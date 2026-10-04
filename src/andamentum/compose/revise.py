"""Revising one section against the work items the checks raised."""

from __future__ import annotations

from ._call import call_model
from .genres import Genre
from .log import ActionLog
from .schemas import Brief, Fact, SectionPlan, WorkItem
from .write import (
    REVISE_INSTRUCTIONS,
    SectionBody,
    brief_block,
    facts_block,
    facts_for,
    plan_block,
)


async def revise_section(
    section_body: str,
    items: list[WorkItem],
    *,
    plan: SectionPlan,
    brief: Brief,
    genre: Genre,
    facts: list[Fact],
    model: str,
    log: ActionLog | None = None,
) -> str:
    """Rewrite the whole section to answer ``items``; same marker rules as drafting."""
    issues = "\n".join(f"- ({i.source}) {i.reason}" for i in items) or "- (none)"
    message = "\n\n".join(
        [
            brief_block(brief, genre),
            plan_block(plan),
            facts_block(facts_for(plan, facts)),
            f"CURRENT SECTION:\n{section_body}",
            f"ISSUES TO RESOLVE:\n{issues}",
        ]
    )
    out = await call_model(
        model=model,
        instructions=REVISE_INSTRUCTIONS,
        output_type=SectionBody,
        user_message=message,
        node="revise",
        log=log,
    )
    return out.body.strip()
