"""Node: EditSections.

For each section, ask the editor agent to propose concrete rewrites
based on the configured ``editor_criteria``. Runs in parallel across
sections (one LLM call per section). Each proposed edit's
``original_text`` is anchored to a specific char span via the chunker's
tiered ``find_anchor``; edits whose original_text can't be located are
silently dropped.

Disabled by default (``state.editor_enabled = False``). Caller opts in
by passing ``editor=True`` (and optionally ``editor_criteria=[...]``)
to ``review_document``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from andamentum.chunker.validation import find_anchor

from ..agents import EditorOutput, build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import Edit, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from ..structural.types import SectionRef
    from .challenge import Challenge


# Cap on parallelism so we don't slam Ollama or hit rate limits.
_MAX_CONCURRENT_EDITORS = 5


@dataclass
class EditSections(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run the editor agent across all sections in parallel."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "Challenge":
        ctx.state.current_phase = "edit"

        if not ctx.state.editor_enabled:
            from .challenge import Challenge

            return Challenge()

        sem = asyncio.Semaphore(_MAX_CONCURRENT_EDITORS)
        criteria = list(ctx.state.editor_criteria)

        async def edit_one(section: "SectionRef") -> list[Edit]:
            async with sem:
                try:
                    return await _run_editor_on_section(ctx.deps, section, criteria)
                except Exception:
                    # Loud-fail-safe: one section's editor crashing must
                    # not abort the whole pipeline.
                    return []

        results = await asyncio.gather(
            *[edit_one(s) for s in ctx.state.sections]
        )
        for edits in results:
            ctx.state.edits.extend(edits)
        ctx.state.llm_calls += sum(1 for r in results if r is not None)

        from .challenge import Challenge

        return Challenge()


async def _run_editor_on_section(
    deps: ReviewDeps,
    section: "SectionRef",
    criteria: list[str],
) -> list[Edit]:
    """One editor call against one section. Returns located Edit objects."""
    prompt = f"""SECTION TITLE: {section.title}
SECTION ID: {section.id}

EDITORIAL CRITERIA TO APPLY:
{", ".join(criteria) or "(none specified — use general editing judgement)"}

SECTION TEXT (quote VERBATIM from below — copy original_text exactly):
--- BEGIN ---
{section.text}
--- END ---

Emit 0–8 EditProposals. Empty list is fine if the section is already strong."""

    agent = build_pydantic_ai_agent("editor", deps.model)
    result = await agent.run(prompt)
    output = cast(EditorOutput, result.output)

    # Anchor each proposal's original_text to a real char span in the section.
    out: list[Edit] = []
    for prop in output.edits:
        if not prop.original_text or not prop.new_text:
            continue
        match = find_anchor(prop.original_text, section.text, search_from=0)
        if match is None:
            # Fabricated quote — skip silently. The agent's prompt
            # explicitly forbids this; if it slips through we drop it
            # rather than guess at offsets.
            continue
        out.append(
            Edit(
                title=prop.title or "(untitled edit)",
                severity=prop.severity,
                confidence=prop.confidence,
                rationale=prop.rationale,
                section_id=section.id,
                char_start=match.start,
                char_end=match.end,
                original_text=section.text[match.start : match.end],
                new_text=prop.new_text,
            )
        )
    return out
