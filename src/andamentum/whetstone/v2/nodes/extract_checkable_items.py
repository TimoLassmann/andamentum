"""Node: ExtractCheckableItems (guidelines mode).

Single LLM call. Reads ``state.guidelines_text`` and produces 10-30
short checkable rule names into ``state.checkable_items``. Each item
is then evaluated by ``EvaluateGuidelineItems`` in the next phase.

Loud-fail-safe: if ``guidelines_text`` is empty when this node runs
(should never happen — ``api.review_document`` validates) we raise a
clear error rather than silently producing zero items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import ExtractedItemsList, build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import CheckableItem, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .evaluate_guideline_items import EvaluateGuidelineItems


logger = logging.getLogger("andamentum.whetstone.v2")


@dataclass
class ExtractCheckableItems(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Turn free-text journal guidelines into checkable item names."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "EvaluateGuidelineItems":
        ctx.state.current_phase = "extract_checkable_items"

        if not ctx.state.guidelines_text.strip():
            raise ValueError(
                "ExtractCheckableItems was reached but state.guidelines_text "
                "is empty. mode='guidelines' requires non-empty guidelines."
            )

        logger.info(
            "[guidelines] extracting checkable items from %d chars of guidelines",
            len(ctx.state.guidelines_text),
        )

        items = await _extract_items(ctx.deps, ctx.state.guidelines_text)
        ctx.state.llm_calls += 1
        ctx.state.checkable_items = [
            CheckableItem(name=name, source="guidelines") for name in items
        ]

        logger.info(
            "[guidelines] extracted %d checkable item(s)",
            len(ctx.state.checkable_items),
        )

        from .evaluate_guideline_items import EvaluateGuidelineItems

        return EvaluateGuidelineItems()


async def _extract_items(deps: ReviewDeps, guidelines_text: str) -> list[str]:
    """Run the extract_checkable_items agent and return its rule names."""
    prompt = f"""JOURNAL AUTHOR GUIDELINES:
{guidelines_text}

Extract 10-30 short checkable rules following the schema. One concept
per item. Skip general editorial prose."""

    agent = build_pydantic_ai_agent("extract_checkable_items", deps.model)
    result = await agent.run(prompt)
    output = cast(ExtractedItemsList, result.output)
    # Strip + dedupe while preserving order.
    seen: set[str] = set()
    items: list[str] = []
    for raw in output.items:
        cleaned = raw.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            items.append(cleaned)
    return items
