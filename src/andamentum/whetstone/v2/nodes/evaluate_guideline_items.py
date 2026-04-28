"""Node: EvaluateGuidelineItems (guidelines mode).

N parallel LLM calls — one per CheckableItem produced by
``ExtractCheckableItems``. Each call asks ``guideline_item_evaluator``
to render a verdict (pass / fail / unclear) plus 1-2 sentences of notes.

Concurrency is capped at 4 to match :class:`CriticalRead`. Per-item
failures degrade to ``status="unclear"`` with the failure message in
``notes`` rather than aborting the whole run — journal-extracted items
come from fuzzy LLM extractor output, so a single bad item shouldn't
stop the rest of the checklist (matches v1's behaviour).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import cast

from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import (
    CheckableItem,
    GuidelineEvaluation,
    ReviewMetrics,
    ReviewResult,
)
from ..state import ReviewState


logger = logging.getLogger("andamentum.whetstone.v2")

_MAX_CONCURRENT_EVALUATIONS = 4
_MAX_DOCUMENT_CHARS = 30_000  # generous; truncation only kicks in for very long docs


@dataclass
class EvaluateGuidelineItems(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Evaluate every checkable item against the manuscript in parallel."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "End[ReviewResult]":
        ctx.state.current_phase = "evaluate_guideline_items"

        items = list(ctx.state.checkable_items)
        if not items:
            logger.warning(
                "[guidelines] no checkable items — emitting empty evaluation list"
            )
            ctx.state.summary = (
                "[Guidelines mode produced no checkable items — supplied "
                "guidelines may be empty or non-actionable.]"
            )
            ctx.state.current_phase = "done"
            return End(_build_result(ctx.state))

        logger.info(
            "[guidelines] evaluating %d item(s) (concurrency=%d)",
            len(items),
            _MAX_CONCURRENT_EVALUATIONS,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT_EVALUATIONS)
        document_view = _build_document_view(ctx.state)

        async def evaluate_one(item: CheckableItem) -> GuidelineEvaluation:
            async with sem:
                try:
                    return await _run_evaluation(ctx.deps, item, document_view)
                except Exception as exc:
                    logger.warning(
                        "[guidelines] evaluation of %r failed: %s", item.name, exc
                    )
                    return GuidelineEvaluation(
                        item_name=item.name,
                        status="unclear",
                        notes=f"Evaluation failed: {exc}",
                        category="",
                    )

        evaluations = await asyncio.gather(*[evaluate_one(it) for it in items])
        # Each successful evaluation is one LLM call; failures already
        # logged but still cost a call attempt.
        ctx.state.llm_calls += len(items)
        ctx.state.guideline_evaluations = list(evaluations)

        # Build a flat textual summary. Mirrors the synthesise-style output:
        # priority-bucketed (fail / unclear / pass) headlines.
        ctx.state.summary = _build_summary(evaluations)
        ctx.state.current_phase = "done"
        return End(_build_result(ctx.state))


async def _run_evaluation(
    deps: ReviewDeps, item: CheckableItem, document_view: str
) -> GuidelineEvaluation:
    """Single guideline_item_evaluator call. Restores item_name if the LLM dropped it."""
    prompt = f"""RULE TO EVALUATE:
{item.name}

DOCUMENT TO CHECK:
{document_view}

Evaluate the rule above. Echo the rule name verbatim in the
``item_name`` field. Pick ``status`` from {{pass, fail, unclear}}.
Provide 1-2 sentences of notes citing evidence."""

    agent = build_pydantic_ai_agent("guideline_item_evaluator", deps.model)
    result = await agent.run(prompt)
    evaluation = cast(GuidelineEvaluation, result.output)
    # Defensive: if the agent forgot the item name, restore it.
    if not evaluation.item_name.strip():
        evaluation = evaluation.model_copy(update={"item_name": item.name})
    return evaluation


def _build_document_view(state: ReviewState) -> str:
    """Document map + truncated markdown — same shape used by other nodes."""
    parts: list[str] = []
    if state.document_map:
        lines = [
            f"  • {c.section_id} — {c.title}: {c.one_line_gist}".rstrip(": ")
            for c in state.document_map
        ]
        parts.append("Document map:\n" + "\n".join(lines))
    if state.markdown:
        if len(state.markdown) <= _MAX_DOCUMENT_CHARS:
            parts.append(f"Full document:\n{state.markdown}")
        else:
            head = state.markdown[:_MAX_DOCUMENT_CHARS]
            parts.append(
                f"Document excerpt (first {len(head)} of "
                f"{len(state.markdown)} chars):\n{head}"
            )
    return "\n\n".join(parts) if parts else "(empty document)"


def _build_summary(evaluations: list[GuidelineEvaluation]) -> str:
    """Bucket evaluations by status; produce a flat narrative summary."""
    fails = [e for e in evaluations if e.status == "fail"]
    unclears = [e for e in evaluations if e.status == "unclear"]
    passes = [e for e in evaluations if e.status == "pass"]

    parts = [
        "## Guideline checklist summary",
        "",
        f"Evaluated **{len(evaluations)}** rule(s) extracted from the "
        f"supplied journal author guidelines: "
        f"**{len(fails)} fail**, **{len(unclears)} unclear**, "
        f"**{len(passes)} pass**.",
    ]
    if fails:
        parts.extend(
            [
                "",
                "### Failing rules (must fix before submission)",
                "",
                *(f"- **{e.item_name}** — {e.notes}".rstrip() for e in fails),
            ]
        )
    if unclears:
        parts.extend(
            [
                "",
                "### Unclear rules (review manually)",
                "",
                *(f"- **{e.item_name}** — {e.notes}".rstrip() for e in unclears),
            ]
        )
    if passes:
        parts.extend(
            [
                "",
                "### Passing rules",
                "",
                *(f"- **{e.item_name}**" for e in passes),
            ]
        )
    return "\n".join(parts)


def _build_result(state: ReviewState) -> ReviewResult:
    """Assemble the final ReviewResult (guidelines mode)."""
    return ReviewResult(
        summary=state.summary,
        findings=list(state.challenged_findings),
        deterministic_findings=list(state.deterministic_findings),
        edits=list(state.edits),
        author_questions=list(state.author_questions),
        document_map=list(state.document_map),
        checkable_items=list(state.checkable_items),
        guideline_evaluations=list(state.guideline_evaluations),
        metrics=ReviewMetrics(
            llm_calls=state.llm_calls,
            wall_seconds=0.0,  # api.py wraps with a timer
            deterministic_findings_count=len(state.deterministic_findings),
            investigated_findings_count=len(state.findings),
            challenged_findings_count=len(state.challenged_findings),
            edits_count=len(state.edits),
            sections_processed=len(state.sections),
            reflection_rounds_used=state.reflection_round,
        ),
    )
