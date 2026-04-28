"""Node: CustomReviewer (custom-criteria mode).

One LLM call. Reads ``state.custom_criteria`` plus the manuscript and
produces a per-criterion verdict + notes for every criterion at once
(matches v1's ``custom_document_reviewer`` pattern). The output schema
is built dynamically by
:func:`dynamic_schemas.create_custom_evaluation_model`; the orchestrator
unpacks the filled model into a flat ``list[CustomEvaluation]`` so
downstream consumers never see the runtime model.

Loud-fail-safe: if ``custom_criteria`` is empty when this node runs
(should never happen — ``api.review_document`` validates) we raise a
clear error rather than silently producing no evaluations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import build_dynamic_output_agent
from ..deps import ReviewDeps
from ..dynamic_schemas import (
    create_custom_evaluation_model,
    unpack_custom_evaluations,
)
from ..schemas import CheckableItem, ReviewMetrics, ReviewResult
from ..state import ReviewState


logger = logging.getLogger("andamentum.whetstone.v2")

_MAX_DOCUMENT_CHARS = 30_000  # generous; truncation only kicks in for very long docs


@dataclass
class CustomReviewer(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run one LLM call evaluating the document against all custom criteria."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "End[ReviewResult]":
        ctx.state.current_phase = "custom_review"

        criteria = list(ctx.state.custom_criteria)
        if not criteria:
            raise ValueError(
                "CustomReviewer was reached but state.custom_criteria is empty. "
                "mode='custom' requires at least one criterion."
            )

        # Mirror v1: surface each criterion as a CheckableItem so the
        # caller can see what was actually evaluated.
        ctx.state.checkable_items = [
            CheckableItem(name=c.strip(), source="custom") for c in criteria
        ]

        logger.info(
            "[custom] evaluating document against %d criterion(s)",
            len(criteria),
        )

        try:
            evaluations = await _run_custom_review(ctx.deps, criteria, ctx.state)
            ctx.state.llm_calls += 1
            ctx.state.custom_evaluations = evaluations
            ctx.state.summary = _build_summary(evaluations)
        except Exception as exc:
            # Loud failure — but still return a valid ReviewResult so the
            # CLI can render the partial state (the criteria the caller
            # supplied, and a clear explanation of why no verdicts).
            logger.warning("[custom] review call failed: %s", exc)
            ctx.state.summary = (
                "[Custom-criteria review call failed — no verdicts available.]\n"
                f"Failure: {exc}"
            )

        ctx.state.current_phase = "done"
        return End(_build_result(ctx.state))


async def _run_custom_review(deps: ReviewDeps, criteria: list[str], state: ReviewState):
    """Build the dynamic output model, run one LLM call, unpack the result."""
    output_model = create_custom_evaluation_model(criteria)

    document_view = _build_document_view(state)
    numbered = "\n".join(f"  {i}. {c.strip()}" for i, c in enumerate(criteria, start=1))
    prompt = f"""REVIEW CRITERIA ({len(criteria)} total):
{numbered}

DOCUMENT TO REVIEW:
{document_view}

Evaluate the document against each criterion above. Fill the
schema completely — every <slug>_status, <slug>_notes, and the
overall_assessment. Status must be one of {{pass, fail, unclear}}."""

    agent = build_dynamic_output_agent("custom_reviewer", deps.model, output_model)
    result = await agent.run(prompt)
    return unpack_custom_evaluations(criteria, result.output)


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


def _build_summary(evaluations) -> str:
    """Bucket custom evaluations by status; produce a flat narrative summary."""
    fails = [e for e in evaluations if e.status == "fail"]
    unclears = [e for e in evaluations if e.status == "unclear"]
    passes = [e for e in evaluations if e.status == "pass"]

    parts = [
        "## Custom-criteria review summary",
        "",
        f"Evaluated **{len(evaluations)}** caller-supplied criterion(s): "
        f"**{len(fails)} fail**, **{len(unclears)} unclear**, "
        f"**{len(passes)} pass**.",
    ]
    if fails:
        parts.extend(
            [
                "",
                "### Failing criteria",
                "",
                *(f"- **{e.criterion}** — {e.notes}".rstrip() for e in fails),
            ]
        )
    if unclears:
        parts.extend(
            [
                "",
                "### Unclear criteria",
                "",
                *(f"- **{e.criterion}** — {e.notes}".rstrip() for e in unclears),
            ]
        )
    if passes:
        parts.extend(
            [
                "",
                "### Passing criteria",
                "",
                *(f"- **{e.criterion}**" for e in passes),
            ]
        )
    return "\n".join(parts)


def _build_result(state: ReviewState) -> ReviewResult:
    """Assemble the final ReviewResult (custom mode)."""
    return ReviewResult(
        summary=state.summary,
        findings=list(state.challenged_findings),
        deterministic_findings=list(state.deterministic_findings),
        edits=list(state.edits),
        author_questions=list(state.author_questions),
        document_map=list(state.document_map),
        checkable_items=list(state.checkable_items),
        custom_evaluations=list(state.custom_evaluations),
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
