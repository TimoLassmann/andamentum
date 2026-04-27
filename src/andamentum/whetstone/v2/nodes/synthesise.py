"""Node: Synthesise.

Final LLM call. Reads challenged_findings + deterministic_findings +
DocumentMap, produces a ReviewSummary, and emits End[ReviewResult].

Loud-fail-safe: if the synthesise call fails, we still emit a result —
just with a generic summary noting the failure. The findings (the most
useful payload) are returned regardless.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import build_pydantic_ai_agent, ReviewSummary
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewMetrics, ReviewResult
from ..state import ReviewState


@dataclass
class Synthesise(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Write the final review prose, return the result."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "End[ReviewResult]":
        ctx.state.current_phase = "synthesise"

        # Combine deterministic + challenged findings. Both are real;
        # the consumer can filter by `source` if they want to separate.
        all_findings = list(ctx.state.deterministic_findings) + list(
            ctx.state.challenged_findings
        )

        try:
            summary = await _run_synthesis(ctx.deps, ctx.state, all_findings)
            ctx.state.llm_calls += 1
            ctx.state.summary = _flatten_summary(summary)
        except Exception as exc:
            ctx.state.summary = (
                "[Synthesis call failed — findings are still valid below.]\n"
                f"Failure: {exc}"
            )

        ctx.state.current_phase = "done"
        return End(_build_result(ctx.state, all_findings))


async def _run_synthesis(
    deps: ReviewDeps,
    state: ReviewState,
    findings: list[Finding],
) -> ReviewSummary:
    map_lines = "\n".join(
        f"  • {c.section_id} — {c.title}: {c.one_line_gist}"
        for c in state.document_map
    )
    finding_lines = "\n".join(
        f"  [{f.severity}|{f.confidence}] {f.title}\n      sections: {', '.join(f.sections_involved)}\n      {f.rationale}"
        for f in findings
    )
    prompt = f"""DOCUMENT MAP:
{map_lines}

FINDINGS ({len(findings)} total):
{finding_lines or "  (no findings — clean document)"}

Write a ReviewSummary with executive_summary (2 paragraphs) and
severity-grouped paragraphs."""

    agent = build_pydantic_ai_agent("synthesise", deps.model)
    result = await agent.run(prompt)
    from typing import cast

    return cast(ReviewSummary, result.output)


def _flatten_summary(summary: ReviewSummary) -> str:
    """Concatenate the four prose fields into the single ``summary`` string
    surfaced on ReviewResult."""
    parts = [
        ("Executive Summary", summary.executive_summary),
        ("Major Findings", summary.major_findings_summary),
        ("Moderate Findings", summary.moderate_findings_summary),
        ("Minor Findings", summary.minor_findings_summary),
    ]
    return "\n\n".join(f"## {title}\n\n{body}".strip() for title, body in parts if body)


def _build_result(state: ReviewState, all_findings: list[Finding]) -> ReviewResult:
    """Assemble the final ReviewResult."""
    return ReviewResult(
        summary=state.summary,
        findings=list(state.challenged_findings),
        deterministic_findings=list(state.deterministic_findings),
        edits=list(state.edits),
        author_questions=list(state.author_questions),
        document_map=list(state.document_map),
        metrics=ReviewMetrics(
            llm_calls=state.llm_calls,
            wall_seconds=0.0,  # api.py wraps with a timer
            deterministic_findings_count=len(state.deterministic_findings),
            investigated_findings_count=len(state.challenged_findings),
            challenged_findings_count=sum(
                1 for f in state.challenged_findings if f.source == "challenged"
            ),
            edits_count=len(state.edits),
            sections_processed=len(state.sections),
            reflection_rounds_used=state.reflection_round,
        ),
    )
