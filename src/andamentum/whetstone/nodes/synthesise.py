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


_DOC_TYPE_VOCAB: dict[str, str] = {
    "academic": (
        "DOCUMENT TYPE: academic. This draft is scholarly writing — a "
        "manuscript, thesis, conference paper, or similar. Use "
        "vocabulary appropriate to academic writing: 'manuscript', "
        "'draft', 'submission', 'methods', 'results', 'discussion'."
    ),
    "external_communication": (
        "DOCUMENT TYPE: external_communication. This draft is written "
        "for a broad non-academic audience — a blog post, LinkedIn "
        "article, email, op-ed, or similar. Use vocabulary appropriate "
        "to that genre: 'post', 'article', 'audience', 'tone', "
        "'call to action', 'reader engagement'. Frame feedback for "
        "publication on a public channel, not for academic submission."
    ),
    "general": (
        "DOCUMENT TYPE: general. This draft is neither academic "
        "writing nor external communication — likely a note, internal "
        "document, book, technical writeup, or similar. Use neutral "
        "vocabulary: 'document', 'draft', 'text', 'section'. Do not "
        "assume academic submission norms apply."
    ),
}


def _document_type_context(document_type: str) -> str:
    return _DOC_TYPE_VOCAB.get(document_type, _DOC_TYPE_VOCAB["general"])


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
        f"  [{f.priority}|{f.severity}|{f.confidence}] {f.title}\n      sections: {', '.join(f.sections_involved)}\n      {f.rationale}"
        for f in findings
    )
    type_context = _document_type_context(state.document_type)
    prompt = f"""{type_context}

DOCUMENT MAP:
{map_lines}

FINDINGS ({len(findings)} total) — each tagged [priority|severity|confidence]:
{finding_lines or "  (no findings — clean draft)"}

Produce a ReviewSummary with executive_summary (2 paragraphs) and
priority-bucketed paragraphs (must_fix / should_fix / consider).
Match vocabulary to the DOCUMENT TYPE shown above."""

    agent = build_pydantic_ai_agent("synthesise", deps.model)
    result = await agent.run(prompt)
    from typing import cast

    return cast(ReviewSummary, result.output)


def _flatten_summary(summary: ReviewSummary) -> str:
    """Concatenate the four prose fields into the single ``summary`` string
    surfaced on ReviewResult."""
    parts = [
        ("Executive Summary", summary.executive_summary),
        ("MUST FIX", summary.must_fix_summary),
        ("SHOULD FIX", summary.should_fix_summary),
        ("CONSIDER", summary.consider_summary),
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
