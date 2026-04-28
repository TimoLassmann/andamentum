"""Node: PanelSynthesise (panel mode).

Final node of the panel pipeline. Single LLM call. Aggregates the
N expert reviews into a ``PanelSynthesis`` and emits End[ReviewResult].

Loud-fail-safe: if the synthesis call crashes, we still emit a result
— the per-expert reviews and profiles are the most useful payload and
get returned regardless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import (
    ExpertReview as ExpertReviewSchema,
    PanelSynthesis,
    ReviewMetrics,
    ReviewResult,
)
from ..state import ReviewState


logger = logging.getLogger("andamentum.whetstone")


@dataclass
class PanelSynthesise(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Aggregate all expert reviews; emit End[ReviewResult]."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "End[ReviewResult]":
        ctx.state.current_phase = "panel_synthesise"

        reviews = list(ctx.state.expert_reviews)
        if not reviews:
            logger.warning("[panel] no expert reviews — skipping synthesis")
            ctx.state.summary = (
                "[Panel synthesis skipped — no expert reviews were produced.]"
            )
            ctx.state.current_phase = "done"
            return End(_build_result(ctx.state))

        try:
            synthesis = await _run_synthesis(ctx.deps, reviews)
            ctx.state.llm_calls += 1
            ctx.state.panel_synthesis = synthesis
            ctx.state.summary = _flatten_synthesis(synthesis)
        except Exception as exc:
            logger.warning("[panel] synthesis call failed: %s", exc)
            ctx.state.summary = (
                "[Panel synthesis call failed — per-expert reviews are still "
                f"valid below.]\nFailure: {exc}"
            )

        ctx.state.current_phase = "done"
        return End(_build_result(ctx.state))


async def _run_synthesis(
    deps: ReviewDeps, reviews: list[ExpertReviewSchema]
) -> PanelSynthesis:
    review_lines: list[str] = []
    for r in reviews:
        review_lines.append(
            f"--- Review by {r.expert_name} ({r.discipline}) ---\n"
            f"  Overall: {r.overall_score}/10 — {r.overall_assessment}\n"
            f"  Scientific rigor: {r.scientific_rigor_score}/10 — "
            f"{r.scientific_rigor_justification}\n"
            f"  Methodology: {r.methodology_score}/10 — "
            f"{r.methodology_justification}\n"
            f"  Novelty: {r.novelty_score}/10 — {r.novelty_justification}\n"
            f"  Clarity: {r.clarity_score}/10 — {r.clarity_justification}\n"
            "  Strengths:\n"
            + "\n".join(f"    • {s}" for s in r.strengths)
            + "\n  Weaknesses:\n"
            + "\n".join(f"    • {w}" for w in r.weaknesses)
            + f"\n  Recommendation: {r.recommendation} — "
            f"{r.recommendation_justification}"
        )

    prompt = f"""EXPERT REVIEWS ({len(reviews)} total):

{chr(10).join(review_lines)}

Synthesise these reviews into a single PanelSynthesis. Calculate
average and range scores; identify consensus strengths/weaknesses;
note divergent opinions if any; summarise per criterion; produce an
aggregated recommendation with justification, confidence level, key
decision factors, and a 5-7 paragraph review summary."""

    agent = build_pydantic_ai_agent("panel_synthesise", deps.model)
    result = await agent.run(prompt)
    return cast(PanelSynthesis, result.output)


def _flatten_synthesis(s: PanelSynthesis) -> str:
    """Concatenate the panel synthesis into the single ``summary`` string."""
    parts = [
        ("Panel Recommendation", _recommendation_block(s)),
        ("Review Summary", s.review_summary),
        ("Consensus Strengths", _bullet_list(s.consensus_strengths)),
        ("Consensus Weaknesses", _bullet_list(s.consensus_weaknesses)),
        ("Divergent Opinions", _bullet_list(s.divergent_opinions) or "None."),
        ("Scientific Rigor", s.scientific_rigor_summary),
        ("Methodology", s.methodology_summary),
        ("Novelty", s.novelty_summary),
        ("Clarity", s.clarity_summary),
        ("Key Decision Factors", _bullet_list(s.key_decision_factors)),
    ]
    return "\n\n".join(f"## {title}\n\n{body}".strip() for title, body in parts if body)


def _recommendation_block(s: PanelSynthesis) -> str:
    return (
        f"**{s.overall_recommendation}** (confidence: {s.confidence_level})\n\n"
        f"Average score: {s.average_overall_score:.1f}/10 "
        f"(range: {s.score_range}, n={s.number_of_experts})\n\n"
        f"{s.recommendation_justification}"
    )


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {it}" for it in items)


def _build_result(state: ReviewState) -> ReviewResult:
    """Assemble the final ReviewResult (panel mode)."""
    return ReviewResult(
        summary=state.summary,
        findings=list(state.challenged_findings),
        deterministic_findings=list(state.deterministic_findings),
        edits=list(state.edits),
        author_questions=list(state.author_questions),
        document_map=list(state.document_map),
        expert_profiles=list(state.expert_profiles),
        expert_reviews=list(state.expert_reviews),
        panel_synthesis=state.panel_synthesis,
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
