"""Node: ExpertReview (panel mode).

N parallel LLM calls — one per ExpertProfile produced by the previous
node. Each call asks the expert_reviewer agent to roleplay as the
profile and produce a structured ExpertReview.

The document view passed to each expert is the document_map plus a
trimmed markdown excerpt — keeping the prompt tractable for any model
while still giving the reviewer enough surface to score rigour /
methodology / novelty / clarity.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import ExpertProfile, ExpertReview as ExpertReviewSchema, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .panel_synthesise import PanelSynthesise


logger = logging.getLogger("andamentum.whetstone")

# Dropped from 5 → 2 to avoid stale-connection / NAT-table saturation.
_MAX_CONCURRENT_REVIEWS = 2
_MAX_DOCUMENT_CHARS = 30_000  # generous — single section excerpt per expert is small


@dataclass
class ExpertReview(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run each expert's review in parallel; populate state.expert_reviews."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "PanelSynthesise":
        ctx.state.current_phase = "expert_review"

        profiles = list(ctx.state.expert_profiles)
        if not profiles:
            logger.warning(
                "[panel] no expert profiles — skipping reviews and "
                "panel synthesis (will be a no-op)"
            )
            from .panel_synthesise import PanelSynthesise

            return PanelSynthesise()

        logger.info(
            "[panel] running %d expert review(s) (concurrency=%d)",
            len(profiles),
            _MAX_CONCURRENT_REVIEWS,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT_REVIEWS)
        document_view = _build_document_view(ctx.state)

        async def review_one(profile: ExpertProfile) -> ExpertReviewSchema | None:
            async with sem:
                try:
                    return await _run_review(ctx.deps, profile, document_view)
                except Exception as exc:
                    logger.warning(
                        "[panel] review by %r (%s) failed: %s",
                        profile.name,
                        profile.discipline,
                        exc,
                    )
                    return None

        results = await asyncio.gather(*[review_one(p) for p in profiles])
        ctx.state.expert_reviews = [r for r in results if r is not None]
        ctx.state.llm_calls += len(ctx.state.expert_reviews)
        logger.info(
            "[panel] %d/%d review(s) completed",
            len(ctx.state.expert_reviews),
            len(profiles),
        )

        from .panel_synthesise import PanelSynthesise

        return PanelSynthesise()


async def _run_review(
    deps: ReviewDeps, profile: ExpertProfile, document_view: str
) -> ExpertReviewSchema:
    """Single expert_reviewer call. Restores expert_name/discipline if dropped."""
    prompt = f"""EXPERT BIOSKETCH (your persona for this review):
  Name: {profile.name}
  Position: {profile.position}
  Education: {profile.education}
  Contributions: {profile.contributions}
  Research: {profile.research}
  Discipline: {profile.discipline}

DOCUMENT TO REVIEW:
{document_view}

Roleplay as the expert above and produce a structured ExpertReview.
Echo the expert's name and discipline back in the corresponding
output fields verbatim."""

    agent = build_pydantic_ai_agent("expert_reviewer", deps.model)
    result = await agent.run(prompt)
    review = cast(ExpertReviewSchema, result.output)
    # Defensive: if the agent forgot the persona fields, restore them.
    fixes: dict[str, str] = {}
    if not review.expert_name.strip():
        fixes["expert_name"] = profile.name
    if not review.discipline.strip():
        fixes["discipline"] = profile.discipline
    if fixes:
        review = review.model_copy(update=fixes)
    return review


def _build_document_view(state: ReviewState) -> str:
    """Document map + truncated markdown — same shape across experts."""
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
