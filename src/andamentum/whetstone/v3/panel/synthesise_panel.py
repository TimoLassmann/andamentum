"""Panel synthesis — one meta-review call over the per-expert reviews."""

from __future__ import annotations

import logging
from typing import cast

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ...schemas import ExpertReview, PanelSynthesis
from .agents import PANEL_SYNTHESISE_DEFN

logger = logging.getLogger("andamentum.whetstone.v3.panel")


def _format_expert_review(r: ExpertReview) -> str:
    """One expert review formatted for the synthesiser's prompt."""
    return (
        f"EXPERT: {r.expert_name} ({r.discipline})\n"
        f"Overall score: {r.overall_score}/10\n"
        f"Recommendation: {r.recommendation}\n"
        f"Rigor: {r.scientific_rigor_score}/10 — {r.scientific_rigor_justification}\n"
        f"Methodology: {r.methodology_score}/10 — {r.methodology_justification}\n"
        f"Novelty: {r.novelty_score}/10 — {r.novelty_justification}\n"
        f"Clarity: {r.clarity_score}/10 — {r.clarity_justification}\n"
        f"Strengths:\n  - " + "\n  - ".join(r.strengths) + "\n"
        "Weaknesses:\n  - " + "\n  - ".join(r.weaknesses) + "\n"
        f"Overall assessment: {r.overall_assessment}\n"
        f"Recommendation rationale: {r.recommendation_justification}"
    )


async def synthesise_panel(
    reviews: list[ExpertReview], *, model: str
) -> tuple[PanelSynthesis | None, str]:
    """Aggregate the expert reviews into one ``PanelSynthesis``.

    Loud-fail-safe: a synthesis crash returns ``(None, failure_note)`` so
    the per-expert reviews still reach the result — the note lands in the
    result's top-level summary. On success the note is ``""`` (renderers
    prefer ``panel_synthesis.review_summary``; a blank top-level summary
    keeps them from double-rendering).
    """
    try:
        review_block = "\n\n---\n\n".join(_format_expert_review(r) for r in reviews)
        agent = build_pydantic_ai_agent(PANEL_SYNTHESISE_DEFN, resolve_model(model))
        result = await agent.run(f"EXPERT REVIEWS TO SYNTHESISE:\n\n{review_block}")
        from .._metrics import bump_from_result

        bump_from_result(result)
        return cast(PanelSynthesis, result.output), ""
    except Exception as exc:
        logger.warning("[panel] synthesis call failed: %s", exc)
        return None, (
            "[Panel synthesis call failed — per-expert reviews are still "
            f"valid below.]\nFailure: {exc}"
        )
