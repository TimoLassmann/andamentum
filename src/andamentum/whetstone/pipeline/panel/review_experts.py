"""Per-expert reviews — N parallel roleplay calls, failure-isolated."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ...schemas import ExpertProfile, ExpertReview
from ..model import Section
from .agents import EXPERT_REVIEWER_DEFN
from .document_view import build_document_view

logger = logging.getLogger("andamentum.whetstone.v3.panel")

_MAX_CONCURRENT_REVIEWS: int = 2


async def review_experts(
    profiles: list[ExpertProfile],
    source: str,
    sections: list[Section],
    *,
    model: str,
) -> list[ExpertReview]:
    """One independent review per expert profile (semaphore=2, matching
    v2's calibration). A crashed review leaves a hole that is filtered
    out — partial success is acceptable; the caller records how many
    were attempted so the aggregate-failure gate can judge the run."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_REVIEWS)
    document_view = build_document_view(source, sections)

    async def _review_one(profile: ExpertProfile) -> ExpertReview | None:
        async with sem:
            try:
                agent = build_pydantic_ai_agent(
                    EXPERT_REVIEWER_DEFN, resolve_model(model)
                )
                result = await agent.run(
                    f"EXPERT BIOSKETCH:\n"
                    f"  Name: {profile.name}\n"
                    f"  Position: {profile.position}\n"
                    f"  Education: {profile.education}\n"
                    f"  Contributions: {profile.contributions}\n"
                    f"  Research: {profile.research}\n"
                    f"  Discipline: {profile.discipline}\n\n"
                    f"DOCUMENT TO REVIEW:\n{document_view}"
                )
                from .._metrics import bump_from_result

                bump_from_result(result)
                review = cast(ExpertReview, result.output)
                # Defensive: agents occasionally drop the identity fields.
                if not review.expert_name:
                    review = review.model_copy(update={"expert_name": profile.name})
                if not review.discipline:
                    review = review.model_copy(
                        update={"discipline": profile.discipline}
                    )
                return review
            except Exception as exc:
                logger.warning(
                    "[panel] review by %r (%s) failed: %s",
                    profile.name,
                    profile.discipline,
                    exc,
                )
                return None

    results = await asyncio.gather(*[_review_one(p) for p in profiles])
    reviews = [r for r in results if r is not None]
    logger.info("[panel] %d/%d review(s) completed", len(reviews), len(profiles))
    return reviews
