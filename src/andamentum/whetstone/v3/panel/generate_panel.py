"""Expert-profile generation — N parallel biosketch calls, failure-isolated."""

from __future__ import annotations

import asyncio
import logging
from typing import cast

from andamentum.core.agents import build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ...schemas import ExpertProfile
from .agents import EXPERT_GENERATOR_DEFN

logger = logging.getLogger("andamentum.whetstone.v3.panel")

_MAX_CONCURRENT_GENERATIONS: int = 2


async def generate_panel(disciplines: list[str], *, model: str) -> list[ExpertProfile]:
    """One fictional biosketch per discipline (semaphore=2, matching v2's
    calibration). A crashed generation leaves a hole that is filtered out
    — partial success is acceptable; the caller records how many were
    attempted so the aggregate-failure gate can judge the run."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)

    async def _gen_one(discipline: str) -> ExpertProfile | None:
        async with sem:
            try:
                agent = build_pydantic_ai_agent(
                    EXPERT_GENERATOR_DEFN, resolve_model(model)
                )
                result = await agent.run(
                    f"Generate a fictional expert biosketch for the discipline: "
                    f"{discipline}"
                )
                from .._metrics import bump_from_result

                bump_from_result(result)
                profile = cast(ExpertProfile, result.output)
                # Defensive: agents occasionally drop the discipline field.
                if not profile.discipline:
                    profile = profile.model_copy(update={"discipline": discipline})
                return profile
            except Exception as exc:
                logger.warning(
                    "[panel] profile generation for %r failed: %s",
                    discipline,
                    exc,
                )
                return None

    results = await asyncio.gather(*[_gen_one(d) for d in disciplines])
    profiles = [p for p in results if p is not None]
    logger.info(
        "[panel] generated %d/%d profile(s) successfully",
        len(profiles),
        len(disciplines),
    )
    return profiles
