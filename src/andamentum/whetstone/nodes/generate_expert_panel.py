"""Node: GenerateExpertPanel (panel mode).

N parallel LLM calls (asyncio.gather). One ``ExpertProfile`` per
discipline, capped at ``state.n_experts`` so a 7-discipline document
doesn't blow the budget.

If a single profile call fails, log a warning and continue — the
remaining profiles still produce a valid (smaller) panel rather than
abort the whole run.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import ExpertProfile, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .expert_review import ExpertReview


logger = logging.getLogger("andamentum.whetstone")

# Cap on parallelism so we don't overload local model servers / hit
# cloud rate limits on tiny accounts.
_MAX_CONCURRENT_GENERATIONS = 5


@dataclass
class GenerateExpertPanel(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Generate one fictional ExpertProfile per discipline (capped at n_experts)."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "ExpertReview":
        ctx.state.current_phase = "generate_panel"

        # Cap disciplines to n_experts so a 7-discipline doc → 4-expert panel.
        disciplines = list(ctx.state.disciplines)[: ctx.state.n_experts]
        if not disciplines:
            logger.warning(
                "[panel] no disciplines available — skipping panel generation"
            )
            from .expert_review import ExpertReview

            return ExpertReview()

        logger.info(
            "[panel] generating %d expert profile(s) (concurrency=%d)",
            len(disciplines),
            _MAX_CONCURRENT_GENERATIONS,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT_GENERATIONS)

        async def gen_one(discipline: str) -> ExpertProfile | None:
            async with sem:
                try:
                    return await _generate_profile(ctx.deps, discipline)
                except Exception as exc:
                    logger.warning(
                        "[panel] profile generation for %r failed: %s",
                        discipline,
                        exc,
                    )
                    return None

        results = await asyncio.gather(*[gen_one(d) for d in disciplines])
        ctx.state.expert_profiles = [p for p in results if p is not None]
        ctx.state.llm_calls += len(ctx.state.expert_profiles)
        logger.info(
            "[panel] generated %d/%d profile(s) successfully",
            len(ctx.state.expert_profiles),
            len(disciplines),
        )

        from .expert_review import ExpertReview

        return ExpertReview()


async def _generate_profile(deps: ReviewDeps, discipline: str) -> ExpertProfile:
    """Single expert_generator call for one discipline."""
    prompt = f"""DISCIPLINE: {discipline}

Generate a realistic but fictional senior-expert biosketch for this
discipline. Echo the discipline back verbatim in the discipline field
of your output."""

    agent = build_pydantic_ai_agent("expert_generator", deps.model)
    result = await agent.run(prompt)
    profile = cast(ExpertProfile, result.output)
    # Defensive: if the agent dropped the discipline, restore it.
    if not profile.discipline.strip():
        profile = profile.model_copy(update={"discipline": discipline})
    return profile
