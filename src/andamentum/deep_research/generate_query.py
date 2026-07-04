"""Worker: draft one candidate search query via the ``query_generator`` agent.

Engine-free (L2): explicit inputs in, a ``GeneratorOutput`` out. Called
once per slot by the ``GenerateOne`` node; on a retry within the same
slot the previous verifier's reason arrives via ``feedback`` so the
generator can correct the rejected query without losing strategic
context.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import UsageLimits

from .build_agent import AgentOverrides, build_agent
from .models import GeneratorOutput
from .reporter import NOOP_REPORTER, SearchReporter

# In-agent (client-level) retry ceiling for the generator call. The
# slot-retry loop in the graph is bounded separately by MAX_SLOT_RETRIES
# in nodes.py.
GENERATOR_REQUEST_LIMIT = 10


def _prompt(
    goal: str,
    *,
    validated_queries: list[str],
    gaps: list[str],
    rejected_queries: list[str],
    feedback: str | None,
) -> str:
    parts = [f"research_goal: {goal}"]
    if validated_queries:
        parts.append(f"validated_queries: {', '.join(validated_queries)}")
    else:
        parts.append("validated_queries: (none yet)")
    if gaps:
        parts.append(f"gaps: {', '.join(gaps)}")
    if rejected_queries:
        parts.append("already_rejected_in_this_slot: " + ", ".join(rejected_queries))
    if feedback:
        parts.append(f"feedback: {feedback}")
    return "\n".join(parts)


async def generate_query(
    goal: str,
    *,
    validated_queries: list[str],
    gaps: list[str],
    rejected_queries: list[str],
    feedback: str | None,
    slot: int,
    attempt: int,
    model: Any,
    overrides: AgentOverrides | None = None,
    reporter: SearchReporter = NOOP_REPORTER,
) -> GeneratorOutput:
    """Produce one search query for the current slot.

    ``validated_queries`` and ``rejected_queries`` are threaded into the
    prompt so the generator can see (and avoid paraphrasing) what it
    already produced. ``slot``/``attempt`` are progress labels for the
    reporter only.
    """
    agent = build_agent("query_generator", model, overrides)
    result = await agent.run(
        _prompt(
            goal,
            validated_queries=validated_queries,
            gaps=gaps,
            rejected_queries=rejected_queries,
            feedback=feedback,
        ),
        usage_limits=UsageLimits(request_limit=GENERATOR_REQUEST_LIMIT),
    )
    out: GeneratorOutput = result.output
    reporter.query_generated(
        slot=slot,
        attempt=attempt,
        query=out.query,
        rationale=out.rationale,
    )
    return out
