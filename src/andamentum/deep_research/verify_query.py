"""Worker: judge one candidate query against the research goal.

Engine-free (L2): the ``topic_verifier`` agent answers accept/reject for
a single query; the ``Verify`` node owns the routing (retry, accept,
skip-and-tighten) that branches on the verdict.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.usage import UsageLimits

from .build_agent import AgentOverrides, build_agent
from .models import VerifierOutput

# In-agent (client-level) retry ceiling for the verifier call.
VERIFIER_REQUEST_LIMIT = 5


async def verify_query(
    query: str,
    *,
    goal: str,
    model: Any,
    overrides: AgentOverrides | None = None,
) -> VerifierOutput:
    """Ask the ``topic_verifier`` agent whether ``query`` helps answer ``goal``."""
    agent = build_agent("topic_verifier", model, overrides)
    result = await agent.run(
        f"research_goal: {goal}\nquery: {query}",
        usage_limits=UsageLimits(request_limit=VERIFIER_REQUEST_LIMIT),
    )
    verdict: VerifierOutput = result.output
    return verdict
