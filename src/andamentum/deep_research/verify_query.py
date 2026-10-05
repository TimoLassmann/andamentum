"""Worker: judge one candidate query against the research goal.

Engine-free (L2): the ``topic_verifier`` agent answers accept/reject for
a single query; the ``Verify`` node owns the routing (retry, accept,
skip-and-tighten) that branches on the verdict.

With ``decision_model`` the local decision model answers one yes/no question
instead of the agent (yes when p >= 0.5); a rejection gives the generator the
fixed ``REJECT_FEEDBACK``.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.usage import UsageLimits

from andamentum.core.decisions import DecisionModel, Question

from .build_agent import AgentOverrides, build_agent
from .models import VerifierOutput

logger = logging.getLogger(__name__)

# In-agent (client-level) retry ceiling for the verifier call.
VERIFIER_REQUEST_LIMIT = 5

# The one question put to the decision model.
QUESTION = Question(
    type="noul",
    instructions=(
        "Does this search query seek information that answers the research "
        "goal? Answer yes when the query has the same intent as the goal, "
        "also in other words or specialist terms. Answer no when it only "
        "shares words with the goal but asks about something else: another "
        "entity, another aspect, or another subject."
    ),
)

# What the generator is told when the gate rejects (the verifier's own reason
# text is not available then, and a fixed string is predictable).
REJECT_FEEDBACK = (
    "The query does not ask for what the research goal asks for; it only "
    "overlaps in vocabulary. Keep the intent of the goal and name the specific "
    "thing it asks about."
)


async def verify_query(
    query: str,
    *,
    goal: str,
    model: Any,
    overrides: AgentOverrides | None = None,
    decision_model: DecisionModel | None = None,
) -> VerifierOutput:
    """Ask whether ``query`` helps answer ``goal``.

    The ``topic_verifier`` agent decides, unless ``decision_model`` is given:
    then its answer to ``QUESTION`` (p >= 0.5) decides and the agent is not
    called.
    """
    if decision_model is not None:
        answers = await decision_model.decide(
            {"research_goal": goal, "query": query}, {"q": QUESTION}
        )
        p_true = answers["q"].p_true
        assert p_true is not None  # a noul answer always has it
        if p_true >= 0.5:
            return VerifierOutput(
                on_topic=True,
                reason=f"decision model: on topic (p={p_true:.2f})",
            )
        return VerifierOutput(on_topic=False, reason=REJECT_FEEDBACK)

    agent = build_agent("topic_verifier", model, overrides)
    result = await agent.run(
        f"research_goal: {goal}\nquery: {query}",
        usage_limits=UsageLimits(request_limit=VERIFIER_REQUEST_LIMIT),
    )
    verdict: VerifierOutput = result.output
    return verdict
