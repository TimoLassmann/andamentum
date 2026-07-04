"""Shared agent builder for the deep-research worker layer.

Every worker that talks to the model resolves its registry
``AgentDefinition`` through this one construction recipe (delegating to
``andamentum.core.agents.build_pydantic_ai_agent``), honouring test-only
overrides so plumbing tests can substitute stub Agents without a live
model.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic_ai.usage import UsageLimits

from .agents import get_agent


class AgentLike(Protocol):
    """Structural surface of a pydantic-ai ``Agent`` as the workers use it.

    A ``Protocol`` (the dialect's test-substitution discipline): plumbing
    tests inject scripted stub agents that satisfy this surface without
    subclassing ``Agent``.
    """

    async def run(
        self,
        user_prompt: str,
        /,
        *,
        usage_limits: UsageLimits | None = None,
    ) -> Any: ...


# Test-only: maps agent name → Agent (or stub). Threaded from
# ``NodeDeps.agent_overrides`` into every worker; production code passes
# ``None``.
AgentOverrides = dict[str, AgentLike]


def build_agent(
    name: str,
    model: Any,
    overrides: AgentOverrides | None = None,
) -> AgentLike:
    """Create a pydantic-ai ``Agent`` from a registry definition.

    If ``overrides`` contains ``name``, the override (typically a stub
    Agent for tests) is returned instead of building from the registry.
    Production code never sets ``overrides``; tests pass it via
    ``NodeDeps.agent_overrides``.
    """
    if overrides and name in overrides:
        return overrides[name]
    from andamentum.core.agents import build_pydantic_ai_agent

    return build_pydantic_ai_agent(get_agent(name), model)
