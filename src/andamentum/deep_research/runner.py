"""DefaultResearchRunner — deep research agent execution via core.AgentRunner.

Wraps andamentum.core.agents.AgentRunner with name-based agent lookup
from the deep research agent registry. Gains PromptedOutput fallback
and bedrock support from core.

Architecture: Layer 1 (standalone package runner)
"""

from typing import Any

# Re-export so orchestrator.py can continue `from .runner import _resolve_model`
from andamentum.core.models import resolve_model as _resolve_model  # noqa: F401


class DefaultResearchRunner:
    """Run deep-research agents using core.AgentRunner with name-based lookup.

    Usage::

        runner = DefaultResearchRunner(model="bedrock:claude-haiku-4-5")
        result = await runner.run("gap_analyzer", evidence="...", question="...")
    """

    def __init__(
        self,
        model: str,
        backend: Any = None,  # SearchBackend — typed loosely to avoid importing backends
    ):
        from andamentum.core.agents import AgentRunner

        self._runner = AgentRunner(model=model)
        self.model = model
        self._backend = backend

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        """Run an agent by name.

        Args:
            agent_name: Registered agent name (e.g. "query_generator")
            **kwargs: Key-value pairs formatted into the user message

        Returns:
            Pydantic model instance matching the agent's output_model
        """
        from .agents import get_agent

        defn = get_agent(agent_name)
        return await self._runner.run(defn, **kwargs)  # type: ignore[arg-type]
