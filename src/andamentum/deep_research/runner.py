"""DefaultResearchRunner — standalone agent execution using pydantic-ai.

More complex than epistemic's runner because deep-research agents use tools
(search, fetch). The runner registers tool functions that call SearchBackend.

Requires the [llm] optional extra: ``pip install mosaic-deep-research[llm]``

Architecture: Layer 1 (standalone package runner)
"""

from typing import Any


def _resolve_model(model: str) -> Any:
    """Resolve model string to a pydantic-ai model object.

    Handles provider-specific setup:
    - ``ollama:`` — sets default OLLAMA_BASE_URL for localhost
    - everything else — passed through to pydantic-ai's infer_model
    """
    import os

    if isinstance(model, str) and model.startswith("ollama:"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model_name = model.split(":", 1)[1]
        return OpenAIChatModel(model_name=model_name, provider=OllamaProvider(base_url=base_url))

    return model


class DefaultResearchRunner:
    """Run deep-research agents using pydantic-ai directly.

    Tool-using agents (search_planner, page_fetcher, lead_agent) get tools
    registered that call the SearchBackend. Pure-output agents (page_summarizer,
    gap_analyzer, novelty_assessor) work like epistemic's runner.

    Usage::

        from deep_research.runner import DefaultResearchRunner
        from deep_research.backends import HttpxSearchBackend

        runner = DefaultResearchRunner()  # uses local Ollama by default
        result = await runner.run("gap_analyzer", evidence="...", question="...")
    """

    def __init__(
        self,
        model: str,
        backend: Any = None,  # SearchBackend — typed loosely to avoid importing backends
    ):
        try:
            from pydantic_ai import Agent
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai is required for DefaultResearchRunner. "
                "Install with: pip install mosaic-deep-research[llm]"
            ) from exc

        self._Agent = Agent
        self.model = _resolve_model(model)
        self._backend = backend
        self._cache: dict[str, Any] = {}

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        """Run an agent by name.

        Args:
            agent_name: Registered agent name (e.g. "search_planner")
            **kwargs: Key-value pairs formatted into the user message

        Returns:
            Pydantic model instance matching the agent's output_model
        """
        from .agents import AGENT_REGISTRY

        defn = AGENT_REGISTRY.get(agent_name)
        if defn is None:
            raise ValueError(
                f"Unknown deep-research agent: {agent_name}. "
                f"Available: {sorted(AGENT_REGISTRY)}"
            )

        if agent_name not in self._cache:
            agent = self._Agent(
                self.model,
                instructions=defn.prompt,
                output_type=defn.output_model,
                retries=defn.retries,
            )
            self._cache[agent_name] = agent

        user_message = "\n".join(f"{k}: {v}" for k, v in kwargs.items())
        result = await self._cache[agent_name].run(user_message)
        return result.output
