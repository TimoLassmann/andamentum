"""Agent definition and execution with structured output fallback.

Provides:
- AgentDefinition: frozen dataclass describing an agent's config
- AgentRunner: executes agents with caching and PromptedOutput fallback
- run_agent_with_fallback: one-shot agent execution with fallback

The PromptedOutput fallback catches both UnexpectedModelBehavior (model
ignores tool definitions) and ModelHTTPError (model returns invalid
responses that cause HTTP errors on retry, e.g. Ollama/Gemma4 sending
content:null after reasoning-only responses). It retries by injecting
the JSON schema directly into the system prompt. Essential for small
models (Ollama locals, nano-tier APIs) that don't reliably support
tool-based structured output.

Architecture: shared infrastructure, lazy pydantic-ai imports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDefinition:
    """Configuration for a pydantic-ai agent.

    Each definition maps to a pydantic-ai Agent with a system prompt
    (via the instructions parameter) and structured output model.

    ``output_model`` may be ``None`` for agents whose output schema is
    determined at runtime (e.g. custom-criteria reviewers). Such agents
    must be executed with an explicitly-provided output type.
    """

    name: str
    prompt: str
    output_model: type[BaseModel] | None
    retries: int = 3
    output_retries: int = 5
    has_tools: bool = False


class AgentRunner:
    """Executes agents with caching and PromptedOutput fallback.

    Usage::

        runner = AgentRunner(model="openai:gpt-4o")
        result = await runner.run(defn, key="value")
    """

    def __init__(self, *, model: Any):
        from .models import resolve_model

        self.model = resolve_model(model) if isinstance(model, str) else model
        self._cache: dict[str, Any] = {}

    async def run(
        self,
        defn: AgentDefinition,
        *,
        validators: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent with PromptedOutput fallback.

        Args:
            defn: Agent definition with prompt and output model
            validators: Optional output validator callables
            **kwargs: Passed as "key: value" lines in the user message

        Returns:
            The agent's structured output (instance of defn.output_model)
        """
        from pydantic_ai import Agent
        from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

        # Guard: output_model must be concrete for direct execution.
        # Dynamic-schema agents (output_model=None) use _run_one_dynamic.
        if defn.output_model is None:
            raise ValueError(
                f"Agent {defn.name}: output_model is None. "
                "Dynamic-schema agents must be executed via _run_one_dynamic()."
            )

        user_message = "\n".join(f"{k}: {v}" for k, v in kwargs.items())

        if defn.name not in self._cache:
            agent = Agent(
                self.model,
                instructions=defn.prompt,
                output_type=defn.output_model,
                retries=defn.retries,
                output_retries=defn.output_retries,
            )
            if validators:
                for v in validators:
                    agent.output_validator(v)
            self._cache[defn.name] = agent

        try:
            result = await self._cache[defn.name].run(user_message)
            return result.output
        except (UnexpectedModelBehavior, ModelHTTPError):
            logger.info(
                "Agent %s: tool-based output failed, falling back to PromptedOutput",
                defn.name,
            )
            return await self._run_prompted_fallback(defn, user_message, validators)

    async def _run_prompted_fallback(
        self,
        defn: AgentDefinition,
        user_message: str,
        validators: list[Callable[..., Any]] | None = None,
    ) -> Any:
        """Retry with PromptedOutput (schema in prompt, not tools).

        Assumes defn.output_model is not None (enforced by run()).
        """
        from pydantic_ai import Agent
        from pydantic_ai.output import PromptedOutput

        # Guard: output_model must be concrete (enforced by run() before fallback).
        if defn.output_model is None:
            raise ValueError(
                f"Agent {defn.name}: output_model is None in fallback. This should never happen; check run()."
            )

        cache_key = f"{defn.name}__prompted"
        if cache_key not in self._cache:
            agent = Agent(
                self.model,
                instructions=defn.prompt,
                output_type=PromptedOutput(defn.output_model),
                retries=defn.retries,
                output_retries=defn.output_retries,
            )
            if validators:
                for v in validators:
                    agent.output_validator(v)
            self._cache[cache_key] = agent

        result = await self._cache[cache_key].run(user_message)
        return result.output

    def clear_cache(self) -> None:
        """Clear the agent cache."""
        self._cache.clear()


async def run_agent_with_fallback(
    model: Any,
    *,
    instructions: str,
    output_type: type[BaseModel],
    user_message: str,
    retries: int = 3,
    output_retries: int = 5,
    validators: list[Callable[..., Any]] | None = None,
) -> Any:
    """One-shot agent execution with PromptedOutput fallback.

    For callsites that don't need a persistent runner (e.g., document_store
    extraction, query planning). Creates a fresh agent each call.

    Args:
        model: Model string or resolved model object
        instructions: System prompt
        output_type: Pydantic BaseModel class for structured output
        user_message: The user prompt
        retries: Max retries for the agent
        output_retries: Max retries for output validation
        validators: Optional output validator callables

    Returns:
        Instance of output_type
    """
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
    from pydantic_ai.output import PromptedOutput

    from .models import resolve_model

    resolved = resolve_model(model) if isinstance(model, str) else model

    agent = Agent(
        resolved,
        instructions=instructions,
        output_type=output_type,
        retries=retries,
        output_retries=output_retries,
    )
    if validators:
        for v in validators:
            agent.output_validator(v)

    try:
        result = await agent.run(user_message)
        return result.output
    except (UnexpectedModelBehavior, ModelHTTPError):
        logger.info("One-shot agent: falling back to PromptedOutput")
        fallback = Agent(
            resolved,
            instructions=instructions,
            output_type=PromptedOutput(output_type),
            retries=retries,
            output_retries=output_retries,
        )
        if validators:
            for v in validators:
                fallback.output_validator(v)
        result = await fallback.run(user_message)
        return result.output
