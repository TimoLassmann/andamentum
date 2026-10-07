"""Agent definition and execution with structured output fallback.

Provides:
- AgentDefinition: frozen dataclass describing an agent's config
- AgentRunner: executes agents with caching and PromptedOutput fallback
- run_agent_with_fallback: one-shot agent execution with fallback
- run_agent_with_usage: the same, returning (output, token usage)

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

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, TypeVar

if TYPE_CHECKING:
    from pydantic_ai.settings import ModelSettings

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Global concurrency caps for in-flight LLM calls per AgentRunner.
# Local models (Ollama) serialise inference per-process anyway — sending
# concurrent requests just queues them upstream and risks timeout
# cascades; the conservative default of 1 forces strict serial execution
# matching the upstream behaviour. Cloud models can sustain real
# concurrency, so the default is higher; tune via the env var below for
# specific tier limits or testing.
#
# Override with ``ANDAMENTUM_LLM_CONCURRENCY`` (integer). Applies to all
# AgentRunner instances created after the env var is set.
_DEFAULT_LOCAL_CONCURRENCY = 1
_DEFAULT_CLOUD_CONCURRENCY = 8
_CONCURRENCY_ENV_VAR = "ANDAMENTUM_LLM_CONCURRENCY"


def is_local_model(model: Any) -> bool:
    """True if ``model`` (an ``ollama:`` id or a resolved model object) runs locally.

    Currently identifies Ollama models.
    """
    # String form: "ollama:llama3" etc.
    if isinstance(model, str):
        return model.startswith("ollama:")
    # Resolved-object form: check the class name to avoid hard-importing
    # pydantic-ai's optional Ollama provider.
    return type(model).__name__.lower().startswith("ollama")


def llm_concurrency(model: Any) -> int:
    """Maximum in-flight LLM calls for ``model``: 1 if local, else 8.

    The one place the rule lives. ``AgentRunner`` sizes its semaphore with it,
    and code that fans out over LLM calls itself (``deep_research``, ``forge``)
    bounds its fan-out with it via :func:`gather_llm`. ``$ANDAMENTUM_LLM_CONCURRENCY``
    overrides it.
    """
    default = (
        _DEFAULT_LOCAL_CONCURRENCY
        if is_local_model(model)
        else _DEFAULT_CLOUD_CONCURRENCY
    )
    env_override = os.environ.get(_CONCURRENCY_ENV_VAR)
    if env_override is None:
        return default
    try:
        return max(1, int(env_override))
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to auto-detected default",
            _CONCURRENCY_ENV_VAR,
            env_override,
        )
        return default


async def gather_llm(calls: Iterable[Awaitable[T]], *, concurrency: int) -> list[T]:
    """``asyncio.gather`` with at most ``concurrency`` of ``calls`` in flight.

    Results keep the order of ``calls``. ``concurrency=1`` runs them strictly one
    after another. A call that raises propagates, as with ``asyncio.gather``.
    """
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def bounded(call: Awaitable[T]) -> T:
        async with semaphore:
            return await call

    return list(await asyncio.gather(*(bounded(c) for c in calls)))


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

        self._model_input = model  # original spec, kept for introspection
        self.model = resolve_model(model) if isinstance(model, str) else model
        self._cache: dict[str, Any] = {}

        # Global semaphore bounding in-flight LLM calls, sized by
        # ``llm_concurrency`` (local 1, cloud 8, env override).
        concurrency = llm_concurrency(model)
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        logger.info(
            "AgentRunner initialised with concurrency=%d (model=%s, local=%s)",
            concurrency,
            self._model_input,
            self.is_local,
        )

    @property
    def concurrency(self) -> int:
        """Maximum in-flight LLM calls. Read-only."""
        return self._concurrency

    @property
    def is_local(self) -> bool:
        """True if the underlying model runs locally and serialises requests.

        Currently identifies Ollama models. Local models can't actually
        execute parallel requests against the same weights — they queue
        them — so callers should run their agent fan-outs sequentially
        to avoid timeout cascades.
        """
        return is_local_model(self._model_input)

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
        # Dynamic-schema agents (output_model=None) are run with an explicit
        # output type instead — via run_agent_with_fallback(output_type=...)
        # or build_pydantic_ai_agent — not through AgentRunner.run.
        if defn.output_model is None:
            raise ValueError(
                f"Agent {defn.name}: output_model is None. Dynamic-schema "
                "agents must be run with an explicit output type — e.g. "
                "run_agent_with_fallback(output_type=...) or "
                "build_pydantic_ai_agent — not via AgentRunner.run."
            )

        user_message = "\n".join(f"{k}: {v}" for k, v in kwargs.items())

        if defn.name not in self._cache:
            agent = Agent(
                self.model,
                instructions=defn.prompt,
                output_type=defn.output_model,
                retries={"tools": defn.retries, "output": defn.output_retries},
            )
            if validators:
                for v in validators:
                    agent.output_validator(v)
            self._cache[defn.name] = agent

        async with self._semaphore:
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
                retries={"tools": defn.retries, "output": defn.output_retries},
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


def build_pydantic_ai_agent(defn: AgentDefinition, model: Any) -> Any:
    """Build a pydantic-ai ``Agent`` from an ``AgentDefinition`` + model.

    Used by sub-modules whose graph nodes need direct access to the agent
    (for ``deps=``, custom ``RunContext``, tools, etc.) — places where
    ``AgentRunner.run`` doesn't fit because the node manages the call
    itself. This helper centralises the (instructions, output_type, retries)
    shape so every node-based caller agrees on the construction recipe.

    Callers are responsible for resolving the model upstream (typically via
    ``core.models.resolve_model``) so this stays a pure shape adapter.

    Raises:
        ValueError: If ``defn.output_model`` is None — node-based agents
            must declare a concrete output type.
    """
    from pydantic_ai import Agent

    if defn.output_model is None:
        raise ValueError(
            f"Agent {defn.name}: output_model is None. "
            "build_pydantic_ai_agent only supports agents with a concrete output_type."
        )
    return Agent(
        model,
        instructions=defn.prompt,
        output_type=defn.output_model,
        retries={"tools": defn.retries, "output": defn.output_retries},
    )


async def run_agent_with_usage(
    model: Any,
    *,
    instructions: str,
    output_type: type[BaseModel],
    user_message: str,
    retries: int = 3,
    output_retries: int = 5,
    validators: list[Callable[..., Any]] | None = None,
    model_settings: ModelSettings | None = None,
) -> tuple[Any, Any]:
    """One-shot agent execution with PromptedOutput fallback, plus token usage.

    Same behaviour and arguments as :func:`run_agent_with_fallback`; returns
    ``(output, usage)`` where ``usage`` is pydantic-ai's ``RunUsage`` for the
    run that produced the output (when the tool-based attempt failed and the
    PromptedOutput retry answered, only the retry's usage is counted).

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
        model_settings: Optional pydantic-ai model settings (e.g. temperature)
            for both the first attempt and the PromptedOutput retry

    Returns:
        ``(instance of output_type, RunUsage)``
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
        retries={"tools": retries, "output": output_retries},
    )
    if validators:
        for v in validators:
            agent.output_validator(v)

    try:
        result = await agent.run(user_message, model_settings=model_settings)
        return result.output, result.usage
    except (UnexpectedModelBehavior, ModelHTTPError):
        logger.info("One-shot agent: falling back to PromptedOutput")
        fallback = Agent(
            resolved,
            instructions=instructions,
            output_type=PromptedOutput(output_type),
            retries={"tools": retries, "output": output_retries},
        )
        if validators:
            for v in validators:
                fallback.output_validator(v)
        result = await fallback.run(user_message, model_settings=model_settings)
        return result.output, result.usage


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

    Returns the instance of ``output_type``; see :func:`run_agent_with_usage`
    for the arguments and for a variant that also returns token usage.
    """
    output, _usage = await run_agent_with_usage(
        model,
        instructions=instructions,
        output_type=output_type,
        user_message=user_message,
        retries=retries,
        output_retries=output_retries,
        validators=validators,
    )
    return output
