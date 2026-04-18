"""DefaultAgentRunner — standalone agent execution using pydantic-ai.

This runner implements the AgentRunner protocol defined in operations.py,
enabling the epistemic package to run agents standalone, without an external SDK.

Architecture: Layer 1 (standalone package runner)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from andamentum.core.models import resolve_model as _resolve_model

if TYPE_CHECKING:
    from .preflight import CheckResult


@dataclass
class UsageSummary:
    """Accumulated token usage across all agent calls.

    Attributes:
        input_tokens: Total input tokens consumed
        output_tokens: Total output tokens consumed
        requests: Total LLM requests made
        tool_calls: Total tool calls made
        details: Per-model breakdown {model_name: token_count}
    """

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    details: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens



class DefaultAgentRunner:
    """Run epistemic agents using pydantic-ai directly.

    Implements the AgentRunner protocol::

        async def run(self, agent_name: str, **kwargs: Any) -> Any

    Token usage is accumulated on ``self.usage`` (a ``UsageSummary``)
    across all agent calls. Read it after the run completes::

        runner = DefaultAgentRunner(model="openai:gpt-4o")
        result = await runner.run("epistemic_clarify_question", question="...")
        print(runner.usage.total_tokens)
    """

    def __init__(self, *, model: str):
        # Load .env from CWD so importing repos get their API keys picked up
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        try:
            from pydantic_ai import Agent  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pydantic-ai is required for DefaultAgentRunner. "
                "Install with: pip install andamentum"
            ) from exc

        self._Agent = Agent
        self.model = _resolve_model(model)
        self._cache: dict[str, Any] = {}
        self.usage = UsageSummary()

    async def __aenter__(self) -> "DefaultAgentRunner":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._cache.clear()

    async def check_health(self) -> "CheckResult":
        """Test LLM connectivity with a minimal inference call."""
        import time

        from .preflight import CheckResult

        t0 = time.monotonic()
        try:
            agent = self._Agent(
                self.model,
                system_prompt="Reply with exactly: ok",
                output_type=str,
            )
            await agent.run("health check")
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="LLM",
                status="pass",
                message=f"Model responded ({elapsed:.0f}ms)",
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="LLM", status="fail", message=str(e), elapsed_ms=elapsed
            )

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        """Run an epistemic agent by name.

        Args:
            agent_name: Registered agent name (e.g. "epistemic_clarify_question")
            **kwargs: Key-value pairs formatted into the user message

        Returns:
            Pydantic model instance matching the agent's output_model
        """
        from .agents import AGENT_REGISTRY

        defn = AGENT_REGISTRY.get(agent_name)
        if defn is None:
            raise ValueError(
                f"Unknown epistemic agent: {agent_name}. "
                f"Available: {sorted(AGENT_REGISTRY)}"
            )

        if agent_name not in self._cache:
            self._cache[agent_name] = self._Agent(
                self.model,
                system_prompt=defn.prompt,
                output_type=defn.output_model,
                retries=defn.retries,
                output_retries=defn.output_retries,
            )

        user_message = "\n".join(f"{k}: {v}" for k, v in kwargs.items())

        try:
            result = await self._cache[agent_name].run(user_message)
        except Exception as first_error:
            # If structured output validation failed after all retries,
            # retry once with PromptedOutput — injects the JSON schema
            # directly into the system prompt instead of using tool calls.
            # This helps small models that ignore tool definitions.
            from pydantic_ai.exceptions import UnexpectedModelBehavior

            if not isinstance(first_error, UnexpectedModelBehavior):
                raise

            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                "Agent %s: tool-based output failed, retrying with prompted output",
                agent_name,
            )

            from pydantic_ai import PromptedOutput

            prompted_key = f"{agent_name}__prompted"
            if prompted_key not in self._cache:
                self._cache[prompted_key] = self._Agent(
                    self.model,
                    system_prompt=defn.prompt,
                    output_type=PromptedOutput(defn.output_model),
                    retries=defn.retries,
                    output_retries=defn.output_retries,
                )

            result = await self._cache[prompted_key].run(user_message)

        # Accumulate token usage
        run_usage = result.usage()
        self.usage.input_tokens += run_usage.input_tokens
        self.usage.output_tokens += run_usage.output_tokens
        self.usage.requests += run_usage.requests
        self.usage.tool_calls += run_usage.tool_calls
        for key, val in run_usage.details.items():
            self.usage.details[key] = self.usage.details.get(key, 0) + val

        return result.output
