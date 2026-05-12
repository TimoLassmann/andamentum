"""DefaultAgentRunner — epistemic agent execution via core.AgentRunner.

Wraps andamentum.core.agents.AgentRunner with name-based agent lookup
from the epistemic agent registry and token usage tracking.

Architecture: Layer 1 (standalone package runner)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    """Run epistemic agents using core.AgentRunner with name-based lookup.

    Implements the AgentRunner protocol::

        async def run(self, agent_name: str, **kwargs: Any) -> Any

    Wraps core.AgentRunner (which provides model resolution, agent caching,
    and PromptedOutput fallback) with epistemic-specific features:
    - Name-based agent lookup from AGENT_REGISTRY
    - Token usage accumulation on ``self.usage``
    - Health check for LLM connectivity
    """

    def __init__(self, *, model: str):
        # Load .env from CWD so importing repos get their API keys picked up
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        from andamentum.core.agents import AgentRunner

        self._runner = AgentRunner(model=model)
        self.model = model
        self.usage = UsageSummary()

    async def __aenter__(self) -> "DefaultAgentRunner":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._runner.clear_cache()

    @property
    def core_runner(self) -> Any:
        """Expose the underlying ``core.agents.AgentRunner``.

        Used by code paths that need the definition-based call shape
        ``.run(defn, **kwargs)`` rather than the name-based wrapper
        this class normally provides — e.g.
        ``dispatch.gather_evidence_new`` calls
        ``formulate_provider_query`` which constructs the definition
        itself and passes it to ``runner.run``.
        """
        return self._runner

    async def check_health(self) -> "CheckResult":
        """Test LLM connectivity with a minimal inference call."""
        import time

        from pydantic_ai import Agent

        from .preflight import CheckResult

        t0 = time.monotonic()
        try:
            agent = Agent(
                self._runner.model,
                instructions="Reply with exactly: ok",
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
        from .agents import get_agent

        defn = get_agent(agent_name)
        return await self._runner.run(defn, **kwargs)
