"""The extractor: per-window calls and the main loop.

Per-window flow:
  1. Build window with lookahead
  2. Build prompt
  3. Call executor (fresh Agent per call so validators register correctly)
  4. Return ExtractionAttempt (success, skip, or error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .types import NextUnitResult
from .validation import make_validator
from .windowing import Window


ExecutorFn = Callable[..., Awaitable[Any]]
"""Async signature: fn(*, instructions, user_message, output_type, validators) -> Any"""


@dataclass
class ExtractionAttempt:
    """Outcome of a single _extract_one call."""

    result: Optional[NextUnitResult]
    calls: int
    error: Optional[Exception]


async def _extract_one(
    *,
    executor: ExecutorFn,
    window: Window,
    domain: str,
    prior_unit_titles: list[str],
) -> ExtractionAttempt:
    """One LLM call to extract the next unit (or report 'nothing here')."""
    user_prompt = build_user_prompt(
        window_text=window.text,
        domain=domain,
        window_size=window.window_end_offset - window.cursor,
        prior_unit_titles=prior_unit_titles,
    )
    validator = make_validator(window)

    try:
        result = await executor(
            instructions=SYSTEM_PROMPT,
            user_message=user_prompt,
            output_type=NextUnitResult,
            validators=[validator],
        )
        return ExtractionAttempt(result=result, calls=1, error=None)
    except Exception as exc:
        return ExtractionAttempt(result=None, calls=1, error=exc)


def make_runner_executor(runner: Any) -> ExecutorFn:
    """Build a production executor from an AgentRunner (uses its model).

    Wraps `core.run_agent_with_fallback` to create a fresh Agent each call,
    so per-call validators register correctly (AgentRunner's caching would
    skip them).
    """
    from andamentum.core.agents import run_agent_with_fallback

    async def executor(*, instructions, user_message, output_type, validators):
        return await run_agent_with_fallback(
            model=runner.model,
            instructions=instructions,
            user_message=user_message,
            output_type=output_type,
            validators=validators,
        )

    return executor
