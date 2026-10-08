"""Per-run LLM call + gap-round counters.

Held in a ``contextvars.ContextVar`` so concurrent ``asyncio.gather``'d
tasks all increment the same object instance (asyncio tasks inherit
their parent's context snapshot; the snapshot is shallow, so the
mutable counter object is shared across tasks).

Used by ``run_review`` to populate ``ReviewMetrics.llm_calls`` and
``ReviewMetrics.reflection_rounds_used`` honestly — before this module
existed those fields silently reported ``0`` because no code
counted the calls.
Panel-mode (``run_panel``) uses the same mechanism: its workers bump
the shared counter after each agent run (a State counter cannot reach
concurrently-gathered calls).

Single-call-per-task contract: ``start_run()`` resets the counter for
the current context. Concurrent ``run_review`` calls from the same
task context would share counters; in practice each top-level call
runs in its own asyncio task and gets its own context, so this is
not a problem.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class ReviewCounters:
    """Mutable counter object — fields are bumped from many concurrent
    coroutines via the shared contextvar reference."""

    llm_calls: int = 0
    gap_rounds: int = 0


_CTX: ContextVar[ReviewCounters | None] = ContextVar("counters", default=None)


def start_run() -> ReviewCounters:
    """Initialise a fresh counter for the current context. Returns the
    counter so the caller can read it after the run completes."""
    c = ReviewCounters()
    _CTX.set(c)
    return c


def current() -> ReviewCounters | None:
    """The active counter for this context, or None if no run is in
    progress. Helpers tolerate None — increments are silently dropped
    so importing this module never breaks an isolated unit test."""
    return _CTX.get()


def increment_llm_calls(n: int = 1) -> None:
    """Bump the LLM-call counter by ``n``. No-op outside a run."""
    c = _CTX.get()
    if c is not None:
        c.llm_calls += n


def increment_gap_rounds(n: int = 1) -> None:
    """Bump the gap-loop-round counter by ``n``. No-op outside a run."""
    c = _CTX.get()
    if c is not None:
        c.gap_rounds += n


def bump_from_result(result) -> None:
    """Bump ``llm_calls`` by pydantic-ai's ``result.usage.requests`` when
    available — that captures tool-call expansion (an agent.run with three
    tool turns is reported as 4 requests). Falls back to +1 when usage is
    absent or unparseable, so the counter is an honest lower bound.

    ``usage`` is a PROPERTY in pydantic-ai v2 (it was a method in v1).
    Reading it as a method raised ``TypeError`` into the except-branch below,
    which silently degraded every count to the +1 fallback — invisible,
    because the fallback is a legitimate outcome. Hence the narrow except:
    a missing/None usage is expected and tolerated, but a wrong-shaped API
    is a bug and must surface."""
    n = 1
    usage = getattr(result, "usage", None)
    if usage is not None:
        candidate = getattr(usage, "requests", None)
        if isinstance(candidate, int) and candidate >= 1:
            n = candidate
    increment_llm_calls(n)
