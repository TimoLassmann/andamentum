"""Per-run LLM call + gap-round counters.

Held in a ``contextvars.ContextVar`` so concurrent ``asyncio.gather``'d
tasks all increment the same object instance (asyncio tasks inherit
their parent's context snapshot; the snapshot is shallow, so the
mutable counter object is shared across tasks).

Used by ``run_review_v3`` to populate ``ReviewMetrics.llm_calls`` and
``ReviewMetrics.reflection_rounds_used`` honestly — before this module
existed those fields silently reported ``0`` in v3 because the v2
manual-increment plumbing was never ported across the consolidation.
Panel-mode (``run_panel_v3``) uses the same mechanism: its workers bump
the shared counter after each agent run (a State counter cannot reach
concurrently-gathered calls).

Single-call-per-task contract: ``start_run()`` resets the counter for
the current context. Concurrent ``run_review_v3`` calls from the same
task context would share counters; in practice each top-level call
runs in its own asyncio task and gets its own context, so this is
not a problem.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class V3Counters:
    """Mutable counter object — fields are bumped from many concurrent
    coroutines via the shared contextvar reference."""

    llm_calls: int = 0
    gap_rounds: int = 0


_CTX: ContextVar[V3Counters | None] = ContextVar("v3_counters", default=None)


def start_run() -> V3Counters:
    """Initialise a fresh counter for the current context. Returns the
    counter so the caller can read it after the run completes."""
    c = V3Counters()
    _CTX.set(c)
    return c


def current() -> V3Counters | None:
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
    """Bump ``llm_calls`` by pydantic-ai's ``result.usage().requests``
    when available — that captures tool-call expansion (an agent.run
    with three tool turns is reported as 4 requests). Falls back to +1
    when the usage call is missing or returns nothing parseable, so the
    counter is an honest lower bound even when usage is unreadable."""
    n = 1
    try:
        usage = result.usage()
        candidate = getattr(usage, "requests", None) or getattr(
            usage, "request_count", None
        )
        if isinstance(candidate, int) and candidate >= 1:
            n = candidate
    except Exception:
        pass
    increment_llm_calls(n)
