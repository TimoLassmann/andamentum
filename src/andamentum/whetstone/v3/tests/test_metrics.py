"""Per-run LLM call + gap-round counters.

Regression tests for the bug where v3 ReviewMetrics always reported
``llm_calls=0`` and ``reflection_rounds_used=0`` regardless of actual
pipeline activity — the v2 manual-increment plumbing was never ported
across the consolidation, so the cli.py summary table lied to the user.

The fix wires every ``agent.run()`` call site in v3 to bump a counter
held in a contextvar, then reads the counter in ``Finalize`` and
passes it to ``ReviewMetrics``. These tests cover the wiring at the
helper level (since end-to-end coverage would need live LLM calls).
"""

from __future__ import annotations

import asyncio


from andamentum.whetstone.v3._metrics import (
    bump_from_result,
    current,
    increment_gap_rounds,
    increment_llm_calls,
    start_run,
)


def test_increments_are_visible_after_start_run() -> None:
    c = start_run()
    increment_llm_calls(3)
    increment_gap_rounds(2)
    assert c.llm_calls == 3
    assert c.gap_rounds == 2
    assert current() is c


def test_increments_outside_a_run_do_not_raise() -> None:
    """When no counter is set in the current context, increments must
    be silent no-ops — helpers that bump after agent.run() should never
    crash a caller that didn't start a run (e.g. an isolated unit test
    of a single helper). Uses a fresh context to guarantee no leak from
    other tests in this file."""
    import contextvars

    def _no_counter():
        # Reset the contextvar to its default by setting None explicitly.
        from andamentum.whetstone.v3 import _metrics

        _metrics._CTX.set(None)
        assert current() is None
        # These must not raise.
        increment_llm_calls(5)
        increment_gap_rounds(1)
        # And still no counter — the no-op shouldn't auto-create one.
        assert current() is None

    contextvars.copy_context().run(_no_counter)


def test_bump_from_result_uses_usage_requests_when_available() -> None:
    """pydantic-ai exposes Usage.requests (incremented per request,
    including tool-call expansions). bump_from_result must consult it."""
    c = start_run()

    class _Usage:
        requests = 4

    class _Result:
        def usage(self) -> _Usage:
            return _Usage()

    bump_from_result(_Result())
    assert c.llm_calls == 4


def test_bump_from_result_falls_back_to_one_when_usage_missing() -> None:
    c = start_run()

    class _ResultNoUsage:
        pass

    bump_from_result(_ResultNoUsage())
    assert c.llm_calls == 1


def test_bump_from_result_handles_usage_raising() -> None:
    """A result whose usage() raises should not propagate — count as +1."""
    c = start_run()

    class _ResultUsageRaises:
        def usage(self):
            raise RuntimeError("usage() not implemented")

    bump_from_result(_ResultUsageRaises())
    assert c.llm_calls == 1


def test_concurrent_gather_tasks_share_counter() -> None:
    """asyncio.gather'd tasks inherit a snapshot of the parent context,
    so the SAME counter instance is reachable from all of them. This is
    load-bearing — many v3 nodes use gather to fan out per-section or
    per-criterion calls, and all of those need to aggregate into one
    total."""

    async def _fanout():
        c = start_run()

        async def _one() -> None:
            increment_llm_calls(1)

        await asyncio.gather(*[_one() for _ in range(10)])
        return c.llm_calls

    total = asyncio.run(_fanout())
    assert total == 10


async def test_to_review_result_passes_through_metrics() -> None:
    """The to_review_result adapter must populate ReviewMetrics.llm_calls
    and .reflection_rounds_used from its new kwargs — the gap in this
    glue was the original 2026-05-26 bug."""
    from andamentum.whetstone.v3.model import (
        Claim,
        DocumentModel,
        Section,
        SectionGist,
        Span,
    )
    from andamentum.whetstone.v3.synth import StructuredReview, to_review_result

    sections = [Section(id="s1", title="Intro", text="Hello", start=0, end=5)]
    claims: list[Claim] = []
    gists = [SectionGist(section_id="s1", title="Intro", gist="Hello")]
    span = Span(section_id="s1", start=0, end=5)  # noqa: F841 — kept for shape parity
    doc = DocumentModel(source="Hello", sections=sections, claims=claims, gists=gists)
    review = StructuredReview(synopsis="Brief review.", strengths=[], weaknesses=[])

    result = to_review_result(
        doc,
        findings=[],
        review=review,
        edits=None,
        llm_calls=17,
        gap_rounds_used=3,
    )
    assert result.metrics.llm_calls == 17
    assert result.metrics.reflection_rounds_used == 3
