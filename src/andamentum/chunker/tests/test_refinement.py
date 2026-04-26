"""Tests for the escalation chain (window halving → executor escalation)."""

import pytest

from andamentum.chunker.refinement import (
    EscalationOutcome,
    escalate,
)
from andamentum.chunker.types import (
    ChunkingFailedError,
    NextUnitResult,
)


def _make_executor(programmed: list, label: str = "fake"):
    """Builds a fake executor that returns/raises items in order."""
    items = list(programmed)

    async def executor(*, instructions, user_message, output_type, validators):
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    setattr(executor, "label", label)  # for diagnostics in failure messages
    return executor


async def test_escalate_succeeds_after_window_halving():
    """First call fails, halved-window call succeeds. No model escalation needed."""
    executor = _make_executor(
        [
            RuntimeError("anchor never matched after retries"),
            NextUnitResult(
                found=True,
                title="t",
                start_anchor="Hello",
                end_anchor="world",
                kind="prose",
            ),
        ],
        label="primary",
    )

    text = "Hello world. More text follows."
    outcome = await escalate(
        primary_executor=executor,
        backup_executors=[],
        source=text,
        cursor=0,
        window_size=20,
        lookahead=10,
        domain="general",
        prior_unit_titles=[],
    )
    assert isinstance(outcome, EscalationOutcome)
    assert outcome.attempt.result is not None
    assert outcome.attempt.result.found is True
    assert outcome.window_size_used == 10  # halved


async def test_escalate_falls_through_to_backup_executor():
    """Primary fails (twice — once full, once halved), backup succeeds."""
    primary = _make_executor(
        [RuntimeError("primary fail 1"), RuntimeError("primary fail 2")],
        label="primary",
    )
    backup = _make_executor(
        [
            NextUnitResult(
                found=True,
                title="t",
                start_anchor="Hello",
                end_anchor="world",
                kind="prose",
            )
        ],
        label="backup",
    )

    outcome = await escalate(
        primary_executor=primary,
        backup_executors=[backup],
        source="Hello world. More text.",
        cursor=0,
        window_size=20,
        lookahead=5,
        domain="general",
        prior_unit_titles=[],
    )
    assert outcome.executor_used is backup
    assert outcome.attempt.result is not None


async def test_escalate_raises_when_all_executors_fail():
    primary = _make_executor([RuntimeError("p1"), RuntimeError("p2")], label="primary")
    backup = _make_executor([RuntimeError("b1"), RuntimeError("b2")], label="backup")
    with pytest.raises(ChunkingFailedError) as excinfo:
        await escalate(
            primary_executor=primary,
            backup_executors=[backup],
            source="some text",
            cursor=0,
            window_size=10,
            lookahead=5,
            domain="general",
            prior_unit_titles=[],
        )
    err = excinfo.value
    assert err.cursor == 0
    # diagnostic should include both executor labels
    assert len(err.attempted_models) >= 2
