"""Tests for whetstone's _run_agents helper (serial vs parallel dispatch)."""

import asyncio

from andamentum.whetstone.orchestrator import _run_agents


class _LocalRunner:
    """Stand-in that claims to be a local (Ollama) runner."""

    is_local = True


class _CloudRunner:
    """Stand-in that claims to be a cloud runner."""

    is_local = False


async def test_local_runner_executes_sequentially():
    """When runner.is_local, _run_agents must await each coroutine in order."""
    order: list[int] = []

    async def make_coro(i: int, delay: float) -> int:
        await asyncio.sleep(delay)
        order.append(i)
        return i

    # If parallel, the longer-delay coro would still finish last by time but
    # `order` would reflect interleaving. Sequentially, items append in
    # the dispatch order regardless of delay.
    coros = [make_coro(0, 0.05), make_coro(1, 0.01), make_coro(2, 0.03)]
    results = await _run_agents(_LocalRunner(), "phase", *coros)  # type: ignore[arg-type]

    assert results == [0, 1, 2]
    assert order == [0, 1, 2]  # serial dispatch preserves call order


async def test_cloud_runner_executes_in_parallel():
    """When runner.is_local is False, coroutines run concurrently via gather."""
    order: list[int] = []

    async def make_coro(i: int, delay: float) -> int:
        await asyncio.sleep(delay)
        order.append(i)
        return i

    coros = [make_coro(0, 0.05), make_coro(1, 0.01), make_coro(2, 0.03)]
    results = await _run_agents(_CloudRunner(), "phase", *coros)  # type: ignore[arg-type]

    # Results preserve coro order (gather guarantees this)
    assert results == [0, 1, 2]
    # Completion order reflects sleep duration → 1 finishes first, then 2, then 0
    assert order == [1, 2, 0]


async def test_failure_wraps_with_phase_context():
    async def boom() -> int:
        raise ValueError("kaboom")

    import pytest

    with pytest.raises(RuntimeError, match="Agent failure during my-phase"):
        await _run_agents(_CloudRunner(), "my-phase", boom())  # type: ignore[arg-type]
