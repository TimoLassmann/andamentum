"""Tests for AgentRunner's global concurrency semaphore.

Phase 2 of the epistemic efficiency plan parallelizes independent LLM
call loops. The plan's open-decision #2 reserved a global per-runner
``Semaphore`` to bound in-flight calls so we don't hammer Ollama (which
serialises inference) or hit cloud rate limits.

These tests pin:

  1. The auto-detection rule: local (Ollama) defaults to concurrency=1,
     cloud defaults to concurrency=8.
  2. ``ANDAMENTUM_LLM_CONCURRENCY`` env var override behaviour.
  3. The semaphore actually fires under concurrent ``run()`` invocations.

The semaphore is *the* control surface for "how many concurrent LLM
calls" — every parallelization site in Phase 2 relies on it. If a
future refactor moves the semaphore or skips it for some path, these
tests fail loud.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from unittest.mock import patch

import pytest


# ── Concurrency auto-detection ───────────────────────────────────────


def _make_runner_with_model_input(model_input: str) -> Any:
    """Build an AgentRunner-like object with a stubbed model so the
    real model resolver isn't invoked. Tests the concurrency
    initialisation path without needing an actual LLM provider."""
    from andamentum.core.agents import AgentRunner

    # Patch resolve_model so we don't need an actual model backend.
    with patch(
        "andamentum.core.models.resolve_model",
        lambda m: f"resolved::{m}",
    ):
        return AgentRunner(model=model_input)


def test_local_model_defaults_to_concurrency_one() -> None:
    """Ollama models should default to concurrency=1 — Ollama serialises
    inference per-process anyway, so concurrent requests just queue."""
    # Clear env var so we test the auto-detection default.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANDAMENTUM_LLM_CONCURRENCY", None)
        runner = _make_runner_with_model_input("ollama:gemma2:2b")
    assert runner.is_local is True
    assert runner.concurrency == 1


def test_cloud_model_defaults_to_concurrency_eight() -> None:
    """Cloud models default to concurrency=8 — sustainable on most
    paid tiers without hitting rate limits."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANDAMENTUM_LLM_CONCURRENCY", None)
        runner = _make_runner_with_model_input("openai:gpt-5.4-nano")
    assert runner.is_local is False
    assert runner.concurrency == 8


def test_env_override_applies_to_local() -> None:
    """ANDAMENTUM_LLM_CONCURRENCY overrides the auto-detected default
    for both local and cloud models."""
    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "4"}):
        runner = _make_runner_with_model_input("ollama:gemma2:2b")
    assert runner.concurrency == 4


def test_env_override_applies_to_cloud() -> None:
    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "20"}):
        runner = _make_runner_with_model_input("openai:gpt-5.4-nano")
    assert runner.concurrency == 20


def test_env_override_invalid_falls_back_to_default() -> None:
    """Garbage values in the env var don't crash — fall back to the
    auto-detected default with a warning logged."""
    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "not-an-int"}):
        runner = _make_runner_with_model_input("openai:gpt-5.4-nano")
    assert runner.concurrency == 8


def test_env_override_clamped_to_at_least_one() -> None:
    """Concurrency=0 would deadlock; the override is clamped to >=1."""
    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "0"}):
        runner = _make_runner_with_model_input("openai:gpt-5.4-nano")
    assert runner.concurrency == 1


# ── Semaphore actually bounds in-flight calls ────────────────────────


@pytest.mark.asyncio
async def test_semaphore_serialises_calls_when_concurrency_one() -> None:
    """With concurrency=1, two concurrently-launched run() calls take
    >= 2x the time of a single call. Proves the semaphore actually
    serialises (not just exists)."""
    from andamentum.core.agents import AgentDefinition, AgentRunner
    from pydantic import BaseModel

    class _Out(BaseModel):
        value: str

    # Stub model resolver and patch the agent runner's behaviour so we
    # don't need a real LLM. The semaphore is acquired in run() before
    # the actual agent.run call, so faking the latter is fine.
    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "1"}):
        with patch(
            "andamentum.core.models.resolve_model",
            lambda m: f"resolved::{m}",
        ):
            runner = AgentRunner(model="openai:gpt-5.4-nano")

    defn = AgentDefinition(
        name="_test",
        prompt="x",
        output_model=_Out,
    )

    call_count = 0

    class _FakeResult:
        def __init__(self, value: str) -> None:
            self.output = _Out(value=value)

    async def _fake_agent_run(self: Any, user_message: str) -> _FakeResult:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)  # simulated LLM latency
        return _FakeResult(f"call-{call_count}")

    # Inject the fake into the cache so run() finds it without
    # constructing a real Agent.
    class _FakeAgent:
        run = _fake_agent_run

    runner._cache[defn.name] = _FakeAgent()

    t0 = time.monotonic()
    await asyncio.gather(
        runner.run(defn, x="a"),
        runner.run(defn, x="b"),
    )
    elapsed = time.monotonic() - t0

    assert call_count == 2
    # Two 100ms calls, fully serialised, take ≥200ms. Allow some
    # scheduler overhead but they should be well above 180ms.
    assert elapsed >= 0.18, (
        f"Two calls with concurrency=1 should serialise; took {elapsed:.3f}s. "
        "If this is < 0.18s, the semaphore isn't actually firing — calls "
        "are running in parallel against Ollama, which would cause queue "
        "buildup or upstream timeouts in production."
    )


@pytest.mark.asyncio
async def test_semaphore_allows_concurrency_when_above_one() -> None:
    """With concurrency=4, four concurrently-launched run() calls take
    roughly the time of a single call (not 4x). Proves the semaphore
    isn't over-bounding when it shouldn't."""
    from andamentum.core.agents import AgentDefinition, AgentRunner
    from pydantic import BaseModel

    class _Out(BaseModel):
        value: str

    with patch.dict(os.environ, {"ANDAMENTUM_LLM_CONCURRENCY": "4"}):
        with patch(
            "andamentum.core.models.resolve_model",
            lambda m: f"resolved::{m}",
        ):
            runner = AgentRunner(model="openai:gpt-5.4-nano")

    defn = AgentDefinition(
        name="_test_par",
        prompt="x",
        output_model=_Out,
    )

    class _FakeResult:
        def __init__(self) -> None:
            self.output = _Out(value="ok")

    async def _fake_agent_run(self: Any, user_message: str) -> _FakeResult:
        await asyncio.sleep(0.1)
        return _FakeResult()

    class _FakeAgent:
        run = _fake_agent_run

    runner._cache[defn.name] = _FakeAgent()

    t0 = time.monotonic()
    await asyncio.gather(*(runner.run(defn, x=str(i)) for i in range(4)))
    elapsed = time.monotonic() - t0

    # Four 100ms calls, fully concurrent, take ~100ms (plus overhead).
    # If this exceeds 200ms, concurrency isn't actually parallel.
    assert elapsed < 0.2, (
        f"Four calls with concurrency=4 should run concurrently; "
        f"took {elapsed:.3f}s. The semaphore is forcing serialisation."
    )
