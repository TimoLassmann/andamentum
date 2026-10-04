"""Shared stub for the model boundary: replaces ``run_agent_with_usage``."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai.usage import RunUsage

from andamentum.compose.schemas import Brief, ReaderQuestion


USAGE = RunUsage(input_tokens=11, output_tokens=7)


def with_usage(fake: Any) -> Any:
    """Wrap an async fake that returns the output so it returns (output, usage)."""

    async def wrapped(*args: Any, **kwargs: Any) -> tuple[Any, RunUsage]:
        return await fake(*args, **kwargs), USAGE

    return wrapped


class StubModel:
    """Returns queued outputs in order and records every call."""

    def __init__(self, outputs: list[BaseModel]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, model: Any, **kwargs: Any) -> tuple[BaseModel, RunUsage]:
        self.calls.append({"model": model, **kwargs})
        out = self.outputs.pop(0)
        assert isinstance(out, kwargs["output_type"])
        return out, USAGE


def install(monkeypatch: pytest.MonkeyPatch, outputs: list[BaseModel]) -> StubModel:
    stub = StubModel(outputs)
    monkeypatch.setattr("andamentum.compose._call.run_agent_with_usage", stub)
    return stub


def make_brief(**overrides: Any) -> Brief:
    base: dict[str, Any] = dict(
        task="Write to Jane Doe about the Apollo Labs pilot",
        genre="memo",
        title="Apollo pilot",
        audience="Jane Doe",
        purpose="Confirm the pilot date",
        key_messages=["Pilot starts 1 March"],
        must_include=["1 March"],
        constraints=[],
        word_range=(300, 1200),
        reader_questions=[
            ReaderQuestion(question="When does it start?", expected_answer="1 March"),
            ReaderQuestion(question="Who runs it?", expected_answer="Apollo Labs"),
        ],
    )
    base.update(overrides)
    return Brief(**base)
