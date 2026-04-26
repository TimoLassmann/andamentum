"""Pytest fixtures for the chunker benchmark.

Model selection:
    CHUNKER_BENCH_MODEL env var, e.g.:
        export CHUNKER_BENCH_MODEL=openai:gpt-4o-mini
        export CHUNKER_BENCH_MODEL=ollama:qwen3.5:9b

    Default: ollama:gemma4:31b-nvfp4
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from andamentum.chunker.extractor import make_runner_executor
from andamentum.core.agents import AgentRunner

DEFAULT_MODEL = "ollama:gemma4:31b-nvfp4"


def get_bench_model() -> str:
    """Get the model to use for the benchmark from env or default."""
    return os.environ.get("CHUNKER_BENCH_MODEL", DEFAULT_MODEL)


@pytest.fixture(scope="session")
def bench_model() -> str:
    return get_bench_model()


@pytest.fixture(scope="session")
def primary_executor(bench_model):
    runner = AgentRunner(model=bench_model)
    executor = make_runner_executor(runner)
    executor.label = bench_model  # type: ignore[attr-defined]
    return executor


def discover_cases() -> list[Path]:
    """Find all cases/*.truth.json (excluding fixtures starting with _)."""
    case_dir = Path(__file__).parent / "cases"
    return sorted(
        p for p in case_dir.glob("*.truth.json") if not p.name.startswith("_")
    )
