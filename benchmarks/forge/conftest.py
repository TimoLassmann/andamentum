"""Pytest fixtures for the forge benchmark.

The live benchmark needs a model. Per forge's explicit-model rule there is **no env-var
default and no hidden default** — the model is supplied on the command line::

    uv run pytest benchmarks/forge -m benchmark --forge-bench-model <id>

The ``bench_model`` fixture skips any test that asks for it when ``--forge-bench-model``
was not given, so the live tests never silently run against a phantom model.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--forge-bench-model",
        action="store",
        default=None,
        help="Model id to drive forge in the live benchmark (no default).",
    )


@pytest.fixture(scope="session")
def bench_model(request: pytest.FixtureRequest) -> str:
    model = request.config.getoption("--forge-bench-model")
    if not model:
        pytest.skip("no --forge-bench-model given; skipping live forge benchmark")
    return str(model)
