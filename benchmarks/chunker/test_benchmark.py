"""Pytest entry for the chunker benchmark.

Each case becomes a parametrised test that asserts F1 >= the case's
declared expected_f1_floor. Strict — a regression below the floor fails.

Marker `benchmark` is deselected by default; run explicitly:

    uv run pytest benchmarks/chunker -m benchmark -v
"""

from __future__ import annotations

import pytest

from .conftest import discover_cases
from .loader import load_case
from .runner import run_case

CASES = discover_cases()


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "truth_path",
    CASES,
    ids=lambda p: p.name.removesuffix(".truth.json"),
)
async def test_chunker_meets_floor(truth_path, primary_executor, bench_model):
    case = load_case(truth_path)
    run = await run_case(
        case, primary_executor=primary_executor, model_label=bench_model
    )

    if run.error is not None:
        pytest.fail(f"Case {case.name!r} crashed on model {bench_model!r}: {run.error}")

    m = run.metrics
    assert m is not None  # appease type checker
    print(
        f"\n  case={case.name}  model={bench_model}  "
        f"F1={m.boundary_f1:.2f}  cov={m.coverage:.2f}  "
        f"granularity={m.granularity_ratio:.2f}  "
        f"calls={m.model_calls}  time={m.wall_clock_seconds:.1f}s"
    )

    if not run.passed_floor:
        pytest.fail(
            f"F1={m.boundary_f1:.3f} below floor {case.expected_f1_floor} "
            f"on case {case.name!r} (model {bench_model!r}). "
            f"Either the chunker regressed or the floor needs adjustment."
        )
