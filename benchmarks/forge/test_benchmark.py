"""Live pytest entry for the forge benchmark.

Each case in :data:`benchmarks.forge.cases.CASES` becomes a parametrised test that drives
forge for real (no stub sink) and asserts a **loose per-case floor** — ``pass_rate >= 0.5``
— so a genuine regression trips but stochastic model noise does not.

The ``benchmark`` marker is deselected by default; this test also needs a model. Run::

    uv run pytest benchmarks/forge -m benchmark --forge-bench-model <id> -v
"""

from __future__ import annotations

import pytest

from .cases import CASES
from .runner import run_case
from .types import Case

#: A run must clear this share of repetitions to pass — loose, to ride out model noise.
PASS_FLOOR = 0.5


@pytest.mark.benchmark
@pytest.mark.parametrize("case", CASES, ids=[c.brief for c in CASES])
async def test_forge_meets_floor(case: Case, bench_model: str) -> None:
    score = await run_case(case, model=bench_model, runs=3)

    sample = score.runs[0] if score.runs else None
    detail = "" if sample is None else f"  first={sample.kind} {sample.error[:80]}"
    print(
        f"\n  brief={case.brief!r}  expected={case.expected}/{case.grammar}  "
        f"model={bench_model}  pass={score.passes}/{score.total}"
        f"  rate={score.pass_rate:.2f}{detail}"
    )

    if score.pass_rate < PASS_FLOOR:
        pytest.fail(
            f"pass_rate {score.pass_rate:.2f} below floor {PASS_FLOOR} for "
            f"brief {case.brief!r} (expected {case.expected}/{case.grammar}, "
            f"model {bench_model!r}). Either forge regressed or the case drifted."
        )
