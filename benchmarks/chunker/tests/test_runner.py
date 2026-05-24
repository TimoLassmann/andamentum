"""Tests for the per-case runner using the structural-first chunker.

The chunker now derives unit boundaries from markdown headings + embeddings,
not from a fake LLM executor. The runner can take any executor (used as the
optional judge stage); for tests we don't need to wire one — passing
`primary_executor=None` is fine because the structural pass handles the
sample inputs.
"""

from benchmarks.chunker.runner import run_case
from benchmarks.chunker.types import (
    BenchmarkCase,
    ResolvedTruth,
    ResolvedTruthUnit,
)


async def test_run_case_produces_metrics_with_structural_split():
    src = (
        "## Section 1\n\n"
        + "Body of section one. " * 10
        + "\n\n## Section 2\n\n"
        + "Body of section two. " * 10
    )
    sec1_end = src.index("\n\n## Section 2")
    sec2_start = src.index("## Section 2")
    case = BenchmarkCase(
        name="x",
        source=src,
        domain="academic",
        expected_f1_floor=0.5,
        boundary_tolerance_chars=20,
        truth=ResolvedTruth(
            convention="t",
            units=[
                ResolvedTruthUnit(
                    title="Section 1", start_offset=0, end_offset=sec1_end
                ),
                ResolvedTruthUnit(
                    title="Section 2", start_offset=sec2_start, end_offset=len(src)
                ),
            ],
        ),
    )

    run = await run_case(case, primary_executor=None, model_label="structural")
    assert run.error is None
    assert run.metrics is not None
    assert run.metrics.unit_count_predicted == 2
    assert run.metrics.unit_count_truth == 2
    # Predicted boundaries match truth within tolerance → strong F1
    assert run.metrics.boundary_f1 > 0.8


async def test_run_case_handles_no_headings_gracefully():
    """A doc without headings becomes one unit; runner should still produce metrics."""
    src = "Plain prose with no markdown structure. " * 10
    case = BenchmarkCase(
        name="x",
        source=src,
        domain="general",
        expected_f1_floor=0.0,
        boundary_tolerance_chars=20,
        truth=ResolvedTruth(
            convention="t",
            units=[ResolvedTruthUnit(title="all", start_offset=0, end_offset=len(src))],
        ),
    )

    run = await run_case(case, primary_executor=None, model_label="structural")
    assert run.error is None
    assert run.metrics is not None
    assert run.metrics.unit_count_predicted >= 1
