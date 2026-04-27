"""Tests for report formatting."""

from benchmarks.chunker.report import to_markdown_table
from benchmarks.chunker.types import (
    BenchmarkCase,
    CaseRun,
    Metrics,
    ResolvedTruth,
)


def _make_case(name: str, floor: float = 0.7) -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        source="x",
        domain="general",
        expected_f1_floor=floor,
        boundary_tolerance_chars=50,
        truth=ResolvedTruth(convention="t", units=[]),
    )


def _make_metrics(f1: float) -> Metrics:
    return Metrics(
        boundary_f1=f1,
        boundary_precision=f1,
        boundary_recall=f1,
        coverage=0.95,
        gap_fraction=0.05,
        granularity_ratio=1.0,
        unit_count_predicted=3,
        unit_count_truth=3,
        fragmentation_rate=0.0,
        anchor_method_exact=3,
        anchor_method_normalised=0,
        anchor_method_fuzzy=0,
        wall_clock_seconds=10.0,
        model_calls=3,
    )


def test_to_markdown_table_renders_header_and_rows():
    runs = [
        CaseRun(
            case=_make_case("c1"), metrics=_make_metrics(0.85), model="m", error=None
        ),
        CaseRun(
            case=_make_case("c2"), metrics=_make_metrics(0.5), model="m", error=None
        ),
    ]
    md = to_markdown_table(runs)
    assert "case" in md.lower()
    assert "f1" in md.lower()
    assert "c1" in md
    assert "c2" in md
    assert "0.85" in md or "0.85" in md


def test_to_markdown_table_marks_failures():
    """A case below its floor gets a visual flag."""
    runs = [
        CaseRun(
            case=_make_case("good", floor=0.5),
            metrics=_make_metrics(0.85),
            model="m",
            error=None,
        ),
        CaseRun(
            case=_make_case("bad", floor=0.9),
            metrics=_make_metrics(0.5),
            model="m",
            error=None,
        ),
    ]
    md = to_markdown_table(runs)
    # Some kind of marker — could be ❌, FAIL, etc.
    assert "❌" in md or "FAIL" in md or "below" in md.lower()


def test_to_markdown_table_includes_errored_runs():
    runs = [
        CaseRun(
            case=_make_case("crashed"),
            metrics=None,
            model="m",
            error="ChunkingFailedError: ...",
        )
    ]
    md = to_markdown_table(runs)
    assert "crashed" in md
    # Error should be visible
    assert "error" in md.lower() or "ChunkingFailedError" in md
