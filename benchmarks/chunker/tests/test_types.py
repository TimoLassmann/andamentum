"""Tests for benchmark types."""

from benchmarks.chunker.types import (
    BenchmarkCase,
    CaseRun,
    Metrics,
    ResolvedTruth,
    TruthUnit,
)


def test_truth_unit_minimal():
    u = TruthUnit(title="Intro", start_anchor="Hello world", end_anchor="end here.")
    assert u.title == "Intro"


def test_resolved_truth_unit_has_offsets():
    from benchmarks.chunker.types import ResolvedTruthUnit

    r = ResolvedTruthUnit(
        title="Intro",
        start_offset=0,
        end_offset=100,
    )
    assert r.length == 100


def test_benchmark_case_minimal():
    c = BenchmarkCase(
        name="academic_short",
        source="Hello world.",
        domain="academic",
        expected_f1_floor=0.7,
        boundary_tolerance_chars=50,
        truth=ResolvedTruth(
            convention="paragraph = unit",
            units=[],
        ),
    )
    assert c.name == "academic_short"


def test_metrics_carries_all_scalars():
    m = Metrics(
        boundary_f1=0.85,
        boundary_precision=0.9,
        boundary_recall=0.8,
        coverage=0.95,
        gap_fraction=0.05,
        granularity_ratio=1.1,
        unit_count_predicted=11,
        unit_count_truth=10,
        fragmentation_rate=0.0,
        anchor_method_exact=10,
        anchor_method_normalised=1,
        anchor_method_fuzzy=0,
        wall_clock_seconds=12.5,
        model_calls=11,
    )
    assert m.boundary_f1 == 0.85
    assert m.coverage > 0.9


def test_case_run_carries_metrics_and_meta():
    c = BenchmarkCase(
        name="x",
        source="x",
        domain="general",
        expected_f1_floor=0.5,
        boundary_tolerance_chars=50,
        truth=ResolvedTruth(convention="t", units=[]),
    )
    metrics = Metrics(
        boundary_f1=0.6,
        boundary_precision=0.6,
        boundary_recall=0.6,
        coverage=0.9,
        gap_fraction=0.1,
        granularity_ratio=1.0,
        unit_count_predicted=1,
        unit_count_truth=1,
        fragmentation_rate=0.0,
        anchor_method_exact=1,
        anchor_method_normalised=0,
        anchor_method_fuzzy=0,
        wall_clock_seconds=1.0,
        model_calls=1,
    )
    run = CaseRun(case=c, metrics=metrics, model="ollama:gemma4:31b-nvfp4", error=None)
    assert run.passed_floor is True  # 0.6 >= 0.5


def test_case_run_failed_floor_when_f1_below():
    c = BenchmarkCase(
        name="x",
        source="x",
        domain="general",
        expected_f1_floor=0.8,
        boundary_tolerance_chars=50,
        truth=ResolvedTruth(convention="t", units=[]),
    )
    metrics = Metrics(
        boundary_f1=0.5,
        boundary_precision=0.5,
        boundary_recall=0.5,
        coverage=0.9,
        gap_fraction=0.1,
        granularity_ratio=1.0,
        unit_count_predicted=1,
        unit_count_truth=1,
        fragmentation_rate=0.0,
        anchor_method_exact=1,
        anchor_method_normalised=0,
        anchor_method_fuzzy=0,
        wall_clock_seconds=1.0,
        model_calls=1,
    )
    run = CaseRun(case=c, metrics=metrics, model="x", error=None)
    assert run.passed_floor is False
