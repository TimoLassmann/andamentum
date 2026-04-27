"""Tests for benchmark metrics — boundary F1, granularity, etc."""

from benchmarks.chunker.metrics import (
    boundary_f1,
    granularity_ratio,
)


def test_boundary_f1_perfect_match():
    """Predicted boundaries exactly match truth boundaries."""
    truth_boundaries = [0, 100, 200]  # 2 units: [0, 100) and [100, 200)
    predicted_boundaries = [0, 100, 200]
    p, r, f = boundary_f1(predicted_boundaries, truth_boundaries, tolerance=50)
    assert p == 1.0
    assert r == 1.0
    assert f == 1.0


def test_boundary_f1_within_tolerance_counts():
    """Boundaries within tolerance count as matched."""
    truth = [0, 100, 200]
    predicted = [0, 110, 195]  # both within tolerance=50
    p, r, f = boundary_f1(predicted, truth, tolerance=50)
    assert p == 1.0
    assert r == 1.0


def test_boundary_f1_missed_boundary():
    """A truth boundary with no nearby predicted boundary hurts recall."""
    truth = [0, 100, 200, 300]  # 3 boundaries
    predicted = [0, 100, 300]  # missed the 200 boundary
    p, r, f = boundary_f1(predicted, truth, tolerance=20)
    assert r < 1.0


def test_boundary_f1_spurious_boundary():
    """A predicted boundary far from any truth boundary hurts precision."""
    truth = [0, 200]
    predicted = [0, 100, 200]  # extra boundary at 100
    p, r, f = boundary_f1(predicted, truth, tolerance=20)
    assert p < 1.0


def test_boundary_f1_handles_empty_predicted():
    truth = [0, 100, 200]
    p, r, f = boundary_f1([], truth, tolerance=50)
    assert p == 0.0
    assert r == 0.0
    assert f == 0.0


def test_boundary_f1_handles_empty_truth():
    truth: list[int] = []
    predicted: list[int] = []
    p, r, f = boundary_f1(predicted, truth, tolerance=50)
    # Both empty → conventionally F1 = 1.0 (vacuously perfect)
    assert f == 1.0


def test_granularity_ratio_perfect():
    assert granularity_ratio(predicted_count=10, truth_count=10) == 1.0


def test_granularity_ratio_under_segmented():
    assert granularity_ratio(predicted_count=5, truth_count=10) == 0.5


def test_granularity_ratio_over_segmented():
    assert granularity_ratio(predicted_count=20, truth_count=10) == 2.0


def test_granularity_ratio_handles_zero_truth():
    # If truth has 0 units (edge case), ratio is undefined. Return inf.
    import math

    assert math.isinf(granularity_ratio(predicted_count=5, truth_count=0))
