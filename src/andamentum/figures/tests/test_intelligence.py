"""Tests for Phase 2: intelligence modules (advisor, auto, stats)."""

import pytest

from andamentum.figures.advisor import check_banned, recommend_kind, validate_kind
from andamentum.figures.auto import (
    detect_column_roles,
    detect_log_scale,
    recommend_sort,
)
from andamentum.figures.stats import aggregate_by, bootstrap_ci, compute_mean_error
from andamentum.figures.types import DataTable


# ── Advisor ──────────────────────────────────────────────────────────────────


class TestCheckBanned:
    def test_pie_banned(self):
        with pytest.raises(ValueError, match="Refused.*pie"):
            check_banned("pie")

    def test_donut_banned(self):
        with pytest.raises(ValueError, match="Refused.*donut"):
            check_banned("donut")

    def test_3d_bar_banned(self):
        with pytest.raises(ValueError, match="Refused.*3d_bar"):
            check_banned("3d_bar")

    def test_bar_allowed(self):
        check_banned("bar")  # should not raise

    def test_line_allowed(self):
        check_banned("line")  # should not raise


class TestRecommendKind:
    def test_single_numeric_histogram(self):
        dt = DataTable.from_dict({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        kind = recommend_kind(dt, x=None, y=["values"], group=None)
        assert kind == "histogram"

    def test_categorical_x_one_per_group_bar(self):
        dt = DataTable.from_dict({"group": ["A", "B", "C"], "value": [10, 20, 30]})
        kind = recommend_kind(dt, x="group", y="value", group=None)
        assert kind == "bar"

    def test_categorical_x_few_per_group_strip(self):
        dt = DataTable.from_dict(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )
        kind = recommend_kind(dt, x="group", y="value", group=None)
        assert kind == "strip"

    def test_categorical_x_many_per_group_box(self):
        groups = ["A"] * 20 + ["B"] * 20
        values = list(range(40))
        dt = DataTable.from_dict({"group": groups, "value": values})
        kind = recommend_kind(dt, x="group", y="value", group=None)
        assert kind == "box"

    def test_categorical_x_very_many_per_group_violin(self):
        groups = ["A"] * 300 + ["B"] * 300
        values = list(range(600))
        dt = DataTable.from_dict({"group": groups, "value": values})
        kind = recommend_kind(dt, x="group", y="value", group=None)
        assert kind == "violin"

    def test_two_numeric_scatter(self):
        dt = DataTable.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
        kind = recommend_kind(dt, x="x", y="y", group=None)
        assert kind == "scatter"

    def test_time_like_x_line(self):
        dt = DataTable.from_dict({"time": [0, 1, 2], "value": [10, 20, 30]})
        kind = recommend_kind(dt, x="time", y="value", group=None)
        assert kind == "line"

    def test_multi_y_line(self):
        dt = DataTable.from_dict({"x": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6]})
        kind = recommend_kind(dt, x="x", y=["a", "b"], group=None)
        assert kind == "line"


class TestValidateKind:
    def test_bar_distribution_warning(self):
        dt = DataTable.from_dict(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "value": list(range(20)),
            }
        )
        warnings = validate_kind("bar", dt, x="group", y="value")
        assert any("hides distribution" in w for w in warnings)

    def test_bar_counts_no_warning(self):
        dt = DataTable.from_dict({"group": ["A", "B", "C"], "count": [10, 20, 30]})
        warnings = validate_kind("bar", dt, x="group", y="count")
        assert not warnings

    def test_too_many_colors_warning(self):
        groups = [f"G{i}" for i in range(12)]
        dt = DataTable.from_dict({"group": groups, "value": list(range(12))})
        warnings = validate_kind("bar", dt, x="group", y="value")
        assert any("colors" in w for w in warnings)

    def test_too_many_lines_warning(self):
        dt = DataTable.from_dict(
            {
                "x": [1, 2, 3],
                "a": [1, 2, 3],
                "b": [2, 3, 4],
                "c": [3, 4, 5],
                "d": [4, 5, 6],
                "e": [5, 6, 7],
            }
        )
        warnings = validate_kind("line", dt, x="x", y=["a", "b", "c", "d", "e"])
        assert any("overlapping series" in w for w in warnings)

    def test_box_few_obs_warning(self):
        dt = DataTable.from_dict(
            {
                "group": ["A", "A", "B", "B"],
                "value": [1, 2, 3, 4],
            }
        )
        warnings = validate_kind("box", dt, x="group", y="value")
        assert any("unreliable" in w for w in warnings)


# ── Auto ─────────────────────────────────────────────────────────────────────


class TestDetectLogScale:
    def test_wide_range(self):
        assert detect_log_scale([1, 10, 100, 1000, 10000])

    def test_narrow_range(self):
        assert not detect_log_scale([1, 2, 3, 4, 5])

    def test_negative_values(self):
        assert not detect_log_scale([-1, 0, 1, 2, 3])

    def test_label_hint_with_range(self):
        # Label hint only triggers when data also spans >1.5 orders of magnitude
        assert detect_log_scale([0.1, 1, 5, 50], label="Concentration (μM)")

    def test_label_hint_without_range(self):
        # Label hint alone should NOT trigger log scale on narrow-range data
        assert not detect_log_scale([1, 2, 3, 4, 5], label="Concentration (μM)")

    def test_skewed_data(self):
        # Right-skewed: most values small, few very large
        values = [1] * 50 + [100] * 5 + [10000] * 2
        assert detect_log_scale(values)

    def test_empty_returns_false(self):
        assert not detect_log_scale([])

    def test_single_value(self):
        assert not detect_log_scale([5])


class TestDetectColumnRoles:
    def test_categorical_x_numeric_y(self):
        dt = DataTable.from_dict({"name": ["A", "B"], "val": [1, 2]})
        x, y_cols, group, error = detect_column_roles(dt)
        assert x == "name"
        assert y_cols == ["val"]

    def test_two_numeric(self):
        dt = DataTable.from_dict({"x": [1, 2], "y": [3, 4]})
        x, y_cols, group, error = detect_column_roles(dt)
        assert x is not None
        assert len(y_cols) == 1

    def test_error_column_auto_detect(self):
        dt = DataTable.from_dict(
            {"group": ["A", "B"], "val": [1, 2], "error": [0.1, 0.2]}
        )
        x, y_cols, group, error = detect_column_roles(dt)
        assert error == "error"

    def test_explicit_overrides(self):
        dt = DataTable.from_dict({"a": [1, 2], "b": [3, 4], "c": ["x", "y"]})
        x, y_cols, group, error = detect_column_roles(dt, x="c", y=["a"])
        assert x == "c"
        assert y_cols == ["a"]


class TestRecommendSort:
    def test_bar_categorical_sort_by_value(self):
        dt = DataTable.from_dict({"group": ["A", "B", "C"], "val": [1, 2, 3]})
        assert recommend_sort(dt, "group", "bar") == "value"

    def test_line_preserve_order(self):
        dt = DataTable.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert recommend_sort(dt, "x", "line") is None

    def test_time_column_preserve_order(self):
        dt = DataTable.from_dict({"time_point": ["T0", "T1", "T2"], "val": [1, 2, 3]})
        assert recommend_sort(dt, "time_point", "bar") is None


# ── Stats ────────────────────────────────────────────────────────────────────


class TestBootstrapCI:
    def test_basic(self):
        mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5], seed=42)
        assert lo <= mean <= hi
        assert abs(mean - 3.0) < 0.01

    def test_single_value(self):
        mean, lo, hi = bootstrap_ci([5.0])
        assert mean == 5.0
        assert lo == hi == 5.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_ci_width(self):
        """CI should be narrower (relative to mean) with more data from same distribution."""
        import random

        rng = random.Random(42)
        small = [rng.gauss(10, 2) for _ in range(5)]
        large = [rng.gauss(10, 2) for _ in range(200)]
        mean_s, lo_s, hi_s = bootstrap_ci(small, seed=42)
        mean_l, lo_l, hi_l = bootstrap_ci(large, seed=42)
        width_small = (hi_s - lo_s) / max(abs(mean_s), 1e-9)
        width_large = (hi_l - lo_l) / max(abs(mean_l), 1e-9)
        assert width_large < width_small


class TestAggregateBy:
    def test_basic(self):
        records = [
            {"method": "A", "score": 0.9},
            {"method": "A", "score": 0.8},
            {"method": "B", "score": 0.7},
        ]
        result = aggregate_by(records, "method", "score")
        assert result == {"A": [0.9, 0.8], "B": [0.7]}


class TestComputeMeanError:
    def test_sem(self):
        mean, err = compute_mean_error([10, 20, 30])
        assert abs(mean - 20.0) < 0.01
        assert err > 0

    def test_sd(self):
        mean, err = compute_mean_error([10, 20, 30], error_type="sd")
        assert abs(mean - 20.0) < 0.01
        assert err > 0

    def test_ci95(self):
        mean, err = compute_mean_error([10, 20, 30], error_type="ci95")
        assert err > compute_mean_error([10, 20, 30], error_type="sem")[1]

    def test_single_value(self):
        mean, err = compute_mean_error([5.0])
        assert mean == 5.0
        assert err == 0.0
