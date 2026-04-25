"""Tests for Phase 3: plots and legend modules."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from andamentum.figures.legend import generate_legend
from andamentum.figures.plots import (
    auto_aggregate_bar,
    bar_plot,
    box_plot,
    grouped_bar,
    grouped_boxplot,
    heatmap_plot,
    histogram_plot,
    line_plot,
    line_with_ci,
    scatter_plot,
    strip_plot,
    swarm_plot,
    violin_plot,
)
from andamentum.figures.style import savefig, setup_style


@pytest.fixture(autouse=True)
def _setup():
    setup_style()


# ── Bar plot ─────────────────────────────────────────────────────────────────


class TestBarPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        bar_plot(ax, ["A", "B", "C"], [10, 20, 30])
        savefig(fig, tmp_path / "bar.png")
        assert (tmp_path / "bar.png").exists()

    def test_with_errors(self, tmp_path):
        fig, ax = plt.subplots()
        bar_plot(ax, ["A", "B"], [10, 20], error_values=[1.5, 2.0], ylabel="Value")
        savefig(fig, tmp_path / "bar_err.png")
        assert (tmp_path / "bar_err.png").exists()

    def test_sorted(self, tmp_path):
        fig, ax = plt.subplots()
        bar_plot(ax, ["C", "A", "B"], [5, 30, 15], sort_by_value=True)
        savefig(fig, tmp_path / "bar_sorted.png")
        assert (tmp_path / "bar_sorted.png").exists()

    def test_horizontal(self, tmp_path):
        fig, ax = plt.subplots()
        bar_plot(ax, ["X", "Y", "Z"], [1, 2, 3], horizontal=True)
        savefig(fig, tmp_path / "bar_h.png")
        assert (tmp_path / "bar_h.png").exists()


class TestGroupedBar:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        grouped_bar(ax, {"Series A": [1, 2, 3], "Series B": [3, 2, 1]}, ["X", "Y", "Z"])
        savefig(fig, tmp_path / "gbar.png")
        assert (tmp_path / "gbar.png").exists()


# ── Line plot ────────────────────────────────────────────────────────────────


class TestLinePlot:
    def test_single_series(self, tmp_path):
        fig, ax = plt.subplots()
        line_plot(ax, [0, 1, 2, 3], {"Drug A": [100, 80, 50, 20]})
        savefig(fig, tmp_path / "line.png")
        assert (tmp_path / "line.png").exists()

    def test_multi_series_with_ci(self, tmp_path):
        fig, ax = plt.subplots()
        x = [0, 1, 2, 3]
        line_plot(
            ax,
            x,
            {"A": [100, 80, 50, 20], "B": [100, 90, 70, 60]},
            ci_series={"A": ([95, 75, 45, 15], [105, 85, 55, 25])},
            ylabel="Viability (%)",
        )
        savefig(fig, tmp_path / "line_ci.png")
        assert (tmp_path / "line_ci.png").exists()


class TestLineWithCI:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        line_with_ci(
            ax, [1, 2, 3], [10, 20, 30], [8, 18, 28], [12, 22, 32], label="Test"
        )
        savefig(fig, tmp_path / "line_ci2.png")
        assert (tmp_path / "line_ci2.png").exists()


# ── Scatter plot ─────────────────────────────────────────────────────────────


class TestScatterPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        scatter_plot(
            ax,
            rng.normal(0, 1, 100).tolist(),
            rng.normal(0, 1, 100).tolist(),
            xlabel="X",
            ylabel="Y",
        )
        savefig(fig, tmp_path / "scatter.png")
        assert (tmp_path / "scatter.png").exists()


# ── Box plot ─────────────────────────────────────────────────────────────────


class TestBoxPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        box_plot(
            ax,
            {
                "Control": rng.normal(10, 2, 50).tolist(),
                "Treatment": rng.normal(15, 3, 50).tolist(),
            },
            ylabel="Response",
        )
        savefig(fig, tmp_path / "box.png")
        assert (tmp_path / "box.png").exists()

    def test_with_points(self, tmp_path):
        fig, ax = plt.subplots()
        box_plot(ax, {"A": [1, 2, 3, 4, 5], "B": [3, 4, 5, 6, 7]}, show_points=True)
        savefig(fig, tmp_path / "box_points.png")
        assert (tmp_path / "box_points.png").exists()

    def test_sorted(self, tmp_path):
        fig, ax = plt.subplots()
        box_plot(
            ax,
            {"Low": [1, 2, 3], "High": [10, 11, 12], "Mid": [5, 6, 7]},
            sort_by_median=True,
        )
        savefig(fig, tmp_path / "box_sorted.png")
        assert (tmp_path / "box_sorted.png").exists()


class TestGroupedBoxplot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = {
            "kalign": {"RV11": [0.9, 0.92, 0.88], "RV12": [0.85, 0.87, 0.83]},
            "mafft": {"RV11": [0.88, 0.90, 0.86], "RV12": [0.82, 0.84, 0.80]},
        }
        grouped_boxplot(ax, data, ["RV11", "RV12"], ["kalign", "mafft"], ylabel="F1")
        savefig(fig, tmp_path / "gbox.png")
        assert (tmp_path / "gbox.png").exists()


# ── Violin plot ──────────────────────────────────────────────────────────────


class TestViolinPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        violin_plot(
            ax,
            {
                "Normal": rng.normal(0, 1, 100).tolist(),
                "Uniform": rng.uniform(-2, 2, 100).tolist(),
            },
        )
        savefig(fig, tmp_path / "violin.png")
        assert (tmp_path / "violin.png").exists()


# ── Histogram ────────────────────────────────────────────────────────────────


class TestHistogramPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        histogram_plot(ax, rng.normal(0, 1, 500).tolist(), xlabel="Value")
        savefig(fig, tmp_path / "hist.png")
        assert (tmp_path / "hist.png").exists()


# ── Heatmap ──────────────────────────────────────────────────────────────────


class TestHeatmapPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        heatmap_plot(ax, [[1, 2, 3], [4, 5, 6]], ["Row1", "Row2"], ["A", "B", "C"])
        savefig(fig, tmp_path / "heatmap.png")
        assert (tmp_path / "heatmap.png").exists()


# ── Strip plot ───────────────────────────────────────────────────────────────


class TestStripPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        strip_plot(ax, {"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]}, ylabel="Score")
        savefig(fig, tmp_path / "strip.png")
        assert (tmp_path / "strip.png").exists()


# ── Swarm plot ───────────────────────────────────────────────────────────────


class TestSwarmPlot:
    def test_basic(self, tmp_path):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(42)
        swarm_plot(
            ax,
            {
                "A": rng.normal(5, 1, 30).tolist(),
                "B": rng.normal(7, 1, 30).tolist(),
            },
        )
        savefig(fig, tmp_path / "swarm.png")
        assert (tmp_path / "swarm.png").exists()


# ── Auto aggregate ───────────────────────────────────────────────────────────


class TestAutoAggregate:
    def test_basic(self):
        cats, means, errors, desc = auto_aggregate_bar(
            {"Control": [10, 12, 11], "Treatment": [20, 22, 21]}
        )
        assert len(cats) == 2
        assert abs(means[0] - 11.0) < 0.01
        assert errors[0] > 0
        assert "SEM" in desc


# ── Legend ───────────────────────────────────────────────────────────────────


class TestLegend:
    def test_basic_bar(self):
        legend = generate_legend(
            "bar",
            y_label="Response Rate (%)",
            x_label="Treatment",
            n_groups=4,
            group_names=["Control", "Drug A", "Drug B", "Drug C"],
        )
        assert "Bar chart" in legend
        assert "Response Rate (%)" in legend
        assert "Treatment" in legend
        assert "4 groups" in legend
        assert legend.endswith(".")

    def test_line_with_ci(self):
        legend = generate_legend(
            "line",
            y_label="Cell Viability (%)",
            x_label="Time (hours)",
            series_names=["Drug A", "Drug B", "Control"],
            error_type="bootstrap",
            log_scale="y",
        )
        assert "Line chart" in legend
        assert "as a function of" in legend
        assert "bootstrap" in legend
        assert "logarithmic" in legend

    def test_box(self):
        legend = generate_legend(
            "box",
            y_label="F1 Score",
            x_label="Method",
            n_groups=5,
            n_per_group=100,
        )
        assert "Box plot" in legend
        assert "interquartile range" in legend
        assert "n = 100 per group" in legend

    def test_histogram(self):
        legend = generate_legend("histogram", y_label="Expression Level")
        assert "Histogram" in legend

    def test_with_aggregation(self):
        legend = generate_legend(
            "bar",
            y_label="Score",
            x_label="Group",
            aggregation="mean ± SEM",
        )
        assert "mean ± SEM" in legend

    def test_scatter(self):
        legend = generate_legend(
            "scatter",
            y_label="Y",
            x_label="X",
            n_observations=500,
        )
        assert "Scatter plot" in legend
        assert "versus" in legend
        assert "500" in legend

    def test_minimal(self):
        legend = generate_legend("bar")
        assert "Bar chart" in legend
        assert legend.endswith(".")
