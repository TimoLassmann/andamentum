"""Integration tests — end-to-end figure() calls."""

import matplotlib
matplotlib.use("Agg")

import pytest

from andamentum.figures import figure
from andamentum.figures.types import FigureResult


class TestFigureBar:
    def test_simple_bar(self, tmp_path):
        result = figure(
            data={"Treatment": ["Control", "Drug A", "Drug B"], "Response": [23.5, 45.2, 67.8]},
            kind="bar",
            x="Treatment",
            y="Response",
            y_label="Response Rate (%)",
            style="npg",
            output=tmp_path / "bar.pdf",
        )
        assert isinstance(result, FigureResult)
        assert (tmp_path / "bar.pdf").exists()
        assert result.kind == "bar"
        assert "Bar chart" in result.legend
        assert result.palette == "npg"

    def test_bar_with_errors(self, tmp_path):
        result = figure(
            data={"Group": ["A", "B", "C"], "Value": [10, 20, 30], "Error": [1, 2, 3]},
            kind="bar",
            x="Group",
            y="Value",
            error="Error",
            error_type="sem",
            output=tmp_path / "bar_err.pdf",
        )
        assert (tmp_path / "bar_err.pdf").exists()

    def test_bar_auto_aggregate(self, tmp_path):
        result = figure(
            data={
                "Group": ["A", "A", "A", "B", "B", "B"],
                "Value": [10, 12, 11, 20, 22, 21],
            },
            kind="bar",
            x="Group",
            y="Value",
            output=tmp_path / "bar_agg.pdf",
        )
        assert (tmp_path / "bar_agg.pdf").exists()
        # Should warn about hiding distribution
        assert any("hides distribution" in n for n in result.advisor_notes)


class TestFigureLine:
    def test_multi_series_line(self, tmp_path):
        result = figure(
            data={
                "Time (h)": [0, 1, 2, 4, 8],
                "Drug A": [100, 85, 60, 30, 10],
                "Drug B": [100, 95, 85, 70, 50],
            },
            kind="line",
            x="Time (h)",
            y=["Drug A", "Drug B"],
            y_label="Cell Viability (%)",
            style="nejm",
            output=tmp_path / "line.pdf",
        )
        assert (tmp_path / "line.pdf").exists()
        assert result.kind == "line"
        assert "Line chart" in result.legend


class TestFigureScatter:
    def test_scatter(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"X": [rng.gauss(0, 1) for _ in range(100)],
                  "Y": [rng.gauss(0, 1) for _ in range(100)]},
            kind="scatter",
            x="X",
            y="Y",
            style="aaas",
            output=tmp_path / "scatter.png",
        )
        assert (tmp_path / "scatter.png").exists()
        assert result.kind == "scatter"


class TestFigureBox:
    def test_box(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={
                "Method": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
                "Score": [rng.gauss(10, 2) for _ in range(30)]
                       + [rng.gauss(15, 3) for _ in range(30)]
                       + [rng.gauss(12, 1) for _ in range(30)],
            },
            kind="box",
            x="Method",
            y="Score",
            y_label="F1 Score",
            style="lancet",
            output=tmp_path / "box.pdf",
        )
        assert (tmp_path / "box.pdf").exists()
        assert "Box plot" in result.legend
        assert "interquartile" in result.legend


class TestFigureViolinStripSwarm:
    def test_violin(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"Group": ["X"] * 50 + ["Y"] * 50,
                  "Val": [rng.gauss(5, 1) for _ in range(100)]},
            kind="violin",
            x="Group",
            y="Val",
            output=tmp_path / "violin.pdf",
        )
        assert (tmp_path / "violin.pdf").exists()

    def test_strip(self, tmp_path):
        result = figure(
            data={"Group": ["A", "A", "A", "B", "B", "B"],
                  "Val": [1, 2, 3, 4, 5, 6]},
            kind="strip",
            x="Group",
            y="Val",
            output=tmp_path / "strip.pdf",
        )
        assert (tmp_path / "strip.pdf").exists()

    def test_swarm(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"Group": ["A"] * 20 + ["B"] * 20,
                  "Val": [rng.gauss(5, 1) for _ in range(40)]},
            kind="swarm",
            x="Group",
            y="Val",
            output=tmp_path / "swarm.pdf",
        )
        assert (tmp_path / "swarm.pdf").exists()


class TestFigureHistogram:
    def test_histogram(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"Expression": [rng.gauss(5, 2) for _ in range(200)]},
            kind="histogram",
            y="Expression",
            y_label="Expression Level",
            output=tmp_path / "hist.pdf",
        )
        assert (tmp_path / "hist.pdf").exists()
        assert "Histogram" in result.legend


class TestFigureHeatmap:
    def test_heatmap(self, tmp_path):
        result = figure(
            data={"A": [1.0, 0.5, 0.3], "B": [0.5, 1.0, 0.7], "C": [0.3, 0.7, 1.0]},
            kind="heatmap",
            output=tmp_path / "heatmap.pdf",
        )
        assert (tmp_path / "heatmap.pdf").exists()


class TestAutoDetection:
    def test_auto_bar_preaggregated(self, tmp_path):
        result = figure(
            data={"Category": ["A", "B", "C"], "Count": [10, 20, 30]},
            output=tmp_path / "auto_bar.pdf",
        )
        assert result.kind == "bar"

    def test_auto_strip_small_n(self, tmp_path):
        result = figure(
            data={"Group": ["A", "A", "A", "B", "B", "B"],
                  "Val": [1, 2, 3, 4, 5, 6]},
            output=tmp_path / "auto_strip.pdf",
        )
        assert result.kind == "strip"

    def test_auto_box_medium_n(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"Group": ["A"] * 20 + ["B"] * 20,
                  "Val": [rng.gauss(5, 1) for _ in range(40)]},
            output=tmp_path / "auto_box.pdf",
        )
        assert result.kind == "box"

    def test_auto_histogram_single_col(self, tmp_path):
        import random
        rng = random.Random(42)
        result = figure(
            data={"values": [rng.gauss(0, 1) for _ in range(100)]},
            y="values",
            output=tmp_path / "auto_hist.pdf",
        )
        assert result.kind == "histogram"


class TestBannedKinds:
    def test_pie_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Refused"):
            figure(data={"A": [1, 2, 3]}, kind="pie", output=tmp_path / "x.pdf")

    def test_donut_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Refused"):
            figure(data={"A": [1, 2, 3]}, kind="donut", output=tmp_path / "x.pdf")


class TestModes:
    def test_showcase(self, tmp_path):
        result = figure(
            data={"Group": ["A", "B", "C"], "Value": [10, 20, 30]},
            kind="bar",
            x="Group",
            y="Value",
            mode="showcase",
            output=tmp_path / "showcase.png",
        )
        assert (tmp_path / "showcase.png").exists()
        # Showcase should be wider than default single-column
        assert result.width_inches > 5.0


class TestDataFormats:
    def test_csv_string(self, tmp_path):
        result = figure(
            data="Group,Value\nA,10\nB,20\nC,30",
            kind="bar",
            x="Group",
            y="Value",
            output=tmp_path / "csv.pdf",
        )
        assert (tmp_path / "csv.pdf").exists()

    def test_records(self, tmp_path):
        result = figure(
            data=[{"Group": "A", "Value": 10}, {"Group": "B", "Value": 20}],
            kind="bar",
            x="Group",
            y="Value",
            output=tmp_path / "records.pdf",
        )
        assert (tmp_path / "records.pdf").exists()


class TestLogScale:
    def test_auto_log(self, tmp_path):
        result = figure(
            data={"Concentration": ["1nM", "10nM", "100nM", "1μM"],
                  "Response": [0.1, 1, 10, 1000]},
            kind="bar",
            x="Concentration",
            y="Response",
            y_label="IC50 (nM)",
            output=tmp_path / "log.pdf",
        )
        # Should auto-detect log scale from label hint and data range
        assert result.log_scale == "y"


class TestPalettes:
    def test_all_palettes_work(self, tmp_path):
        for palette in ["npg", "nejm", "lancet", "jama", "aaas", "d3", "okabe_ito"]:
            result = figure(
                data={"X": ["A", "B", "C"], "Y": [1, 2, 3]},
                kind="bar",
                x="X",
                y="Y",
                style=palette,
                output=tmp_path / f"{palette}.pdf",
            )
            assert (tmp_path / f"{palette}.pdf").exists()
            assert result.palette == palette
