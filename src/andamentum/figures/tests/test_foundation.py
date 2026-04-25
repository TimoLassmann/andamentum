"""Tests for Phase 1: foundation modules (types, palettes, standards, style)."""

import matplotlib.pyplot as plt
import pytest

from andamentum.figures.palettes import PALETTES, get_palette, list_palettes
from andamentum.figures.standards import get_preset, list_presets, resolve_width
from andamentum.figures.style import (
    despine,
    panel_label,
    savefig,
    setup_style,
    shared_legend,
)
from andamentum.figures.types import BANNED_KINDS, DataTable, FigureResult, PlotKind


# ── DataTable ────────────────────────────────────────────────────────────────


class TestDataTable:
    def test_from_dict(self):
        dt = DataTable.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]})
        assert dt.column_names == ["x", "y"]
        assert dt.n_rows == 3
        assert dt.n_cols == 2

    def test_from_records(self):
        dt = DataTable.from_records([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        assert dt.column_names == ["a", "b"]
        assert dt.n_rows == 2

    def test_from_csv(self):
        dt = DataTable.from_csv("x,y\n1,10\n2,20\n3,30")
        assert dt.column_names == ["x", "y"]
        assert dt.n_rows == 3
        assert dt.is_numeric("x")
        assert dt.is_numeric("y")

    def test_normalize_dict(self):
        dt = DataTable.normalize({"x": [1, 2]})
        assert dt.n_rows == 2

    def test_normalize_records(self):
        dt = DataTable.normalize([{"x": 1}, {"x": 2}])
        assert dt.n_rows == 2

    def test_normalize_csv(self):
        dt = DataTable.normalize("a,b\n1,2\n3,4")
        assert dt.n_rows == 2

    def test_unequal_columns_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            DataTable({"x": [1, 2], "y": [1]})

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="empty"):
            DataTable.from_records([])

    def test_empty_csv_raises(self):
        with pytest.raises(ValueError, match="no data"):
            DataTable.from_csv("x,y\n")

    def test_is_categorical(self):
        dt = DataTable.from_dict({"name": ["a", "b"], "val": [1, 2]})
        assert dt.is_categorical("name")
        assert not dt.is_categorical("val")

    def test_unique_count(self):
        dt = DataTable.from_dict({"x": [1, 1, 2, 3]})
        assert dt.unique_count("x") == 3

    def test_values_per_category(self):
        dt = DataTable.from_dict({"group": ["A", "A", "B"], "val": [1, 2, 3]})
        grouped = dt.values_per_category("group", "val")
        assert grouped == {"A": [1, 2], "B": [3]}

    def test_csv_mixed_types(self):
        dt = DataTable.from_csv("name,score\nAlice,95.5\nBob,87")
        assert dt.is_categorical("name")
        assert dt.is_numeric("score")
        assert dt.columns["score"] == [95.5, 87]


# ── Palettes ─────────────────────────────────────────────────────────────────


class TestPalettes:
    def test_all_palettes_exist(self):
        for name in ["npg", "nejm", "lancet", "jama", "aaas", "d3", "okabe_ito"]:
            colors = get_palette(name)
            assert len(colors) >= 7
            assert all(c.startswith("#") for c in colors)

    def test_get_palette_subset(self):
        colors = get_palette("npg", 3)
        assert len(colors) == 3

    def test_get_palette_cycling(self):
        base = get_palette("jama")
        extended = get_palette("jama", 14)
        assert len(extended) == 14
        assert extended[0] == base[0]
        assert extended[len(base)] == base[0]  # cycles

    def test_unknown_palette_raises(self):
        with pytest.raises(ValueError, match="Unknown palette"):
            get_palette("nonexistent")

    def test_list_palettes(self):
        result = list_palettes()
        assert "npg" in result
        assert result["npg"] == 10

    def test_hex_format(self):
        for name, colors in PALETTES.items():
            for c in colors:
                assert len(c) == 7, f"Color {c} in {name} is not 7 chars"
                assert c[0] == "#"
                int(c[1:], 16)  # valid hex


# ── Standards ────────────────────────────────────────────────────────────────


class TestStandards:
    def test_all_presets_exist(self):
        for name in ["default", "nature", "science", "cell", "plos", "showcase"]:
            preset = get_preset(name)
            assert preset.dpi >= 150
            assert preset.single_column_width > 0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown journal preset"):
            get_preset("fake_journal")

    def test_resolve_width_named(self):
        preset = get_preset("nature")
        assert resolve_width("single", preset) == 3.5
        assert resolve_width("double", preset) == 7.2
        assert resolve_width("1.5", preset) == 5.5

    def test_resolve_width_numeric(self):
        preset = get_preset("default")
        assert resolve_width(5.0, preset) == 5.0
        assert resolve_width(3, preset) == 3.0

    def test_resolve_width_invalid(self):
        preset = get_preset("default")
        with pytest.raises(ValueError, match="Unknown width"):
            resolve_width("triple", preset)

    def test_list_presets(self):
        result = list_presets()
        assert "nature" in result
        assert "3.5" in result["nature"]

    def test_showcase_larger_fonts(self):
        pub = get_preset("default")
        show = get_preset("showcase")
        assert show.body_font_size > pub.body_font_size
        assert show.label_font_size > pub.label_font_size
        assert show.line_width > pub.line_width


# ── Style ────────────────────────────────────────────────────────────────────


class TestStyle:
    def test_setup_style_returns_preset(self):
        preset = setup_style()
        assert preset.name == "default"

    def test_setup_style_sets_rcparams(self):
        setup_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
        assert plt.rcParams["legend.frameon"] is False
        assert plt.rcParams["savefig.dpi"] == 300.0

    def test_setup_style_journal(self):
        preset = setup_style(journal="nature")
        assert preset.name == "nature"
        assert plt.rcParams["font.size"] == 7

    def test_despine(self):
        fig, ax = plt.subplots()
        despine(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["left"].get_visible()
        plt.close(fig)

    def test_despine_all(self):
        fig, ax = plt.subplots()
        despine(ax, left=True, bottom=True)
        assert not ax.spines["left"].get_visible()
        assert not ax.spines["bottom"].get_visible()
        plt.close(fig)

    def test_panel_label(self):
        fig, ax = plt.subplots()
        panel_label(ax, "A")
        # Check that a text object was added
        texts = [t for t in ax.texts if t.get_text() == "A"]
        assert len(texts) == 1
        assert texts[0].get_fontweight() == "bold"
        plt.close(fig)

    def test_savefig(self, tmp_path):
        setup_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        path = tmp_path / "test.png"
        result = savefig(fig, path)
        assert path.exists()
        assert path.stat().st_size > 0
        assert result == str(path)

    def test_savefig_pdf(self, tmp_path):
        setup_style()
        fig, ax = plt.subplots()
        ax.bar(["A", "B"], [1, 2])
        path = tmp_path / "test.pdf"
        savefig(fig, path)
        assert path.exists()

    def test_shared_legend(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        shared_legend(fig, ["Method A", "Method B"], ["#E64B35", "#4DBBD5"])
        path = tmp_path / "legend_test.png"
        savefig(fig, path)
        assert path.exists()


# ── Types ────────────────────────────────────────────────────────────────────


class TestTypes:
    def test_plot_kind_enum(self):
        assert PlotKind.BAR.value == "bar"
        assert PlotKind.AUTO.value == "auto"

    def test_banned_kinds(self):
        assert "pie" in BANNED_KINDS
        assert "donut" in BANNED_KINDS
        assert "3d_bar" in BANNED_KINDS

    def test_figure_result_model(self):
        result = FigureResult(
            path="test.pdf",
            legend="Test legend",
            kind="bar",
            width_inches=3.5,
            height_inches=3.0,
            dpi=300,
            palette="npg",
            data_summary="3 groups",
        )
        assert result.path == "test.pdf"
        assert result.advisor_notes == []
        assert result.log_scale is None
