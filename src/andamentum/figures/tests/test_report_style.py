"""Tests for the report look: andamentum palette, report preset, inline SVG,
the stacked kind, and the ``--group`` fix on plain bar charts.

The report figure that motivated this work: three arms, three outcomes
each, one row per (arm, outcome), equal totals.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

from andamentum.figures import figure, get_palette, get_preset, inline_svg
from andamentum.figures.advisor import stacked_totals_note
from andamentum.figures.cli import _parse_series_colors
from andamentum.figures.palettes import (
    ANDAMENTUM_TONES,
    assign_series_colors,
    palette_capacity_note,
    resolve_tone,
)
from andamentum.figures.plots import _contrast, _relative_luminance, stacked_bar
from andamentum.figures.style import setup_style

ARMS_CSV = """Arm,Outcome,Claims
Epistemic pipeline,wrong,31
Epistemic pipeline,refused,60
Epistemic pipeline,right,62
Frontier single call,wrong,55
Frontier single call,refused,2
Frontier single call,right,96
Small model single call,wrong,58
Small model single call,refused,0
Small model single call,right,95
"""

TONES = ["#6e7a8c", "#9a8559", "#5e7a6a", "#9c4a36", "#555555", "#999999"]
INK = "#1a1a1a"
RULE = "#d4d0c8"
PAPER = "#f9f7f4"
MUTED = "#777777"

_FILL = re.compile(r"fill:\s*(#[0-9a-fA-F]{6}|none|[a-z]+)")


def _fills(svg: str) -> set[str]:
    return {m.lower() for m in _FILL.findall(svg)}


def _stacked(tmp_path: Path, **kwargs: Any):
    defaults: dict[str, Any] = dict(
        kind="stacked",
        x="Arm",
        y="Claims",
        group="Outcome",
        group_order=["wrong", "refused", "right"],
        style="andamentum",
        journal="report",
        y_label="Claims (n = 153 per arm)",
        out=str(tmp_path / "arms.svg"),
    )
    defaults.update(kwargs)
    return figure(ARMS_CSV, **defaults)


# ── 1. Palette ────────────────────────────────────────────────────────────


class TestAndamentumPalette:
    def test_six_hexes_in_order(self):
        assert get_palette("andamentum") == TONES

    def test_never_cycles(self):
        with pytest.raises(ValueError, match="does not cycle"):
            get_palette("andamentum", 7)

    def test_capacity_note_past_four(self):
        assert palette_capacity_note("andamentum", 4) is None
        note = palette_capacity_note("andamentum", 5)
        assert note is not None and "Merge or facet" in note
        assert palette_capacity_note("npg", 12) is None

    def test_resolve_tone_names_and_aliases(self):
        assert resolve_tone("rust") == "#9c4a36"
        assert resolve_tone("danger") == "#9c4a36"
        assert resolve_tone("grey_light") == "#999999"
        with pytest.raises(ValueError, match="Unknown tone"):
            resolve_tone("#9c4a36")

    def test_assign_series_colors_skips_taken_hexes(self):
        out = assign_series_colors(
            ["a", "b", "c"], list(ANDAMENTUM_TONES.values()), {"b": "slate"}
        )
        assert out[1] == "#6e7a8c"
        assert out[0] != "#6e7a8c" and out[2] != "#6e7a8c"
        assert len(set(out)) == 3

    def test_assign_series_colors_unknown_name(self):
        with pytest.raises(ValueError, match="not in the data"):
            assign_series_colors(["a"], TONES, {"zzz": "rust"})


# ── 2. Preset ─────────────────────────────────────────────────────────────


class TestReportPreset:
    def test_fields(self):
        p = get_preset("report")
        assert p.background == "none"
        assert p.min_font_size == 9.75
        assert p.font_family.startswith("Inter")
        assert p.ink == INK
        assert p.rule == RULE
        assert p.single_column_width == 7.5
        assert p.fit_layout is True
        assert p.pad_inches == 0.0

    def test_journal_presets_keep_their_defaults(self):
        for name in ("default", "nature", "science", "cell", "plos", "showcase"):
            p = get_preset(name)
            assert p.background == "white"
            assert p.ink == "#000000"
            assert p.fit_layout is False
            assert p.pad_inches == 0.05

    def test_setup_style_applies_colours_and_svg_text(self):
        setup_style("report")
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["text.color"] == INK
        assert plt.rcParams["axes.edgecolor"] == RULE
        assert plt.rcParams["savefig.facecolor"] == "none"
        assert list(plt.rcParams["font.family"])[0] == "Inter"
        # And a journal preset restores white paper.
        setup_style("nature")
        assert plt.rcParams["savefig.facecolor"] == "white"
        assert plt.rcParams["svg.fonttype"] == "none"


# ── 3. Report SVG output ─────────────────────────────────────────────────


class TestReportSvg:
    def test_only_token_fills_no_white_and_real_text(self, tmp_path):
        result = _stacked(tmp_path)
        svg = Path(result.path).read_text()
        fills = _fills(svg)
        assert "#ffffff" not in fills
        allowed = set(TONES) | {INK, RULE, PAPER, MUTED, "none"}
        assert fills <= allowed, fills - allowed
        # The three tones used are exactly the first three.
        assert set(TONES[:3]) <= fills
        assert "<text" in svg
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        assert "Epistemic pipeline" in texts
        assert "153" in texts
        assert "'Inter'" in svg

    def test_text_at_least_13px_at_report_width(self, tmp_path):
        """9.75 pt is 13 px only if the viewBox is no wider than 540 pt (7.5 in)."""
        result = _stacked(tmp_path, horizontal=False)
        svg = Path(result.path).read_text()
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        assert m is not None
        width_pt = float(m.group(1))
        assert width_pt <= 540.0, width_pt
        sizes = [float(s) for s in re.findall(r"font-size:\s*([\d.]+)px", svg)]
        assert sizes and min(sizes) * 720.0 / width_pt >= 13.0

    def test_journal_path_untouched(self, tmp_path):
        result = _stacked(
            tmp_path, style="npg", journal="nature", out=str(tmp_path / "arms.pdf")
        )
        assert Path(result.path).exists()
        png = _stacked(
            tmp_path, style="npg", journal="nature", out=str(tmp_path / "arms.png")
        )
        assert Path(png.path).exists()


# ── 4. inline_svg ────────────────────────────────────────────────────────


class TestInlineSvg:
    def test_trimmed_root(self, tmp_path):
        result = _stacked(tmp_path)
        text = inline_svg(result.path)
        assert text.startswith("<svg")
        root = text[: text.index(">")]
        assert "viewBox" in root
        assert "xmlns=" in root
        assert " width=" not in root and " height=" not in root
        assert "<?xml" not in text
        assert "<!DOCTYPE" not in text
        assert "<metadata>" not in text
        assert text.rstrip().endswith("</svg>")

    def test_svg_inline_flag_writes_trimmed_file(self, tmp_path):
        result = _stacked(tmp_path, svg_inline=True)
        text = Path(result.path).read_text()
        assert text.startswith("<svg")
        assert "<?xml" not in text
        # Idempotent: inlining the trimmed file gives the same element back.
        assert inline_svg(result.path) == text

    def test_svg_inline_needs_svg_output(self, tmp_path):
        with pytest.raises(ValueError, match="svg_inline"):
            _stacked(tmp_path, svg_inline=True, out=str(tmp_path / "a.png"))


# ── 5–7. Stacked kind ───────────────────────────────────────────────────


class TestStackedKind:
    def test_long_form_one_bar_per_arm_three_segments(self, tmp_path):
        result = _stacked(tmp_path)
        assert result.kind == "stacked"
        assert result.series_names == ["wrong", "refused", "right"]
        svg = Path(result.path).read_text()
        # Each tone is one series; every series draws three bars.
        for tone in TONES[:3]:
            assert svg.count(f"fill: {tone}") >= 3
        assert "3 groups, 3 series" in result.data_summary
        assert "segmented by Outcome (wrong, refused, right)" in result.legend
        assert "(n = 153)" in result.legend
        assert not any("hides distribution" in n for n in result.advisor_notes)

    def test_default_series_order_is_first_appearance(self, tmp_path):
        result = _stacked(tmp_path, group_order=None)
        assert result.series_names == ["wrong", "refused", "right"]

    def test_wide_form(self, tmp_path):
        data = {"Arm": ["A", "B"], "wrong": [1, 2], "right": [3, 4]}
        result = figure(
            data,
            kind="stacked",
            x="Arm",
            y=["wrong", "right"],
            out=str(tmp_path / "w.svg"),
        )
        assert result.series_names == ["wrong", "right"]
        assert "segmented by series (wrong, right)" in result.legend
        assert "n = 4 to 6" in result.legend

    def test_unequal_totals_note_and_normalize(self, tmp_path):
        data = {
            "Arm": ["A", "A", "B", "B"],
            "Outcome": ["x", "y", "x", "y"],
            "n": [1, 2, 3, 4],
        }
        result = figure(
            data,
            kind="stacked",
            x="Arm",
            y="n",
            group="Outcome",
            out=str(tmp_path / "u.svg"),
        )
        assert any(
            "Category totals differ (min 3, max 7)" in n for n in result.advisor_notes
        )
        normalized = figure(
            data,
            kind="stacked",
            x="Arm",
            y="n",
            group="Outcome",
            normalize=True,
            out=str(tmp_path / "n.svg"),
        )
        assert not any("totals differ" in n for n in normalized.advisor_notes)
        assert "n = 3 to 7" in normalized.legend

    def test_negative_value_raises(self, tmp_path):
        data = {"Arm": ["A", "A"], "Outcome": ["x", "y"], "n": [1, -2]}
        with pytest.raises(ValueError, match="negative"):
            figure(
                data,
                kind="stacked",
                x="Arm",
                y="n",
                group="Outcome",
                out=str(tmp_path / "neg.svg"),
            )

    def test_duplicate_pair_raises(self, tmp_path):
        data = {"Arm": ["A", "A"], "Outcome": ["x", "x"], "n": [1, 2]}
        with pytest.raises(ValueError, match="Two rows"):
            figure(
                data,
                kind="stacked",
                x="Arm",
                y="n",
                group="Outcome",
                out=str(tmp_path / "dup.svg"),
            )

    def test_missing_pair_drawn_as_zero_with_note(self, tmp_path):
        data = {"Arm": ["A", "A", "B"], "Outcome": ["x", "y", "x"], "n": [1, 2, 3]}
        result = figure(
            data,
            kind="stacked",
            x="Arm",
            y="n",
            group="Outcome",
            out=str(tmp_path / "m.svg"),
        )
        assert any("No row for (B, y)" in n for n in result.advisor_notes)

    def test_group_order_must_cover_every_series(self, tmp_path):
        with pytest.raises(ValueError, match="every series exactly once"):
            _stacked(tmp_path, group_order=["wrong", "right"])
        with pytest.raises(ValueError, match="every series exactly once"):
            _stacked(tmp_path, group_order=["wrong", "refused", "right", "extra"])

    def test_stacked_needs_series(self, tmp_path):
        with pytest.raises(ValueError, match="needs a group column"):
            figure(
                {"A": ["x", "y"], "n": [1, 2]},
                kind="stacked",
                x="A",
                y="n",
                out=str(tmp_path / "s.svg"),
            )

    def test_horizontal_honoured_both_ways(self, tmp_path):
        h = _stacked(tmp_path, horizontal=True, out=str(tmp_path / "h.svg"))
        v = _stacked(tmp_path, horizontal=False, out=str(tmp_path / "v.svg"))
        assert not any("auto-selected" in n.lower() for n in h.advisor_notes)
        assert any("Consider horizontal=True" in n for n in v.advisor_notes)

    def test_sort_by_total(self, tmp_path):
        data = {
            "Arm": ["A", "A", "B", "B"],
            "Outcome": ["x", "y", "x", "y"],
            "n": [1, 2, 3, 4],
        }
        result = figure(
            data,
            kind="stacked",
            x="Arm",
            y="n",
            group="Outcome",
            sort="value",
            out=str(tmp_path / "s.svg"),
        )
        svg = Path(result.path).read_text()
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        assert texts.index("B") < texts.index("A")

    def test_series_colors_by_meaning(self, tmp_path):
        result = _stacked(
            tmp_path,
            series_colors={"wrong": "rust", "refused": "ochre", "right": "olive"},
        )
        svg = Path(result.path).read_text()
        assert "#9c4a36" in svg and "#5e7a6a" in svg and "#9a8559" in svg
        assert "#6e7a8c" not in svg

    def test_series_colors_rejects_hex(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown tone"):
            _stacked(tmp_path, series_colors={"wrong": "#ff0000"})

    def test_andamentum_past_four_series_notes_past_six_raises(self, tmp_path):
        arms = ["A"] * 5
        data = {"Arm": arms, "Outcome": list("abcde"), "n": [1, 1, 1, 1, 1]}
        result = figure(
            data,
            kind="stacked",
            x="Arm",
            y="n",
            group="Outcome",
            style="andamentum",
            out=str(tmp_path / "5.svg"),
        )
        assert any("carries 4 tones" in n for n in result.advisor_notes)
        data7 = {"Arm": ["A"] * 7, "Outcome": list("abcdefg"), "n": [1] * 7}
        with pytest.raises(ValueError, match="does not cycle"):
            figure(
                data7,
                kind="stacked",
                x="Arm",
                y="n",
                group="Outcome",
                style="andamentum",
                out=str(tmp_path / "7.svg"),
            )


class TestStackedBarPrimitive:
    def test_returns_totals_and_labels(self, tmp_path):
        setup_style("report")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        totals = stacked_bar(
            ax, {"x": [10, 20], "y": [30, 5]}, ["A", "B"], colors=TONES[:2], ink=INK
        )
        assert totals == [40.0, 25.0]
        texts = [t.get_text() for t in ax.texts]
        assert "40" in texts and "25" in texts
        assert "10" in texts and "30" in texts
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["x", "y"]
        plt.close(fig)

    def test_tiny_segment_gets_no_label(self):
        setup_style("report")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        stacked_bar(ax, {"x": [100.0], "y": [0.5]}, ["A"], colors=TONES[:2])
        texts = [t.get_text() for t in ax.texts]
        assert "0.5" not in texts
        plt.close(fig)

    def test_wrong_length_raises(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="values for 2 categories"):
            stacked_bar(ax, {"x": [1]}, ["A", "B"])
        plt.close(fig)

    def test_label_colour_by_contrast(self):
        assert _relative_luminance("#ffffff") == pytest.approx(1.0)
        assert _relative_luminance("#000000") == 0.0
        # Rust is dark: paper wins. Ochre is lighter: ink wins.
        assert _contrast("#9c4a36", PAPER) > _contrast("#9c4a36", INK)
        assert _contrast("#9a8559", INK) > _contrast("#9a8559", PAPER)


class TestStackedTotalsNote:
    def test_equal_totals_none(self):
        assert stacked_totals_note([153, 153, 153]) is None

    def test_unequal_totals(self):
        note = stacked_totals_note([148, 153, 150])
        assert note == (
            "Category totals differ (min 148, max 153). "
            "State the totals in the caption or use --normalize."
        )


# ── 8. The --group fix on plain bar ─────────────────────────────────────


class TestBarWithGroup:
    def test_group_draws_one_bar_per_series(self, tmp_path):
        result = figure(
            ARMS_CSV,
            kind="bar",
            x="Arm",
            y="Claims",
            group="Outcome",
            style="okabe_ito",
            out=str(tmp_path / "g.svg"),
        )
        assert result.series_names == ["wrong", "refused", "right"]
        assert result.aggregation is None
        assert not any("averaged" in n for n in result.advisor_notes)
        assert not any("hides distribution" in n for n in result.advisor_notes)
        svg = Path(result.path).read_text()
        # Three series colours, three bars each.
        for tone in ("#e69f00", "#56b4e9", "#009e73"):
            assert svg.count(f"fill: {tone}") >= 3
        texts = re.findall(r"<text[^>]*>([^<]*)</text>", svg)
        assert {"wrong", "refused", "right"} <= set(texts)

    def test_no_group_still_averages_and_says_so(self, tmp_path):
        result = figure(
            ARMS_CSV, kind="bar", x="Arm", y="Claims", out=str(tmp_path / "avg.svg")
        )
        assert result.aggregation == "mean ± SEM"
        assert any(
            "Rows per category were averaged (mean ± SEM)" in n
            for n in result.advisor_notes
        )
        assert "mean ± SEM" in result.legend

    def test_group_order_on_bar(self, tmp_path):
        result = figure(
            ARMS_CSV,
            kind="bar",
            x="Arm",
            y="Claims",
            group="Outcome",
            group_order=["right", "refused", "wrong"],
            out=str(tmp_path / "go.svg"),
        )
        assert result.series_names == ["right", "refused", "wrong"]

    def test_group_order_without_series_raises(self, tmp_path):
        with pytest.raises(ValueError, match="group_order applies"):
            figure(
                {"A": ["x", "y"], "n": [1, 2]},
                kind="bar",
                x="A",
                y="n",
                group_order=["x", "y"],
                out=str(tmp_path / "x.svg"),
            )

    def test_bar_series_colors_map_categories(self, tmp_path):
        result = figure(
            {"Group": ["ctl", "trt"], "n": [1, 5]},
            kind="bar",
            x="Group",
            y="n",
            sort="value",
            series_colors={"trt": "rust"},
            out=str(tmp_path / "c.svg"),
        )
        assert result.series_names is None
        svg = Path(result.path).read_text()
        assert "#9c4a36" in svg


# ── CLI parsing ──────────────────────────────────────────────────────────


class TestCliParsing:
    def test_series_colors(self):
        assert _parse_series_colors("wrong=rust, right=olive") == {
            "wrong": "rust",
            "right": "olive",
        }
        assert _parse_series_colors(None) is None

    def test_help_strings_come_from_registries(self):
        from andamentum.figures.cli import build_parser

        parser = build_parser()
        subparsers = [
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
        ]
        help_text = subparsers[0].choices["plot"].format_help()
        assert "stacked" in help_text
        assert "andamentum" in help_text
        assert "report" in help_text
        assert "--svg-inline" in help_text
        assert "--group-order" in help_text
        assert "--normalize" in help_text
        assert "--series-colors" in help_text
        assert "--horizontal" in help_text


class TestCliCommand:
    CSV = "group,value\nA,1\nB,2\n"

    def _data(self, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text(self.CSV)
        return f

    def test_out_is_required(self, tmp_path, capsys):
        from andamentum.figures.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["plot", str(self._data(tmp_path))])
        assert exc.value.code == 2

    def test_json_output_and_exit_codes(self, tmp_path, capsys):
        import json as _json

        from andamentum.figures.cli import main

        out = tmp_path / "f.png"
        rc = main(["plot", str(self._data(tmp_path)), "-o", str(out), "--json"])
        assert rc == 0
        assert _json.loads(capsys.readouterr().out)["path"] == str(out)
        assert out.exists()
        assert main(["plot", str(tmp_path / "missing.csv"), "-o", str(out)]) == 3
        assert main(["plot", str(self._data(tmp_path)), "-o", str(out), "--kind", "pie"]) == 2

    def test_plot_is_the_default_command(self, tmp_path, capsys):
        from andamentum.figures.cli import main

        out = tmp_path / "default.svg"
        rc = main(
            [str(self._data(tmp_path)), "--x", "group", "--y", "value", "-o", str(out)]
        )
        assert rc == 0 and out.exists()
        assert main(["palettes"]) == 0
        assert main(["journals"]) == 0
