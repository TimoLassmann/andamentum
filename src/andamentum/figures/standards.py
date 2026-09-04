"""Journal format presets — dimensions, fonts, DPI, and line widths.

These are independent from color palettes. They control the physical
properties of the figure for journal submission compliance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JournalPreset:
    """Format preset for a specific journal or context."""

    name: str
    single_column_width: float  # inches
    double_column_width: float  # inches
    one_half_column_width: float  # inches
    max_height: float  # inches
    font_family: str
    body_font_size: float  # points — axis tick labels, legend
    label_font_size: float  # points — axis labels
    title_font_size: float  # points — figure title
    panel_label_font_size: float  # points — A, B, C panel labels
    min_font_size: float  # points — smallest allowed text
    dpi: int
    line_width: float  # points
    marker_size: float  # points
    # Colours. Defaults keep every journal preset on its previous output:
    # white paper, black text, black spines, black error bars.
    background: str = "white"  # figure + axes face; "none" for transparent
    ink: str = "#000000"  # text, axis labels, tick labels
    rule: str = "#000000"  # axis spines and tick marks
    muted_ink: str = "#000000"  # error bars and other secondary strokes
    # Layout. Journal presets keep matplotlib's default margins and a small
    # pad around the tight crop. A preset whose width is a promise to a
    # page (report: 7.5 in = 720 CSS px, so that 9.75 pt is 13 px) sets
    # fit_layout so every artist is packed inside the figure box, and no
    # pad, so the output is never wider than the figure.
    fit_layout: bool = False
    pad_inches: float = 0.05


# ── Journal presets ──────────────────────────────────────────────────────────

DEFAULT = JournalPreset(
    name="default",
    single_column_width=3.5,
    double_column_width=7.2,
    one_half_column_width=5.5,
    max_height=9.5,
    font_family="sans-serif",
    body_font_size=8,
    label_font_size=10,
    title_font_size=11,
    panel_label_font_size=13,
    min_font_size=6,
    dpi=300,
    line_width=1.5,
    marker_size=5,
)

NATURE = JournalPreset(
    name="nature",
    single_column_width=3.5,  # 89mm
    double_column_width=7.2,  # 183mm
    one_half_column_width=5.5,  # ~140mm
    max_height=6.7,  # 170mm
    font_family="Helvetica, Arial, sans-serif",
    body_font_size=7,
    label_font_size=10,
    title_font_size=11,
    panel_label_font_size=13,
    min_font_size=5,
    dpi=300,
    line_width=1.5,
    marker_size=5,
)

SCIENCE = JournalPreset(
    name="science",
    single_column_width=3.5,
    double_column_width=7.2,
    one_half_column_width=5.5,
    max_height=9.25,
    font_family="Helvetica, Arial, sans-serif",
    body_font_size=7,
    label_font_size=9,
    title_font_size=10,
    panel_label_font_size=12,
    min_font_size=6,
    dpi=300,
    line_width=1.5,
    marker_size=5,
)

CELL = JournalPreset(
    name="cell",
    single_column_width=3.35,  # 85mm
    double_column_width=6.85,  # 174mm
    one_half_column_width=5.0,
    max_height=9.0,
    font_family="Arial, sans-serif",
    body_font_size=7,
    label_font_size=9,
    title_font_size=10,
    panel_label_font_size=12,
    min_font_size=6,
    dpi=300,
    line_width=1.5,
    marker_size=5,
)

PLOS = JournalPreset(
    name="plos",
    single_column_width=5.2,  # 13.2cm
    double_column_width=7.5,  # 19cm
    one_half_column_width=6.3,
    max_height=8.75,
    font_family="Arial, sans-serif",
    body_font_size=8,
    label_font_size=10,
    title_font_size=11,
    panel_label_font_size=13,
    min_font_size=8,
    dpi=300,
    line_width=1.5,
    marker_size=5,
)

SHOWCASE = JournalPreset(
    name="showcase",
    single_column_width=10.0,  # presentation-friendly
    double_column_width=13.33,  # 16:9 slide width
    one_half_column_width=10.0,
    max_height=7.5,
    font_family="sans-serif",
    body_font_size=12,
    label_font_size=16,
    title_font_size=22,
    panel_label_font_size=20,
    min_font_size=10,
    dpi=150,
    line_width=2.5,
    marker_size=8,
)

# The andamentum report look. Sizes are chosen so that an SVG shown at the
# report column width (720 CSS px = 7.5 in at 96 px/in) has no text smaller
# than 13 px, the floor the report catalogue permits inside a figure.
# Colours are design tokens from docs/design/components.css: text is
# --am-ink (#1a1a1a), spines --am-rule-3 (#d4d0c8), error bars --am-ink-4
# (#777777). The background is transparent so the figure sits on the
# page's own paper in light and dark theme alike.
#
# Inter ships with the report's font link, not with matplotlib. Raster
# output falls back to the next family; SVG output keeps the name as text
# and the browser resolves it. Nothing is bundled.
REPORT = JournalPreset(
    name="report",
    single_column_width=7.5,  # 720 CSS px, the full report column
    double_column_width=7.5,  # no other width exists in a report
    one_half_column_width=7.5,
    max_height=5.0,
    font_family="Inter, system-ui, sans-serif",
    body_font_size=9.75,  # 13 px
    label_font_size=10.5,  # 14 px
    title_font_size=11.25,  # 15 px; the caption carries the title
    panel_label_font_size=11.25,
    min_font_size=9.75,  # 13 px
    dpi=96,  # CSS pixel density; matters only for raster output
    line_width=1.4,  # matches .svg-line
    marker_size=5,
    background="none",
    ink="#1a1a1a",
    rule="#d4d0c8",
    muted_ink="#777777",
    fit_layout=True,
    pad_inches=0.0,
)

# ── Preset registry ──────────────────────────────────────────────────────────

JOURNAL_PRESETS: dict[str, JournalPreset] = {
    "default": DEFAULT,
    "nature": NATURE,
    "science": SCIENCE,
    "cell": CELL,
    "plos": PLOS,
    "showcase": SHOWCASE,
    "report": REPORT,
}


def get_preset(name: str) -> JournalPreset:
    """Get a journal format preset by name.

    Args:
        name: Preset name (default, nature, science, cell, plos, showcase, report).

    Returns:
        JournalPreset with all formatting parameters.

    Raises:
        ValueError: If preset name is not recognized.
    """
    key = name.lower()
    if key not in JOURNAL_PRESETS:
        available = ", ".join(sorted(JOURNAL_PRESETS.keys()))
        raise ValueError(f"Unknown journal preset '{name}'. Available: {available}")
    return JOURNAL_PRESETS[key]


def resolve_width(width: str | float, preset: JournalPreset) -> float:
    """Resolve width specification to inches.

    Args:
        width: "single", "1.5", "double", or a float in inches.
        preset: Journal preset for named width lookup.

    Returns:
        Width in inches.
    """
    if isinstance(width, (int, float)):
        return float(width)
    mapping = {
        "single": preset.single_column_width,
        "1.5": preset.one_half_column_width,
        "double": preset.double_column_width,
    }
    key = str(width).lower()
    if key not in mapping:
        available = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"Unknown width '{width}'. Use {available} or a number in inches."
        )
    return mapping[key]


def list_presets() -> dict[str, str]:
    """List available presets with descriptions.

    Returns:
        Dict mapping preset name to single-column width description.
    """
    return {
        name: f'{p.single_column_width}" single / {p.double_column_width}" double, {p.font_family.split(",")[0]}, {p.body_font_size}pt'
        for name, p in JOURNAL_PRESETS.items()
    }
