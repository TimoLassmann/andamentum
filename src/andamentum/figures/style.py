"""Matplotlib style setup and aesthetic helpers.

Configures matplotlib rcParams for publication-quality figures and provides
helper functions used across all plot types: panel labels, spine removal,
shared legends, and figure saving.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .errors import FiguresError
from .standards import JournalPreset, get_preset

# Use non-interactive backend by default
matplotlib.use("Agg")

_log = logging.getLogger(__name__)

# Families matplotlib resolves through its own rc lists rather than by name.
_GENERIC_FAMILIES = {
    "serif",
    "sans-serif",
    "sans serif",
    "cursive",
    "fantasy",
    "monospace",
}


class _WebFontFilter(logging.Filter):
    """Drop matplotlib's per-text "font family not found" warning for web-only fonts.

    A preset can name a family the page supplies (Inter through the report's
    font link) that is not installed where matplotlib runs. Matplotlib then
    logs one warning per text object per draw — hundreds per figure — for a
    fallback that is the intended behaviour. The filter drops exactly those
    messages, for exactly the families ``setup_style`` registered, and
    nothing else the font manager says.
    """

    def __init__(self) -> None:
        super().__init__()
        self.families: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        m = re.match(r"findfont: Font family '(.+)' not found\.?$", msg)
        if m and m.group(1) in self.families:
            return False
        return True


_web_font_filter = _WebFontFilter()
logging.getLogger("matplotlib.font_manager").addFilter(_web_font_filter)


def _register_web_fonts(families: list[str]) -> None:
    """Note which of the preset's families are not installed and silence their warning."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    installed |= {f.name for f in font_manager.fontManager.afmlist}
    for fam in families:
        if fam in _GENERIC_FAMILIES or fam in installed:
            continue
        if fam not in _web_font_filter.families:
            _log.info(
                "Font family '%s' is not installed; raster output falls back to "
                "the next family in the preset. SVG output keeps the name.",
                fam,
            )
        _web_font_filter.families.add(fam)


def setup_style(journal: str = "default") -> JournalPreset:
    """Configure matplotlib rcParams for publication-quality figures.

    Sets font sizes, line widths, spine visibility, legend style, colours,
    and save parameters according to the specified journal preset.

    The whole ``font_family`` list goes into ``font.family``, so matplotlib
    resolves the first installed family in order. SVG output keeps text as
    text (``svg.fonttype: none``), written with the same family list, so a
    page can supply the font and a reader can search the labels.

    Args:
        journal: Journal preset name (default, nature, science, cell, plos,
            showcase, report).

    Returns:
        The JournalPreset that was applied (useful for downstream width/height decisions).
    """
    preset = get_preset(journal)
    families = [f.strip() for f in preset.font_family.split(",") if f.strip()]
    _register_web_fonts(families)

    params: dict[str, Any] = {
        # Font
        "font.family": families,
        "font.size": preset.body_font_size,
        "axes.titlesize": preset.title_font_size,
        "axes.labelsize": preset.label_font_size,
        "xtick.labelsize": preset.body_font_size,
        "ytick.labelsize": preset.body_font_size,
        "legend.fontsize": preset.body_font_size,
        # Lines
        "lines.linewidth": preset.line_width,
        "lines.markersize": preset.marker_size,
        # Axes — remove top and right spines
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": preset.rule,
        "axes.labelcolor": preset.ink,
        "axes.facecolor": preset.background,
        # Text
        "text.color": preset.ink,
        # Ticks
        "xtick.color": preset.rule,
        "xtick.labelcolor": preset.ink,
        "xtick.major.width": 0.8,
        "xtick.major.size": 4,
        "xtick.minor.width": 0.5,
        "ytick.color": preset.rule,
        "ytick.labelcolor": preset.ink,
        "ytick.major.width": 0.8,
        "ytick.major.size": 4,
        "ytick.minor.width": 0.5,
        # Grid — off by default
        "axes.grid": False,
        # Legend — clean, no border
        "legend.frameon": False,
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": preset.background,
        "savefig.dpi": preset.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": preset.pad_inches,
        "savefig.facecolor": preset.background,
        # SVG — keep text as text so the page's font and the reader's
        # search both work. Safe for every preset.
        "svg.fonttype": "none",
        # Patch — solid fill, no outline (Nature Methods recommendation)
        "patch.linewidth": 0,
    }

    # matplotlib-stubs type RcParams keys as the RcKeyType Literal; our keys
    # are valid rc keys typed as plain str, so cast to satisfy update().
    plt.rcParams.update(cast("dict[Any, Any]", params))
    return preset


def despine(ax: Axes, *, left: bool = False, bottom: bool = False) -> None:
    """Remove spines from axes.

    By default removes top and right spines (already handled by setup_style rcParams).
    Optionally also removes left and/or bottom spines for minimal designs.

    Args:
        ax: Matplotlib axes.
        left: Also remove left spine.
        bottom: Also remove bottom spine.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)


def panel_label(ax: Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    """Add a bold panel label (A, B, C...) to axes.

    Positioned in axes coordinates so it stays in place regardless of data.
    Follows Nature/Science convention: bold, slightly larger than body text.

    Args:
        ax: Matplotlib axes.
        label: Label text (typically "A", "B", "C" or "a", "b", "c").
        x: X position in axes fraction (negative = left of axes).
        y: Y position in axes fraction (> 1 = above axes).
    """
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def shared_legend(
    fig: Figure,
    labels: list[str],
    colors: list[str],
    *,
    ncol: int = 6,
    marker: str = "s",
    y_offset: float = -0.02,
) -> None:
    """Add a shared legend below all panels.

    Creates a unified legend at the bottom of the figure, useful for
    multi-panel figures where all panels share the same color coding.

    Args:
        fig: Matplotlib figure.
        labels: Legend labels.
        colors: Colors corresponding to each label.
        ncol: Number of columns in the legend.
        marker: Marker style for legend entries.
        y_offset: Vertical position (fraction of figure height, negative = below).
    """
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="w",
            markerfacecolor=c,
            markersize=8,
            label=lab,
        )
        for lab, c in zip(labels, colors)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=min(ncol, len(labels)),
        frameon=False,
    )


def savefig(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int | None = None,
    pad: float = 0.05,
    facecolor: str | None = None,
) -> str:
    """Save figure with publication-quality settings.

    Args:
        fig: Matplotlib figure.
        path: Output file path. Format inferred from extension.
        dpi: Override DPI (default: uses rcParams savefig.dpi).
        pad: Padding around figure in inches.
        facecolor: Background colour. Defaults to the active preset's
            ``background`` (set by ``setup_style`` as ``savefig.facecolor``),
            which is white for every journal preset and ``"none"`` for
            ``report``.

    Returns:
        The resolved output path as string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if facecolor is None:
        facecolor = str(plt.rcParams["savefig.facecolor"])
        if facecolor == "auto":
            facecolor = "white"

    save_kwargs: dict[str, Any] = {
        "bbox_inches": "tight",
        "pad_inches": pad,
        "facecolor": facecolor,
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi

    fig.savefig(str(path), **save_kwargs)
    plt.close(fig)
    return str(path)


_XML_DECL = re.compile(r"<\?xml[^>]*\?>\s*", re.DOTALL)
_DOCTYPE = re.compile(r"<!DOCTYPE[^>]*>\s*", re.DOTALL)
_METADATA = re.compile(r"\s*<metadata>.*?</metadata>", re.DOTALL)
_ROOT_SIZE = re.compile(r'\s(?:width|height)="[^"]*"')


def inline_svg(path: str | Path) -> str:
    """Read a saved SVG and return the bare ``<svg>`` element for inlining.

    Removes the XML declaration, the DOCTYPE, matplotlib's ``<metadata>``
    block, and the ``width`` and ``height`` attributes on the root element.
    ``viewBox`` stays, so the element scales to its container: a report
    drops the string inside ``<div class="figure">`` and the page's CSS
    sets the width. The result still opens in a browser on its own,
    because the ``xmlns`` declaration is on the root and is kept.

    Args:
        path: Path to an SVG file written by ``savefig``.

    Returns:
        The trimmed element as a string, starting with ``<svg``.

    Raises:
        FiguresError: If the file has no ``<svg`` root or no ``viewBox``.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = _XML_DECL.sub("", text, count=1)
    text = _DOCTYPE.sub("", text, count=1)
    text = _METADATA.sub("", text, count=1)
    text = text.strip()

    start = text.find("<svg")
    if start < 0:
        raise FiguresError(f"{path} has no <svg> root element")
    text = text[start:]
    root_end = text.index(">")
    root = text[:root_end]
    if "viewBox" not in root:
        raise FiguresError(f"{path}: the <svg> root has no viewBox; cannot inline")
    root = _ROOT_SIZE.sub("", root)
    return root + text[root_end:]
