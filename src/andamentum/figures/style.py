"""Matplotlib style setup and aesthetic helpers.

Configures matplotlib rcParams for publication-quality figures and provides
helper functions used across all plot types: panel labels, spine removal,
shared legends, and figure saving.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .standards import JournalPreset, get_preset

# Use non-interactive backend by default
matplotlib.use("Agg")


def setup_style(journal: str = "default") -> JournalPreset:
    """Configure matplotlib rcParams for publication-quality figures.

    Sets font sizes, line widths, spine visibility, legend style, and
    save parameters according to the specified journal preset.

    Args:
        journal: Journal preset name (default, nature, science, cell, plos, showcase).

    Returns:
        The JournalPreset that was applied (useful for downstream width/height decisions).
    """
    preset = get_preset(journal)

    params: dict[str, Any] = {
        # Font
        "font.family": preset.font_family.split(",")[0].strip(),
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
        # Ticks
        "xtick.major.width": 0.8,
        "xtick.major.size": 4,
        "xtick.minor.width": 0.5,
        "ytick.major.width": 0.8,
        "ytick.major.size": 4,
        "ytick.minor.width": 0.5,
        # Grid — off by default
        "axes.grid": False,
        # Legend — clean, no border
        "legend.frameon": False,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": preset.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Patch — solid fill, no outline (Nature Methods recommendation)
        "patch.linewidth": 0,
    }

    plt.rcParams.update(params)
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
    fig: Figure, path: str | Path, *, dpi: int | None = None, pad: float = 0.05
) -> str:
    """Save figure with publication-quality settings.

    Args:
        fig: Matplotlib figure.
        path: Output file path. Format inferred from extension.
        dpi: Override DPI (default: uses rcParams savefig.dpi).
        pad: Padding around figure in inches.

    Returns:
        The resolved output path as string.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {
        "bbox_inches": "tight",
        "pad_inches": pad,
        "facecolor": "white",
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi

    fig.savefig(str(path), **save_kwargs)
    plt.close(fig)
    return str(path)
