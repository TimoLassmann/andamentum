"""andamentum.figures — Publication-quality scientific figure rendering.

Deterministic plotting of your data with journal-matched sizing.
No generative AI — figure() plots only the numbers you give it.

Quick start::

    from andamentum.figures import figure

    result = figure(
        data={"Group": ["A", "B", "C"], "Value": [10, 25, 18]},
        kind="bar",
        title="Comparison",
        y_label="Value (units)",
        style="npg",
        output="fig.pdf",
    )

Low-level API for custom multi-panel figures::

    from andamentum.figures import setup_style, get_palette, panel_label, savefig

    setup_style()
    colors = get_palette("npg")
    # ... build figure with matplotlib ...
    savefig(fig, "output.pdf")

For plot primitives, import from submodules::

    from andamentum.figures.plots import grouped_boxplot, line_with_ci
    from andamentum.figures.stats import bootstrap_ci
"""

# === Functions you can wrap as agent tools ===
from .palettes import get_palette, list_palettes
from .render import figure
from .standards import get_preset, list_presets, resolve_width
from .style import despine, panel_label, savefig, setup_style, shared_legend

# === Result/data types (returned by the above; not tools themselves) ===
from .types import DataTable, FigureMode, FigureResult, PlotKind

__all__ = [
    # Functions / callables
    "figure",
    "setup_style",
    "get_palette",
    "panel_label",
    "savefig",
    "shared_legend",
    "despine",
    "get_preset",
    "list_presets",
    "list_palettes",
    "resolve_width",
    # Data types
    "PlotKind",
    "FigureMode",
    "FigureResult",
    "DataTable",
]
