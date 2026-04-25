"""Chart selection advisor — encodes visualization science as decision rules.

Provides chart type recommendation based on data shape and size, enforces
hard rules (no pie charts, no 3D, no dual y-axes), and emits soft warnings
when a chart choice is likely suboptimal.

Rules grounded in:
- Cleveland & McGill (1984) — perceptual hierarchy of visual encodings
- Streit & Gehlenborg, Nature Methods 2014 — bar charts for counts only
- Wilke, "Fundamentals of Data Visualization" (2019) — chart selection
- Rougier et al., PLOS 2014 — ten simple rules for better figures
"""

from __future__ import annotations

from .types import BANNED_KINDS, DataTable, PlotKind


def check_banned(kind: str) -> None:
    """Raise ValueError if the requested kind is explicitly banned.

    Args:
        kind: Plot kind string.

    Raises:
        ValueError: If kind is banned (pie, donut, 3d_bar, 3d_pie).
    """
    key = kind.lower().replace("-", "_")
    if key in BANNED_KINDS:
        raise ValueError(f"Refused to create '{kind}' chart. {BANNED_KINDS[key]}")


def recommend_kind(
    table: DataTable,
    x: str | None,
    y: str | list[str] | None,
    group: str | None,
) -> str:
    """Recommend the best chart type based on data shape.

    Uses the decision tree from the PRD, grounded in visualization science.

    Args:
        table: Normalized data table.
        x: X-axis column name (may be None for auto-detection).
        y: Y-axis column name(s) (may be None for auto-detection).
        group: Grouping column name (may be None).

    Returns:
        Recommended PlotKind value string.
    """
    cols = table.column_names
    numeric_cols = [c for c in cols if table.is_numeric(c)]
    categorical_cols = [c for c in cols if table.is_categorical(c)]

    # Resolve x and y if not provided
    x_col = x
    y_cols: list[str] = []
    if isinstance(y, str):
        y_cols = [y]
    elif isinstance(y, list):
        y_cols = list(y)

    # Auto-detect x column
    if x_col is None:
        if categorical_cols:
            x_col = categorical_cols[0]
        elif len(numeric_cols) >= 2:
            x_col = numeric_cols[0]

    # Auto-detect y columns
    if not y_cols:
        y_cols = [c for c in numeric_cols if c != x_col and c != group]

    # If y was explicitly specified with multiple columns, prefer line
    if isinstance(y, list) and len(y) > 1 and x_col is not None:
        return PlotKind.LINE.value

    # Single numeric column, no x → histogram
    if len(y_cols) == 1 and x_col is None and not categorical_cols:
        return PlotKind.HISTOGRAM.value

    # All numeric columns, roughly square → heatmap (only when auto-detecting)
    if not categorical_cols and len(numeric_cols) >= 3 and not y:
        if table.n_rows >= 3 and 0.3 <= table.n_rows / len(numeric_cols) <= 3.0:
            return PlotKind.HEATMAP.value

    # X is categorical, Y is numeric
    if x_col and table.is_categorical(x_col) and y_cols:
        y_col = y_cols[0]
        if table.is_numeric(y_col):
            n_per_group = _n_per_group(table, x_col, y_col)
            if n_per_group == 1:
                return PlotKind.BAR.value
            elif n_per_group < 7:
                return PlotKind.STRIP.value
            elif n_per_group <= 30:
                return PlotKind.BOX.value
            elif n_per_group <= 200:
                return PlotKind.BOX.value
            else:
                return PlotKind.VIOLIN.value

    # X is numeric, Y is numeric
    if x_col and table.is_numeric(x_col) and y_cols:
        if len(y_cols) > 1:
            return PlotKind.LINE.value
        # Check if x looks ordered/time-like
        if _looks_ordered(x_col):
            return PlotKind.LINE.value
        return PlotKind.SCATTER.value

    # Fallback
    if y_cols:
        return PlotKind.BAR.value

    return PlotKind.BAR.value


def validate_kind(
    kind: str,
    table: DataTable,
    x: str | None,
    y: str | list[str] | None,
) -> list[str]:
    """Validate a chosen chart type and return warnings if suboptimal.

    Does NOT raise errors for suboptimal choices — only check_banned() does that.
    Returns a list of advisory notes.

    Args:
        kind: The chosen plot kind.
        table: Normalized data table.
        x: X-axis column.
        y: Y-axis column(s).

    Returns:
        List of warning strings. Empty if choice is fine.
    """
    warnings: list[str] = []

    y_cols: list[str] = []
    if isinstance(y, str):
        y_cols = [y]
    elif isinstance(y, list):
        y_cols = list(y)

    # Bar chart for continuous distributions
    if kind == PlotKind.BAR.value and x and y_cols:
        if table.is_categorical(x):
            y_col = y_cols[0]
            if table.is_numeric(y_col):
                n_per = _n_per_group(table, x, y_col)
                if n_per > 1:
                    warnings.append(
                        f"Bar chart hides distribution shape (n={n_per} per group). "
                        "Consider box, violin, or strip plot to show the underlying distribution "
                        "(Streit & Gehlenborg, Nature Methods 2014)."
                    )

    # Too many categorical colors
    if x and table.is_categorical(x):
        n_cats = table.unique_count(x)
        if n_cats > 8:
            warnings.append(
                f"This figure uses {n_cats} categorical colors. Beyond 8 colors, humans cannot "
                "reliably distinguish categories (Rougier et al. 2014). Consider faceting or "
                "highlighting key groups."
            )

    # Too many overlapping line series
    if kind == PlotKind.LINE.value and len(y_cols) > 4:
        warnings.append(
            f"Line chart with {len(y_cols)} overlapping series may be difficult to read. "
            "Consider facets/small multiples for clearer comparison (Wilke 2019)."
        )

    # Box plot with very few observations
    if kind in (PlotKind.BOX.value, PlotKind.VIOLIN.value) and x and y_cols:
        if table.is_categorical(x):
            y_col = y_cols[0]
            if table.is_numeric(y_col):
                n_per = _n_per_group(table, x, y_col)
                if n_per < 5:
                    warnings.append(
                        f"Box/violin plot with only {n_per} observations per group is unreliable. "
                        "Consider strip plot to show individual data points."
                    )

    return warnings


def _n_per_group(table: DataTable, cat_col: str, val_col: str) -> int:
    """Median number of values per category."""
    grouped = table.values_per_category(cat_col, val_col)
    if not grouped:
        return 0
    counts = [len(v) for v in grouped.values()]
    counts.sort()
    mid = len(counts) // 2
    return counts[mid]


def _looks_ordered(col_name: str) -> bool:
    """Heuristic: does a column name suggest ordered/time-like data?"""
    hints = {
        "time",
        "date",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "age",
        "dose",
        "concentration",
        "step",
        "epoch",
    }
    name_lower = col_name.lower().replace("_", " ").replace("-", " ")
    return any(hint in name_lower for hint in hints)
