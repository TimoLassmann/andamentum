"""High-level figure() orchestrator.

Ties together all modules: data normalization → column detection → chart
selection → style setup → plotting → legend generation → save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .advisor import (
    check_banned,
    recommend_horizontal_bars,
    recommend_kind,
    recommend_label_rotation,
    validate_kind,
)
from .auto import detect_column_roles, detect_log_scale, recommend_sort
from .legend import generate_legend
from .palettes import get_palette
from .plots import (
    auto_aggregate_bar,
    bar_plot,
    box_plot,
    heatmap_plot,
    histogram_plot,
    line_plot,
    scatter_plot,
    strip_plot,
    swarm_plot,
    violin_plot,
)
from .standards import resolve_width
from .style import savefig, setup_style
from .types import DataTable, FigureResult, PlotKind


def figure(
    data: dict[str, list[Any]] | list[dict[str, Any]] | str,
    *,
    kind: str = "auto",
    x: str | None = None,
    y: str | list[str] | None = None,
    group: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    style: str = "npg",
    journal: str = "default",
    mode: str = "publication",
    width: str | float = "single",
    height: float | None = None,
    dpi: int = 300,
    log_scale: str | None = None,
    sort: str | None = None,
    horizontal: bool | None = None,
    output: str | Path = "figure.pdf",
) -> FigureResult:
    """Generate a publication-quality figure from data.

    This is the main entry point. Accepts data in multiple formats,
    auto-detects the best chart type, applies journal styling, and
    saves the figure with an auto-generated legend.

    Args:
        data: Data as columnar dict, list of row dicts, or CSV string.
        kind: Plot type ("bar", "line", "scatter", "box", "violin",
              "histogram", "heatmap", "strip", "swarm", "auto").
        x: X-axis column name (auto-detected if None).
        y: Y-axis column name(s) (auto-detected if None).
        group: Grouping column for color coding.
        error: Error bar column name.
        error_type: Error bar type ("sem", "sd", "ci95").
        title: Figure title.
        x_label: X-axis label (defaults to column name).
        y_label: Y-axis label (defaults to column name).
        style: Color palette name (npg, nejm, lancet, jama, aaas, d3, okabe_ito).
        journal: Journal format preset (default, nature, science, cell, plos).
        mode: "publication" or "showcase".
        width: Figure width ("single", "1.5", "double", or float in inches).
        height: Figure height in inches (auto-calculated if None).
        dpi: Resolution.
        log_scale: "x", "y", "both", or None (auto-detect).
        sort: "value", "name", or None (auto-detect for bar/box).
        horizontal: For bar charts only. True forces horizontal bars,
            False forces vertical, None (default) auto-decides based on
            label count and length. Many bars with long labels become
            illegible mush in vertical orientation; the auto-decider
            picks horizontal for those cases and emits an advisor note.
        output: Output file path.

    Returns:
        FigureResult with path, legend, advisor notes, and metadata.

    Raises:
        ValueError: If kind is banned (pie, 3D, etc.) or data is invalid.

    Caption guidance:
        ``figure()`` plots only — it neither generates nor modifies
        data points. When ``FigureResult.advisor_notes`` is non-empty,
        those notes describe auto-applied visual changes (bar
        orientation flip, log-scale auto-pick, sort order, x-tick
        rotation) that the renderer chose for you. **Mirror them in
        your figure caption** so reviewers can see what was decided
        automatically. Example: "Bars plotted horizontally because of
        label length; y-axis log-scaled because the data spans three
        orders of magnitude."
    """
    # ── 1. Check banned kinds ────────────────────────────────────────────
    check_banned(kind)

    # ── 2. Normalize data ────────────────────────────────────────────────
    table = DataTable.normalize(data)

    # ── 3. Detect column roles ───────────────────────────────────────────
    x_col, y_cols, group_col, error_col = detect_column_roles(table, x, y, group, error)

    # ── 4. Resolve chart type ────────────────────────────────────────────
    resolved_kind = kind
    if kind == "auto" or kind == PlotKind.AUTO.value:
        resolved_kind = recommend_kind(table, x_col, y_cols, group_col)

    # ── 5. Validate and get advisor warnings ─────────────────────────────
    advisor_notes = validate_kind(resolved_kind, table, x_col, y_cols)

    # ── 6. Setup style ──────────────────────────────────────────────────
    effective_journal = "showcase" if mode == "showcase" else journal
    preset = setup_style(journal=effective_journal)
    colors = get_palette(style)

    # ── 7. Resolve dimensions ────────────────────────────────────────────
    fig_width = resolve_width(width, preset)
    fig_height = height if height is not None else fig_width * 0.75  # 4:3 default

    # ── 8. Detect log scale ──────────────────────────────────────────────
    resolved_log = log_scale
    if resolved_log is None and y_cols:
        y_values = []
        for yc in y_cols:
            if yc in table.columns:
                y_values.extend(
                    [v for v in table.columns[yc] if isinstance(v, (int, float))]
                )
        if y_values and detect_log_scale(y_values, label=y_label):
            resolved_log = "y"

    # ── 9. Resolve sort order ────────────────────────────────────────────
    resolved_sort = sort
    if resolved_sort is None:
        resolved_sort = recommend_sort(table, x_col, resolved_kind)

    # ── 10. Decide bar orientation (auto-pick horizontal for long labels) ──
    effective_horizontal = horizontal
    bar_categories: list[str] | None = None
    if resolved_kind == "bar" and x_col and table.is_categorical(x_col):
        bar_categories = [str(v) for v in dict.fromkeys(table.columns[x_col])]
        should_h, reason = recommend_horizontal_bars(bar_categories)
        if effective_horizontal is None:
            effective_horizontal = should_h
            if should_h and reason:
                advisor_notes.append(
                    f"Auto-selected horizontal bars: {reason}. "
                    "Pass horizontal=False to override."
                )
        elif effective_horizontal is False and should_h and reason:
            advisor_notes.append(
                f"Bar chart uses vertical orientation but {reason}. "
                "Consider horizontal=True."
            )

    # ── 11. Create figure and plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if title:
        ax.set_title(title)

    _dispatch_plot(
        ax,
        resolved_kind,
        table,
        x_col,
        y_cols,
        group_col,
        error_col,
        error_type=error_type,
        colors=colors,
        x_label=x_label,
        y_label=y_label,
        resolved_sort=resolved_sort,
        horizontal=bool(effective_horizontal),
    )

    # ── 11b. Auto-rotate x-tick labels for vertical bars with long labels ──
    if (
        resolved_kind == "bar"
        and bar_categories is not None
        and not effective_horizontal
    ):
        rotation = recommend_label_rotation(bar_categories)
        if rotation:
            ha = "right" if rotation < 90 else "center"
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                label.set_horizontalalignment(ha)

    # ── 12. Apply log scale ──────────────────────────────────────────────
    if resolved_log in ("y", "both"):
        ax.set_yscale("log")
    if resolved_log in ("x", "both"):
        ax.set_xscale("log")

    # ── 13. Set axis labels (default to column names if not explicitly provided) ──
    # Heatmaps use tick labels instead of axis labels, so skip auto-labeling.
    # For horizontal bars the visual axes are flipped: values are on x, categories
    # on y, so the category column labels the y-axis and the value column labels x.
    if resolved_kind != "heatmap":
        category_label = x_label or x_col
        value_label = y_label or (y_cols[0] if len(y_cols) == 1 else None)
        if effective_horizontal and resolved_kind == "bar":
            effective_x_label, effective_y_label = value_label, category_label
        else:
            effective_x_label, effective_y_label = category_label, value_label
        if effective_x_label and not ax.get_xlabel():
            ax.set_xlabel(effective_x_label)
        if effective_y_label and not ax.get_ylabel():
            ax.set_ylabel(effective_y_label)

    # ── 14. Generate legend text ─────────────────────────────────────────
    aggregation_desc: str | None = None
    data_summary = _data_summary(table, x_col, y_cols, resolved_kind)

    # Get group info for legend
    n_groups: int | None = None
    group_names_list: list[str] | None = None
    if x_col and table.is_categorical(x_col):
        unique_cats = list(dict.fromkeys(str(v) for v in table.columns[x_col]))
        n_groups = len(unique_cats)
        group_names_list = unique_cats

    n_per_group: int | None = None
    if x_col and y_cols and table.is_categorical(x_col) and table.is_numeric(y_cols[0]):
        grouped = table.values_per_category(x_col, y_cols[0])
        counts = [len(v) for v in grouped.values()]
        if counts and all(c == counts[0] for c in counts):
            n_per_group = counts[0]

    legend_text = generate_legend(
        resolved_kind,
        title=title,
        x_label=x_label or (x_col if x_col else None),
        y_label=y_label or (y_cols[0] if y_cols else None),
        n_groups=n_groups,
        group_names=group_names_list,
        n_observations=table.n_rows,
        n_per_group=n_per_group if n_per_group and n_per_group > 1 else None,
        series_names=y_cols if len(y_cols) > 1 else None,
        error_type=error_type,
        log_scale=resolved_log,
        aggregation=aggregation_desc,
    )

    # ── 15. Save figure ──────────────────────────────────────────────────
    output_path = savefig(fig, output, dpi=dpi)

    return FigureResult(
        path=output_path,
        legend=legend_text,
        advisor_notes=advisor_notes,
        kind=resolved_kind,
        width_inches=fig_width,
        height_inches=fig_height,
        dpi=dpi,
        palette=style,
        log_scale=resolved_log,
        data_summary=data_summary,
        aggregation=aggregation_desc,
    )


def _dispatch_plot(
    ax: Any,
    kind: str,
    table: DataTable,
    x_col: str | None,
    y_cols: list[str],
    group_col: str | None,
    error_col: str | None,
    *,
    error_type: str | None,
    colors: list[str],
    x_label: str | None,
    y_label: str | None,
    resolved_sort: str | None,
    horizontal: bool = False,
) -> None:
    """Dispatch to the appropriate plot function based on kind."""
    sort_by_value = resolved_sort == "value"

    if kind == "histogram":
        vals = table.columns[y_cols[0]] if y_cols else list(table.columns.values())[0]
        numeric_vals = [float(v) for v in vals if isinstance(v, (int, float))]
        histogram_plot(
            ax,
            numeric_vals,
            color=colors[0],
            xlabel=x_label or (y_cols[0] if y_cols else None),
        )
        return

    if kind == "heatmap":
        numeric_cols = [c for c in table.column_names if table.is_numeric(c)]
        matrix = [[float(v) for v in table.columns[c]] for c in numeric_cols]
        # Transpose: rows = data rows, cols = numeric columns
        if matrix:
            transposed = list(map(list, zip(*matrix)))
            # For square matrices (correlation-like), use column names as both labels
            if len(transposed) == len(numeric_cols):
                row_labels = list(numeric_cols)
            elif x_col and table.is_categorical(x_col):
                row_labels = [str(v) for v in table.columns[x_col]]
            else:
                row_labels = [str(i) for i in range(len(transposed))]
            heatmap_plot(ax, transposed, row_labels, numeric_cols)
        return

    if not x_col or not y_cols:
        # Fallback: plot first available data
        if y_cols:
            vals = [
                float(v)
                for v in table.columns[y_cols[0]]
                if isinstance(v, (int, float))
            ]
            histogram_plot(ax, vals, xlabel=y_cols[0])
        return

    # Get error values if available
    error_values: list[float] | None = None
    if error_col and error_col in table.columns:
        error_values = [
            float(v) for v in table.columns[error_col] if isinstance(v, (int, float))
        ]

    if kind == "scatter":
        x_vals = [float(v) for v in table.columns[x_col] if isinstance(v, (int, float))]
        y_vals = [
            float(v) for v in table.columns[y_cols[0]] if isinstance(v, (int, float))
        ]
        scatter_plot(
            ax, x_vals, y_vals, color=colors[0], xlabel=x_label, ylabel=y_label
        )
        return

    if kind == "line":
        x_data = table.columns[x_col]
        y_series: dict[str, list[float]] = {}
        for yc in y_cols:
            y_series[yc] = [
                float(v) for v in table.columns[yc] if isinstance(v, (int, float))
            ]
        line_plot(ax, x_data, y_series, colors=colors, xlabel=x_label, ylabel=y_label)
        return

    # Categorical x-axis plots
    if table.is_categorical(x_col):
        y_col = y_cols[0]
        grouped = table.values_per_category(x_col, y_col)

        if kind == "bar":
            # Check if we need to auto-aggregate
            n_per = max(len(v) for v in grouped.values()) if grouped else 0
            if n_per == 1:
                categories = list(grouped.keys())
                values = [float(v[0]) for v in grouped.values()]
                bar_plot(
                    ax,
                    categories,
                    values,
                    colors=colors,
                    ylabel=y_label,
                    error_values=error_values,
                    sort_by_value=sort_by_value,
                    horizontal=horizontal,
                )
            else:
                # Auto-aggregate
                et = error_type or "sem"
                categories, means, errors, _ = auto_aggregate_bar(
                    grouped, error_type=et
                )
                bar_plot(
                    ax,
                    categories,
                    means,
                    colors=colors,
                    ylabel=y_label,
                    error_values=errors,
                    sort_by_value=sort_by_value,
                    horizontal=horizontal,
                )
            return

        if kind == "box":
            box_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return

        if kind == "violin":
            violin_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return

        if kind == "strip":
            strip_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return

        if kind == "swarm":
            swarm_plot(ax, grouped, colors=colors, ylabel=y_label)
            return

    # Numeric x fallback to bar
    categories = [str(v) for v in table.columns[x_col]]
    values = [float(v) for v in table.columns[y_cols[0]] if isinstance(v, (int, float))]
    bar_plot(
        ax,
        categories,
        values,
        colors=colors,
        ylabel=y_label,
        sort_by_value=sort_by_value,
        horizontal=horizontal,
    )


def _data_summary(
    table: DataTable, x_col: str | None, y_cols: list[str], kind: str
) -> str:
    """Generate a brief data summary string."""
    parts: list[str] = []
    if x_col and table.is_categorical(x_col):
        parts.append(f"{table.unique_count(x_col)} groups")
    if len(y_cols) > 1:
        parts.append(f"{len(y_cols)} series")
    parts.append(f"{table.n_rows} observations")
    return ", ".join(parts)
