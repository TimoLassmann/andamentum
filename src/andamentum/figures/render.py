"""High-level figure() orchestrator.

Ties together all modules: data normalization → column detection → chart
selection → style setup → plotting → legend generation → save.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .advisor import (
    check_banned,
    recommend_horizontal_bars,
    recommend_kind,
    recommend_label_rotation,
    stacked_totals_note,
    validate_kind,
)
from .auto import detect_column_roles, detect_log_scale, recommend_sort
from .legend import generate_legend
from .palettes import assign_series_colors, get_palette, palette_capacity_note
from .plots import (
    auto_aggregate_bar,
    bar_plot,
    box_plot,
    grouped_bar,
    heatmap_plot,
    histogram_plot,
    line_plot,
    scatter_plot,
    stacked_bar,
    strip_plot,
    swarm_plot,
    violin_plot,
)
from .standards import JournalPreset, resolve_width
from .style import inline_svg, savefig, setup_style
from .types import DataTable, FigureResult, PlotKind

# Kinds that draw one bar per category and share the orientation logic.
_BAR_KINDS = {PlotKind.BAR.value, PlotKind.STACKED.value}


@dataclass
class _Series:
    """Multi-series data in plot order: one value per category per series."""

    categories: list[str]
    data: dict[str, list[float]]
    segment_label: str  # the series column name, or "series" for wide data
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.data.keys())


@dataclass
class _PlotInfo:
    """What the dispatch step learned while drawing, for the legend and result."""

    series_names: list[str] | None = None
    totals: list[float] | None = None
    aggregation: str | None = None
    segment_label: str | None = None


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
    group_order: list[str] | None = None,
    normalize: bool = False,
    series_colors: dict[str, str] | None = None,
    svg_inline: bool = False,
) -> FigureResult:
    """Generate a publication-quality figure from data.

    This is the main entry point. Accepts data in multiple formats,
    auto-detects the best chart type, applies journal styling, and
    saves the figure with an auto-generated legend.

    Args:
        data: Data as columnar dict, list of row dicts, or CSV string.
        kind: Plot type ("bar", "line", "scatter", "box", "violin",
              "histogram", "heatmap", "strip", "swarm", "stacked", "auto").
        x: X-axis column name (auto-detected if None).
        y: Y-axis column name(s) (auto-detected if None).
        group: Series column. For ``bar`` this draws one bar per
            (category, series) side by side; for ``stacked`` one segment
            per series. Long-form data has one row per (category, series).
        error: Error bar column name.
        error_type: Error bar type ("sem", "sd", "ci95").
        title: Figure title.
        x_label: X-axis label (defaults to column name).
        y_label: Y-axis label (defaults to column name).
        style: Color palette name (npg, nejm, lancet, jama, aaas, d3,
            okabe_ito, andamentum).
        journal: Journal format preset (default, nature, science, cell,
            plos, report).
        mode: "publication" or "showcase".
        width: Figure width ("single", "1.5", "double", or float in inches).
        height: Figure height in inches (auto-calculated if None).
        dpi: Resolution.
        log_scale: "x", "y", "both", or None (auto-detect).
        sort: "value", "name", or None (auto-detect for bar/box). For
            ``stacked``, "value" sorts bars by their total.
        horizontal: For bar and stacked charts. True forces horizontal
            bars, False forces vertical, None (default) auto-decides based
            on label count and length. Many bars with long labels become
            illegible mush in vertical orientation; the auto-decider
            picks horizontal for those cases and emits an advisor note.
        output: Output file path.
        group_order: Series order for ``stacked`` and grouped ``bar``.
            Must name every series exactly once. Default is the order of
            first appearance in the data.
        normalize: For ``stacked``: draw segments as fractions of the
            category total, value axis 0 to 1. Totals are still labelled
            as counts at the bar ends.
        series_colors: ``{series_name: tone}`` mapping a series (or, for a
            plain bar chart, a category) to a design-system tone by name:
            slate, ochre, olive, rust, grey, grey-light, or the aliases
            info, warn, success, danger. Unmapped names take the palette
            in order. Hex values are refused.
        svg_inline: Write the bare ``<svg>`` element (no XML declaration,
            DOCTYPE, metadata, width or height) instead of the full file,
            so a report can drop it inside a ``<div>``. Output must be
            ``.svg``.

    Returns:
        FigureResult with path, legend, advisor notes, and metadata.

    Raises:
        ValueError: If kind is banned (pie, 3D, etc.), unknown, or the
            data does not fit it; if ``stacked`` data has a negative value
            or two rows for one (category, series); if ``group_order`` or
            ``series_colors`` names something not in the data; or if the
            ``andamentum`` palette is asked for more than six colours.

    Caption guidance:
        ``figure()`` plots only — it neither generates nor modifies
        data points. When ``FigureResult.advisor_notes`` is non-empty,
        those notes describe auto-applied visual changes (bar
        orientation flip, log-scale auto-pick, sort order, x-tick
        rotation, per-category averaging) that the renderer chose for
        you. **Mirror them in your figure caption** so reviewers can see
        what was decided automatically. Example: "Bars plotted
        horizontally because of label length; y-axis log-scaled because
        the data spans three orders of magnitude."
    """
    # ── 1. Check the kind ────────────────────────────────────────────────
    check_banned(kind)
    known = {k.value for k in PlotKind}
    if kind not in known:
        raise ValueError(
            f"Unknown kind '{kind}'. Available: {', '.join(sorted(known))}"
        )
    if svg_inline and Path(output).suffix.lower() != ".svg":
        raise ValueError(f"svg_inline needs a .svg output path, got '{output}'")

    # ── 2. Normalize data ────────────────────────────────────────────────
    table = DataTable.normalize(data)

    # ── 3. Detect column roles ───────────────────────────────────────────
    x_col, y_cols, group_col, error_col = detect_column_roles(table, x, y, group, error)

    # ── 4. Resolve chart type ────────────────────────────────────────────
    resolved_kind = kind
    if kind == PlotKind.AUTO.value:
        resolved_kind = recommend_kind(table, x_col, y_cols, group_col)

    # ── 5. Validate and get advisor warnings ─────────────────────────────
    advisor_notes = validate_kind(resolved_kind, table, x_col, y_cols, group=group_col)

    # ── 6. Series table for multi-series kinds ───────────────────────────
    series: _Series | None = None
    if resolved_kind == PlotKind.STACKED.value:
        series = _build_series(table, x_col, y_cols, group_col, group_order)
        if series is None:
            raise ValueError(
                "kind='stacked' needs a group column (long form: x, y, group) "
                "or two or more y columns (wide form)."
            )
    elif resolved_kind == PlotKind.BAR.value and group_col is not None:
        series = _build_series(table, x_col, y_cols, group_col, group_order)
    elif group_order is not None:
        raise ValueError(
            "group_order applies to kind='stacked' or a bar chart with a group column"
        )
    if series is not None:
        advisor_notes.extend(series.notes)

    # ── 7. Setup style and colours ──────────────────────────────────────
    effective_journal = "showcase" if mode == "showcase" else journal
    preset = setup_style(journal=effective_journal)
    colored_names = _colored_names(resolved_kind, table, x_col, y_cols, series)
    colors = _resolve_colors(style, colored_names, series_colors, advisor_notes)

    # ── 8. Resolve dimensions ────────────────────────────────────────────
    fig_width = resolve_width(width, preset)
    fig_height = height if height is not None else fig_width * 0.75  # 4:3 default

    # ── 9. Detect log scale ──────────────────────────────────────────────
    resolved_log = log_scale
    if resolved_log is None and y_cols and resolved_kind != PlotKind.STACKED.value:
        y_values = []
        for yc in y_cols:
            if yc in table.columns:
                y_values.extend(
                    [v for v in table.columns[yc] if isinstance(v, (int, float))]
                )
        if y_values and detect_log_scale(y_values, label=y_label):
            resolved_log = "y"

    # ── 10. Resolve sort order ───────────────────────────────────────────
    resolved_sort = sort
    if resolved_sort is None:
        resolved_sort = recommend_sort(table, x_col, resolved_kind)
    if series is not None and resolved_sort == "value":
        _sort_series_by_total(series)

    # ── 11. Decide bar orientation (auto-pick horizontal for long labels) ──
    effective_horizontal = horizontal
    bar_categories: list[str] | None = None
    if resolved_kind in _BAR_KINDS and x_col and table.is_categorical(x_col):
        if series is not None:
            bar_categories = list(series.categories)
        else:
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

    # ── 12. Create figure and plot ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if title:
        ax.set_title(title)

    info = _dispatch_plot(
        ax,
        resolved_kind,
        table,
        x_col,
        y_cols,
        error_col,
        series=series,
        preset=preset,
        error_type=error_type,
        colors=colors,
        x_label=x_label,
        y_label=y_label,
        resolved_sort=resolved_sort,
        horizontal=bool(effective_horizontal),
        normalize=normalize,
    )
    if info.aggregation:
        advisor_notes.append(
            f"Rows per category were averaged ({info.aggregation}). "
            "Pass group= to draw one bar per series instead."
        )
    if resolved_kind == PlotKind.STACKED.value and info.totals and not normalize:
        note = stacked_totals_note(info.totals)
        if note:
            advisor_notes.append(note)

    # ── 12b. Auto-rotate x-tick labels for vertical bars with long labels ──
    if (
        resolved_kind in _BAR_KINDS
        and bar_categories is not None
        and not effective_horizontal
    ):
        rotation = recommend_label_rotation(bar_categories)
        if rotation:
            ha = "right" if rotation < 90 else "center"
            for label in ax.get_xticklabels():
                label.set_rotation(rotation)
                label.set_horizontalalignment(ha)

    # ── 13. Apply log scale ──────────────────────────────────────────────
    if resolved_log in ("y", "both"):
        ax.set_yscale("log")
    if resolved_log in ("x", "both"):
        ax.set_xscale("log")

    # ── 14. Set axis labels (default to column names if not explicitly provided) ──
    # Heatmaps use tick labels instead of axis labels, so skip auto-labeling.
    # For horizontal bars the visual axes are flipped: values are on x, categories
    # on y, so the category column labels the y-axis and the value column labels x.
    if resolved_kind != PlotKind.HEATMAP.value:
        category_label = x_label or x_col
        if normalize and resolved_kind == PlotKind.STACKED.value:
            value_label = y_label or (f"Fraction of {y_cols[0]}" if y_cols else None)
        else:
            value_label = y_label or (y_cols[0] if len(y_cols) == 1 else None)
        if effective_horizontal and resolved_kind in _BAR_KINDS:
            effective_x_label, effective_y_label = value_label, category_label
        else:
            effective_x_label, effective_y_label = category_label, value_label
        if effective_x_label and not ax.get_xlabel():
            ax.set_xlabel(effective_x_label)
        if effective_y_label and not ax.get_ylabel():
            ax.set_ylabel(effective_y_label)

    # ── 15. Generate legend text ─────────────────────────────────────────
    data_summary = _data_summary(table, x_col, y_cols, resolved_kind, series)

    # Get group info for legend
    n_groups: int | None = None
    group_names_list: list[str] | None = None
    if x_col and table.is_categorical(x_col):
        unique_cats = list(dict.fromkeys(str(v) for v in table.columns[x_col]))
        n_groups = len(unique_cats)
        group_names_list = unique_cats

    n_per_group: int | None = None
    if (
        series is None
        and x_col
        and y_cols
        and table.is_categorical(x_col)
        and table.is_numeric(y_cols[0])
    ):
        grouped = table.values_per_category(x_col, y_cols[0])
        counts = [len(v) for v in grouped.values()]
        if counts and all(c == counts[0] for c in counts):
            n_per_group = counts[0]

    legend_series = info.series_names or (y_cols if len(y_cols) > 1 else None)
    legend_text = generate_legend(
        resolved_kind,
        title=title,
        x_label=x_label or (x_col if x_col else None),
        y_label=y_label or (y_cols[0] if y_cols else None),
        n_groups=n_groups,
        group_names=group_names_list,
        n_observations=table.n_rows,
        n_per_group=n_per_group if n_per_group and n_per_group > 1 else None,
        series_names=legend_series,
        error_type=error_type,
        log_scale=resolved_log,
        aggregation=info.aggregation,
        segment_label=info.segment_label,
        category_totals=info.totals,
    )

    # ── 16. Save figure ──────────────────────────────────────────────────
    if preset.fit_layout:
        # Pack tick labels, axis labels and an above-axes legend inside the
        # figure box, so the crop is never wider than the preset width.
        fig.tight_layout()
    output_path = savefig(fig, output, dpi=dpi, pad=preset.pad_inches)
    if svg_inline:
        trimmed = inline_svg(output_path)
        Path(output_path).write_text(trimmed, encoding="utf-8")

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
        aggregation=info.aggregation,
        series_names=info.series_names,
    )


def _build_series(
    table: DataTable,
    x_col: str | None,
    y_cols: list[str],
    group_col: str | None,
    group_order: list[str] | None,
) -> _Series | None:
    """Turn long-form (x, y, group) or wide-form (x, y1..yn) data into series.

    Long form is one row per (category, series). Two rows for one pair is
    refused rather than summed: aggregation is a decision the caller must
    make visibly. A missing pair reads as 0 and is reported in a note.

    Returns None when the data has neither shape.
    """
    if not x_col:
        return None

    categories = [str(v) for v in dict.fromkeys(table.columns[x_col])]
    data: dict[str, list[float]] = {}
    notes: list[str] = []

    if group_col is not None and y_cols:
        y_col = y_cols[0]
        segment_label = group_col
        cells: dict[tuple[str, str], float] = {}
        series_order: list[str] = []
        for cat, ser, val in zip(
            table.columns[x_col], table.columns[group_col], table.columns[y_col]
        ):
            key = (str(cat), str(ser))
            if key in cells:
                raise ValueError(
                    f"Two rows for ({key[0]}, {key[1]}) in columns "
                    f"'{x_col}' and '{group_col}'. One row per (category, series) "
                    "is required; aggregate first if that is what you want."
                )
            if str(ser) not in series_order:
                series_order.append(str(ser))
            cells[key] = float(val) if isinstance(val, (int, float)) else 0.0
        missing: list[str] = []
        for ser in series_order:
            row: list[float] = []
            for cat in categories:
                if (cat, ser) in cells:
                    row.append(cells[(cat, ser)])
                else:
                    row.append(0.0)
                    missing.append(f"({cat}, {ser})")
            data[ser] = row
        if missing:
            notes.append(
                f"No row for {', '.join(missing)}; drawn as 0. "
                "Add the rows if the value is not zero."
            )
    elif group_col is None and len(y_cols) >= 2:
        segment_label = "series"
        if table.n_rows != len(categories):
            raise ValueError(
                f"Wide-form data needs one row per '{x_col}' value; "
                f"got {table.n_rows} rows for {len(categories)} categories."
            )
        for yc in y_cols:
            data[yc] = [
                float(v) if isinstance(v, (int, float)) else 0.0
                for v in table.columns[yc]
            ]
    else:
        return None

    if group_order is not None:
        present = list(data.keys())
        if sorted(group_order) != sorted(present) or len(group_order) != len(present):
            raise ValueError(
                f"group_order must name every series exactly once. "
                f"Given {group_order}; series present: {present}"
            )
        data = {name: data[name] for name in group_order}

    return _Series(
        categories=categories, data=data, segment_label=segment_label, notes=notes
    )


def _sort_series_by_total(series: _Series) -> None:
    totals = [
        sum(series.data[name][i] for name in series.names)
        for i in range(len(series.categories))
    ]
    order = sorted(range(len(totals)), key=lambda i: totals[i], reverse=True)
    series.categories = [series.categories[i] for i in order]
    series.data = {name: [vals[i] for i in order] for name, vals in series.data.items()}


def _colored_names(
    kind: str,
    table: DataTable,
    x_col: str | None,
    y_cols: list[str],
    series: _Series | None,
) -> list[str] | None:
    """The names that carry the chart's colours: series if any, else categories."""
    if series is not None:
        return series.names
    if kind == PlotKind.LINE.value and len(y_cols) > 1:
        return list(y_cols)
    if (
        kind in ("bar", "box", "violin", "strip", "swarm")
        and x_col
        and table.is_categorical(x_col)
    ):
        return [str(v) for v in dict.fromkeys(table.columns[x_col])]
    return None


def _resolve_colors(
    style: str,
    names: list[str] | None,
    series_colors: dict[str, str] | None,
    advisor_notes: list[str],
) -> list[str]:
    """One colour per name, honouring series_colors; palette capacity checked."""
    palette = get_palette(style)
    if names is None:
        if series_colors:
            raise ValueError(
                "series_colors applies to charts with named series or categories"
            )
        return palette
    # Raises for the andamentum palette past six colours; cycles otherwise.
    get_palette(style, len(names))
    note = palette_capacity_note(style, len(names))
    if note:
        advisor_notes.append(note)
    return assign_series_colors(names, palette, series_colors)


def _dispatch_plot(
    ax: Any,
    kind: str,
    table: DataTable,
    x_col: str | None,
    y_cols: list[str],
    error_col: str | None,
    *,
    series: _Series | None,
    preset: JournalPreset,
    error_type: str | None,
    colors: list[str],
    x_label: str | None,
    y_label: str | None,
    resolved_sort: str | None,
    horizontal: bool = False,
    normalize: bool = False,
) -> _PlotInfo:
    """Dispatch to the appropriate plot function based on kind."""
    sort_by_value = resolved_sort == "value"
    info = _PlotInfo()

    if kind == PlotKind.STACKED.value:
        if series is None:
            raise ValueError("kind='stacked' needs series data")
        totals = stacked_bar(
            ax,
            series.data,
            series.categories,
            colors=colors,
            ylabel=y_label,
            horizontal=horizontal,
            normalize=normalize,
            ink=preset.ink,
        )
        info.series_names = series.names
        info.totals = totals
        info.segment_label = series.segment_label
        return info

    if kind == PlotKind.BAR.value and series is not None:
        grouped_bar(
            ax,
            series.data,
            series.categories,
            colors=colors,
            ylabel=y_label,
            horizontal=horizontal,
        )
        info.series_names = series.names
        return info

    if kind == "histogram":
        vals = table.columns[y_cols[0]] if y_cols else list(table.columns.values())[0]
        numeric_vals = [float(v) for v in vals if isinstance(v, (int, float))]
        histogram_plot(
            ax,
            numeric_vals,
            color=colors[0],
            xlabel=x_label or (y_cols[0] if y_cols else None),
        )
        return info

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
        return info

    if not x_col or not y_cols:
        # Fallback: plot first available data
        if y_cols:
            vals = [
                float(v)
                for v in table.columns[y_cols[0]]
                if isinstance(v, (int, float))
            ]
            histogram_plot(ax, vals, xlabel=y_cols[0])
        return info

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
        return info

    if kind == "line":
        x_data = table.columns[x_col]
        y_series: dict[str, list[float]] = {}
        for yc in y_cols:
            y_series[yc] = [
                float(v) for v in table.columns[yc] if isinstance(v, (int, float))
            ]
        line_plot(ax, x_data, y_series, colors=colors, xlabel=x_label, ylabel=y_label)
        if len(y_cols) > 1:
            info.series_names = list(y_cols)
        return info

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
                    error_color=preset.muted_ink,
                )
            else:
                # Auto-aggregate
                et = error_type or "sem"
                categories, means, errors, desc = auto_aggregate_bar(
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
                    error_color=preset.muted_ink,
                )
                info.aggregation = desc
            return info

        if kind == "box":
            box_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return info

        if kind == "violin":
            violin_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return info

        if kind == "strip":
            strip_plot(
                ax, grouped, colors=colors, ylabel=y_label, sort_by_median=sort_by_value
            )
            return info

        if kind == "swarm":
            swarm_plot(ax, grouped, colors=colors, ylabel=y_label)
            return info

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
        error_color=preset.muted_ink,
    )
    return info


def _data_summary(
    table: DataTable,
    x_col: str | None,
    y_cols: list[str],
    kind: str,
    series: _Series | None,
) -> str:
    """Generate a brief data summary string."""
    parts: list[str] = []
    if x_col and table.is_categorical(x_col):
        parts.append(f"{table.unique_count(x_col)} groups")
    if series is not None:
        parts.append(f"{len(series.names)} series")
    elif len(y_cols) > 1:
        parts.append(f"{len(y_cols)} series")
    parts.append(f"{table.n_rows} observations")
    return ", ".join(parts)
