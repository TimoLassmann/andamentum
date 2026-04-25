"""Plot type implementations.

Each function takes a matplotlib Axes and draws a specific plot type.
These are the building blocks used by both the high-level figure() API
and directly by users building custom multi-panel figures.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes

from .palettes import get_palette
from .stats import compute_mean_error


def bar_plot(
    ax: Axes,
    categories: list[str],
    values: list[float],
    *,
    colors: list[str] | None = None,
    error_values: list[float] | None = None,
    ylabel: str | None = None,
    horizontal: bool = False,
    alpha: float = 0.8,
    sort_by_value: bool = False,
) -> None:
    """Simple bar chart for pre-aggregated data.

    Args:
        ax: Matplotlib axes.
        categories: Category labels.
        values: Bar heights.
        colors: Color per bar (cycles palette if None).
        error_values: Error bar values.
        ylabel: Y-axis label.
        horizontal: If True, draw horizontal bars.
        alpha: Bar transparency.
        sort_by_value: Sort bars by value (descending).
    """
    if sort_by_value:
        paired = sorted(zip(values, categories, range(len(values))), reverse=True)
        values = [p[0] for p in paired]
        categories = [p[1] for p in paired]
        if error_values is not None:
            idx = [p[2] for p in paired]
            error_values = [error_values[i] for i in idx]

    resolved_colors = (
        colors if colors is not None else get_palette("npg", len(categories))
    )

    x = np.arange(len(categories))
    err_kw = {"capsize": 3, "capthick": 0.8, "elinewidth": 0.8, "color": "black"}

    if horizontal:
        ax.barh(
            x,
            values,
            color=resolved_colors[: len(categories)],
            alpha=alpha,
            xerr=error_values,
            error_kw=err_kw,
        )
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        if ylabel:
            ax.set_xlabel(ylabel)
    else:
        ax.bar(
            x,
            values,
            color=resolved_colors[: len(categories)],
            alpha=alpha,
            yerr=error_values,
            error_kw=err_kw,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        if ylabel:
            ax.set_ylabel(ylabel)


def grouped_bar(
    ax: Axes,
    data: dict[str, list[float]],
    categories: list[str],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    alpha: float = 0.8,
) -> None:
    """Grouped bar chart with multiple series per category.

    Args:
        ax: Matplotlib axes.
        data: {series_name: [values_per_category]}.
        categories: Category labels.
        colors: Color per series.
        ylabel: Y-axis label.
        alpha: Bar transparency.
    """
    series_names = list(data.keys())
    n_series = len(series_names)
    n_cats = len(categories)

    resolved_colors = colors if colors is not None else get_palette("npg", n_series)

    x = np.arange(n_cats)
    width = 0.8 / n_series

    for i, name in enumerate(series_names):
        offset = (i - n_series / 2 + 0.5) * width
        ax.bar(
            x + offset,
            data[name],
            width,
            label=name,
            color=resolved_colors[i % len(resolved_colors)],
            alpha=alpha,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()


def line_plot(
    ax: Axes,
    x: list[Any],
    y_series: dict[str, list[float]],
    *,
    colors: list[str] | None = None,
    ci_series: dict[str, tuple[list[float], list[float]]] | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    marker: str = "o",
    ci_alpha: float = 0.1,
) -> None:
    """Line chart with optional confidence bands.

    Args:
        ax: Matplotlib axes.
        x: X-axis values.
        y_series: {series_name: [y_values]}.
        colors: Color per series.
        ci_series: {series_name: (ci_lower, ci_upper)} for confidence bands.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        marker: Marker style.
        ci_alpha: Alpha for confidence bands.
    """
    series_names = list(y_series.keys())
    resolved_colors = (
        colors if colors is not None else get_palette("npg", len(series_names))
    )

    x_arr = np.array(x)
    for i, name in enumerate(series_names):
        color = resolved_colors[i % len(resolved_colors)]
        ax.plot(x_arr, y_series[name], marker=marker, color=color, label=name)
        if ci_series and name in ci_series:
            lo, hi = ci_series[name]
            ax.fill_between(x_arr, lo, hi, color=color, alpha=ci_alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(series_names) > 1:
        ax.legend()


def line_with_ci(
    ax: Axes,
    x: list[Any],
    y: list[float],
    ci_lo: list[float],
    ci_hi: list[float],
    *,
    color: str = "#1F77B4",
    label: str | None = None,
    marker: str = "o",
    ci_alpha: float = 0.1,
) -> None:
    """Single line with confidence band — convenient for loop-based plotting.

    Args:
        ax: Matplotlib axes.
        x: X-axis values.
        y: Y-axis values (e.g., means).
        ci_lo: Lower bound of confidence interval.
        ci_hi: Upper bound of confidence interval.
        color: Line and band color.
        label: Legend label.
        marker: Marker style.
        ci_alpha: Band transparency.
    """
    x_arr = np.array(x)
    ax.plot(x_arr, y, marker=marker, color=color, label=label)
    ax.fill_between(x_arr, ci_lo, ci_hi, color=color, alpha=ci_alpha)


def scatter_plot(
    ax: Axes,
    x: list[float],
    y: list[float],
    *,
    color: str | None = None,
    alpha: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Scatter plot with auto-alpha for large datasets.

    Args:
        ax: Matplotlib axes.
        x: X values.
        y: Y values.
        color: Point color.
        alpha: Point transparency (auto-calculated if None).
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    n = len(x)
    if alpha is None:
        if n < 100:
            alpha = 0.8
        elif n < 1000:
            alpha = 0.5
        elif n < 10000:
            alpha = 0.2
        else:
            alpha = 0.05

    if color is None:
        color = get_palette("npg", 1)[0]

    ax.scatter(x, y, color=color, alpha=alpha, s=20, edgecolors="none")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def box_plot(
    ax: Axes,
    data: dict[str, list[float]],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    show_points: bool = False,
    sort_by_median: bool = False,
    alpha: float = 0.7,
) -> None:
    """Box plot for distribution comparison.

    Follows Nature Methods conventions: solid fill, no outline, black median,
    whiskers at 1.5×IQR, outliers as small dots.

    Args:
        ax: Matplotlib axes.
        data: {group_name: [values]}.
        colors: Color per group.
        ylabel: Y-axis label.
        show_points: Overlay individual data points (strip).
        sort_by_median: Sort groups by median value.
        alpha: Box fill transparency.
    """
    names = list(data.keys())
    values = [data[n] for n in names]

    if sort_by_median:
        medians = [float(np.median(v)) for v in values]
        paired = sorted(zip(medians, names, values), reverse=True)
        names = [p[1] for p in paired]
        values = [p[2] for p in paired]

    resolved_colors = colors if colors is not None else get_palette("npg", len(names))

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], resolved_colors[: len(names)]):
        patch.set_facecolor(c)
        patch.set_alpha(alpha)
        patch.set_linewidth(0)  # solid fill, no outline (Nature Methods)

    if show_points:
        for i, vals in enumerate(values):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                np.full(len(vals), i + 1) + jitter,
                vals,
                color=resolved_colors[i % len(resolved_colors)],
                alpha=0.5,
                s=10,
                edgecolors="none",
                zorder=3,
            )

    ax.set_xticklabels(names)
    if ylabel:
        ax.set_ylabel(ylabel)


def grouped_boxplot(
    ax: Axes,
    data: dict[str, dict[str, list[float]]],
    categories: list[str],
    groups: list[str],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    alpha: float = 0.7,
) -> None:
    """Grouped box plots — multiple groups within each category.

    Args:
        ax: Matplotlib axes.
        data: {group_name: {category: [values]}}.
        categories: Category labels (x-axis).
        groups: Group names (colored series).
        colors: Color per group.
        ylabel: Y-axis label.
        alpha: Box fill transparency.
    """
    n_groups = len(groups)
    n_cats = len(categories)

    resolved_colors = colors if colors is not None else get_palette("npg", n_groups)

    width = 0.8 / n_groups
    positions: list[float] = []
    box_data: list[list[float]] = []
    box_colors: list[str] = []

    for i, cat in enumerate(categories):
        for j, grp in enumerate(groups):
            pos = i + (j - n_groups / 2 + 0.5) * width
            positions.append(pos)
            box_data.append(data.get(grp, {}).get(cat, []))
            box_colors.append(resolved_colors[j % len(resolved_colors)])

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=width * 0.8,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
    )
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(alpha)
        patch.set_linewidth(0)  # solid fill, no outline (Nature Methods)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories)
    if ylabel:
        ax.set_ylabel(ylabel)


def violin_plot(
    ax: Axes,
    data: dict[str, list[float]],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    sort_by_median: bool = False,
    alpha: float = 0.7,
) -> None:
    """Violin plot for rich distribution comparison.

    Args:
        ax: Matplotlib axes.
        data: {group_name: [values]}.
        colors: Color per group.
        ylabel: Y-axis label.
        sort_by_median: Sort groups by median.
        alpha: Fill transparency.
    """
    names = list(data.keys())
    values = [data[n] for n in names]

    if sort_by_median:
        medians = [float(np.median(v)) for v in values]
        paired = sorted(zip(medians, names, values), reverse=True)
        names = [p[1] for p in paired]
        values = [p[2] for p in paired]

    resolved_colors = colors if colors is not None else get_palette("npg", len(names))

    parts = ax.violinplot(values, showmeans=False, showmedians=True)

    bodies = cast(list[Any], parts["bodies"])
    for i, pc in enumerate(bodies):
        pc.set_facecolor(resolved_colors[i % len(resolved_colors)])
        pc.set_alpha(alpha)

    # Style median lines
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    if ylabel:
        ax.set_ylabel(ylabel)


def histogram_plot(
    ax: Axes,
    values: list[float],
    *,
    bins: int | str = "auto",
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Count",
    alpha: float = 0.8,
) -> None:
    """Histogram for single variable distribution.

    Args:
        ax: Matplotlib axes.
        values: Numeric values.
        bins: Number of bins or "auto".
        color: Bar color.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        alpha: Fill transparency.
    """
    if color is None:
        color = get_palette("npg", 1)[0]

    ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor="none")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def heatmap_plot(
    ax: Axes,
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap: str = "RdYlGn",
    annotate: bool = True,
    fmt: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Annotated heatmap.

    Args:
        ax: Matplotlib axes.
        matrix: 2D values.
        row_labels: Row names.
        col_labels: Column names.
        cmap: Colormap name.
        annotate: Show text values in cells.
        fmt: Number format for annotations.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
    """
    arr = np.array(matrix, dtype=float)

    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if np.isnan(val):
                    continue
                # Choose text color based on background brightness
                norm_val = (val - (vmin or arr.min())) / (
                    (vmax or arr.max()) - (vmin or arr.min()) + 1e-9
                )
                text_color = "white" if norm_val > 0.7 or norm_val < 0.3 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7,
                )

    ax.figure.colorbar(im, ax=ax, shrink=0.8)  # type: ignore[union-attr]


def annotated_heatmap(
    ax: Axes,
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    **kwargs: Any,
) -> None:
    """Alias for heatmap_plot with annotation defaults."""
    heatmap_plot(ax, matrix, row_labels, col_labels, annotate=True, **kwargs)


def strip_plot(
    ax: Axes,
    data: dict[str, list[float]],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    jitter: float = 0.15,
    alpha: float = 0.7,
    size: float = 20,
    sort_by_median: bool = False,
) -> None:
    """Jittered strip plot showing individual data points.

    Best for small sample sizes (n < 20 per group).

    Args:
        ax: Matplotlib axes.
        data: {group_name: [values]}.
        colors: Color per group.
        ylabel: Y-axis label.
        jitter: Horizontal jitter amount.
        alpha: Point transparency.
        size: Point size.
        sort_by_median: Sort groups by median.
    """
    names = list(data.keys())
    values = [data[n] for n in names]

    if sort_by_median:
        medians = [float(np.median(v)) if v else 0 for v in values]
        paired = sorted(zip(medians, names, values), reverse=True)
        names = [p[1] for p in paired]
        values = [p[2] for p in paired]

    resolved_colors = colors if colors is not None else get_palette("npg", len(names))

    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(values, resolved_colors[: len(names)])):
        j = rng.uniform(-jitter, jitter, len(vals))
        ax.scatter(
            np.full(len(vals), i) + j,
            vals,
            color=c,
            alpha=alpha,
            s=size,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    if ylabel:
        ax.set_ylabel(ylabel)


def swarm_plot(
    ax: Axes,
    data: dict[str, list[float]],
    *,
    colors: list[str] | None = None,
    ylabel: str | None = None,
    size: float = 20,
    alpha: float = 0.8,
) -> None:
    """Beeswarm plot — non-overlapping point placement.

    Simple implementation using sorted offset placement.
    For large n (>100), falls back to jittered strip.

    Args:
        ax: Matplotlib axes.
        data: {group_name: [values]}.
        colors: Color per group.
        ylabel: Y-axis label.
        size: Point size.
        alpha: Point transparency.
    """
    names = list(data.keys())
    resolved_colors = colors if colors is not None else get_palette("npg", len(names))

    for i, name in enumerate(names):
        vals = sorted(data[name])
        n = len(vals)
        if n == 0:
            continue

        # Simple beeswarm: offset points that would overlap
        if n > 100:
            # Too many points — fall back to jittered strip
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.2, 0.2, n)
            ax.scatter(
                np.full(n, i) + jitter,
                vals,
                color=resolved_colors[i % len(resolved_colors)],
                alpha=alpha * 0.5,
                s=size * 0.5,
                edgecolors="none",
                zorder=3,
            )
        else:
            x_positions = _beeswarm_positions(vals, i, point_size=size)
            ax.scatter(
                x_positions,
                vals,
                color=resolved_colors[i % len(resolved_colors)],
                alpha=alpha,
                s=size,
                edgecolors="none",
                zorder=3,
            )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    if ylabel:
        ax.set_ylabel(ylabel)


def _beeswarm_positions(
    values: list[float], center: float, point_size: float = 20
) -> list[float]:
    """Compute non-overlapping x positions for beeswarm layout."""
    if not values:
        return []

    n = len(values)
    positions = [center] * n

    # Simple offset algorithm: for each point, check overlap with previous
    # and offset horizontally if needed
    radius = (point_size**0.5) * 0.01  # approximate point radius in data coords
    for i in range(1, n):
        for j in range(i):
            dy = abs(values[i] - values[j])
            if dy < radius * 2:
                # Overlap — offset horizontally
                direction = 1 if (i % 2 == 0) else -1
                offset = radius * (1 + (i - j) * 0.5) * direction
                positions[i] = center + offset

    return positions


def auto_aggregate_bar(
    data: dict[str, list[float]],
    error_type: str = "sem",
) -> tuple[list[str], list[float], list[float], str]:
    """Auto-aggregate grouped data for bar charts.

    When data has multiple values per category, compute means and error bars.

    Args:
        data: {category: [values]}.
        error_type: Error bar type ("sem", "sd", "ci95").

    Returns:
        Tuple of (categories, means, errors, description).
    """
    categories = list(data.keys())
    means: list[float] = []
    errors: list[float] = []

    for cat in categories:
        mean, err = compute_mean_error(data[cat], error_type=error_type)
        means.append(mean)
        errors.append(err)

    type_labels = {"sem": "SEM", "sd": "SD", "ci95": "95% CI"}
    desc = f"mean ± {type_labels.get(error_type, error_type)}"

    return categories, means, errors, desc
