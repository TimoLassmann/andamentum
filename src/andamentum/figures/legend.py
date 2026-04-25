"""Template-based figure legend generation.

Produces a descriptive caption from figure metadata — no LLM required.
The legend text follows scientific conventions for describing what a
figure shows, including data summary, error bar type, and scale information.
"""

from __future__ import annotations


# Chart type descriptions for legend text
_KIND_DESCRIPTIONS = {
    "bar": "Bar chart",
    "line": "Line chart",
    "scatter": "Scatter plot",
    "box": "Box plot",
    "violin": "Violin plot",
    "histogram": "Histogram",
    "heatmap": "Heatmap",
    "strip": "Strip plot",
    "swarm": "Beeswarm plot",
}

# Error type descriptions
_ERROR_DESCRIPTIONS = {
    "sem": "standard error of the mean",
    "sd": "standard deviation",
    "ci95": "95% confidence interval",
    "ci": "confidence interval",
    "bootstrap": "95% bootstrap confidence interval",
}


def generate_legend(
    kind: str,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    n_groups: int | None = None,
    group_names: list[str] | None = None,
    n_observations: int | None = None,
    n_per_group: int | None = None,
    series_names: list[str] | None = None,
    error_type: str | None = None,
    log_scale: str | None = None,
    aggregation: str | None = None,
    sort_description: str | None = None,
) -> str:
    """Generate a template-based figure legend.

    Args:
        kind: Plot type used.
        title: Figure title (included if provided).
        x_label: X-axis label.
        y_label: Y-axis label.
        n_groups: Number of groups/categories.
        group_names: Names of groups (included if ≤ 6).
        n_observations: Total number of data points.
        n_per_group: Observations per group (if uniform).
        series_names: Names of data series (for multi-series plots).
        error_type: Type of error bars shown.
        log_scale: Which axes use log scale.
        aggregation: Aggregation description (e.g., "mean ± SEM").
        sort_description: How data was sorted.

    Returns:
        Legend text string.
    """
    parts: list[str] = []

    # Opening: chart type + what it shows
    chart_desc = _KIND_DESCRIPTIONS.get(kind, kind.capitalize())

    if y_label and x_label:
        if kind in ("bar", "box", "violin", "strip", "swarm"):
            opening = f"{chart_desc} showing {y_label} across {x_label}"
        elif kind in ("line",):
            opening = f"{chart_desc} showing {y_label} as a function of {x_label}"
        elif kind in ("scatter",):
            opening = f"{chart_desc} of {y_label} versus {x_label}"
        elif kind == "histogram":
            opening = f"{chart_desc} of {y_label}"
        else:
            opening = f"{chart_desc} showing {y_label}"
    elif y_label:
        opening = f"{chart_desc} showing {y_label}"
    elif title:
        opening = f"{chart_desc}: {title}"
    else:
        opening = chart_desc

    parts.append(opening)

    # Data description
    data_parts: list[str] = []
    if n_groups and group_names and len(group_names) <= 6:
        data_parts.append(f"{n_groups} groups ({', '.join(group_names)})")
    elif n_groups:
        data_parts.append(f"{n_groups} groups")

    if n_per_group and n_per_group > 1:
        data_parts.append(f"n = {n_per_group} per group")
    elif n_observations:
        data_parts.append(f"n = {n_observations}")

    if series_names and len(series_names) > 1:
        if len(series_names) <= 6:
            data_parts.append(f"{len(series_names)} series ({', '.join(series_names)})")
        else:
            data_parts.append(f"{len(series_names)} series")

    if data_parts:
        parts.append(". ".join(data_parts))

    # Aggregation note
    if aggregation:
        parts.append(f"Values represent {aggregation}")

    # Box plot specifics
    if kind == "box":
        parts.append(
            "Box spans interquartile range; whiskers extend to 1.5× IQR; points beyond whiskers are outliers"
        )

    # Error bars
    if error_type:
        err_desc = _ERROR_DESCRIPTIONS.get(error_type, error_type)
        if kind == "line":
            parts.append(f"Shaded bands represent {err_desc}")
        else:
            parts.append(f"Error bars represent {err_desc}")

    # Log scale
    if log_scale:
        if log_scale == "both":
            parts.append("Both axes use logarithmic scale")
        elif log_scale == "x":
            parts.append("X-axis uses logarithmic scale")
        elif log_scale == "y":
            parts.append("Y-axis uses logarithmic scale")

    # Sort order
    if sort_description:
        parts.append(sort_description)

    # Join with periods
    text = ". ".join(parts)
    if not text.endswith("."):
        text += "."

    return text
