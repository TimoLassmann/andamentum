"""Auto-detection logic for scales, column roles, and sort order.

Provides heuristics for:
- Log scale detection (range, skewness, label hints)
- Column role inference (x, y, group, error)
- Sort order recommendation
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .types import DataTable


def detect_log_scale(
    values: Sequence[float | int],
    label: str | None = None,
) -> bool:
    """Detect whether logarithmic scale is appropriate for the data.

    Rules:
    1. All values must be positive (log requires positive)
    2. Range check: max/min > 100 suggests log scale
    3. Skewness > 2 suggests log scale
    4. Label contains known log-scale keywords

    Args:
        values: Numeric values to assess.
        label: Axis label (used for keyword hints).

    Returns:
        True if log scale is recommended.
    """
    # Filter to finite positive values
    positive = [
        v for v in values if isinstance(v, (int, float)) and v > 0 and math.isfinite(v)
    ]

    if len(positive) < 3:
        return False

    # Guard: any non-positive values means no log scale
    all_numeric = [
        v for v in values if isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if any(v <= 0 for v in all_numeric):
        return False

    # Range check: spans > 2 orders of magnitude
    vmin, vmax = min(positive), max(positive)
    if vmin > 0 and vmax / vmin > 100:
        return True

    # Skewness check (simplified: using mean/median ratio as proxy)
    sorted_vals = sorted(positive)
    median = sorted_vals[len(sorted_vals) // 2]
    mean = sum(positive) / len(positive)
    if median > 0 and mean / median > 2.0:
        return True

    # Label hint check — only triggers if data also spans at least 1.5 orders of magnitude.
    # Without the range requirement, labels like "fold change" trigger log scale on data
    # that spans 1–5, producing ugly "2×10⁰" tick labels for no benefit.
    if label and vmin > 0 and vmax / vmin > 30:
        log_hints = {
            "concentration",
            "dose",
            "fold change",
            "fold_change",
            "foldchange",
            "ic50",
            "ec50",
            "ki",
            "kd",
            "ka",
            "km",
            "frequency",
            "abundance",
            "copy number",
            "titer",
        }
        label_lower = label.lower().replace("-", " ").replace("_", " ")
        if any(hint in label_lower for hint in log_hints):
            return True

    return False


def detect_column_roles(
    table: DataTable,
    x: str | None = None,
    y: str | list[str] | None = None,
    group: str | None = None,
    error: str | None = None,
) -> tuple[str | None, list[str], str | None, str | None]:
    """Detect column roles (x, y, group, error) from data.

    Auto-fills unspecified roles based on column types.

    Args:
        table: Data table.
        x: Explicit x column.
        y: Explicit y column(s).
        group: Explicit grouping column.
        error: Explicit error column.

    Returns:
        Tuple of (x_col, y_cols, group_col, error_col).
    """
    cols = table.column_names
    categorical = [c for c in cols if table.is_categorical(c)]
    numeric = [c for c in cols if table.is_numeric(c)]

    x_col = x
    y_cols: list[str] = []
    group_col = group
    error_col = error

    if isinstance(y, str):
        y_cols = [y]
    elif isinstance(y, list):
        y_cols = list(y)

    # Auto-detect x
    if x_col is None:
        if categorical:
            x_col = categorical[0]
        elif len(numeric) >= 2:
            # First numeric column as x, or one with ordering hints
            ordered = [c for c in numeric if _has_ordering_hint(c)]
            x_col = ordered[0] if ordered else numeric[0]

    # Auto-detect error column by name pattern BEFORE y detection
    # so error columns don't get swept into y_cols
    if error_col is None:
        error_hints = {"error", "err", "sem", "sd", "std", "stdev", "ci", "uncertainty"}
        for c in cols:
            if c != x_col and c != group_col and c not in y_cols:
                c_lower = c.lower().replace("_", " ").replace("-", " ")
                if any(hint in c_lower for hint in error_hints):
                    error_col = c
                    break

    # Auto-detect y
    used = {x_col, group_col, error_col}
    if not y_cols:
        y_cols = [c for c in numeric if c not in used]

    return x_col, y_cols, group_col, error_col


def recommend_sort(
    table: DataTable,
    x_col: str | None,
    kind: str,
) -> str | None:
    """Recommend sort order for the data.

    Rules:
    - Bar charts: sort by value (descending) unless x has natural order
    - Box plots: sort by median (descending) unless x has natural order
    - Other charts: preserve data order

    Args:
        table: Data table.
        x_col: X-axis column name.
        kind: Plot kind.

    Returns:
        "value" for value-based sorting, "name" for alphabetical, None for preserve order.
    """
    if kind not in ("bar", "box", "violin", "strip", "swarm"):
        return None

    if x_col is None:
        return None

    # If x has natural ordering (numeric-like, time-like), preserve order
    if table.is_numeric(x_col):
        return None
    if _has_ordering_hint(x_col):
        return None

    # Check if the category VALUES contain dose/concentration/time patterns
    # e.g., "10 nM", "1 μM", "Vehicle", "Day 1", "Week 4"
    if x_col in table.columns and _values_have_ordering(table.columns[x_col]):
        return None

    # Categorical x: sort by value for cleaner presentation
    return "value"


def _has_ordering_hint(col_name: str) -> bool:
    """Check if column name suggests natural ordering."""
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
        "iteration",
        "position",
        "distance",
        "depth",
        "index",
    }
    name_lower = col_name.lower().replace("_", " ").replace("-", " ")
    return any(hint in name_lower for hint in hints)


def _values_have_ordering(values: list) -> bool:
    """Check if category values contain dose/concentration/time patterns.

    Detects patterns like "10 nM", "1 μM", "Vehicle", "Day 1", "Week 4",
    "Baseline", "T0", etc. that suggest the data has a natural scientific order.
    """
    unit_patterns = {
        "nm",
        "μm",
        "um",
        "mm",
        "m",
        "μg",
        "ug",
        "mg",
        "kg",
        "nm",
        "pm",
        "fm",  # concentrations
        "ml",
        "μl",
        "ul",
        "l",  # volumes
        "day",
        "week",
        "month",
        "hour",
        "hr",
        "min",  # time
    }
    ordinal_patterns = {
        "vehicle",
        "control",
        "baseline",
        "placebo",
        "untreated",
        "dmso",
        "pre",
        "post",
        "before",
        "after",
    }

    str_vals = [str(v).lower().strip() for v in values if v is not None]
    if not str_vals:
        return False

    # If any value matches an ordinal pattern (e.g., "Vehicle", "Control")
    if any(any(p in v for p in ordinal_patterns) for v in str_vals):
        return True

    # If any value contains a unit suffix (e.g., "10 nM", "1 μM")
    if any(any(v.endswith(u) or f" {u}" in v for u in unit_patterns) for v in str_vals):
        return True

    return False
