"""Statistical helpers for figure rendering.

Provides bootstrap confidence intervals and data aggregation utilities.
These are the minimal statistics needed for figure rendering — not a
statistics library.
"""

from __future__ import annotations

import random
from typing import Any


def bootstrap_ci(
    values: list[float | int],
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: Numeric values.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 = 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, ci_lower, ci_upper).

    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot compute bootstrap CI on empty values")

    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        raise ValueError("No numeric values found")

    rng = random.Random(seed)
    n = len(clean)
    mean = sum(clean) / n

    if n == 1:
        return (mean, mean, mean)

    # Bootstrap
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(clean) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))

    return (mean, boot_means[lo_idx], boot_means[hi_idx])


def aggregate_by(
    records: list[dict[str, Any]],
    group_key: str,
    value_key: str,
) -> dict[str, list[Any]]:
    """Group records by a key and collect values.

    Args:
        records: List of dicts (row-oriented data).
        group_key: Key to group by.
        value_key: Key whose values to collect.

    Returns:
        Dict mapping group values to lists of collected values.
    """
    result: dict[str, list[Any]] = {}
    for r in records:
        group = str(r.get(group_key, ""))
        val = r.get(value_key)
        if group not in result:
            result[group] = []
        result[group].append(val)
    return result


def compute_mean_error(
    values: list[float | int],
    error_type: str = "sem",
) -> tuple[float, float]:
    """Compute mean and error bar value.

    Args:
        values: Numeric values.
        error_type: Type of error: "sem" (standard error), "sd" (standard deviation),
                    or "ci95" (95% CI half-width).

    Returns:
        Tuple of (mean, error_value).
    """
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return (0.0, 0.0)

    n = len(clean)
    mean = sum(clean) / n

    if n < 2:
        return (mean, 0.0)

    # Standard deviation
    variance = sum((x - mean) ** 2 for x in clean) / (n - 1)
    sd = variance**0.5

    if error_type == "sd":
        return (mean, sd)
    elif error_type == "sem":
        return (mean, sd / n**0.5)
    elif error_type == "ci95":
        return (mean, 1.96 * sd / n**0.5)
    else:
        return (mean, sd / n**0.5)  # default to SEM
