"""Small statistics for comparing arms on the same items: no I/O, no model.

``paired_bootstrap_ci`` resamples the per-item differences between two arms
(variant minus baseline, one number per item) and reports the mean difference
with a percentile interval. Items are the unit of resampling, so the interval
reflects how much the verdict depends on which questions were drawn.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

DEFAULT_CONFIDENCE = 0.95


class BootstrapCI(BaseModel):
    """The mean of the paired differences and its bootstrap interval."""

    model_config = ConfigDict(frozen=True)

    mean: float
    low: float
    high: float
    n_items: int
    n_resamples: int
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval is on one side of zero."""
        return self.low > 0.0 or self.high < 0.0


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending sequence, ``q`` in [0, 1]."""
    position = q * (len(sorted_values) - 1)
    below = math.floor(position)
    above = math.ceil(position)
    weight = position - below
    return sorted_values[below] * (1.0 - weight) + sorted_values[above] * weight


def paired_bootstrap_ci(
    differences: Sequence[float],
    n_resamples: int,
    seed: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapCI:
    """Percentile bootstrap interval for the mean of *differences*.

    Each resample draws ``len(differences)`` items with replacement from a
    ``random.Random(seed)``, so the same inputs and seed give the same interval.
    One item gives a zero-width interval at its value (there is nothing to
    resample); no items is an error, because no interval exists.
    """
    if not differences:
        raise ValueError("paired_bootstrap_ci needs at least one difference")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = [float(d) for d in differences]
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError("differences must be finite numbers")
    n = len(values)
    rng = random.Random(seed)
    means = sorted(
        math.fsum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return BootstrapCI(
        mean=math.fsum(values) / n,
        low=_percentile(means, tail),
        high=_percentile(means, 1.0 - tail),
        n_items=n,
        n_resamples=n_resamples,
        confidence=confidence,
    )
