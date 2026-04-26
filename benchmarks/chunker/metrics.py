"""Pure-function metrics for the chunker benchmark.

All functions take primitives and return primitives — no dataclass dependencies,
no LLM calls, no I/O. Easy to unit-test, easy to reason about.
"""

from __future__ import annotations

import math


def boundary_f1(
    predicted: list[int],
    truth: list[int],
    *,
    tolerance: int,
) -> tuple[float, float, float]:
    """Compute (precision, recall, F1) of predicted boundaries vs truth.

    A predicted boundary "matches" a truth boundary if it's within `tolerance`
    chars. Each truth boundary can be matched by at most one predicted
    boundary (greedy assignment by closest distance).
    """
    if not predicted and not truth:
        return (1.0, 1.0, 1.0)
    if not predicted or not truth:
        return (0.0, 0.0, 0.0)

    # Greedy matching: for each predicted, find nearest unmatched truth.
    truth_used = [False] * len(truth)
    matched_predicted = 0
    for p in predicted:
        best_idx = -1
        best_dist = tolerance + 1
        for i, t in enumerate(truth):
            if truth_used[i]:
                continue
            d = abs(p - t)
            if d <= tolerance and d < best_dist:
                best_idx = i
                best_dist = d
        if best_idx >= 0:
            truth_used[best_idx] = True
            matched_predicted += 1

    precision = matched_predicted / len(predicted)
    recall = matched_predicted / len(truth)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return (precision, recall, f1)


def granularity_ratio(*, predicted_count: int, truth_count: int) -> float:
    """Return predicted_count / truth_count.

    1.0 = perfect granularity. <1 = under-segmented. >1 = over-segmented.
    Returns +inf if truth_count is 0 (can't divide).
    """
    if truth_count == 0:
        return math.inf
    return predicted_count / truth_count
