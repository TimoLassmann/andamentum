"""Retrieval metrics over passage ids, and the paired bootstrap between runs.

Gold is the set of supporting passages; relevance is binary. ``full_support@k``
asks whether ALL supporting passages are in the top k — the multi-hop question
is answerable from the retrieved set only then — and is the headline number;
recall@k gives partial credit and MRR rewards the first hit.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

CUTOFFS = (5, 10, 20)


def question_metrics(ranked: Sequence[str], gold: frozenset[str]) -> dict[str, float]:
    if not gold:
        raise ValueError("a question without gold passages cannot be scored")
    out: dict[str, float] = {}
    for k in CUTOFFS:
        hit = len(gold.intersection(ranked[:k]))
        out[f"recall@{k}"] = hit / len(gold)
        out[f"full_support@{k}"] = float(hit == len(gold))
    first = next((i for i, pid in enumerate(ranked) if pid in gold), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, pid in enumerate(ranked[:10]) if pid in gold
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    out["ndcg@10"] = dcg / idcg
    return out


def mean_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], *, n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (b - a) over paired questions with a 95% percentile interval."""
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap needs two equal, non-empty samples")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples)
    )
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return sum(diffs) / n, lo, hi
