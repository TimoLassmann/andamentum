"""Retrieval metrics over passage ids, and the paired bootstrap between runs.

Gold is the set of supporting passages; relevance is binary. ``full_support@k``
asks whether ALL supporting passages are in the top k — the multi-hop question
is answerable from the retrieved set only then — and is the headline number;
recall@k gives partial credit and MRR rewards the first hit.
"""

from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
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


def _resampled_means(
    diffs: Sequence[float], labels: Sequence[str], *, n_resamples: int, seed: int
) -> list[float]:
    units: dict[str, list[float]] = {}
    for label, d in zip(labels, diffs, strict=True):
        units.setdefault(label, []).append(d)
    sums = [(sum(v), len(v)) for _, v in sorted(units.items())]
    k = len(sums)
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = count = 0.0
        for _ in range(k):
            s_, n_ = sums[rng.randrange(k)]
            total += s_
            count += n_
        means.append(total / count)
    means.sort()
    return means


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    n_resamples: int = 10_000,
    seed: int = 1234,
) -> tuple[float, float, float]:
    """Mean of (b - a) over paired questions with a 95% percentile interval.

    With ``clusters`` (one label per question), whole clusters are resampled,
    so questions that share supporting passages count as one unit and the
    interval is not made too narrow by their correlation.
    """
    diff, lo, hi, _ = paired_bootstrap_test(
        a, b, clusters=clusters, n_resamples=n_resamples, seed=seed
    )
    return diff, lo, hi


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    clusters: Sequence[str] | None = None,
    n_resamples: int = 10_000,
    seed: int = 1234,
) -> tuple[float, float, float, float]:
    """As :func:`paired_bootstrap`, plus a two-sided bootstrap p-value.

    The p-value is twice the smaller share of resampled means on either side
    of zero, at most 1; it is 1.0 when every difference is zero.
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap needs two equal, non-empty samples")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    labels = (
        list(clusters) if clusters is not None else [str(i) for i in range(len(diffs))]
    )
    if len(labels) != len(diffs):
        raise ValueError("clusters needs one label per question")
    means = _resampled_means(diffs, labels, n_resamples=n_resamples, seed=seed)
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    if not any(diffs):
        p = 1.0
    else:
        below = sum(m <= 0 for m in means) / n_resamples
        above = sum(m >= 0 for m in means) / n_resamples
        p = min(1.0, 2 * min(below, above))
    return sum(diffs) / len(diffs), lo, hi, p


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, drop punctuation and articles, collapse spaces."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def _token_f1(prediction: str, answer: str) -> float:
    p, a = normalize_answer(prediction).split(), normalize_answer(answer).split()
    if not p or not a:
        return float(p == a)
    common = sum((Counter(p) & Counter(a)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(a)
    return 2 * precision * recall / (precision + recall)


def answer_metrics(prediction: str, answers: Sequence[str]) -> dict[str, float]:
    """Exact match and token F1, each the maximum over the accepted answers."""
    if not answers:
        raise ValueError("a question without answers cannot be scored")
    norm = normalize_answer(prediction)
    return {
        "em": float(any(norm == normalize_answer(a) for a in answers)),
        "f1": max(_token_f1(prediction, a) for a in answers),
        "unknown": float(norm == "unknown"),
    }
