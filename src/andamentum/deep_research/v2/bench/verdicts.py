"""Applying a pre-registered rule to results: the interval and the verdict.

Pure functions over per-item values, no I/O. A rule (``experiment.Rule``) is
fixed before the run; this module only reads it. The verdict is ``accepted``
when the whole bootstrap interval is on the right side of the margin or
threshold, ``not accepted`` when the whole interval is on the wrong side, and
``inconclusive`` when it straddles the line (or there is nothing to compare):
a small sample is allowed to say "I cannot tell".

Items are the unit of resampling. A paired rule uses the items both arms have
a value for; a missing value never counts as zero here (the runner records an
item that failed as a zero on the outcome metrics, so a crashing arm is
already penalised before it gets here).
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .experiment import Aggregate, Rule
from .stats import paired_bootstrap_ci

Outcome = Literal["accepted", "not accepted", "inconclusive"]
ItemValues = Mapping[str, float | None]


class Interval(BaseModel):
    """A point estimate with a percentile bootstrap interval."""

    model_config = ConfigDict(frozen=True)

    estimate: float
    low: float
    high: float
    n_items: int


class Verdict(BaseModel):
    """What one rule decided, and why."""

    rule_id: str
    question: str
    outcome: Outcome
    gate: bool
    interval: Interval | None
    statement: str


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    below, above = math.floor(position), math.ceil(position)
    weight = position - below
    return sorted_values[below] * (1.0 - weight) + sorted_values[above] * weight


def aggregate_interval(
    values: Sequence[float],
    aggregate: Aggregate,
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Interval | None:
    """The mean or median of *values* with a bootstrap interval; None for no values."""
    if not values:
        return None
    if aggregate == "max":
        # no sampling: the largest value is the largest value
        top = float(max(values))
        return Interval(estimate=top, low=top, high=top, n_items=len(values))
    if aggregate == "mean":
        ci = paired_bootstrap_ci(values, resamples, seed, confidence=confidence)
        return Interval(estimate=ci.mean, low=ci.low, high=ci.high, n_items=ci.n_items)
    rng = random.Random(seed)
    n = len(values)
    medians = sorted(
        statistics.median(values[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    tail = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(statistics.median(values)),
        low=_percentile(medians, tail),
        high=_percentile(medians, 1.0 - tail),
        n_items=n,
    )


def paired_values(arm: ItemValues, reference: ItemValues) -> list[float]:
    """Per-item ``arm - reference`` over the items both have a value for."""
    return [
        a - b
        for item, a in arm.items()
        if a is not None and (b := reference.get(item)) is not None
    ]


def reductions(arm: ItemValues, reference: ItemValues) -> list[float]:
    """Per item ``1 - arm / reference`` over the items both have a value for
    and whose reference is above zero (there is nothing to reduce otherwise)."""
    return [
        1.0 - a / b
        for item, a in arm.items()
        if a is not None and (b := reference.get(item)) is not None and b > 0
    ]


def paired_interval(
    arm: ItemValues, reference: ItemValues, *, resamples: int, seed: int
) -> Interval | None:
    """The mean paired difference (arm minus reference) and its interval."""
    return aggregate_interval(
        paired_values(arm, reference), "mean", resamples=resamples, seed=seed
    )


def single_interval(
    values: ItemValues, aggregate: Aggregate, *, resamples: int, seed: int
) -> Interval | None:
    """The aggregate of one arm's item values and its interval."""
    return aggregate_interval(
        [v for v in values.values() if v is not None],
        aggregate,
        resamples=resamples,
        seed=seed,
    )


def _fmt(value: float) -> str:
    return f"{value:+.3f}" if abs(value) < 1000 else f"{value:+.0f}"


def judge_rule(
    rule: Rule,
    *,
    arm: ItemValues,
    reference: ItemValues | None,
    resamples: int,
    seed: int,
) -> Verdict:
    """Apply *rule* to the item values of its arm (and reference arm)."""
    if rule.test == "reduced_by":
        return _judge_reduction(
            rule, arm=arm, reference=reference, resamples=resamples, seed=seed
        )
    paired = rule.test in ("noninferior", "superior")
    if paired:
        if reference is None:
            raise ValueError(f"rule {rule.id} needs the reference arm's values")
        raw = paired_values(arm, reference)
        sign = 1.0 if rule.higher_is_better else -1.0
        interval = aggregate_interval(
            [sign * d for d in raw], "mean", resamples=resamples, seed=seed
        )
        if interval is None:
            return _verdict(
                rule, "inconclusive", None, "no item has a value for both arms"
            )
        bound = -rule.margin if rule.test == "noninferior" else rule.margin
        outcome: Outcome
        if interval.low > bound:
            outcome = "accepted"
        elif interval.high <= bound:
            outcome = "not accepted"
        else:
            outcome = "inconclusive"
        bar = (
            f"not worse than -{rule.margin:g}"
            if rule.test == "noninferior"
            else f"better than +{rule.margin:g}"
        )
        statement = (
            f"{rule.arm} minus {rule.reference}, oriented so that higher is better: "
            f"{_fmt(interval.estimate)} (95% interval {_fmt(interval.low)} to "
            f"{_fmt(interval.high)}, {interval.n_items} items); bar: {bar}"
        )
        return _verdict(rule, outcome, interval, statement)
    threshold = rule.threshold
    if threshold is None:  # the rule validator guarantees one
        raise ValueError(f"rule {rule.id} has no threshold")
    interval = single_interval(arm, rule.aggregate, resamples=resamples, seed=seed)
    if interval is None:
        return _verdict(rule, "inconclusive", None, "no item has a value")
    if rule.test == "at_most" and rule.aggregate == "max":
        outcome = "accepted" if interval.high <= threshold else "not accepted"
        statement = (
            f"{rule.arm} largest {rule.metric} over {interval.n_items} items: "
            f"{interval.estimate:g}; bar: <= {threshold:g}"
        )
        return _verdict(rule, outcome, interval, statement)
    if rule.test == "at_least":
        outcome = (
            "accepted"
            if interval.low >= threshold
            else "not accepted"
            if interval.high < threshold
            else "inconclusive"
        )
        bar = f">= {threshold:g}"
    else:
        outcome = (
            "accepted"
            if interval.high <= threshold
            else "not accepted"
            if interval.low > threshold
            else "inconclusive"
        )
        bar = f"<= {threshold:g}"
    statement = (
        f"{rule.arm} {rule.aggregate} {rule.metric}: {interval.estimate:.3f} "
        f"(95% interval {interval.low:.3f} to {interval.high:.3f}, "
        f"{interval.n_items} items); bar: {bar}"
    )
    return _verdict(rule, outcome, interval, statement)


def _judge_reduction(
    rule: Rule,
    *,
    arm: ItemValues,
    reference: ItemValues | None,
    resamples: int,
    seed: int,
) -> Verdict:
    """``reduced_by``: the aggregate share by which *arm* uses less than
    *reference* per item, against the rule's threshold."""
    if reference is None:
        raise ValueError(f"rule {rule.id} needs the reference arm's values")
    threshold = rule.threshold
    if threshold is None:  # the rule validator guarantees one
        raise ValueError(f"rule {rule.id} has no threshold")
    interval = aggregate_interval(
        reductions(arm, reference), rule.aggregate, resamples=resamples, seed=seed
    )
    if interval is None:
        return _verdict(
            rule,
            "inconclusive",
            None,
            "no item has a value above zero for the reference arm",
        )
    # the registered rule is the aggregate itself ("median over items") against
    # the bar, so the point value decides; the interval is reported beside it
    outcome: Outcome = "accepted" if interval.estimate >= threshold else "not accepted"
    statement = (
        f"{rule.arm} reduces {rule.metric} against {rule.reference} by "
        f"{interval.estimate:.1%} ({rule.aggregate} over items; 95% interval "
        f"{interval.low:.1%} to {interval.high:.1%}, {interval.n_items} items); "
        f"bar: the {rule.aggregate} at least {threshold:.0%}"
    )
    return _verdict(rule, outcome, interval, statement)


def _verdict(
    rule: Rule, outcome: Outcome, interval: Interval | None, statement: str
) -> Verdict:
    return Verdict(
        rule_id=rule.id,
        question=rule.question,
        outcome=outcome,
        gate=rule.gate,
        interval=interval,
        statement=statement,
    )
