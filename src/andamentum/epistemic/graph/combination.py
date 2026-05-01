"""Claim-verdict combination — pure functions consumed by the graph.

Phase 4 of the multi-seed-claim refactor. The combination logic was
previously in ``decomposed_runner.combine_sub_verdicts``, which operated
on per-child PipelineResult objects. Under multi-seed-claim, all claims
live on one Objective, so combination operates on Claim entities directly.

This module:

* ``CombinedVerdict`` — dataclass capturing the aggregate outcome.
* ``combine_claim_verdicts(claims, rule, weights) -> CombinedVerdict`` —
  the pure function the graph node calls. Honours
  AND / OR / WEIGHTED_AND / UNION combination rules. Drops claims that
  are abandoned, cycle_capped, or have no integration verdict from
  numeric combination; surfaces the dropped subset in the diagnostic.

Combination semantics (Mill / Lakatos / Lipton):

* AND          → min of per-claim posteriors (weakest-link bound)
* OR           → max of per-claim posteriors (best-evidence bound)
* WEIGHTED_AND → weighted mean of per-claim posteriors (Phase 5 weights
  from ``decomposition.sub_investigations[i].weight``)
* UNION        → posterior=None; the combined view is the set of
  per-claim verdicts (exploratory questions where each sub-investigation
  contributes a facet rather than a value)

A claim's posterior is derived from its ``integrated_assessment`` /
``integrated_confidence`` (the IBE chain's output):

* "supports"     → posterior = 0.5 + confidence/2
* "contradicts"  → posterior = 0.5 - confidence/2
* "insufficient" → posterior = 0.5

Mirrors the same mapping ``compute_posterior`` uses (confidence.py),
keeping the two entry points consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..entities.claim import Claim


@dataclass
class CombinedVerdict:
    """The aggregate outcome over a set of Claims.

    posterior is None when:
      * combination_rule is UNION (no scalar verdict by design)
      * all input claims are excluded (capped, abandoned, no verdict)
    """

    posterior: float | None
    verdict: str  # "supports" | "contradicts" | "insufficient" | "no_data" | "union"
    combination_rule: str
    claim_posteriors: list[float | None]  # one entry per input Claim, in order
    n_capped: int
    n_no_verdict: int
    n_abandoned: int
    explanation: str


def _verdict_label(p: float) -> str:
    """Map a combined posterior to a verdict label using the same
    breakpoints the per-claim integration uses."""
    if p > 0.66:
        return "supports"
    if p < 0.34:
        return "contradicts"
    return "insufficient"


def _claim_posterior(claim: Claim) -> float | None:
    """Derive a claim's posterior from its integration verdict.

    Returns None when the claim has no integration verdict — IBE
    didn't run on it (e.g. it's still HYPOTHESIS, abandoned, or
    cycle-capped before promotion).
    """
    if claim.integrated_assessment is None:
        return None
    confidence = claim.integrated_confidence or 0.0
    if claim.integrated_assessment == "supports":
        return 0.5 + confidence / 2
    if claim.integrated_assessment == "contradicts":
        return 0.5 - confidence / 2
    # "insufficient" or any other value — treat as neutral
    return 0.5


def combine_claim_verdicts(
    claims: list[Claim],
    combination_rule: str,
    weights: list[float] | None = None,
) -> CombinedVerdict:
    """Aggregate per-claim posteriors into a combined verdict.

    Args:
        claims: claims in decomposition order. Abandoned, cycle_capped,
            and no-verdict claims are excluded from numeric combination
            but recorded in the diagnostic.
        combination_rule: AND / OR / WEIGHTED_AND / UNION (case-insensitive).
        weights: optional per-claim weights, same length as ``claims``,
            consumed only by WEIGHTED_AND. None or all-equal makes
            WEIGHTED_AND degenerate to a simple mean. Negative weights
            raise ValueError.
    """
    rule = combination_rule.upper()

    claim_posteriors: list[float | None] = []
    n_abandoned = 0
    n_capped = 0
    n_no_verdict = 0
    eligible_indices: list[int] = []

    for i, c in enumerate(claims):
        if c.abandoned:
            n_abandoned += 1
            claim_posteriors.append(None)
            continue
        if getattr(c, "cycle_capped", False):
            n_capped += 1
            claim_posteriors.append(None)
            continue
        p = _claim_posterior(c)
        claim_posteriors.append(p)
        if p is None:
            n_no_verdict += 1
            continue
        eligible_indices.append(i)

    numeric = [claim_posteriors[i] for i in eligible_indices]
    # mypy/pyright: claim_posteriors[i] is not None for i in eligible_indices
    numeric = [p for p in numeric if p is not None]

    diag = (
        f"{len(claims)} claims: {len(numeric)} aggregated, "
        f"{n_capped} cycle-capped, {n_abandoned} abandoned, "
        f"{n_no_verdict} no-verdict"
    )

    if rule == "UNION":
        # Set-collection semantics: each claim's verdict contributes a
        # facet rather than a value to be averaged. No scalar verdict.
        return CombinedVerdict(
            posterior=None,
            verdict="union",
            combination_rule="UNION",
            claim_posteriors=claim_posteriors,
            n_capped=n_capped,
            n_no_verdict=n_no_verdict,
            n_abandoned=n_abandoned,
            explanation=(
                f"UNION over {diag}. Render each claim's verdict "
                "individually; there is no scalar verdict."
            ),
        )

    if not numeric:
        return CombinedVerdict(
            posterior=None,
            verdict="no_data",
            combination_rule=rule,
            claim_posteriors=claim_posteriors,
            n_capped=n_capped,
            n_no_verdict=n_no_verdict,
            n_abandoned=n_abandoned,
            explanation=f"No claim produced a numeric posterior ({diag}).",
        )

    if rule == "AND":
        combined = min(numeric)
        method = "min (weakest-link bound on conjunction)"
    elif rule == "OR":
        combined = max(numeric)
        method = "max (best-evidence bound on disjunction)"
    elif rule == "WEIGHTED_AND":
        combined, method = _weighted_mean(claim_posteriors, weights)
    else:
        raise ValueError(
            f"Unknown combination_rule {combination_rule!r}; "
            "expected AND / OR / WEIGHTED_AND / UNION"
        )

    return CombinedVerdict(
        posterior=combined,
        verdict=_verdict_label(combined),
        combination_rule=rule,
        claim_posteriors=claim_posteriors,
        n_capped=n_capped,
        n_no_verdict=n_no_verdict,
        n_abandoned=n_abandoned,
        explanation=(
            f"{rule} over {diag} via {method}: "
            f"{[round(p, 3) for p in numeric]} → {round(combined, 3)}"
        ),
    )


def _weighted_mean(
    claim_posteriors: list[float | None], weights: list[float] | None
) -> tuple[float, str]:
    """Compute weighted mean over numeric per-claim posteriors.

    Claims with posterior=None are dropped along with their weight.
    None weights → simple mean over numeric subset. All-zero weights on
    numeric claims also fall back to simple mean.
    """
    if weights is None:
        numeric = [p for p in claim_posteriors if p is not None]
        return sum(numeric) / len(numeric), "mean (no weights provided)"
    if len(weights) != len(claim_posteriors):
        raise ValueError(
            f"weights length {len(weights)} does not match "
            f"claims length {len(claim_posteriors)}"
        )
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    paired = [
        (p, w)
        for p, w in zip(claim_posteriors, weights, strict=True)
        if p is not None
    ]
    weight_sum = sum(w for _, w in paired)
    if weight_sum == 0.0:
        numeric = [p for p, _ in paired]
        return sum(numeric) / len(numeric), "mean (all weights zero)"
    weighted = sum(p * w for p, w in paired) / weight_sum
    return (
        weighted,
        f"weighted mean (weights={[round(w, 2) for _, w in paired]})",
    )


def resolve_combination_rule(objective: Any) -> str | None:
    """Single source of truth for the combination_rule lookup.

    Reads from BOTH ``objective.combination_rule`` (the dedicated
    Objective field) AND ``objective.decomposition["combination_rule"]``
    (the rule the decomposer wrote into the decomposition dict).
    Returns the first non-None value, or None if neither is set.

    Used by both ``CombineClaimVerdicts`` (graph node) and
    ``compute_posterior`` (confidence.py) so the two paths can never
    disagree on which rule to apply. Without this helper, one path
    might default to "AND" while the other reads OR/WEIGHTED_AND/UNION
    from the decomposition dict — silently producing different
    posteriors on the same claims.
    """
    rule = getattr(objective, "combination_rule", None)
    if rule:
        return rule
    decomposition = getattr(objective, "decomposition", None) or {}
    return decomposition.get("combination_rule")


def extract_weights_from_decomposition(
    decomposition: dict | None, claims: list[Claim]
) -> list[float] | None:
    """Pull per-claim weights from the decomposition's sub_investigations.

    Returns None when:
    * decomposition is missing or has no sub_investigations
    * any claim lacks a sub_investigation_id (open-research claims)
    * a claim's sub_investigation_id has no matching entry

    Returned list aligns with ``claims`` (same length, same order).
    """
    if not decomposition:
        return None
    subs = decomposition.get("sub_investigations") or []
    if not subs:
        return None
    sub_weights = {s.get("id"): float(s.get("weight", 1.0)) for s in subs}
    weights: list[float] = []
    for c in claims:
        sid = c.sub_investigation_id
        if sid is None or sid not in sub_weights:
            return None  # any unmatched claim → fall back to no-weights
        weights.append(sub_weights[sid])
    return weights
