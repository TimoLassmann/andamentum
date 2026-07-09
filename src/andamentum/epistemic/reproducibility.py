"""Reproducibility tripwire for confident evidence judgments (Peirce, demand-gated).

The verbalized-confidence experiment (``experiments/dirichlet_confidence``) found
one thing a single call is *structurally blind* to: a confident judgment that
flips under paraphrase. Each individual call looks sure (one-hot), so entropy —
the normal wrong-answer detector — carries no information; only re-asking under
different wording exposes the brittleness. It does not lower total error, it
*reallocates* it (trading silent irreproducible flips for stable ones): a
**reproducibility** property, not an accuracy one.

That check is expensive, so it is applied narrowly (P7 lazy escalation): only at
a load-bearing moment (a decisive posterior about to finalize), and only to the
judgments that are actually blind — the one-hot ones. When a paraphrase flips a
verdict, the caller emits a Demand for more inquiry rather than finalizing; the
tripwire never asserts a new direction, it only withholds a confident one
(suspend-only).

This module holds the pure selectivity/agreement predicates plus a bounded,
strictly-sequential re-judgment driver (never two concurrent local-model calls).

Architecture: Layer 1 (framework-agnostic, async driver + pure predicates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .judge import judge_evidence


class _EvidenceLike(Protocol):
    entity_id: str
    support_judgment: Optional[str]
    judgment_one_hot: Optional[bool]
    extracted_content: str
    source_ref: str
    invalidated: bool
    cluster_status: str


# The directional verdicts. A one-hot ``no_bearing`` is not load-bearing on the
# posterior direction, so the tripwire only checks directional judgments.
_DIRECTIONAL = frozenset({"supports", "contradicts"})


def warrants_reproducibility_check(evidence: _EvidenceLike) -> bool:
    """True if this judgment is a paraphrase-flip blind spot worth re-checking.

    The judgment must be **one-hot** (confident → entropy uninformative → the
    exact case a single call cannot self-diagnose) and **directional**
    (supports/contradicts — the verdicts that move the posterior). Evidence
    without a captured distribution (``judgment_one_hot is None``) is skipped:
    no verbalized confidence was measured, so there is nothing to call brittle.
    """
    return (
        evidence.judgment_one_hot is True and evidence.support_judgment in _DIRECTIONAL
    )


def verdicts_agree(original: Optional[str], paraphrased: Optional[str]) -> bool:
    """Whether a re-judgment under paraphrase preserved the verdict.

    Disagreement is only load-bearing when it crosses a directional boundary; a
    drift to/from ``no_bearing`` is treated as agreement-enough here because the
    tripwire's job is to catch confident *direction* flips (supports↔contradicts),
    the reproducibility landmine the experiment isolated.
    """
    if original == paraphrased:
        return True
    return not (original in _DIRECTIONAL and paraphrased in _DIRECTIONAL)


@dataclass
class ReproducibilityOutcome:
    """Result of a bounded reproducibility sweep over load-bearing judgments."""

    flip_found: bool = False
    checks_run: int = 0
    flipped_evidence_id: Optional[str] = None
    detail: str = ""
    claim_ids_checked: list[str] = field(default_factory=list)


def _eligible_one_hot(evidence_items: list[_EvidenceLike]) -> list[_EvidenceLike]:
    return [
        ev
        for ev in evidence_items
        if ev.extracted_content
        and not ev.invalidated
        and ev.cluster_status not in ("corroborative", "deferred")
        and warrants_reproducibility_check(ev)
    ]


async def check_claims_reproducibility(
    claims: list[Any],
    repo: Any,
    runner: Any,
    *,
    max_checks: int = 4,
) -> ReproducibilityOutcome:
    """Bounded, sequential paraphrase-flip sweep over claims' one-hot judgments.

    For each claim with load-bearing one-hot judgments, paraphrase the claim
    once (``epistemic_paraphrase_claim``) and re-judge its one-hot evidence
    against the paraphrase (existing ``epistemic_judge_evidence``), comparing to
    the stored verdict. Returns on the FIRST directional flip — one is enough to
    warrant more inquiry. Strictly sequential (never two concurrent local-model
    calls); total re-judgments capped at ``max_checks`` so the synthesis-loop
    cost stays bounded.

    Pure driver: it does NOT mutate any evidence or claim — its only product is
    the outcome the caller routes on.
    """
    outcome = ReproducibilityOutcome()
    if runner is None or max_checks <= 0:
        return outcome

    for claim in claims:
        if outcome.checks_run >= max_checks:
            break

        evidence_items: list[_EvidenceLike] = []
        for eid in getattr(claim, "evidence_ids", []):
            ev = await repo.get("evidence", eid)
            if ev is not None:
                evidence_items.append(ev)
        one_hot = _eligible_one_hot(evidence_items)
        if not one_hot:
            continue

        outcome.claim_ids_checked.append(claim.entity_id)
        paraphrase_result = await runner.run(
            "epistemic_paraphrase_claim",
            claim=claim.statement,
            scope=getattr(claim, "scope", "") or "",
        )
        paraphrased_claim = paraphrase_result.paraphrase

        for ev in one_hot:
            if outcome.checks_run >= max_checks:
                break
            rejudged = await judge_evidence(
                claim_statement=paraphrased_claim,
                claim_scope=getattr(claim, "scope", "") or "",
                evidence_content=ev.extracted_content,
                evidence_source=ev.source_ref,
                runner=runner,
            )
            outcome.checks_run += 1
            if not verdicts_agree(ev.support_judgment, rejudged.verdict):
                outcome.flip_found = True
                outcome.flipped_evidence_id = ev.entity_id
                outcome.detail = (
                    f"Evidence {ev.entity_id} judged '{ev.support_judgment}' "
                    f"flipped to '{rejudged.verdict}' when claim "
                    f"{claim.entity_id} was paraphrased — a confident but "
                    f"non-reproducible judgment underpinning a decisive verdict."
                )
                return outcome

    return outcome


__all__ = [
    "ReproducibilityOutcome",
    "warrants_reproducibility_check",
    "verdicts_agree",
    "check_claims_reproducibility",
]
