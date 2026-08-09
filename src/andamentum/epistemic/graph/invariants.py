"""Reusable post-condition invariants for graph nodes.

Phase 0 of the Move-3 plan. Invariants are predicates over
``(EpistemicGraphState, list[Claim])`` that return ``None`` when they
hold, or a non-empty violation message when they don't.

Each node's ``post_invariants`` tuple references one or more of these,
and the contract validator asserts they hold after the node runs.

The canonical invariant — and the only one we need at the start of the
migration — is ``no_stranded_claims``: no Claim ends in a state where
IBE should have run on it but didn't. That's the bug shape we've hit
three times in the recurring routing-bug class.

Origin: ``stranded_claims`` was first written inline in
``tests/test_graph_reachability.py`` (commit ``d280573``); moving it
here lets nodes reference it as ``post_invariants`` and lets the
contract test reuse it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..entities.claim import ClaimStage

if TYPE_CHECKING:
    from ..entities import Claim
    from .state import EpistemicGraphState


def stranded_claims(
    claims: "list[Claim]", state: "EpistemicGraphState"
) -> "list[Claim]":
    """Return the subset of ``claims`` that are stranded.

    A stranded claim has been promoted past HYPOTHESIS, has no
    integration verdict, and is NOT marked verification_done. Such a
    claim should still reach the IBE chain to receive a calibrated
    verdict. If the graph routes it to a terminal in this state, the
    posterior falls back to no-data 0.5 and the report is silently
    miscalibrated.

    Refute-promoted claims have ``integrated_assessment="contradicts"``
    set by the operation and ARE marked verification_done — the IBE
    chain shouldn't overwrite that pre-set verdict. Soft-promoted
    claims have ``integrated_assessment=None`` and are NOT marked
    verification_done — IBE is exactly what should populate the
    verdict.

    Cycle-capped and abandoned claims are excluded — both are
    legitimate terminal states where no integration verdict is
    expected.
    """
    out: list[Claim] = []
    for c in claims:
        if c.abandoned or getattr(c, "cycle_capped", False):
            continue
        if c.stage == ClaimStage.HYPOTHESIS:
            continue
        if c.integrated_assessment is not None:
            continue
        if c.entity_id in state.verification_done:
            continue
        out.append(c)
    return out


def no_stranded_claims(
    state: "EpistemicGraphState", claims: "list[Claim]"
) -> "str | None":
    """Invariant: no claim is stranded.

    Returns ``None`` when the invariant holds, or a violation message
    listing the stranded claim IDs when it doesn't. Suitable for use as
    an entry in a node's ``post_invariants`` tuple.
    """
    stranded = stranded_claims(claims, state)
    if not stranded:
        return None
    ids = ", ".join(c.entity_id[:12] for c in stranded)
    return (
        f"{len(stranded)} claim(s) stranded at SUPPORTED with no "
        f"integration verdict and not in verification_done: {ids}. "
        "These claims should have entered the IBE chain to receive a "
        "calibrated verdict; instead they reached a terminal in a "
        "state that makes the posterior fall back to no-data 0.5."
    )


def capped_claims_carry_a_cap_evaluation(
    state: "EpistemicGraphState", claims: "list[Claim]"
) -> "str | None":
    """Invariant: every cycle-capped claim carries a ``cap_evaluation``.

    ``_mark_cycle_capped`` is the single funnel both cap-firing sites go
    through, and it unconditionally dispatches
    ``ConcludeAtCycleCapOperation``, which always writes the record. So a
    capped claim reaching a terminal *without* one means some site set
    ``cycle_capped=True`` while bypassing the funnel — exactly the
    regression shape this guards.

    This is a DIFFERENT guarantee from ``no_stranded_claims``. That one
    asks "should this claim have entered the IBE chain and didn't?", and
    correctly exempts capped claims — a capped claim is legitimately not
    an IBE candidate. This one asks "did the system look at the evidence
    before terminating this claim?", which applies precisely to the
    capped set. Do not merge them, and do not tighten
    ``stranded_claims``' capped exemption on the strength of this.

    Note this is about presence of the *record*, not presence of a
    verdict: ``cap_evaluation.verdict is None`` is a valid, informative
    outcome (the evidence was read and did not decide).
    """
    del state  # signature matches the Invariant typedef; state is unused
    missing = [
        c
        for c in claims
        if getattr(c, "cycle_capped", False)
        and getattr(c, "cap_evaluation", None) is None
    ]
    if not missing:
        return None
    ids = ", ".join(c.entity_id[:12] for c in missing)
    return (
        f"{len(missing)} cycle-capped claim(s) reached a terminal with no "
        f"cap_evaluation record: {ids}. Every cap firing goes through "
        "_mark_cycle_capped, which dispatches ConcludeAtCycleCapOperation "
        "unconditionally — so this means a site set cycle_capped=True "
        "while bypassing that funnel, and the claim's terminal outcome is "
        "unaudited."
    )
