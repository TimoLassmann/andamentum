"""Named adjudication between competing principles (plan §4.2).

Where two principles compete for the same claim, the decision becomes a
standalone, tested function instead of a buried conditional. Today that is
exactly the Peirce-cap-vs-AGM-demotion conflict in
``graph/nodes.py:AbandonOrDemote``.

``adjudicate()`` asks every principle, in the fixed order declared below,
whether it wants a claim. The FIRST principle whose ``eligible()`` returns
True wins. Checking Peirce before AGM encodes the CURRENT "cap wins over
demotion" policy — a cycle-capped claim's cap-derived verdict must never be
erased by demotion — without changing it: this diff adds visibility, not
reachability. Every principle's verdict is preserved in the returned
``AdjudicationDecision.considered``, so the losing principle's ``reason`` is
never silently dropped — a claim's adjudication history can show "AGM
wanted to demote; Peirce's cap took precedence because X."

Pure: no I/O, no operation dispatch. ``AbandonOrDemote`` (and any future
node with the same shape) calls this, then ACTS on the
``AdjudicationDecision`` — dispatching the winning principle's operation is
the caller's job, not this module's (P1: operations are pure transforms;
P2: the graph is the sole flow controller).

Architecture: Layer 1 (framework-agnostic; pure function, no I/O).
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from ..entities.claim import Claim
from ..graph.state import EpistemicGraphState
from . import agm, peirce
from .eligibility import EligibilityVerdict


class AdjudicationDecision(BaseModel):
    """The outcome of asking every principle whether it wants a claim.

    ``winner`` is ``None`` when no principle claimed eligibility — a case
    the caller must handle explicitly rather than silently doing nothing:
    a claim scrutiny already flagged as failing, that no principle wants to
    act on, is a fact worth failing loud over, not a no-op.
    """

    winner: str | None = Field(
        description=(
            "Name of the principle that won the adjudication (e.g. "
            "'peirce', 'agm'), or None if no principle claimed eligibility."
        )
    )
    reason: str = Field(
        min_length=1,
        description=(
            "The winning principle's EligibilityVerdict.reason, or — when "
            "winner is None — an explanation that nothing claimed the claim."
        ),
    )
    considered: dict[str, EligibilityVerdict] = Field(
        description=(
            "Every principle's EligibilityVerdict, keyed by name, in the "
            "declared order asked. Preserves the losing principle's reason "
            "even when it did not win."
        )
    )


# Fixed declared order (plan §4.2): Peirce is asked before AGM. This
# ordering IS the "cap wins over demotion" policy from
# graph/nodes.py:AbandonOrDemote's original comment — encoded here as data,
# not changed: a cycle-capped claim's cap-derived verdict must not be
# erased by demotion (record_demotion resets scrutiny_verdict and nulls
# integrated_assessment / confidence / reasoning, which for a capped claim
# hold the terminal verdict ConcludeAtCycleCapOperation already derived
# from the evidence).
_PRINCIPLES: tuple[
    tuple[str, Callable[[Claim, EpistemicGraphState], EligibilityVerdict]], ...
] = (
    ("peirce", peirce.eligible),
    ("agm", agm.eligible),
)


def adjudicate(claim: Claim, state: EpistemicGraphState) -> AdjudicationDecision:
    """Ask every principle, in the fixed order above, whether it wants ``claim``.

    Pure: no I/O, no operation dispatch — just calls each principle's
    ``eligible()`` and picks a winner. Returns the first eligible
    principle's name as ``winner`` (or ``None`` if none are eligible), plus
    every principle's verdict in ``considered``.
    """
    considered: dict[str, EligibilityVerdict] = {
        name: eligible_fn(claim, state) for name, eligible_fn in _PRINCIPLES
    }

    for name, _eligible_fn in _PRINCIPLES:
        verdict = considered[name]
        if verdict.eligible:
            return AdjudicationDecision(
                winner=name, reason=verdict.reason, considered=considered
            )

    return AdjudicationDecision(
        winner=None,
        reason="no principle claimed eligibility for this claim",
        considered=considered,
    )


__all__ = ["AdjudicationDecision", "adjudicate"]
