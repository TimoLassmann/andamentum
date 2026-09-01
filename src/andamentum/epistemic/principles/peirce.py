"""Peirce's cycle-cap termination — the eligibility contract (plan §4.1).

Thin by design: this module does not relocate the operation it dispatches.
``ConcludeAtCycleCapOperation`` stays exactly where it lives
(``operations/stage_management.py``), dispatched through the shared
``_mark_cycle_capped`` funnel in ``graph/nodes.py``, which both cap-firing
sites — ``ResolveUncertainties`` (primary loop-breaker) and ``Scrutinize``
(defense in depth) — call. This module exports exactly three things:
``eligible``, the re-exported operation, and ``ACTIVATION_NAME``.

``eligible`` answers one question: does Peirce's cap already hold a
terminal, evidence-derived verdict over this claim that must not be
erased? That is True exactly when ``claim.cycle_capped`` is set —
``ConcludeAtCycleCapOperation`` has already run (via ``_mark_cycle_capped``)
and recorded the verdict its evidence supports (``Claim.cap_evaluation``).
This is the predicate ``principles/adjudicate.py`` asks FIRST when
``graph/nodes.py:AbandonOrDemote`` adjudicates a cap-vs-demotion conflict —
asking it first, and treating a True answer as decisive, is what encodes
the current "cap wins over demotion" policy (plan §2) as data instead of a
buried conditional. This module does not change that policy.

Architecture: Layer 1 (framework-agnostic; pure function, no I/O).
"""

from __future__ import annotations

from ..entities.claim import Claim
from ..graph.state import EpistemicGraphState
from ..operations.stage_management import ConcludeAtCycleCapOperation
from .eligibility import EligibilityVerdict

#: The exact operation-log name this principle's firing shows up as in
#: ``operations.jsonl``'s ``operation`` field — the literal string passed to
#: ``_run_op`` at the ``_mark_cycle_capped`` call site
#: (``graph/nodes.py``), so the firing-rate canary (plan §4.3) can count it.
ACTIVATION_NAME: str = "conclude_at_cycle_cap"


def eligible(claim: Claim, state: EpistemicGraphState) -> EligibilityVerdict:
    """True when Peirce's cap already holds a terminal verdict over ``claim``.

    Pure: reads only ``claim.cycle_capped`` and ``claim.stage`` (for the
    reason string). No I/O, no side effects — testable with a bare ``Claim``
    and nothing else running, per plan §4.1.

    ``state`` is accepted to match the shared eligibility contract every
    principle module implements (a caller dispatches all principles the
    same way); this principle's eligibility does not depend on it.
    """
    del state
    if claim.cycle_capped:
        return EligibilityVerdict(
            eligible=True,
            reason=(
                f"claim is cycle-capped at {claim.stage.value}; a "
                "cap-derived terminal verdict already exists "
                "(ConcludeAtCycleCapOperation, see Claim.cap_evaluation) "
                "and must not be erased by demotion"
            ),
        )
    return EligibilityVerdict(
        eligible=False,
        reason="claim is not cycle-capped",
    )


__all__ = ["eligible", "ConcludeAtCycleCapOperation", "ACTIVATION_NAME"]
