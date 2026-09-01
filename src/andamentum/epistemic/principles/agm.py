"""AGM belief contraction — the eligibility contract (plan §4.1).

Thin by design: this module does not relocate the operations it dispatches.
Both stay exactly where they already live — two dispatch sites for one
belief-contraction mechanism (both call ``Claim.record_demotion``, per
``entities/claim.py``'s own docstring: "Both DemoteClaimOperation and
RevalidateClaimOperation call this to ensure consistent behavior regardless
of demotion cause"):

* ``RevalidateClaimOperation`` (``operations/belief_maintenance.py``) — plan
  §2's own words, "the only AGM-shaped gate in the system": dispatched from
  the TMS sweep (``graph/nodes.py:_run_tms_sweep``) when invalidated
  evidence cascades onto an already-promoted claim. This is the operation
  the fidelity canary counts as the ``agm`` signature firing
  (``docs/epistemic/FIDELITY_LEDGER.md``), and the one this module's
  ``ACTIVATION_NAME`` names.
* ``DemoteClaimOperation`` (``operations/stage_management.py``) —
  dispatched from ``graph/nodes.py:AbandonOrDemote`` when a promoted
  claim's most recent scrutiny comes back failing and nothing protects it
  (i.e. Peirce's cap has not already claimed the claim — see
  ``principles/adjudicate.py``).

Both are re-exported here so "what does AGM own" has one answer, per §4.1 —
neither import is moved; ``AbandonOrDemote`` keeps dispatching
``DemoteClaimOperation`` from its own existing import (this module does not
change that wiring, only documents it).

``eligible`` answers: has this claim's justification been undermined enough
that contraction is warranted? True when the claim has been promoted past
HYPOTHESIS and its most recent scrutiny verdict indicates the grounds for
that stage no longer hold. This is the predicate
``principles/adjudicate.py`` asks SECOND in ``AbandonOrDemote``'s
cap-vs-demotion adjudication: AGM can be — and in that call site always is —
eligible on a claim Peirce also claims; the adjudicator's fixed declared
order is what breaks the tie (Peirce first), not this function declining.

Architecture: Layer 1 (framework-agnostic; pure function, no I/O).
"""

from __future__ import annotations

from ..entities.claim import Claim, ClaimStage
from ..graph.state import EpistemicGraphState
from ..operations.belief_maintenance import RevalidateClaimOperation
from ..operations.stage_management import DemoteClaimOperation
from .eligibility import EligibilityVerdict

#: The exact operation-log name this principle's firing shows up as in
#: ``operations.jsonl``'s ``operation`` field — ``RevalidateClaimOperation``'s
#: dispatch string at its ``_run_tms_sweep`` call site (``graph/nodes.py``),
#: which is what the firing-rate canary counts as ``agm``
#: (``FIDELITY_LEDGER.md``: "a `revalidate_claim` operations row whose
#: message contains 'TMS demoted from'"). ``DemoteClaimOperation``'s own
#: dispatch string, ``"demote_claim"``, is explicitly EXCLUDED from that
#: canary signature ("NOT the scrutiny-driven `demote_claim`") — the two
#: operations are both AGM-shaped contraction, but only the TMS-sweep
#: instance is the one plan §2 identifies as starved by construction, so it
#: is the one this constant names.
ACTIVATION_NAME: str = "revalidate_claim"

#: Scrutiny verdicts that indicate a claim's current stage is no longer
#: justified — the AGM contraction trigger.
_CONTRACTING_VERDICTS: tuple[str, ...] = ("needs_resolution", "fail")


def eligible(claim: Claim, state: EpistemicGraphState) -> EligibilityVerdict:
    """True when ``claim``'s current stage is no longer justified by its
    most recent scrutiny — belief contraction is warranted.

    Pure: reads only ``claim.stage`` and ``claim.scrutiny_verdict``. No I/O,
    no side effects — testable with a bare ``Claim`` and nothing else
    running, per plan §4.1.

    ``state`` is accepted to match the shared eligibility contract every
    principle module implements; this principle's eligibility does not
    depend on it.
    """
    del state
    if (
        claim.stage != ClaimStage.HYPOTHESIS
        and claim.scrutiny_verdict in _CONTRACTING_VERDICTS
    ):
        return EligibilityVerdict(
            eligible=True,
            reason=(
                f"claim is at {claim.stage.value} with scrutiny_verdict="
                f"{claim.scrutiny_verdict!r}; its grounds for the current "
                "stage no longer hold"
            ),
        )
    if claim.stage == ClaimStage.HYPOTHESIS:
        return EligibilityVerdict(
            eligible=False,
            reason="claim is at HYPOTHESIS; nothing promoted to contract",
        )
    return EligibilityVerdict(
        eligible=False,
        reason=(
            f"scrutiny_verdict={claim.scrutiny_verdict!r} does not warrant contraction"
        ),
    )


__all__ = [
    "eligible",
    "RevalidateClaimOperation",
    "DemoteClaimOperation",
    "ACTIVATION_NAME",
]
