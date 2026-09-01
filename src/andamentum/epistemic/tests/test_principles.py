"""Tier-1 contract tests for ``epistemic.principles`` (plan §4.1-4.2).

Plan §4.2b: "Tier 1 needs no fake because the plan's own design made the
contracts pure... Construct a ``Claim``, assert the ``EligibilityVerdict``.
Nothing to mock, because nothing is called. This is the majority of the new
tests." Every test here constructs a bare ``Claim`` and calls a pure
function directly — no repo, no agent, no graph, no event loop beyond what
``asyncio_mode = auto`` gives every test file for free.

These tests pin two things Stage 2a claims:

1. Each principle module's ``eligible()`` is total (always returns an
   ``EligibilityVerdict`` with a non-empty ``reason``, on both the True and
   False path — plan §4.1's "never a bare boolean").
2. ``adjudicate()`` reproduces the CURRENT, unchanged "cap wins over
   demotion" policy from ``graph/nodes.py:AbandonOrDemote`` exactly: when a
   claim is both cycle-capped (Peirce) and has failing scrutiny at
   SUPPORTED+ (AGM), Peirce wins — and the losing principle's reason is
   still visible in ``AdjudicationDecision.considered``, never dropped.
"""

from __future__ import annotations

from andamentum.epistemic.entities.claim import Claim, ClaimStage
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.belief_maintenance import (
    RevalidateClaimOperation,
)
from andamentum.epistemic.operations.stage_management import (
    ConcludeAtCycleCapOperation,
    DemoteClaimOperation,
)
from andamentum.epistemic.principles import agm, peirce
from andamentum.epistemic.principles.adjudicate import AdjudicationDecision, adjudicate
from andamentum.epistemic.principles.eligibility import EligibilityVerdict


def _claim(
    *,
    stage: ClaimStage = ClaimStage.HYPOTHESIS,
    scrutiny_verdict: str | None = None,
    cycle_capped: bool = False,
) -> Claim:
    claim = Claim(
        objective_id="obj1",
        statement="a claim",
        scope="x",
        stage=stage,
        scrutiny_verdict=scrutiny_verdict,
    )
    claim.cycle_capped = cycle_capped
    return claim


_STATE = EpistemicGraphState(objective_id="obj1")


# ── peirce.eligible ─────────────────────────────────────────────────────


def test_peirce_eligible_when_cycle_capped() -> None:
    claim = _claim(
        stage=ClaimStage.SUPPORTED, scrutiny_verdict="fail", cycle_capped=True
    )
    verdict = peirce.eligible(claim, _STATE)
    assert isinstance(verdict, EligibilityVerdict)
    assert verdict.eligible is True
    assert "cycle-capped" in verdict.reason
    assert "supported" in verdict.reason


def test_peirce_ineligible_when_not_cycle_capped() -> None:
    claim = _claim(
        stage=ClaimStage.SUPPORTED, scrutiny_verdict="fail", cycle_capped=False
    )
    verdict = peirce.eligible(claim, _STATE)
    assert verdict.eligible is False
    assert verdict.reason  # never empty — plan §4.1: never a bare boolean
    assert "not cycle-capped" in verdict.reason


def test_peirce_reexports_conclude_at_cycle_cap_operation_unmoved() -> None:
    """The operation is re-exported, not relocated — same class object."""
    assert peirce.ConcludeAtCycleCapOperation is ConcludeAtCycleCapOperation


def test_peirce_activation_name_is_the_operation_log_string() -> None:
    assert peirce.ACTIVATION_NAME == "conclude_at_cycle_cap"


# ── agm.eligible ─────────────────────────────────────────────────────────


def test_agm_eligible_when_supported_and_scrutiny_fails() -> None:
    claim = _claim(stage=ClaimStage.SUPPORTED, scrutiny_verdict="fail")
    verdict = agm.eligible(claim, _STATE)
    assert verdict.eligible is True
    assert "supported" in verdict.reason
    assert "fail" in verdict.reason


def test_agm_eligible_when_supported_and_needs_resolution() -> None:
    claim = _claim(stage=ClaimStage.SUPPORTED, scrutiny_verdict="needs_resolution")
    verdict = agm.eligible(claim, _STATE)
    assert verdict.eligible is True


def test_agm_ineligible_at_hypothesis() -> None:
    """Nothing has been promoted yet — there is nothing to contract."""
    claim = _claim(stage=ClaimStage.HYPOTHESIS, scrutiny_verdict="fail")
    verdict = agm.eligible(claim, _STATE)
    assert verdict.eligible is False
    assert verdict.reason
    assert "HYPOTHESIS" in verdict.reason


def test_agm_ineligible_when_scrutiny_passes() -> None:
    claim = _claim(stage=ClaimStage.SUPPORTED, scrutiny_verdict="pass")
    verdict = agm.eligible(claim, _STATE)
    assert verdict.eligible is False
    assert verdict.reason


def test_agm_ineligible_when_scrutiny_verdict_is_none() -> None:
    claim = _claim(stage=ClaimStage.SUPPORTED, scrutiny_verdict=None)
    verdict = agm.eligible(claim, _STATE)
    assert verdict.eligible is False


def test_agm_reexports_both_dispatch_sites_unmoved() -> None:
    """Both operations AGM owns are re-exported, not relocated — same
    class objects at their original homes (belief_maintenance.py /
    stage_management.py)."""
    assert agm.RevalidateClaimOperation is RevalidateClaimOperation
    assert agm.DemoteClaimOperation is DemoteClaimOperation


def test_agm_activation_name_is_the_canary_counted_signature() -> None:
    """FIDELITY_LEDGER.md's ``agm`` signature counts ``revalidate_claim``
    rows specifically, excluding the scrutiny-driven ``demote_claim`` —
    ACTIVATION_NAME names the one the canary counts."""
    assert agm.ACTIVATION_NAME == "revalidate_claim"


# ── adjudicate() — the cap-wins-over-demotion policy, named ─────────────


def test_adjudicate_peirce_wins_when_both_eligible() -> None:
    """The exact conflict plan §2 documents: a claim that is BOTH
    cycle-capped (Peirce) and has failing scrutiny at SUPPORTED+ (AGM).
    Peirce wins — this is the current policy, unchanged."""
    claim = _claim(
        stage=ClaimStage.SUPPORTED, scrutiny_verdict="fail", cycle_capped=True
    )
    decision = adjudicate(claim, _STATE)
    assert isinstance(decision, AdjudicationDecision)
    assert decision.winner == "peirce"
    assert "cycle-capped" in decision.reason

    # AGM's own verdict is preserved even though it lost — the losing
    # principle's reason must never silently disappear.
    assert set(decision.considered) == {"peirce", "agm"}
    assert decision.considered["peirce"].eligible is True
    assert decision.considered["agm"].eligible is True
    assert decision.considered["agm"].reason  # AGM's reason survives the loss


def test_adjudicate_agm_wins_when_not_cycle_capped() -> None:
    """Sanity mirror of test_graph_reachability.py's
    test_demoted_claim_routes_to_rescrutiny: SUPPORTED + fail, not capped
    -> AGM wins, and demotion is the correct action."""
    claim = _claim(
        stage=ClaimStage.SUPPORTED, scrutiny_verdict="fail", cycle_capped=False
    )
    decision = adjudicate(claim, _STATE)
    assert decision.winner == "agm"
    assert decision.considered["peirce"].eligible is False
    assert decision.considered["agm"].eligible is True


def test_adjudicate_no_winner_when_neither_eligible() -> None:
    """Outside AbandonOrDemote's own guard (which only calls adjudicate()
    once scrutiny has already failed at stage != HYPOTHESIS), neither
    principle need be eligible — adjudicate() must say so explicitly
    rather than silently picking one."""
    claim = _claim(
        stage=ClaimStage.SUPPORTED, scrutiny_verdict="pass", cycle_capped=False
    )
    decision = adjudicate(claim, _STATE)
    assert decision.winner is None
    assert decision.reason
    assert decision.considered["peirce"].eligible is False
    assert decision.considered["agm"].eligible is False


def test_adjudicate_considered_preserves_every_principle_even_when_moot() -> None:
    """Even a decisive, single-eligible case reports both principles'
    verdicts in ``considered`` — the record is never partial."""
    claim = _claim(stage=ClaimStage.HYPOTHESIS, scrutiny_verdict="fail")
    decision = adjudicate(claim, _STATE)
    assert decision.winner is None
    assert len(decision.considered) == 2
