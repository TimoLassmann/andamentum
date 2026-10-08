"""Regression tests for the post-verification revalidation routing path.

Originally investigated the hypothesised IBE-skip routing bug surfaced in
the Phase 2 benchmark run (commits e770e31 / d280573 era):

  Suspected path:
    soft_promote (SUPPORTED, integrated_assessment=None)
    → PromoteToSupported (routes to ClusterEvidence)
    → ClusterEvidence (HDBSCAN clustering)
    → RunVerification (adversarial / convergence / etc fire)
    → TMS sweep: adversarial counter-evidence invalidates support;
       RevalidateClaim demotes claim back to HYPOTHESIS, clears
       scrutiny_verdict
    → ??? (the suspected bug: IBE chain fails to run on the now-
       HYPOTHESIS claim because EnumerateCandidates only fires on
       stage == SUPPORTED)

The original mitigation (``_run_tms_sweep`` demoting mid-verification and
flagging ``state.claims_needing_rescrutiny`` so the claim re-cycled through
Scrutinize before IBE) has been SUPERSEDED by certify-before-contract
(docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md D1): the defect that
mitigation was routing around — a claim's stage revision racing IBE for
the same claim, sometimes winning and erasing the verdict IBE was about
to produce — is now avoided at the source. ``_run_tms_sweep`` cascades
evidence invalidation ONLY (Step 1); it no longer revalidates or contracts
claims. The revalidate-then-contract check that used to run mid-verification
now runs exactly once, in ``PromoteSupported``, AFTER
``SelectBestExplanation`` has already produced this run's IBE verdict — so
a claim is certified before its stage is ever revised, and a contraction
preserves rather than restarts the claim's verdict (``Claim.contract``,
not ``record_demotion``). The graph-level tests for that new path (S1, S2,
S3, S4) live in ``test_certify_before_contract.py``.

The tests below now pin the NEGATIVE side of that change at the two old
call sites: ``_run_tms_sweep`` and ``RunVerification`` no longer revalidate
or contract a claim reactively mid-verification. The
``ResolveUncertainties`` routing test (unrelated to TMS — it exercises the
generic rescrutiny-flag guard that ``Investigate`` still populates) is
unchanged.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import Store
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    ResolveUncertainties,
    RunVerification,
    Scrutinize,
    _run_tms_sweep,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def test_tms_sweep_cascades_but_does_not_contract(
    tmp_path: Path, fake_runner
) -> None:
    """``_run_tms_sweep`` cascades invalidated evidence (Step 1, unchanged)
    but must NOT revalidate or contract the claim any more — that check
    moved to ``PromoteSupported`` (certify-before-contract D1).

    Setup: a SUPPORTED claim with integrated_assessment=None, a few
    pieces of evidence already invalidated, and an adversarial_balance
    that would fail ``validate_current_stage`` if revalidated. Before
    this stage, ``_run_tms_sweep`` would have demoted the claim to
    HYPOTHESIS and flagged it for rescrutiny; now it must leave the
    claim's stage, ``claims_needing_rescrutiny``, and
    ``verification_done`` untouched — only the evidence cascade fires.
    """
    store = await Store.open(tmp_path / "post_verify_demote.db")
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # SUPPORTED claim — soft-promoted, no integration verdict yet.
    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim under test",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",  # passed scrutiny prior to TMS
        integrated_assessment=None,
        adversarial_balance=0.2,  # bad balance: would fail revalidation
    )
    await repo.save(claim)

    # Evidence: one supporting (about to be invalidated), one
    # contradicting (the adversarial result that survived).
    ev_sup = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup",
        extracted_content="weak support",
        extracted=True,
        support_judgment="supports",
        invalidated=True,  # adversarial sweep already invalidated this
        invalidation_cascaded=False,  # ← Step 1 should cascade this
    )
    ev_con = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con",
        extracted_content="strong counter",
        extracted=True,
        support_judgment="contradicts",
    )
    await repo.save(ev_sup)
    await repo.save(ev_con)
    claim.evidence_ids = [ev_sup.entity_id, ev_con.entity_id]
    claim.evidence_count = 2
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.verification_done.add(claim.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    await _run_tms_sweep(deps, state)

    # Step 1 still cascades the invalidated evidence.
    reloaded_sup = await repo.get("evidence", ev_sup.entity_id)
    assert reloaded_sup.invalidation_cascaded is True

    # No revalidate_claim op fired — the sweep no longer touches claims.
    revalidate_ops = [
        op for op in state.operations_log if op.operation == "revalidate_claim"
    ]
    assert revalidate_ops == [], (
        "_run_tms_sweep dispatched revalidate_claim — Steps 2/3 were "
        "supposed to be removed under certify-before-contract D1. The "
        "revalidate-then-contract check belongs solely to PromoteSupported "
        "now, which runs it after IBE, not reactively mid-verification."
    )

    # The claim's stage is untouched: nothing revalidated it, so nothing
    # could have contracted it, regardless of how badly its evidence
    # would fare against the SUPPORTED gate.
    reloaded_claim = await repo.get("claim", claim.entity_id)
    assert reloaded_claim.stage == ClaimStage.SUPPORTED, (
        f"_run_tms_sweep contracted this claim to "
        f"{reloaded_claim.stage.value} — Step 1 (cascade) is the only "
        "thing left in this function; stage revision is PromoteSupported's "
        "job now, and only after IBE has run."
    )
    assert reloaded_claim.contraction_count == 0

    # Neither rescrutiny nor verification_done bookkeeping fires — there
    # is no contraction here for them to react to.
    assert claim.entity_id not in state.claims_needing_rescrutiny
    assert claim.entity_id in state.verification_done


async def test_resolve_uncertainties_routes_to_scrutiny_on_rescrutiny_flag(
    tmp_path: Path, fake_runner
) -> None:
    """Sibling assertion, unrelated to TMS: when ResolveUncertainties runs
    with the rescrutiny flag set (the state ``Investigate`` still
    produces on new evidence), it must route to Scrutinize so the
    flagged claim's inquiry cycle continues.
    """
    store = await Store.open(tmp_path / "resolve_to_scrutiny.db")
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim needing rescrutiny",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="needs_resolution",
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.claims_needing_rescrutiny.add(claim.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    next_node = await ResolveUncertainties(next_on_clear="integrate").run(
        _FakeRunContext(state, deps)  # type: ignore[arg-type]
    )

    # No blocking uncertainties + claims_needing_rescrutiny set →
    # Scrutinize. The "no blocking uncertainties" branch returns
    # PromoteToSupported / EnumerateCandidates if rescrutiny is empty,
    # but Scrutinize when rescrutiny is set (so the flagged claim
    # continues its inquiry cycle).
    assert isinstance(next_node, Scrutinize), (
        f"ResolveUncertainties should route to Scrutinize when "
        f"claims_needing_rescrutiny is non-empty, regardless of "
        f"next_on_clear. Got {type(next_node).__name__}. Claims flagged "
        "for rescrutiny must not be stranded in a verdict-less state "
        "before the IBE chain gets a chance to run on them."
    )


# ── The certify-before-contract negative: RunVerification no longer
# contracts a claim mid-verification ──────────────────────────────────
#
# docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md D1. Before this stage,
# RunVerification flagged a claim into a now-deleted
# `claims_needing_tms` state set (G3) the moment its adversarial track
# dispatched, and _run_tms_sweep's Step 3
# then revalidated (and could contract) it in the SAME pass — racing
# IBE, which had not run yet, for that claim's verdict. That flagging
# and Step 3's consumption of it are both gone: the revalidate-then-
# contract check now runs exactly once, in PromoteSupported, after IBE.


async def test_run_verification_no_longer_contracts_after_adversarial(
    tmp_path: Path,
) -> None:
    """RunVerification's adversarial track must dispatch as before, but
    must NOT cause a revalidate_claim op or a stage contraction any more
    — that used to be the (buggy, pre-IBE) reachability fix; it is now
    PromoteSupported's job, run once, after IBE.
    """
    store = await Store.open(tmp_path / "run_verification_flags_tms.db")
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # SUPPORTED claim, zero evidence (the SUPPORTED gate's min_evidence=1
    # would fail on revalidation, if anything revalidated it), every
    # OTHER verification track pre-marked checked so "adversarial"
    # (PRIMARY under the verificatory profile) is the only one left to
    # dispatch — isolates the mechanism under test from the other six
    # tracks.
    claim = Claim(
        objective_id=obj.entity_id,
        statement="promoted, evidence-thin claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[],
        evidence_count=0,
        adversarial_checked=False,
        convergence_checked=True,
        deductive_checked=True,
        computational_checked=True,
        argument_analyzed=True,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await RunVerification().run(
        _FakeRunContext(state, deps)  # type: ignore[arg-type]
    )

    adversarial_ops = [
        op for op in state.operations_log if op.operation == "adversarial_search"
    ]
    assert len(adversarial_ops) == 1

    revalidate_ops = [
        op for op in state.operations_log if op.operation == "revalidate_claim"
    ]
    assert revalidate_ops == [], (
        "AdversarialSearchOperation dispatched and RevalidateClaimOperation "
        "also ran — RunVerification (directly or via _run_tms_sweep) must "
        "not revalidate/contract a claim any more. Certify-before-contract "
        "moved that check to PromoteSupported, which runs after IBE."
    )

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.stage == ClaimStage.SUPPORTED, (
        "The claim's stage was revised mid-verification — nothing in "
        "RunVerification's path is allowed to do that any more."
    )
    assert reloaded.contraction_count == 0
    assert claim.entity_id not in state.claims_needing_rescrutiny, (
        "Nothing here contracted the claim, so nothing should have "
        "queued it for rescrutiny."
    )
    # With no rescrutiny flag and convergence_verdict never set (the
    # convergence track is pre-marked checked=True but never actually
    # produced a terminal verdict here), RunVerification's own fallthrough
    # — not the rescrutiny guard — routes to ResolveUncertainties.
    assert isinstance(next_node, ResolveUncertainties), (
        f"Expected the plain fallthrough to ResolveUncertainties (no "
        f"blocking uncertainties, no terminal convergence verdict). Got "
        f"{type(next_node).__name__}."
    )
