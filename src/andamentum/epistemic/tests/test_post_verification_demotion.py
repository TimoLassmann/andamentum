"""Regression test for the post-verification demotion routing path.

Investigates the hypothesised IBE-skip routing bug surfaced in the
Phase 2 benchmark run (commits e770e31 / d280573 era):

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

The mitigation already in place (in ``_run_tms_sweep`` at
graph/nodes.py:331-333): when revalidate demotes, the claim id is
added to ``state.claims_needing_rescrutiny`` so RunVerification's
tail routes through ResolveUncertainties → Scrutinize, restoring
the inquiry cycle. After re-scrutiny, the claim either passes
(re-routes through verification → IBE) or hits the cycle cap.

This test verifies the mitigation actually fires: when a SUPPORTED
claim gets TMS-demoted, the rescrutiny flag is set, and the routing
out of RunVerification goes to ResolveUncertainties (not to
EnumerateCandidates fast-path). If this test ever fails, the
demotion-bypasses-rescrutiny bug class has resurfaced.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
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


async def test_tms_demotion_sets_rescrutiny_flag(tmp_path: Path, fake_runner) -> None:
    """Construct the post-verification demotion scenario directly and
    assert the TMS sweep correctly flags the claim for rescrutiny.

    Setup: a SUPPORTED claim with integrated_assessment=None, a few
    pieces of evidence already invalidated. ``_run_tms_sweep`` will
    cascade those invalidations and call RevalidateClaim, which (per
    the gate logic) should demote the claim back to HYPOTHESIS
    because the evidence pool no longer satisfies the SUPPORTED gate.

    The mitigation: state.claims_needing_rescrutiny.add(claim.entity_id).
    Without this, the demoted claim sits at HYPOTHESIS with
    scrutiny_verdict=None and PromoteToSupported's idempotent check
    skips it, leading to the IBE-skip terminal path.
    """
    store = DocumentStore.for_database("post_verify_demote", db_dir=tmp_path)
    await store.initialize()
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
        adversarial_balance=0.2,  # bad balance: counter-evidence won
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
        invalidation_cascaded=False,  # ← TMS sweep should cascade
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

    # Run the TMS sweep directly (the same call RunVerification makes
    # after its track dispatches).
    await _run_tms_sweep(deps, state)

    # Assertion 1: the supporting evidence was cascaded.
    reloaded_sup = await repo.get("evidence", ev_sup.entity_id)
    assert reloaded_sup.invalidation_cascaded is True

    # Assertion 2: the claim MUST be demoted — deterministically, not
    # "either demoted or stayed SUPPORTED". adversarial_balance=0.2 is
    # below ADVERSARIAL_REFUTED_THRESHOLD (0.3), and
    # validate_current_stage (gates.py) fails the SUPPORTED gate on
    # that check alone, unconditionally, regardless of how the evidence
    # cascade above landed. Nothing in this path (InvalidateEvidence /
    # RevalidateClaim / ConcludeAtCycleCap are all documented no-LLM,
    # purely structural operations) can make the outcome vary between
    # runs, so there is exactly one correct outcome to assert (Stage 2b,
    # docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md §5: "tighten it to
    # assert the actual, now-deterministic outcome").
    reloaded_claim = await repo.get("claim", claim.entity_id)
    assert reloaded_claim.stage == ClaimStage.HYPOTHESIS, (
        f"Expected TMS to demote this claim back to HYPOTHESIS — "
        f"adversarial_balance=0.2 is below ADVERSARIAL_REFUTED_THRESHOLD "
        f"and validate_current_stage fails on that check alone — but it "
        f"stayed at {reloaded_claim.stage.value}."
    )
    # The rescrutiny flag MUST be set, otherwise PromoteToSupported
    # skips the claim and IBE never runs.
    assert claim.entity_id in state.claims_needing_rescrutiny, (
        "TMS demoted the claim back to HYPOTHESIS but did NOT add "
        "it to claims_needing_rescrutiny. This is the IBE-skip "
        "routing bug: without the rescrutiny flag, "
        "PromoteToSupported's idempotent check will skip this "
        "claim (it's no longer SUPPORTED), it'll never re-enter "
        "the inquiry cycle, and the IBE chain will never run on "
        "it. The claim stays at HYPOTHESIS forever, contributing "
        "nothing to compute_posterior, and the headline falls "
        "back to no-data 0.5."
    )
    # And verification_done MUST be cleared so PromoteToSupported
    # re-routes through verification on the way back up.
    assert claim.entity_id not in state.verification_done, (
        "TMS demoted the claim but didn't clear "
        "verification_done. PromoteToSupported's check "
        "(stage == SUPPORTED and not in verification_done) won't "
        "re-route this claim through verification when it gets "
        "re-promoted, so the second-pass evidence-cluster step "
        "won't re-cluster the now-changed evidence pool."
    )


async def test_resolve_uncertainties_routes_to_scrutiny_on_rescrutiny_flag(
    tmp_path: Path, fake_runner
) -> None:
    """Sibling assertion: when ResolveUncertainties runs with the
    rescrutiny flag set (the state TMS demotion produces), it must
    route to Scrutinize so the demoted claim's inquiry cycle continues.
    """
    store = DocumentStore.for_database("resolve_to_scrutiny", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="demoted claim",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict=None,  # cleared by record_demotion
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
    # but Scrutinize when rescrutiny is set (so the demoted claim
    # continues its inquiry cycle).
    assert isinstance(next_node, Scrutinize), (
        f"ResolveUncertainties should route to Scrutinize when "
        f"claims_needing_rescrutiny is non-empty, regardless of "
        f"next_on_clear. Got {type(next_node).__name__}. The TMS "
        "demotion-then-rescrutiny path depends on this routing — "
        "without it, demoted claims are stranded in a verdict-less "
        "HYPOTHESIS state and the IBE chain never gets a chance to "
        "run on them."
    )


# ── Stage 2b: the AGM reachability fix ───────────────────────────────
#
# docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md §5, "2b — the reachability
# fix". §2's defect: Investigate (the thing that grows a claim's
# evidence) only ever runs on HYPOTHESIS claims, so RevalidateClaimOperation
# (stage != HYPOTHESIS) could never be reached by anything that changed a
# PROMOTED claim's evidence foundation — the two preconditions are
# empirically disjoint by construction. The fix: RunVerification now adds
# a claim to state.claims_needing_tms whenever its adversarial track
# actually dispatches (the one verification track that writes fields
# validate_current_stage reads — claim.adversarial_balance and
# support_judgment on any evidence it appends — without ever setting
# evidence.invalidated=True), and _run_tms_sweep's Step 3 ("claims
# flagged for TMS by graph nodes" — pre-existing, previously had no real
# producer) now applies the same post-demotion bookkeeping Step 2
# already had.
#
# All three tests below are LLM-free by construction: RunVerification's
# AdversarialSearchOperation degrades to a clean structural no-op with
# agent_runner=None (it still marks adversarial_checked=True; it just
# never reaches the agent-calling branches), and InvalidateEvidence /
# RevalidateClaim / ConcludeAtCycleCap are all documented no-LLM,
# purely-structural operations.


async def test_run_verification_flags_promoted_claim_for_tms_after_adversarial(
    tmp_path: Path,
) -> None:
    """RunVerification must route a SUPPORTED claim into the TMS sweep
    once its adversarial track actually dispatches — not only when
    something else has already marked evidence invalidated=True. This is
    the actual reachability fix: before it, a promoted claim whose
    adversarial track just ran had no path back to
    RevalidateClaimOperation at all.
    """
    store = DocumentStore.for_database("run_verification_flags_tms", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # SUPPORTED claim, zero evidence (the SUPPORTED gate's min_evidence=1
    # will fail on revalidation), every OTHER verification track
    # pre-marked checked so "adversarial" (PRIMARY under the verificatory
    # profile) is the only one left to dispatch — isolates the mechanism
    # under test from the other six tracks.
    claim = Claim(
        objective_id=obj.entity_id,
        statement="promoted, about to lose its evidence gate",
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

    # claims_needing_tms is populated AND drained inside this one call
    # (RunVerification calls _run_tms_sweep itself), so the observable
    # proof is the revalidate_claim op firing, not the set's contents
    # after the fact.
    revalidate_ops = [
        op for op in state.operations_log if op.operation == "revalidate_claim"
    ]
    assert len(revalidate_ops) == 1, (
        "AdversarialSearchOperation dispatched but RevalidateClaimOperation "
        "never ran. This is the Stage 2b reachability defect: a promoted "
        "claim whose adversarial track just ran never reached "
        "RevalidateClaimOperation, because nothing routed it into the TMS "
        "sweep's meaningful (Step 2/Step 3) path."
    )

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.stage == ClaimStage.HYPOTHESIS, (
        "revalidate_claim ran but did not demote a claim with zero "
        "evidence, which cannot satisfy the SUPPORTED gate's min_evidence=1."
    )
    assert claim.entity_id in state.claims_needing_rescrutiny, (
        "The claim was demoted but not queued for rescrutiny — it would "
        "be stranded at HYPOTHESIS with scrutiny_verdict=None forever."
    )
    assert isinstance(next_node, ResolveUncertainties), (
        f"A claim just demoted mid-pass must route through "
        f"ResolveUncertainties (which itself routes to Scrutinize while "
        f"claims_needing_rescrutiny is non-empty), not fast-path past it. "
        f"Got {type(next_node).__name__}."
    )


async def test_tms_step3_demotion_gets_the_same_bookkeeping_as_step2(
    tmp_path: Path,
) -> None:
    """_run_tms_sweep's Step 3 must apply the SAME post-demotion
    bookkeeping Step 2 already has. Isolates Step 3 from Step 2
    deliberately: no evidence is invalidated=True anywhere in this
    fixture, so Step 1 cascades nothing, ``had_cascades`` stays False,
    and Step 2's own "revalidate every promoted claim" loop never runs —
    the only thing that can revalidate this claim is Step 3, driven
    purely by state.claims_needing_tms (what RunVerification's fix
    populates). Before this stage, Step 3 ran RevalidateClaimOperation
    but never followed up on a successful demotion — a claim demoted
    through Step 3 alone would have been stranded (record_demotion clears
    scrutiny_verdict, and nothing would route it back through Scrutinize
    to re-establish one).
    """
    store = DocumentStore.for_database("tms_step3_bookkeeping", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="flagged for TMS with no evidence to invalidate",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[],
        evidence_count=0,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.verification_done.add(claim.entity_id)
    # The signal RunVerification's fix produces — set directly here to
    # isolate Step 3 from the node body that populates it.
    state.claims_needing_tms.add(claim.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    await _run_tms_sweep(deps, state)

    # No evidence was ever invalidated: Step 1 cascaded nothing and
    # Step 2's all-claims loop never ran — confirms the demotion below
    # came from Step 3 alone.
    assert [
        op for op in state.operations_log if op.operation == "invalidate_evidence"
    ] == []
    revalidate_ops = [
        op for op in state.operations_log if op.operation == "revalidate_claim"
    ]
    assert len(revalidate_ops) == 1

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.stage == ClaimStage.HYPOTHESIS
    assert claim.entity_id in state.claims_needing_rescrutiny, (
        "Step 3 demoted the claim but did not add it to "
        "claims_needing_rescrutiny — the same stranding bug Step 2's "
        "bookkeeping already guards against, now reachable through "
        "Step 3 too."
    )
    assert claim.entity_id not in state.verification_done, (
        "Step 3 demoted the claim but did not discard it from "
        "verification_done — PromoteToSupported would skip re-routing it "
        "through ClusterEvidence/RunVerification on re-promotion."
    )
    assert state.claims_needing_tms == set()


async def test_new_tms_path_shares_the_same_cycle_cap(tmp_path: Path) -> None:
    """L5 (bounded loops): the demote -> rescrutinise path RunVerification's
    fix newly makes reachable through _run_tms_sweep's Step 3 must be
    bounded by the SAME state.scrutiny_resolve_cycles /
    SCRUTINY_RESOLVE_CYCLE_CAP that already bounds every other
    Peirce-cycling path (Investigate, ResolveUncertainties, TMS Step 2) —
    there is no separate counter for it, and nothing about Step 3's new
    bookkeeping (added in this stage) creates one.

    Drives a claim through SCRUTINY_RESOLVE_CYCLE_CAP rounds of "TMS
    Step 3 demotes it, adds it to claims_needing_rescrutiny" — each round
    advances state.scrutiny_resolve_cycles exactly the way Scrutinize's
    own rescrutiny branch does, without dispatching
    ScrutiniseClaimOperation (which needs a model — this test is pinning
    the CAP's boundedness, not the scrutiny LLM call, so no LLM call
    happens anywhere in it). It then confirms Scrutinize's own
    (pre-existing, unmodified) cap-check branch recognises the count and
    cycle-caps the claim on the (CAP + 1)-th attempt, via the same
    defense-in-depth branch that already guards the Investigate path and
    the TMS Step 2 path (test_tms_demote_repromote.py).
    """
    store = DocumentStore.for_database("tms_step3_shares_cap", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="repeatedly TMS-demoted via the new Step 3 path",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        evidence_ids=[],
        evidence_count=0,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    for round_num in range(1, SCRUTINY_RESOLVE_CYCLE_CAP + 1):
        # Re-promote directly (no LLM — this test isolates the cap, not
        # promotion) so each round starts from the SUPPORTED state
        # RunVerification's adversarial-track fix would flag.
        current = await repo.get("claim", claim.entity_id)
        current.stage = ClaimStage.SUPPORTED
        current.scrutiny_verdict = "pass"
        await repo.save(current)

        # The new path: RunVerification would add this claim to
        # claims_needing_tms; _run_tms_sweep's Step 3 revalidates,
        # demotes (zero evidence never satisfies the SUPPORTED gate),
        # and — the bookkeeping this stage adds — flags it for
        # rescrutiny.
        state.claims_needing_tms.add(claim.entity_id)
        await _run_tms_sweep(deps, state)
        assert claim.entity_id in state.claims_needing_rescrutiny

        cycles = state.scrutiny_resolve_cycles.get(claim.entity_id, 0)
        assert cycles < SCRUTINY_RESOLVE_CYCLE_CAP, (
            f"round {round_num}: already at the cap before Scrutinize's "
            "own bookkeeping would run on it — the loop under test ended "
            "earlier than SCRUTINY_RESOLVE_CYCLE_CAP rounds."
        )
        # Scrutinize's own (unmodified) rescrutiny bookkeeping: increment
        # the shared counter and clear the rescrutiny flag. Replicated
        # here rather than calling Scrutinize().run() directly because
        # the under-cap branch falls through to ScrutiniseClaimOperation,
        # which needs a real agent — this test pins the CAP, not
        # scrutiny's own LLM call.
        state.scrutiny_resolve_cycles[claim.entity_id] = cycles + 1
        state.claims_needing_rescrutiny.discard(claim.entity_id)

    # One more round: SUPPORTED again, flagged again — but
    # scrutiny_resolve_cycles is now AT the cap.
    current = await repo.get("claim", claim.entity_id)
    current.stage = ClaimStage.SUPPORTED
    current.scrutiny_verdict = "pass"
    await repo.save(current)
    state.claims_needing_tms.add(claim.entity_id)
    await _run_tms_sweep(deps, state)
    assert claim.entity_id in state.claims_needing_rescrutiny
    assert (
        state.scrutiny_resolve_cycles.get(claim.entity_id, 0)
        >= SCRUTINY_RESOLVE_CYCLE_CAP
    )

    # Scrutinize's pre-existing, unmodified cap-check must now refuse
    # further rescrutiny — deterministically, no LLM call (the
    # cap-exceeded branch `continue`s before ScrutiniseClaimOperation).
    await Scrutinize().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    reloaded = await repo.get("claim", claim.entity_id)
    assert reloaded.cycle_capped is True, (
        "The new TMS Step 3 -> rescrutiny path was not stopped by "
        "SCRUTINY_RESOLVE_CYCLE_CAP — it is a separate, unbounded loop, "
        "not one that shares the existing Peirce-cycling cap."
    )
    assert claim.entity_id not in state.claims_needing_rescrutiny
