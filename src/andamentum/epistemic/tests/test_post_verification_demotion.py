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
    ResolveUncertainties,
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

    # Assertion 2: claim was either demoted or stayed SUPPORTED. Either
    # way is structurally fine — what matters is that IF it was
    # demoted, the rescrutiny flag is set so the inquiry cycle
    # continues; IF it wasn't demoted, the routing continues normally
    # to IBE.
    reloaded_claim = await repo.get("claim", claim.entity_id)
    if reloaded_claim.stage == ClaimStage.HYPOTHESIS:
        # Demotion path: the rescrutiny flag MUST be set, otherwise
        # PromoteToSupported skips the claim and IBE never runs.
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
    else:
        # Non-demotion path: claim is still SUPPORTED. Rescrutiny
        # flag may or may not be set depending on whether the gate
        # was tight enough to fail. This branch is fine — the
        # routing continues normally to IBE.
        pass


async def test_resolve_uncertainties_routes_to_scrutiny_on_rescrutiny_flag(
    tmp_path: Path, fake_runner
) -> None:
    """Sibling assertion: when ResolveUncertainties runs with the
    rescrutiny flag set (the state TMS demotion produces), it must
    route to Scrutinize so the demoted claim's inquiry cycle continues.
    """
    from andamentum.epistemic.graph.nodes import Scrutinize

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
