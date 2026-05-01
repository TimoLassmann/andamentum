"""Regression test: soft-promoted claims must reach the IBE chain.

The bug, found by running the system on a real query (the
intermittent-fasting case): when scrutiny fails, investigation
exhausts, refute-promotion declines, and the claim soft-promotes —
the AbandonOrDemote node added the claim to ``state.verification_done``
and returned to ``CheckCompletion``. That short-circuited the
verification path, whose terminal step is the IBE chain
(EnumerateCandidates → ScoreLoveliness → ScoreLikeliness →
SelectBestExplanation). Result: a SUPPORTED claim with
``integrated_assessment=None`` made it to Synthesize, ``compute_posterior``
fell back to its no-data 0.5 prior, and the report's headline number
disagreed with its own narrative prose.

The contract divergence:

    refute-promote → integrated_assessment="contradicts" (set by op)
                  → verification_done.add() is correct (skip IBE)
    soft-promote  → integrated_assessment=None (deferred to IBE)
                  → verification_done.add() is WRONG (IBE must run)

This test exercises the AbandonOrDemote node with a HYPOTHESIS claim
that has directional evidence and 3 investigation rounds, then asserts:

  1. Soft-promote ran (the precondition).
  2. AbandonOrDemote returned PromoteToSupported (the routing fix).
  3. The claim is NOT in verification_done (the flag fix).

Together those three guarantee the soft-promoted claim continues into
ClusterEvidence → RunVerification → EnumerateCandidates and gets a
real integration verdict.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import AbandonOrDemote, PromoteToSupported
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def test_soft_promote_routes_to_promote_to_supported(
    tmp_path: Path, fake_runner
) -> None:
    """The end-to-end routing assertion: a HYPOTHESIS claim with
    directional evidence that exhausts investigation must, on
    soft-promotion, route to PromoteToSupported (which feeds IBE)."""
    store = DocumentStore.for_database("soft_promote_ibe", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="some claim",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="needs_resolution",
        # No integration verdict — this is what soft-promote leaves and
        # what IBE is supposed to populate.
        integrated_assessment=None,
    )
    await repo.save(claim)

    # Two pieces of evidence: one supporting, one contradicting. This
    # gives soft-promote something to work with (n_sup + n_con > 0)
    # but doesn't pass the refutation threshold.
    ev_sup = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/sup",
        extracted_content="supports",
        extracted=True,
        support_judgment="supports",
    )
    ev_con = Evidence(
        objective_id=obj.entity_id,
        source_type="web",
        source_ref="https://ex.com/con",
        extracted_content="contradicts",
        extracted=True,
        support_judgment="contradicts",
    )
    await repo.save(ev_sup)
    await repo.save(ev_con)
    claim.evidence_ids = [ev_sup.entity_id, ev_con.entity_id]
    claim.evidence_count = 2
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.investigation_counts[claim.entity_id] = 3  # exhausted
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    ctx = _FakeRunContext(state, deps)
    next_node = await AbandonOrDemote().run(ctx)  # type: ignore[arg-type]

    # 1. Soft-promote ran (the precondition for the routing fix to matter).
    soft_calls = [
        op for op in state.operations_log if op["operation"] == "soft_promote"
    ]
    assert len(soft_calls) == 1, (
        f"Expected one soft_promote call, got {len(soft_calls)}: {soft_calls}"
    )
    assert soft_calls[0]["success"], (
        f"Soft promote should have succeeded with directional evidence: "
        f"{soft_calls[0]}"
    )

    # 2. The routing fix: AbandonOrDemote returned PromoteToSupported,
    #    not CheckCompletion. This is what feeds the IBE chain.
    assert isinstance(next_node, PromoteToSupported), (
        f"After soft-promote, AbandonOrDemote should return "
        f"PromoteToSupported (which routes to ClusterEvidence → "
        f"RunVerification → EnumerateCandidates → IBE). Got "
        f"{type(next_node).__name__} instead — the soft-promoted "
        f"claim would terminate without an integration verdict and "
        f"compute_posterior would fall back to no-data 0.5."
    )

    # 3. The flag fix: the soft-promoted claim must NOT be in
    #    verification_done, otherwise PromoteToSupported's reachability
    #    check skips ClusterEvidence and IBE never runs.
    assert claim.entity_id not in state.verification_done, (
        "Soft-promoted claim must not be in verification_done — "
        "verification is the path to IBE, and IBE is what populates "
        "the deferred integrated_assessment. Setting verification_done "
        "here is the bug from the intermittent-fasting reproduction."
    )

    # And the claim itself: SUPPORTED with no integration verdict yet
    # (IBE hasn't run — that's the next node's job, not AbandonOrDemote's).
    reloaded = await repo.get("claim", claim.entity_id)
    assert isinstance(reloaded, Claim)
    assert reloaded.stage == ClaimStage.SUPPORTED
    assert reloaded.integrated_assessment is None, (
        "Soft-promote must leave integrated_assessment=None for IBE to "
        "populate — pre-setting it would short-circuit the IBE chain."
    )


async def test_refute_promote_still_skips_verification(
    tmp_path: Path, fake_runner
) -> None:
    """Sibling assertion: the refute-promote path's verification_done
    flag is correct (the claim has a verdict already, IBE shouldn't
    overwrite it). The fix to soft-promote must not break this."""
    from andamentum.epistemic.graph.nodes import CheckCompletion

    store = DocumentStore.for_database("refute_skip_verify", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="weak claim",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="needs_resolution",
    )
    await repo.save(claim)

    # Strong contradicting evidence (3 con, 0 sup) → refute-promote
    # should fire instead of soft-promote.
    for i in range(3):
        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="web",
            source_ref=f"https://ex.com/con{i}",
            extracted_content=f"contradicts {i}",
            extracted=True,
            support_judgment="contradicts",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    claim.evidence_count = 3
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    state.investigation_counts[claim.entity_id] = 3
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    ctx = _FakeRunContext(state, deps)
    next_node = await AbandonOrDemote().run(ctx)  # type: ignore[arg-type]

    refute_calls = [
        op for op in state.operations_log if op["operation"] == "promote_as_refuted"
    ]
    assert len(refute_calls) == 1 and refute_calls[0]["success"]

    # Refute-promote correctly adds to verification_done — the verdict
    # is already "contradicts", IBE shouldn't overwrite it.
    assert claim.entity_id in state.verification_done

    # And no soft-promote got triggered (claim was handled by refute).
    assert not any(
        op["operation"] == "soft_promote" for op in state.operations_log
    )

    # Routing: nothing soft-promoted → CheckCompletion (terminal),
    # not PromoteToSupported.
    assert isinstance(next_node, CheckCompletion)
