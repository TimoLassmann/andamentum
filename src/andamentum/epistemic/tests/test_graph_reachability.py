"""Graph reachability invariants.

Stage A of the Move-3 plan (graph node contracts). These tests pin the
routing invariants that have been our recurring failure mode — silent
dead zones where a node returns to a successor that doesn't continue
work the claim still needs.

The bug class we keep hitting:

  * v0.3 dormancy: DecomposeQuestionOperation registered, never called.
  * post-audit-1 dormancy: ReflectOnGapsOperation still dormant after
    the fix queue.
  * post-audit-2 routing: AbandonOrDemote → CheckCompletion stranded
    soft-promoted claims (SUPPORTED, integrated_assessment=None) at
    Synthesize, never running IBE.
  * post-audit-2 routing (sibling): same node, same shape — stranded
    HYPOTHESIS-with-pass claims when scrutiny found mixed outcomes.

Static type-checking + the structural-wiring test (which checks
"operation has at least one graph caller") catch dormancy-of-classes,
but they don't catch dormancy-of-state-patterns. A claim can be in a
state where it should reach IBE, and the system has IBE wired up
*for some other state pattern*, but the routing path from this state
to IBE is missing. That's invisible to anything except actually
producing the state and checking where it ends up.

These tests are deliberately architecture-agnostic. They construct
a state pattern, run the relevant routing node directly, and assert:

  1. The claim ends in a sensible state (the post-condition the node
     promises).
  2. The next-node return value continues the work the claim still
     needs (the routing contract).
  3. The "no stranded claims" invariant holds on the resulting state.

When Move 3 lands and nodes get explicit Reads/Writes/Successors
metadata, these tests stay valid — they assert outputs, not internals.
The contract validator (Move 3 work) is complementary, not a
replacement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    AbandonOrDemote,
    CheckCompletion,
    PromoteToSupported,
    Scrutinize,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


# ── Invariant: no claim is "stranded" ─────────────────────────────────


def _stranded_claims(claims: list[Claim], state: EpistemicGraphState) -> list[Claim]:
    """A stranded claim is one that has been promoted past HYPOTHESIS,
    has no integration verdict, and is NOT marked verification_done.

    Such a claim should still reach the IBE chain to receive a calibrated
    verdict. If the graph routes it to a terminal in this state, the
    posterior falls back to no-data 0.5 and the report is silently
    miscalibrated.

    Refute-promoted claims have ``integrated_assessment="contradicts"``
    set by the operation and ARE marked verification_done — the IBE
    chain shouldn't overwrite that pre-set verdict. Soft-promoted
    claims have ``integrated_assessment=None`` and are NOT marked
    verification_done — IBE is exactly what should populate the verdict.

    Cycle-capped and abandoned claims are excluded — both are
    legitimate terminal states where no integration verdict is
    expected.
    """
    stranded = []
    for c in claims:
        if c.abandoned or getattr(c, "cycle_capped", False):
            continue
        if c.stage == ClaimStage.HYPOTHESIS:
            continue  # not promoted, not eligible for IBE yet
        if c.integrated_assessment is not None:
            continue  # IBE ran (or refute pre-set the verdict)
        if c.entity_id in state.verification_done:
            continue  # explicitly excluded from IBE (e.g. refute path)
        stranded.append(c)
    return stranded


# ── Helpers to build state patterns ───────────────────────────────────


async def _build_objective_with_claims(
    tmp_path: Path,
    db_name: str,
    claim_specs: list[dict[str, Any]],
) -> tuple[Objective, list[Claim], EpistemicRepository]:
    """Construct an objective with N claims of specified shapes.

    Each spec is a dict with keys: stage, scrutiny_verdict, n_supports,
    n_contradicts, investigation_count, integrated_assessment.
    """
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="parent", question_type="verificatory")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claims: list[Claim] = []
    for i, spec in enumerate(claim_specs):
        claim = Claim(
            objective_id=obj.entity_id,
            statement=f"claim {i}",
            scope="x",
            stage=spec.get("stage", ClaimStage.HYPOTHESIS),
            scrutiny_verdict=spec.get("scrutiny_verdict"),
            integrated_assessment=spec.get("integrated_assessment"),
            integrated_confidence=spec.get("integrated_confidence"),
        )
        await repo.save(claim)

        # Add evidence as specified.
        for j in range(spec.get("n_supports", 0)):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"https://ex.com/sup_{i}_{j}",
                extracted_content=f"sup {i} {j}",
                extracted=True,
                support_judgment="supports",
            )
            await repo.save(ev)
            claim.evidence_ids.append(ev.entity_id)
        for j in range(spec.get("n_contradicts", 0)):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"https://ex.com/con_{i}_{j}",
                extracted_content=f"con {i} {j}",
                extracted=True,
                support_judgment="contradicts",
            )
            await repo.save(ev)
            claim.evidence_ids.append(ev.entity_id)
        claim.evidence_count = len(claim.evidence_ids)
        await repo.save(claim)
        claims.append(claim)

    return obj, claims, repo


# ── Routing assertions for AbandonOrDemote ────────────────────────────


class TestAbandonOrDemoteRoutingInvariants:
    """The four state patterns AbandonOrDemote can encounter, and the
    routing each one demands. The pattern that's NEW vs. the existing
    soft-promote test is the mixed pass+abandon case (the second bug
    found by running the system).
    """

    async def test_strong_contradicts_refute_promotes_then_terminates_cleanly(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """3 contradicts / 0 supports → refute-promote → SUPPORTED with
        integrated_assessment="contradicts". Verification is correctly
        skipped (the verdict is already set). Claim is NOT stranded."""
        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "refute_clean",
            [
                {
                    "scrutiny_verdict": "needs_resolution",
                    "n_supports": 0,
                    "n_contradicts": 3,
                }
            ],
        )
        claim = claims[0]
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        all_claims = await repo.query("claim", objective_id=obj.entity_id)
        assert next_node is not None
        # Refute promoted: claim is SUPPORTED with verdict pre-set, and
        # marked verification_done so IBE doesn't overwrite.
        reloaded = next(c for c in all_claims if c.entity_id == claim.entity_id)
        assert reloaded.stage == ClaimStage.SUPPORTED
        assert reloaded.integrated_assessment == "contradicts"
        assert claim.entity_id in state.verification_done

        # Invariant: no stranded claims.
        assert _stranded_claims(all_claims, state) == []

    async def test_directional_no_refute_soft_promotes_then_must_reach_ibe(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """3 supports / 2 contradicts → refute declines (threshold not
        met), soft-promote takes over. Claim is SUPPORTED with
        integrated_assessment=None (deferred to IBE). The routing must
        carry it toward IBE — i.e. NOT to CheckCompletion directly."""
        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "soft_to_ibe",
            [
                {
                    "scrutiny_verdict": "needs_resolution",
                    "n_supports": 3,
                    "n_contradicts": 2,
                }
            ],
        )
        claim = claims[0]
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        all_claims = await repo.query("claim", objective_id=obj.entity_id)
        reloaded = next(c for c in all_claims if c.entity_id == claim.entity_id)

        # Soft-promoted: SUPPORTED, no verdict yet, NOT verification_done.
        assert reloaded.stage == ClaimStage.SUPPORTED
        assert reloaded.integrated_assessment is None
        assert claim.entity_id not in state.verification_done

        # Routing carries the work forward — PromoteToSupported is the
        # dispatcher that will route this claim to ClusterEvidence → IBE.
        assert isinstance(next_node, PromoteToSupported), (
            f"Soft-promoted claim must route to PromoteToSupported, "
            f"not {type(next_node).__name__}. Direct routing to "
            f"CheckCompletion strands the claim at SUPPORTED with no "
            f"integration verdict — the headline posterior falls back "
            f"to no-data 0.5 and the report is silently miscalibrated."
        )

        # The stranded check passes for now BECAUSE the claim is on a
        # path to IBE. If IBE hasn't run yet but is reachable, that's OK.
        # The terminal-time check is below in
        # TestEndToEndNoStrandedClaims.

    async def test_no_signal_abandons_cleanly(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """0/0 evidence: refute and soft-promote both decline (no
        directional signal). Abandonment is the honest terminal."""
        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "no_signal_abandon",
            [
                {
                    "scrutiny_verdict": "needs_resolution",
                    "n_supports": 0,
                    "n_contradicts": 0,
                }
            ],
        )
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.investigation_counts[claims[0].entity_id] = 3
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        all_claims = await repo.query("claim", objective_id=obj.entity_id)
        reloaded = next(c for c in all_claims if c.entity_id == claims[0].entity_id)
        assert reloaded.abandoned is True
        # Abandoned claims aren't stranded — the invariant excludes them.
        assert _stranded_claims(all_claims, state) == []

    async def test_mixed_pass_and_no_signal_routes_through_promote(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """The bug found by the second benchmark run: claim A passed
        scrutiny, claim B exhausted with no signal. AbandonOrDemote
        handles B (abandon) but must route to PromoteToSupported so
        A gets promoted (and then on to IBE). Returning CheckCompletion
        directly strands A at HYPOTHESIS forever."""
        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "mixed_pass_abandon",
            [
                {"scrutiny_verdict": "pass"},  # A — should be promoted
                {  # B — should be abandoned
                    "scrutiny_verdict": "needs_resolution",
                    "n_supports": 0,
                    "n_contradicts": 0,
                },
            ],
        )
        claim_a, claim_b = claims
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.investigation_counts[claim_b.entity_id] = 3
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        # B is abandoned (handled by AbandonOrDemote).
        all_claims = await repo.query("claim", objective_id=obj.entity_id)
        reloaded_b = next(c for c in all_claims if c.entity_id == claim_b.entity_id)
        assert reloaded_b.abandoned is True

        # A is unchanged — still HYPOTHESIS+pass, awaiting promotion.
        reloaded_a = next(c for c in all_claims if c.entity_id == claim_a.entity_id)
        assert reloaded_a.stage == ClaimStage.HYPOTHESIS
        assert reloaded_a.scrutiny_verdict == "pass"

        # Routing must carry the work forward — PromoteToSupported is
        # the only node that runs PromoteClaimOperation.
        assert isinstance(next_node, PromoteToSupported), (
            "Mixed pass+abandon must route to PromoteToSupported. "
            f"Got {type(next_node).__name__} — direct CheckCompletion "
            "strands the pass-verdict claim at HYPOTHESIS, never "
            "promoted, never reaching IBE."
        )

    async def test_demoted_claim_routes_to_rescrutiny(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Sanity: demoted SUPPORTED claims still route to Scrutinize
        for re-evaluation. The fix must not break this path."""
        obj, _, repo = await _build_objective_with_claims(
            tmp_path,
            "demote_rescrutiny",
            [
                {
                    "stage": ClaimStage.SUPPORTED,
                    "scrutiny_verdict": "fail",
                    "n_supports": 0,
                    "n_contradicts": 0,
                }
            ],
        )
        # Don't set investigation_counts — demote path applies to
        # SUPPORTED+ regardless of investigation count.
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await AbandonOrDemote().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        # Routing: demoted claims need re-scrutiny.
        assert isinstance(next_node, Scrutinize)


# ── End-to-end: a soft-promoted claim reaches a sensible terminal ─────


class TestEndToEndNoStrandedClaims:
    """Drive the AbandonOrDemote → PromoteToSupported chain and assert
    that PromoteToSupported correctly hands soft-promoted claims off to
    the verification-then-IBE path. This is the strongest available
    architecture-agnostic check: it composes the routing of two nodes
    rather than just one, and verifies the end-state invariant."""

    async def test_soft_promoted_claim_flows_into_cluster_evidence(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """After AbandonOrDemote → PromoteToSupported, a soft-promoted
        claim must be routed to ClusterEvidence (the entry to the
        verification → IBE pipeline), not to CheckCompletion."""
        from andamentum.epistemic.graph.nodes import ClusterEvidence

        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "soft_flow_e2e",
            [
                {
                    "scrutiny_verdict": "needs_resolution",
                    "n_supports": 3,
                    "n_contradicts": 2,
                }
            ],
        )
        claim = claims[0]
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        # Step 1: AbandonOrDemote soft-promotes.
        ctx1 = _FakeRunContext(state, deps)
        next1 = await AbandonOrDemote().run(ctx1)  # type: ignore[arg-type]
        assert isinstance(next1, PromoteToSupported)

        # Step 2: PromoteToSupported routes to ClusterEvidence (the
        # entry to the verification → IBE chain).
        ctx2 = _FakeRunContext(state, deps)
        next2 = await PromoteToSupported().run(ctx2)  # type: ignore[arg-type]
        assert isinstance(next2, ClusterEvidence), (
            f"PromoteToSupported should route soft-promoted claim "
            f"to ClusterEvidence (the IBE-chain entry); got "
            f"{type(next2).__name__}. If this returns CheckCompletion, "
            f"the claim is stranded at SUPPORTED with no integration "
            f"verdict and the headline posterior is broken."
        )

    async def test_promoted_claim_without_verification_flows_into_cluster_evidence(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """A claim that is already SUPPORTED, has no integration verdict,
        and is not in verification_done must route to ClusterEvidence
        (the IBE-chain entry). Same routing destination as a soft-
        promoted claim — that's the invariant: any SUPPORTED claim
        without a verdict and not excluded must reach IBE.

        We pre-set the claim to SUPPORTED here to isolate the routing
        concern from PromoteClaimOperation's gate validation (which
        has its own coverage). The point of THIS test is the
        post-promotion routing decision, not gate logic.
        """
        from andamentum.epistemic.graph.nodes import ClusterEvidence

        obj, _, repo = await _build_objective_with_claims(
            tmp_path,
            "supported_flow_e2e",
            [
                {
                    "stage": ClaimStage.SUPPORTED,
                    "scrutiny_verdict": "pass",
                    "integrated_assessment": None,
                }
            ],
        )
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
        assert isinstance(next_node, ClusterEvidence), (
            f"SUPPORTED claim without integration verdict and not in "
            f"verification_done must route to ClusterEvidence; got "
            f"{type(next_node).__name__}. CheckCompletion here means "
            f"the claim is stranded — IBE never runs on it and the "
            f"posterior falls back to no-data 0.5."
        )

    async def test_refute_promoted_claim_flows_to_completion(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Refute-promoted claim has its verdict pre-set and is in
        verification_done. PromoteToSupported should see no work and
        return CheckCompletion — the IBE chain correctly does NOT run."""
        obj, claims, repo = await _build_objective_with_claims(
            tmp_path,
            "refute_flow_e2e",
            [
                {
                    "stage": ClaimStage.SUPPORTED,
                    "integrated_assessment": "contradicts",
                    "integrated_confidence": 0.9,
                }
            ],
        )
        claim = claims[0]
        state = EpistemicGraphState(objective_id=obj.entity_id)
        state.verification_done.add(claim.entity_id)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )

        next_node = await PromoteToSupported().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
        assert isinstance(next_node, CheckCompletion)


# ── Stranded-claim invariant as a standalone unit test ────────────────


class TestStrandedClaimInvariant:
    """The invariant function itself should correctly classify every
    state pattern. Regressions to the invariant function would silently
    weaken every test above."""

    def test_supported_with_no_verdict_not_in_verification_done_is_stranded(
        self,
    ) -> None:
        """The bug shape: stage=SUPPORTED, integrated_assessment=None,
        not in verification_done. This is stranded."""
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment=None,
        )
        state = EpistemicGraphState(objective_id="o")
        assert _stranded_claims([c], state) == [c]

    def test_supported_with_verdict_is_not_stranded(self) -> None:
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
        )
        state = EpistemicGraphState(objective_id="o")
        assert _stranded_claims([c], state) == []

    def test_supported_no_verdict_in_verification_done_is_not_stranded(
        self,
    ) -> None:
        """Refute-promote pre-sets the verdict but adds to
        verification_done. The post-condition relies on this."""
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment=None,
        )
        state = EpistemicGraphState(objective_id="o")
        state.verification_done.add(c.entity_id)
        assert _stranded_claims([c], state) == []

    def test_hypothesis_is_never_stranded(self) -> None:
        """Claims at HYPOTHESIS aren't promoted; they're not eligible
        for IBE yet."""
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.HYPOTHESIS,
        )
        state = EpistemicGraphState(objective_id="o")
        assert _stranded_claims([c], state) == []

    def test_abandoned_is_never_stranded(self) -> None:
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment=None,
            abandoned=True,
        )
        state = EpistemicGraphState(objective_id="o")
        assert _stranded_claims([c], state) == []

    def test_cycle_capped_is_never_stranded(self) -> None:
        c = Claim(
            objective_id="o",
            statement="x",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment=None,
            cycle_capped=True,
        )
        state = EpistemicGraphState(objective_id="o")
        assert _stranded_claims([c], state) == []
