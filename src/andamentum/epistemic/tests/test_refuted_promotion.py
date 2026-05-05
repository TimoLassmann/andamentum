"""Tests for the refuted claim promotion path."""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import compute_posterior
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.gates import (
    count_support_contradict,
    is_refuted_by_evidence,
)
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import AbandonOrDemote
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.stage_management import (
    PromoteAsRefutedOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


async def _setup_claim_with_evidence(
    tmp_path: Path,
    n_supports: int,
    n_contradicts: int,
    n_unjudged: int = 0,
) -> tuple[Claim, EpistemicRepository]:
    """Create an objective + claim linked to N supporting + M contradicting evidence."""
    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(description="test objective", question_type="predictive")
    # Objectives are self-referential: objective_id == entity_id. Production
    # code (graph/__init__.py) constructs Objective with both set explicitly;
    # without this, _build_metadata writes objective_id="" and round-trip
    # lookups via repo.get_objective(...) fail.
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claim = Claim(
        statement="Test claim.",
        scope="test scope",
        objective_id=obj.entity_id,
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)
    for i in range(n_supports):
        ev = Evidence(
            source_type="web",
            source_ref=f"https://ex.com/s{i}",
            extracted_content="supports",
            objective_id=obj.entity_id,
            support_judgment="supports",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    for i in range(n_contradicts):
        ev = Evidence(
            source_type="web",
            source_ref=f"https://ex.com/c{i}",
            extracted_content="contradicts",
            objective_id=obj.entity_id,
            support_judgment="contradicts",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    for i in range(n_unjudged):
        ev = Evidence(
            source_type="web",
            source_ref=f"https://ex.com/u{i}",
            extracted_content="x",
            objective_id=obj.entity_id,
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    claim.evidence_count = len(claim.evidence_ids)
    await repo.save(claim)
    return claim, repo


class TestCountSupportContradict:
    async def test_counts_directional_judgments(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 5)
        n_sup, n_con = await count_support_contradict(claim, repo)
        assert n_sup == 2
        assert n_con == 5

    async def test_ignores_unjudged(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 1, n_unjudged=3)
        n_sup, n_con = await count_support_contradict(claim, repo)
        assert (n_sup, n_con) == (1, 1)

    async def test_empty_evidence(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 0)
        assert await count_support_contradict(claim, repo) == (0, 0)

    async def test_skips_invalidated_evidence(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 2)
        # Invalidate one supporting and one contradicting evidence item
        evs = []
        for eid in claim.evidence_ids:
            ev = await repo.get("evidence", eid)
            evs.append(ev)
        sup_ev = next(e for e in evs if e.support_judgment == "supports")
        con_ev = next(e for e in evs if e.support_judgment == "contradicts")
        sup_ev.invalidated = True
        con_ev.invalidated = True
        await repo.save(sup_ev)
        await repo.save(con_ev)

        n_sup, n_con = await count_support_contradict(claim, repo)
        assert (n_sup, n_con) == (1, 1)


class TestIsRefutedByEvidence:
    async def test_refuted_when_contradicts_dominate(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_not_refuted_when_balanced(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 4, 5)
        assert await is_refuted_by_evidence(claim, repo) is False

    async def test_not_refuted_when_too_few_contradicts(self, tmp_path: Path) -> None:
        # Threshold requires at least 3 contradicts.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 2)
        assert await is_refuted_by_evidence(claim, repo) is False

    async def test_refuted_with_zero_supports(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 3)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_refuted_at_minimum_ratio_edge(self, tmp_path: Path) -> None:
        # (n_con=3, n_sup=1) sits on the tightest passing edge:
        # 3 >= 3 AND 3 >= 2 * max(1, 1) = 2. Must be True.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 3)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_not_refuted_just_below_ratio(self, tmp_path: Path) -> None:
        # (n_con=3, n_sup=2) is the just-below-ratio case:
        # 3 >= 3 passes but 3 >= 2 * 2 = 4 fails. Must be False.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 3)
        assert await is_refuted_by_evidence(claim, repo) is False


class TestPromoteAsRefutedOperation:
    async def test_promotes_to_supported_with_contradicts(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)
        op = PromoteAsRefutedOperation(repo=repo, agent_runner=None)
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="promote_as_refuted",
            )
        )
        assert result.success is True
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.SUPPORTED
        assert reloaded.integrated_assessment == "contradicts"
        assert reloaded.integrated_confidence is not None
        assert 0.5 <= reloaded.integrated_confidence <= 0.95
        assert reloaded.integrated_reasoning is not None
        assert reloaded.abandoned is False
        assert reloaded.confidence_score == reloaded.integrated_confidence
        assert len(reloaded.promotion_history) == 1
        assert reloaded.promotion_history[-1].to_stage == ClaimStage.SUPPORTED

    async def test_refuses_when_not_refuted(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 5, 1)
        op = PromoteAsRefutedOperation(repo=repo, agent_runner=None)
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="promote_as_refuted",
            )
        )
        assert result.success is False
        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.stage == ClaimStage.HYPOTHESIS
        assert reloaded.integrated_assessment is None

    async def test_refuses_when_not_hypothesis(self, tmp_path: Path) -> None:
        # Only HYPOTHESIS claims are eligible; SUPPORTED claims should be refused.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)
        claim.stage = ClaimStage.SUPPORTED
        await repo.save(claim)
        op = PromoteAsRefutedOperation(repo=repo, agent_runner=None)
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="promote_as_refuted",
            )
        )
        assert result.success is False
        reloaded = await repo.get("claim", claim.entity_id)
        # Stage unchanged; assessment still None.
        assert reloaded.stage == ClaimStage.SUPPORTED
        assert reloaded.integrated_assessment is None


class _FakeRunContext:
    """Duck-typed GraphRunContext for tests — AbandonOrDemote only reads .state and .deps."""

    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


class TestAbandonOrDemoteRoutesToRefutedPromotion:
    async def test_refuted_hypothesis_is_promoted_not_abandoned(
        self, tmp_path: Path
    ) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)
        claim.scrutiny_verdict = "fail"
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(repo=repo, agent_runner=None)

        node = AbandonOrDemote()
        ctx = _FakeRunContext(state, deps)
        await node.run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.abandoned is False
        assert reloaded.stage == ClaimStage.SUPPORTED
        assert reloaded.integrated_assessment == "contradicts"
        assert claim.entity_id in state.verification_done
        assert claim.entity_id not in state.terminal_claims

    async def test_no_directional_evidence_is_still_abandoned(
        self, tmp_path: Path
    ) -> None:
        # 0/0 evidence: refute declines AND soft-promote declines (no
        # directional signal at all). Abandonment is the correct terminal.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 0)
        claim.scrutiny_verdict = "needs_resolution"
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(repo=repo, agent_runner=None)

        node = AbandonOrDemote()
        ctx = _FakeRunContext(state, deps)
        await node.run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.abandoned is True
        assert claim.entity_id in state.terminal_claims
        assert claim.entity_id not in state.verification_done

    async def test_directional_evidence_below_refute_threshold_is_soft_promoted(
        self, tmp_path: Path
    ) -> None:
        # 3/2: refute threshold is n_con >= 2*max(1, n_sup) → 2 < 6, declines.
        # Directional signal exists (3+2 > 0), so soft-promote takes over
        # instead of abandonment.
        #
        # SoftPromote does not pre-set the integration verdict — that was a
        # pre-IBE optimization. After the IBE 4-stage refactor, the verdict
        # is produced by the IBE chain (EnumerateCandidates → ... →
        # SelectBestExplanation), so SoftPromote leaves
        # integrated_assessment / integrated_confidence as None for IBE to
        # populate.
        #
        # CRITICAL: the soft-promoted claim must NOT be added to
        # ``verification_done``, and AbandonOrDemote must return
        # PromoteToSupported (not CheckCompletion) to route the claim
        # through ClusterEvidence → RunVerification → IBE. An earlier
        # version of this test asserted ``in state.verification_done``,
        # which baked the IBE-unreachability bug into the suite — see
        # test_soft_promote_reaches_ibe.py for the full diagnosis.
        from andamentum.epistemic.graph.nodes import PromoteToSupported

        claim, repo = await _setup_claim_with_evidence(tmp_path, 3, 2)
        claim.scrutiny_verdict = "needs_resolution"
        await repo.save(claim)

        state = EpistemicGraphState(objective_id=claim.objective_id)
        state.investigation_counts[claim.entity_id] = 3
        deps = EpistemicDeps(repo=repo, agent_runner=None)

        node = AbandonOrDemote()
        ctx = _FakeRunContext(state, deps)
        next_node = await node.run(ctx)  # type: ignore[arg-type]

        reloaded = await repo.get("claim", claim.entity_id)
        assert reloaded.abandoned is False
        assert reloaded.stage == ClaimStage.SUPPORTED
        # SoftPromote does not pre-set the verdict; IBE will produce it.
        assert reloaded.integrated_assessment is None
        assert reloaded.integrated_confidence is None
        # Verification path must remain open so the claim reaches IBE.
        assert claim.entity_id not in state.verification_done
        assert claim.entity_id not in state.terminal_claims
        assert isinstance(next_node, PromoteToSupported)


class TestPosteriorIncludesRefuted:
    async def test_refuted_claim_drives_posterior_low(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)

        # Verify objective exists in repo before continuing
        objective = await repo.get_objective(claim.objective_id)
        assert objective is not None

        op = PromoteAsRefutedOperation(repo=repo, agent_runner=None)
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="promote_as_refuted",
            )
        )
        assert result.success

        posterior = await compute_posterior(repo, objective_id=claim.objective_id)
        assert posterior is not None
        assert posterior.posterior < 0.2, (
            f"Expected low posterior (evidence contradicts), got {posterior.posterior:.3f}"
        )
        assert posterior.integration_verdict == "contradicts"
