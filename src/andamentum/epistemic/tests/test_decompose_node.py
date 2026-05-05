"""Tests for the Decompose graph node.

The Decompose node sits between PrepareObjective and PlanEvidence,
gated by ``obj.claim_to_verify``: when set (verify mode — the user
named the exact claim), the node is a no-op pass-through. Otherwise
(research mode), DecomposeQuestionOperation runs and populates
``objective.decomposition``; downstream PlanEvidence picks the per-claim
query formulation branch and CreateClaims routes to MultiSeedClaim. If
the decomposer produces no usable sub-investigations, the
MultiSeedClaim → ProposeClaims fallback in CreateClaims kicks in — the
open-research path emerges naturally without a separate flag.

History: the gate was originally ``state.decompose`` (a function
argument propagated into graph state). The 2026-05-04 entry-point
consolidation flipped it to ``obj.claim_to_verify`` so callers no longer
have two parallel ways to declare mode.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import Decompose, PlanEvidence
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state: EpistemicGraphState, deps: EpistemicDeps) -> None:
        self.state = state
        self.deps = deps


async def _setup_objective(
    tmp_path: Path,
    db_name: str,
    *,
    phase: str = "analyzed",
    claim_to_verify: str | None = None,
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="Are podocytes motile in injury?",
        clarified_question="Are podocytes motile in injury?",
        question_type="verificatory",
        phase=phase,
        claim_to_verify=claim_to_verify,
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj, repo


class TestDecomposeNode:
    async def test_claim_to_verify_passthrough(self, tmp_path: Path) -> None:
        """When obj.claim_to_verify is set (verify mode), Decompose
        returns PlanEvidence without running DecomposeQuestionOperation.
        The objective's decomposition stays None."""
        obj, repo = await _setup_objective(
            tmp_path,
            "decompose_passthrough",
            claim_to_verify="Podocytes are motile in injury.",
        )
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        ctx = _FakeRunContext(state, deps)
        next_node = await Decompose().run(ctx)  # type: ignore[arg-type]
        assert isinstance(next_node, PlanEvidence)
        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is None
        # No execution_step for decompose_question.
        decompose_ops = [
            op for op in state.operations_log if op["operation"] == "decompose_question"
        ]
        assert decompose_ops == []

    async def test_research_mode_runs_decomposition(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Without claim_to_verify (research mode), the node runs
        DecomposeQuestionOperation and populates objective.decomposition
        with the conftest fake's 3 sub-investigations."""
        obj, repo = await _setup_objective(tmp_path, "decompose_active")
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
        ctx = _FakeRunContext(state, deps)
        next_node = await Decompose().run(ctx)  # type: ignore[arg-type]
        assert isinstance(next_node, PlanEvidence)
        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is not None
        assert len(reloaded.decomposition.sub_investigations) == 3
        assert reloaded.decomposition.combination_rule == "AND"
        decompose_ops = [
            op for op in state.operations_log if op["operation"] == "decompose_question"
        ]
        assert len(decompose_ops) == 1

    async def test_research_mode_idempotent(self, tmp_path: Path, fake_runner) -> None:
        """Re-running Decompose on an objective that already has a
        decomposition is a did_work=False short-circuit (the underlying
        DecomposeQuestionOperation is idempotent)."""
        obj, repo = await _setup_objective(tmp_path, "decompose_idempotent")
        state = EpistemicGraphState(objective_id=obj.entity_id)
        deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")
        ctx = _FakeRunContext(state, deps)
        await Decompose().run(ctx)  # type: ignore[arg-type]
        # Second pass.
        await Decompose().run(ctx)  # type: ignore[arg-type]
        reloaded = await repo.get("objective", obj.entity_id)
        # Still 3 sub-investigations (not 6 — the op short-circuits).
        assert reloaded.decomposition is not None
        assert len(reloaded.decomposition.sub_investigations) == 3
        # Two op records but the second was a did_work=False short-circuit.
        decompose_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_decompose_question"
        ]
        assert len(decompose_calls) == 1
