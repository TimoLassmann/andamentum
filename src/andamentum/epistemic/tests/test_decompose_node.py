"""Tests for the Decompose graph node and the --decompose CLI flag.

Phase D of the post-audit fix queue. Without this, multi-seed-claim
mode was unreachable in production: ``DecomposeQuestionOperation`` had
no graph caller and no CLI invocation, so ``objective.decomposition``
was always None at CreateClaims time and the third branch
(MultiSeedClaim) never fired.

The Decompose node sits between PrepareObjective and PlanEvidence,
gated by ``state.decompose``. With ``decompose=True``,
DecomposeQuestionOperation runs and populates
``objective.decomposition``; downstream PlanEvidence picks the
per-claim query formulation branch and CreateClaims routes to
MultiSeedClaim.
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
    tmp_path: Path, db_name: str, *, phase: str = "analyzed"
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="Are podocytes motile in injury?",
        clarified_question="Are podocytes motile in injury?",
        question_type="verificatory",
        phase=phase,
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj, repo


class TestDecomposeNode:
    async def test_decompose_false_is_passthrough(self, tmp_path: Path) -> None:
        """When state.decompose is False (the default open-research
        path), Decompose returns PlanEvidence without running
        DecomposeQuestionOperation. The objective's decomposition
        stays None."""
        obj, repo = await _setup_objective(tmp_path, "decompose_passthrough")
        state = EpistemicGraphState(objective_id=obj.entity_id, decompose=False)
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

    async def test_decompose_true_runs_decomposition(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """With state.decompose=True, the node runs
        DecomposeQuestionOperation and populates
        objective.decomposition with the conftest fake's 3 sub-
        investigations."""
        obj, repo = await _setup_objective(tmp_path, "decompose_active")
        state = EpistemicGraphState(objective_id=obj.entity_id, decompose=True)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        ctx = _FakeRunContext(state, deps)
        next_node = await Decompose().run(ctx)  # type: ignore[arg-type]
        assert isinstance(next_node, PlanEvidence)
        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is not None
        assert len(reloaded.decomposition.sub_investigations) == 3
        assert reloaded.combination_rule == "AND"
        decompose_ops = [
            op for op in state.operations_log if op["operation"] == "decompose_question"
        ]
        assert len(decompose_ops) == 1

    async def test_decompose_true_idempotent(
        self, tmp_path: Path, fake_runner
    ) -> None:
        """Re-running Decompose on an objective that already has a
        decomposition is a did_work=False short-circuit (the underlying
        DecomposeQuestionOperation is idempotent)."""
        obj, repo = await _setup_objective(tmp_path, "decompose_idempotent")
        state = EpistemicGraphState(objective_id=obj.entity_id, decompose=True)
        deps = EpistemicDeps(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
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


class TestStateDecomposeFieldDefault:
    def test_default_is_false(self) -> None:
        """Sanity: the new state field defaults to False so existing
        callers (open-research path) get unchanged behavior without
        opt-in."""
        s = EpistemicGraphState()
        assert s.decompose is False
