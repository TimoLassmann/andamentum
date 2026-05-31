"""Regression: execution_step file_path uniqueness across concurrent graph runs.

Bug context: ``_run_op`` in ``graph/nodes.py`` stored an execution-trace
document per operation with ``file_path=f"execution_step_{step_number}"``.
``step_number`` was ``len(state.operations_log)`` — local to each graph
run. The decomposed orchestrator (Phase 3+) shares one DocumentStore
across N child graph runs, so all children write
``execution_step_1, execution_step_2, ...`` and the second child crashes
on ``UNIQUE constraint failed: documents.file_path`` after burning ~7-8
minutes per case. Pre-decomposition, every run had a fresh DB and the
collision was masked.

These tests pin the fix:
1. Two graph states sharing one DocumentStore can both write
   execution_step entries without collision.
2. The same objective re-run on the same DB doesn't collide either
   (per-run ``run_id`` disambiguator).
"""

from __future__ import annotations

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import _run_op
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.operations.base import (
    BaseOperation,
    OperationInput,
    OperationResult,
)
from andamentum.epistemic.repository import EpistemicRepository


class _AlwaysSucceedsOp(BaseOperation):
    """Minimal op for exercising _run_op without invoking real machinery."""

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        return OperationResult(success=True, entity_id=work.entity_id, message="ok")


@pytest.fixture
async def shared_repo(tmp_path):
    store = DocumentStore.for_database("test_step_unique", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _seed_objective(repo: EpistemicRepository, oid: str) -> Objective:
    obj = Objective(entity_id=oid, objective_id=oid, description=f"q-{oid}")
    await repo.save(obj)
    return obj


async def test_two_graph_runs_on_shared_db_do_not_collide(shared_repo):
    """Two children of a decomposed run share the DocumentStore. Both
    must be able to write execution_step entries from step 1 onward."""
    await _seed_objective(shared_repo, "objA_alpha")
    await _seed_objective(shared_repo, "objB_beta")

    deps = EpistemicDeps(repo=shared_repo, agent_runner=None)
    state_a = EpistemicGraphState(objective_id="objA_alpha")
    state_b = EpistemicGraphState(objective_id="objB_beta")

    # Child A writes 3 ops.
    for _ in range(3):
        await _run_op(
            _AlwaysSucceedsOp, deps, state_a, "objA_alpha", "objective", "noop"
        )
    # Child B writes 3 ops on the SAME DocumentStore. Pre-fix, this
    # raised "UNIQUE constraint failed: documents.file_path" on the
    # very first call (both starting from step 1).
    for _ in range(3):
        await _run_op(
            _AlwaysSucceedsOp, deps, state_b, "objB_beta", "objective", "noop"
        )

    # Both graph runs logged 3 successes in their local state.
    assert state_a.successful == 3
    assert state_b.successful == 3
    assert state_a.failed == 0
    assert state_b.failed == 0

    # The execution_step rows for both objectives are present in the DB.
    backend = shared_repo.store
    rows = await backend.find_by_metadata(
        {"epistemic_type": "execution_step"}, limit=100
    )
    assert len(rows) == 6
    objective_ids_seen = {row.metadata["entity_id"] for row in rows}
    assert objective_ids_seen == {"objA_alpha", "objB_beta"}
    # File paths are unique (the bug we're pinning).
    file_paths = {row.file_path for row in rows}
    assert len(file_paths) == 6


async def test_same_objective_rerun_does_not_collide(shared_repo):
    """Two consecutive runs targeting the SAME objective on the SAME DB
    must not collide. The per-run ``run_id`` disambiguates step paths."""
    await _seed_objective(shared_repo, "obj_repeat")
    deps = EpistemicDeps(repo=shared_repo, agent_runner=None)

    # First run.
    state1 = EpistemicGraphState(objective_id="obj_repeat")
    for _ in range(2):
        await _run_op(
            _AlwaysSucceedsOp, deps, state1, "obj_repeat", "objective", "noop"
        )

    # Second run on the same objective + same DB. Without per-run id,
    # step_number resets and would collide with run 1's entries.
    state2 = EpistemicGraphState(objective_id="obj_repeat")
    # Sanity: each fresh state has a distinct run_id so file_paths differ.
    assert state1.run_id != state2.run_id
    for _ in range(2):
        await _run_op(
            _AlwaysSucceedsOp, deps, state2, "obj_repeat", "objective", "noop"
        )

    backend = shared_repo.store
    rows = await backend.find_by_metadata(
        {"epistemic_type": "execution_step"}, limit=100
    )
    assert len(rows) == 4
    file_paths = {row.file_path for row in rows}
    assert len(file_paths) == 4


async def test_step_number_metadata_preserved(shared_repo):
    """The fix changes file_path but step_number metadata stays a clean
    1..N sequence per run — needed by execution-step queries used in
    cli_handlers.print_profile and integration tests."""
    await _seed_objective(shared_repo, "obj_seq")
    deps = EpistemicDeps(repo=shared_repo, agent_runner=None)
    state = EpistemicGraphState(objective_id="obj_seq")
    for _ in range(4):
        await _run_op(_AlwaysSucceedsOp, deps, state, "obj_seq", "objective", "noop")

    backend = shared_repo.store
    rows = await backend.find_by_metadata(
        {"epistemic_type": "execution_step"}, limit=100
    )
    step_numbers = sorted(row.metadata["step_number"] for row in rows)
    assert step_numbers == [1, 2, 3, 4]
