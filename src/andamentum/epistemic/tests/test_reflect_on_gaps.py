"""Tests for ReflectOnGapsOperation (Phase 4).

Covers the operation in isolation:
- failure modes (no decomposition, no children, no agent runner)
- sufficient verdict path (did_work=False, history records sufficiency)
- insufficient + new sub-investigations path (decomposition grows,
  reflection_rounds bumps, deterministic re-keying)
- insufficient + empty additions (did_work=False, history records gap)

End-to-end orchestrator loop tests live in test_decomposed_runner.py.
"""

from __future__ import annotations

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.preplanning import (
    ReflectOnGapsOperation,
    SpawnSubObjectivesOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


@pytest.fixture
async def repo(tmp_path):
    store = DocumentStore.for_database("test_reflect", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _seed_parent_with_children(
    repo: EpistemicRepository,
) -> Objective:
    """Build a parent in the post-spawn state: decomposition + 2 children
    with sub_investigation_ids A and B."""
    parent = Objective(
        description="parent",
        clarified_question="parent",
        question_type="verificatory",
        phase="analyzed",
        decomposition={
            "sub_investigations": [
                {"id": "A", "seed_claim": "alpha", "rationale": "ra", "weight": 1.0},
                {"id": "B", "seed_claim": "beta", "rationale": "rb", "weight": 1.0},
            ],
            "combination_rule": "AND",
            "rationale": "both must hold",
        },
        combination_rule="AND",
    )
    parent.objective_id = parent.entity_id
    await repo.save(parent)

    spawn = SpawnSubObjectivesOperation(repo=repo, agent_runner=None, embedding_model="t")
    await spawn.execute(
        OperationInput(
            entity_id=parent.entity_id,
            entity_type="objective",
            operation="spawn_sub_objectives",
        )
    )
    return await repo.get("objective", parent.entity_id)


# ── Failure modes ─────────────────────────────────────────────────────


class TestFailureModes:
    async def test_no_decomposition_returns_failure(self, repo, fake_runner):
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is False
        assert result.did_work is False
        assert "decomposition" in result.message.lower()

    async def test_no_children_spawned_returns_failure(self, repo, fake_runner):
        obj = Objective(
            description="parent",
            question_type="verificatory",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra", "weight": 1.0}
                ],
                "combination_rule": "AND",
                "rationale": "trivial",
            },
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is False
        assert result.did_work is False
        assert "spawned children" in result.message.lower()

    async def test_no_agent_runner_returns_failure(self, repo):
        parent = await _seed_parent_with_children(repo)
        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is False
        assert result.did_work is False
        assert "agent_runner" in result.message.lower()


# ── Sufficient verdict path ───────────────────────────────────────────


class TestSufficientPath:
    async def test_sufficient_verdict_records_history_no_additions(
        self, repo, fake_runner
    ):
        parent = await _seed_parent_with_children(repo)
        # Default conftest fake declares sufficient=True.

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is True
        assert result.did_work is False

        reloaded = await repo.get("objective", parent.entity_id)
        assert reloaded.reflection_rounds == 1
        assert len(reloaded.reflection_history) == 1
        entry = reloaded.reflection_history[0]
        assert entry["sufficient"] is True
        assert entry["added_count"] == 0
        # No new sub-investigations should have been added.
        assert len(reloaded.decomposition["sub_investigations"]) == 2


# ── Insufficient + additions path ─────────────────────────────────────


class TestGapFillingPath:
    async def test_insufficient_with_additions_appends_and_rekeys(
        self, repo, fake_runner
    ):
        parent = await _seed_parent_with_children(repo)
        # Override fake to declare insufficient and propose 2 additions.
        fake_runner._overrides["epistemic_reflect_on_gaps"] = {
            "sufficient": False,
            "gap_description": "Confounder check missing.",
            "additional_sub_investigations": [
                {
                    "id": "?",  # placeholder; op re-keys
                    "seed_claim": "Confounders ruled out.",
                    "rationale": "Causal interpretation depends on this.",
                    "weight": 2.0,
                },
                {
                    "id": "?",
                    "seed_claim": "Effect size is robust to model choice.",
                    "rationale": "Robustness check.",
                    "weight": 1.0,
                },
            ],
            "rationale": "Two missing checks would close the gap.",
        }

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is True
        assert result.did_work is True

        reloaded = await repo.get("objective", parent.entity_id)
        assert reloaded.reflection_rounds == 1
        # Decomposition grew from 2 to 4.
        subs = reloaded.decomposition["sub_investigations"]
        assert len(subs) == 4
        # New ids re-keyed deterministically: existing A, B → next is C, D.
        assert subs[2]["id"] == "C"
        assert subs[3]["id"] == "D"
        assert subs[2]["seed_claim"] == "Confounders ruled out."
        assert subs[2]["weight"] == 2.0
        assert subs[3]["weight"] == 1.0
        # History captured the gap.
        entry = reloaded.reflection_history[0]
        assert entry["sufficient"] is False
        assert entry["added_count"] == 2
        assert "Confounder" in entry["gap_description"]

    async def test_insufficient_with_empty_additions_is_no_op_with_history(
        self, repo, fake_runner
    ):
        """Agent says 'gap exists' but proposes no fix → record it and
        return did_work=False rather than silently bumping rounds."""
        parent = await _seed_parent_with_children(repo)
        fake_runner._overrides["epistemic_reflect_on_gaps"] = {
            "sufficient": False,
            "gap_description": "There is a gap but I cannot articulate it.",
            "additional_sub_investigations": [],
            "rationale": "Gap suspected but no concrete additions proposed.",
        }

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        assert result.success is True
        assert result.did_work is False

        reloaded = await repo.get("objective", parent.entity_id)
        assert reloaded.reflection_rounds == 1
        entry = reloaded.reflection_history[0]
        assert entry["sufficient"] is False
        assert entry["added_count"] == 0
        # Decomposition unchanged.
        assert len(reloaded.decomposition["sub_investigations"]) == 2

    async def test_two_rounds_of_reflection_continue_re_keying(
        self, repo, fake_runner
    ):
        """ID assignment continues from the last existing ID across rounds."""
        parent = await _seed_parent_with_children(repo)
        fake_runner._overrides["epistemic_reflect_on_gaps"] = {
            "sufficient": False,
            "gap_description": "First gap.",
            "additional_sub_investigations": [
                {
                    "id": "?",
                    "seed_claim": "first add",
                    "rationale": "r",
                    "weight": 1.0,
                },
            ],
            "rationale": "round 1",
        }

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        # Round 1: adds C
        await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        # Round 2: should add D, not C again
        await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        reloaded = await repo.get("objective", parent.entity_id)
        subs = reloaded.decomposition["sub_investigations"]
        assert [s["id"] for s in subs] == ["A", "B", "C", "D"]
        assert reloaded.reflection_rounds == 2
        assert len(reloaded.reflection_history) == 2


# ── SpawnSubObjectives delta-spawning ─────────────────────────────────


class TestDeltaSpawn:
    async def test_delta_spawn_skips_already_spawned_and_creates_new(self, repo):
        """After reflection adds a new sub-investigation, re-running
        SpawnSubObjectivesOperation creates only the new child — A and
        B already exist."""
        parent = await _seed_parent_with_children(repo)  # spawns A and B
        # Mutate decomposition to simulate reflection adding C.
        assert parent.decomposition is not None
        parent.decomposition["sub_investigations"].append(
            {"id": "C", "seed_claim": "gamma", "rationale": "rc", "weight": 1.0}
        )
        await repo.save(parent)

        spawn = SpawnSubObjectivesOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        )
        result = await spawn.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="spawn_sub_objectives",
            )
        )
        assert result.success is True
        assert result.did_work is True
        assert len(result.created_entities) == 1

        reloaded = await repo.get("objective", parent.entity_id)
        # Three children total: A, B from initial spawn, C from delta.
        assert len(reloaded.sub_objective_ids) == 3
        # The new child references C.
        new_child = await repo.get("objective", result.created_entities[0])
        assert new_child.sub_investigation_id == "C"

    async def test_delta_spawn_full_match_is_did_work_false(self, repo):
        parent = await _seed_parent_with_children(repo)
        spawn = SpawnSubObjectivesOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        )
        # Second call with no decomposition changes.
        result = await spawn.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="spawn_sub_objectives",
            )
        )
        assert result.success is True
        assert result.did_work is False
        assert "Already spawned" in result.message


# ── Objective entity round-trip ───────────────────────────────────────


class TestObjectivePersistsReflectionFields:
    async def test_reflection_fields_round_trip(self, repo):
        obj = Objective(
            description="parent",
            reflection_rounds=2,
            reflection_history=[
                {
                    "round": 1,
                    "sufficient": False,
                    "gap_description": "A gap.",
                    "added_count": 1,
                    "rationale": "First reflection.",
                },
                {
                    "round": 2,
                    "sufficient": True,
                    "gap_description": "",
                    "added_count": 0,
                    "rationale": "Now sufficient.",
                },
            ],
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        loaded = await repo.get("objective", obj.entity_id)
        assert loaded.reflection_rounds == 2
        assert len(loaded.reflection_history) == 2
        assert loaded.reflection_history[0]["round"] == 1
        assert loaded.reflection_history[1]["sufficient"] is True

    async def test_reflection_defaults_when_unset(self, repo):
        obj = Objective(description="parent")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        loaded = await repo.get("objective", obj.entity_id)
        assert loaded.reflection_rounds == 0
        assert loaded.reflection_history == []
