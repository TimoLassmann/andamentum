"""Tests for Phase 2: sub-objective infrastructure.

Verifies:
- Objective entity round-trips the new fields (parent_objective_id,
  sub_investigation_id, decomposition, sub_objective_ids,
  combination_rule)
- DecomposeQuestionOperation persists the decomposition
- SpawnSubObjectivesOperation creates correct child objectives
- Idempotence at both stages
- Failure modes (no decomposition, missing agent runner)
"""

from __future__ import annotations

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.preplanning import (
    DecomposeQuestionOperation,
    SpawnSubObjectivesOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


# ── Persistence round-trip tests ──────────────────────────────────────


@pytest.fixture
async def repo(tmp_path):
    store = DocumentStore.for_database("test_p2", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


class TestObjectivePersistence:
    async def test_new_fields_round_trip(self, repo):
        """All five Phase-2 fields persist and reload correctly."""
        obj = Objective(
            description="parent question",
            parent_objective_id="parent-id-12345",
            sub_investigation_id="A",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "claim-a", "rationale": "rat-a"},
                    {"id": "B", "seed_claim": "claim-b", "rationale": "rat-b"},
                ],
                "combination_rule": "AND",
                "rationale": "all must hold",
            },
            sub_objective_ids=["child-1", "child-2"],
            combination_rule="AND",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        loaded = await repo.get("objective", obj.entity_id)
        assert loaded.parent_objective_id == "parent-id-12345"
        assert loaded.sub_investigation_id == "A"
        assert loaded.decomposition is not None
        assert loaded.decomposition["combination_rule"] == "AND"
        assert len(loaded.decomposition["sub_investigations"]) == 2
        assert loaded.sub_objective_ids == ["child-1", "child-2"]
        assert loaded.combination_rule == "AND"

    async def test_unset_fields_default_cleanly(self, repo):
        """Existing objectives without the new fields load with defaults."""
        obj = Objective(description="root question")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        loaded = await repo.get("objective", obj.entity_id)
        assert loaded.parent_objective_id is None
        assert loaded.sub_investigation_id is None
        assert loaded.decomposition is None
        assert loaded.sub_objective_ids == []
        assert loaded.combination_rule is None


# ── DecomposeQuestionOperation persistence ───────────────────────────


class TestDecomposeQuestionPersists:
    async def test_decomposition_written_to_objective(self, repo, fake_runner):
        """After successful decomposition, the parent has decomposition + combination_rule set."""
        obj = Objective(description="test", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = DecomposeQuestionOperation(repo=repo, agent_runner=fake_runner, embedding_model="t")
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )
        assert result.success

        reloaded = await repo.get("objective", obj.entity_id)
        assert reloaded.decomposition is not None
        assert reloaded.combination_rule == "AND"
        # The conftest mock returns 3 sub-investigations
        assert len(reloaded.decomposition["sub_investigations"]) == 3

    async def test_idempotent_on_second_call(self, repo, fake_runner):
        """Re-running on an already-decomposed objective is a did_work=False no-op."""
        obj = Objective(description="test", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = DecomposeQuestionOperation(repo=repo, agent_runner=fake_runner, embedding_model="t")
        await op.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )
        # Second call
        result = await op.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )
        assert result.success is True
        assert result.did_work is False
        assert "already decomposed" in result.message.lower()

        # Agent fired only on the first call.
        decompose_calls = [c for c in fake_runner.calls if c[0] == "epistemic_decompose_question"]
        assert len(decompose_calls) == 1


# ── SpawnSubObjectivesOperation ──────────────────────────────────────


class TestSpawnSubObjectives:
    async def test_spawns_one_child_per_sub_investigation(self, repo, fake_runner):
        """3-sub-investigation decomposition spawns 3 children."""
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # First decompose, then spawn.
        decompose = DecomposeQuestionOperation(repo=repo, agent_runner=fake_runner, embedding_model="t")
        await decompose.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )

        spawn = SpawnSubObjectivesOperation(repo=repo, agent_runner=None, embedding_model="t")
        result = await spawn.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )

        assert result.success
        assert len(result.created_entities) == 3

        reloaded_parent = await repo.get("objective", obj.entity_id)
        assert reloaded_parent.sub_objective_ids == result.created_entities

    async def test_each_child_has_correct_parent_linkage(self, repo, fake_runner):
        """Each child has parent_objective_id, sub_investigation_id, claim_to_verify set."""
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        await DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )
        result = await SpawnSubObjectivesOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )

        sub_ids_seen: set[str] = set()
        for child_id in result.created_entities:
            child = await repo.get("objective", child_id)
            assert child.parent_objective_id == obj.entity_id
            assert child.sub_investigation_id in {"A", "B", "C"}
            sub_ids_seen.add(child.sub_investigation_id)
            # claim_to_verify is set, so the child runs in seed_claim mode.
            assert child.claim_to_verify is not None
            assert child.claim_to_verify != ""
            # Question type inherited from parent.
            assert child.question_type == "verificatory"

        assert sub_ids_seen == {"A", "B", "C"}

    async def test_children_run_in_seed_claim_mode(self, repo, fake_runner):
        """A child objective's claim_to_verify is the sub-investigation's seed_claim verbatim."""
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        await DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )
        result = await SpawnSubObjectivesOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )

        # The conftest mock's first sub_investigation has seed_claim
        # "There is a plausible mechanism for the claim."
        first_child = await repo.get("objective", result.created_entities[0])
        assert first_child.claim_to_verify == "There is a plausible mechanism for the claim."

    async def test_idempotent_does_not_double_spawn(self, repo, fake_runner):
        """Second call with already-spawned children is did_work=False."""
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        await DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="decompose_question")
        )
        spawn = SpawnSubObjectivesOperation(repo=repo, agent_runner=None, embedding_model="t")
        await spawn.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )
        # Second call
        result = await spawn.execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )
        assert result.success is True
        assert result.did_work is False

        reloaded = await repo.get("objective", obj.entity_id)
        # Still exactly 3 children, not 6.
        assert len(reloaded.sub_objective_ids) == 3

    async def test_fails_cleanly_when_no_decomposition_present(self, repo):
        """Spawn without prior decomposition returns success=False, did_work=False."""
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        result = await SpawnSubObjectivesOperation(
            repo=repo, agent_runner=None, embedding_model="t"
        ).execute(
            OperationInput(entity_id=obj.entity_id, entity_type="objective", operation="spawn_sub_objectives")
        )
        assert result.success is False
        assert result.did_work is False
        assert "decomposition" in result.message.lower()
