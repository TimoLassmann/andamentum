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
)
from andamentum.epistemic.repository import EpistemicRepository


@pytest.fixture
async def repo(tmp_path):
    store = DocumentStore.for_database("test_reflect", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _seed_parent_with_claims(
    repo: EpistemicRepository,
    *,
    claim_verdicts: list[tuple[str, str | None, float]] | None = None,
) -> Objective:
    """Build a parent in the post-MultiSeedClaim state: decomposition + N
    Claims linked via sub_investigation_id.

    Args:
        repo: target repo.
        claim_verdicts: optional list of (sub_id, integrated_assessment,
            integrated_confidence) triples. If None, defaults to two
            unverdicted claims (sub_ids A and B) — same shape as the
            v0.2 helper. The integration verdict simulates IBE having
            run on each claim before reflection fires.
    """
    if claim_verdicts is None:
        claim_verdicts = [("A", None, 0.0), ("B", None, 0.0)]
    parent = Objective(
        description="parent",
        clarified_question="parent",
        question_type="verificatory",
        phase="analyzed",
        decomposition={
            "sub_investigations": [
                {
                    "id": sub_id,
                    "seed_claim": f"seed for {sub_id}",
                    "rationale": f"rationale for {sub_id}",
                    "weight": 1.0,
                }
                for sub_id, _, _ in claim_verdicts
            ],
            "combination_rule": "AND",
            "rationale": "all must hold",
        },
        combination_rule="AND",
    )
    parent.objective_id = parent.entity_id
    await repo.save(parent)

    from andamentum.epistemic.entities import Claim
    from andamentum.epistemic.entities.claim import ClaimStage

    for sub_id, verdict, confidence in claim_verdicts:
        claim = Claim(
            objective_id=parent.entity_id,
            statement=f"seed for {sub_id}",
            scope=f"rationale for {sub_id}",
            stage=ClaimStage.SUPPORTED,
            sub_investigation_id=sub_id,
            integrated_assessment=verdict,
            integrated_confidence=confidence if verdict else None,
        )
        await repo.save(claim)
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

    async def test_no_claims_returns_failure(self, repo, fake_runner):
        """Decomposition exists but no claims have been minted yet
        (MultiSeedClaim hasn't run). Reflection has nothing to read."""
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
        assert "claims" in result.message.lower()

    async def test_no_agent_runner_returns_failure(self, repo):
        parent = await _seed_parent_with_claims(repo)
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
        parent = await _seed_parent_with_claims(repo)
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
        parent = await _seed_parent_with_claims(repo)
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
        parent = await _seed_parent_with_claims(repo)
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
        parent = await _seed_parent_with_claims(repo)
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


# NOTE: TestDeltaSpawn (testing SpawnSubObjectivesOperation's delta-add
# behavior for child Objectives) was removed in Phase 6. Under
# multi-seed-claim there are no child Objectives — sub-investigations
# are Claims on the parent, materialized by MultiSeedClaimOperation
# (which has its own idempotence tests in test_multi_seed_claim.py).
# SpawnSubObjectivesOperation remains in the codebase as dormant code;
# it's no longer wired into the graph.


# ── Phase 6: per-claim summary in agent prompt ────────────────────────


class TestReflectionReadsClaimState:
    """The Phase 6 rewire: ReflectOnGaps consumes per-Claim integration
    state directly off the parent Objective rather than iterating child
    Objectives. The agent's `current_decomposition` input must reflect
    each Claim's integrated_assessment / integrated_confidence /
    cycle_capped state."""

    async def test_supports_verdict_surfaces_in_summary(
        self, repo, fake_runner
    ):
        """A claim with integrated_assessment=supports + confidence=0.8
        appears as 'supports (p=0.90, confidence=0.80)' in the prompt."""
        await _seed_parent_with_claims(
            repo,
            claim_verdicts=[
                ("A", "supports", 0.8),
                ("B", "contradicts", 0.6),
            ],
        )
        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        # Re-load to get the saved entity_id from the helper.
        objs = await repo.query("objective")
        parent = next(o for o in objs if o.decomposition is not None)
        await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        # The agent received a current_decomposition string with both
        # claims' verdicts. fake_runner records calls; the most recent
        # reflect_on_gaps call should have the rendered prompt.
        reflect_calls = [
            kwargs
            for name, kwargs in fake_runner.calls
            if name == "epistemic_reflect_on_gaps"
        ]
        assert reflect_calls
        prompt_decomp = reflect_calls[-1]["current_decomposition"]
        assert "supports" in prompt_decomp
        assert "contradicts" in prompt_decomp
        # Confidence-derived posteriors: 0.5+0.8/2=0.90, 0.5-0.6/2=0.20.
        assert "p=0.90" in prompt_decomp
        assert "p=0.20" in prompt_decomp

    async def test_cycle_capped_claim_surfaces_in_summary(
        self, repo, fake_runner
    ):
        """Cycle-capped claims surface explicitly so the agent can decide
        whether to add a tie-breaker rather than silently treating them
        as no-data."""
        from andamentum.epistemic.entities import Claim
        from andamentum.epistemic.entities.claim import ClaimStage

        await _seed_parent_with_claims(
            repo,
            claim_verdicts=[("A", "supports", 0.8)],
        )
        objs = await repo.query("objective")
        parent = next(o for o in objs if o.decomposition is not None)
        # Add a B claim that hit the cap.
        capped = Claim(
            objective_id=parent.entity_id,
            statement="seed for B",
            scope="rationale for B",
            stage=ClaimStage.HYPOTHESIS,
            sub_investigation_id="B",
            cycle_capped=True,
            persistent_concerns=["unc-1", "unc-2"],
        )
        await repo.save(capped)
        # Add B to decomposition so it shows up in the per-sub loop.
        assert parent.decomposition is not None
        parent.decomposition["sub_investigations"].append(
            {
                "id": "B",
                "seed_claim": "seed for B",
                "rationale": "rationale for B",
                "weight": 1.0,
            }
        )
        await repo.save(parent)

        op = ReflectOnGapsOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="t"
        )
        await op.execute(
            OperationInput(
                entity_id=parent.entity_id,
                entity_type="objective",
                operation="reflect_on_gaps",
            )
        )
        reflect_calls = [
            kwargs
            for name, kwargs in fake_runner.calls
            if name == "epistemic_reflect_on_gaps"
        ]
        prompt_decomp = reflect_calls[-1]["current_decomposition"]
        assert "cycle_capped" in prompt_decomp
        assert "persistent_concerns=2" in prompt_decomp


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
