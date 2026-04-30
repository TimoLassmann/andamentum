"""Tests for the top-down question decomposition (Phase 1, standalone).

Phase 1 ships only the agent + operation. No graph integration yet; that
comes in Phase 2 (sub-objective infrastructure) and Phase 3 (graph wiring).
These tests verify the agent registration, the operation's execute()
behaviour, and the seed_claim-mode bypass.
"""

from __future__ import annotations

import pytest

from andamentum.epistemic.agents.output_models import (
    QuestionDecomposition,
    SubInvestigation,
)
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.preplanning import DecomposeQuestionOperation


# ── Output model tests ────────────────────────────────────────────────


class TestQuestionDecompositionSchema:
    def test_minimum_2_sub_investigations(self):
        """Schema enforces at least 2 sub-investigations."""
        with pytest.raises(ValueError):
            QuestionDecomposition(
                sub_investigations=[
                    SubInvestigation(
                        id="A", seed_claim="only one", rationale="not enough"
                    ),
                ],
                combination_rule="AND",
                rationale="too few",
            )

    def test_maximum_5_sub_investigations(self):
        """Schema enforces at most 5 sub-investigations."""
        with pytest.raises(ValueError):
            QuestionDecomposition(
                sub_investigations=[
                    SubInvestigation(id=str(i), seed_claim=f"c{i}", rationale=f"r{i}")
                    for i in range(6)
                ],
                combination_rule="AND",
                rationale="too many",
            )

    def test_valid_decomposition_constructs(self):
        """A 3-item decomposition with valid combination_rule is accepted."""
        d = QuestionDecomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="alpha", rationale="reason a"),
                SubInvestigation(id="B", seed_claim="beta", rationale="reason b"),
                SubInvestigation(id="C", seed_claim="gamma", rationale="reason c"),
            ],
            combination_rule="AND",
            rationale="three pillars must hold",
        )
        assert len(d.sub_investigations) == 3
        assert d.combination_rule == "AND"

    def test_combination_rule_must_be_one_of_four(self):
        """combination_rule is a Literal; invalid values rejected at runtime."""
        with pytest.raises(ValueError):
            QuestionDecomposition(
                sub_investigations=[
                    SubInvestigation(id="A", seed_claim="x", rationale="y"),
                    SubInvestigation(id="B", seed_claim="x", rationale="y"),
                ],
                combination_rule="MAJORITY",  # type: ignore[arg-type]  # intentional: testing runtime rejection
                rationale="invalid rule",
            )


# ── Operation tests ───────────────────────────────────────────────────


@pytest.fixture
async def repo(tmp_path):
    from andamentum.document_store import DocumentStore
    from andamentum.epistemic.repository import EpistemicRepository

    store = DocumentStore.for_database("test_decompose", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


async def _make_objective(
    repo,
    description: str,
    *,
    question_type: str | None = "verificatory",
    claim_to_verify: str | None = None,
) -> Objective:
    obj = Objective(
        description=description,
        clarified_question=description,
        question_type=question_type,
        claim_to_verify=claim_to_verify,
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj


class TestDecomposeQuestionOperation:
    async def test_calls_agent_and_returns_decomposition_summary(
        self, repo, fake_runner
    ):
        """Operation calls the agent and surfaces the decomposition in the message."""
        obj = await _make_objective(
            repo,
            "Are podocytes motile in injury?",
            question_type="verificatory",
        )

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        assert result.success is True
        assert "Decomposed into 3 sub-investigations" in result.message
        assert "combination=AND" in result.message
        # Each sub-investigation's id should appear in the summary
        for sub_id in ("A:", "B:", "C:"):
            assert sub_id in result.message

    async def test_seed_claim_mode_bypasses_decomposition(self, repo, fake_runner):
        """When claim_to_verify is set, the operation skips with did_work=False."""
        obj = await _make_objective(
            repo,
            "Are podocytes motile in injury?",
            question_type="verificatory",
            claim_to_verify="Podocytes are motile in the presence of injury.",
        )

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        assert result.success is True
        assert result.did_work is False
        assert "claim_to_verify" in result.message
        assert "seed_claim" in result.message.lower()
        # The agent should NOT have been called.
        assert not any(
            call[0] == "epistemic_decompose_question" for call in fake_runner.calls
        )

    async def test_no_agent_runner_returns_failure(self, repo):
        """Without an agent runner, decomposition can't proceed; clear failure."""
        obj = await _make_objective(repo, "test", question_type="verificatory")

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=None, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        assert result.success is False
        assert result.did_work is False
        assert "agent_runner" in result.message.lower()

    async def test_falls_back_to_verificatory_when_question_type_missing(
        self, repo, fake_runner
    ):
        """Missing question_type defaults to verificatory; agent still receives a value."""
        obj = await _make_objective(repo, "test", question_type=None)

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        assert result.success is True
        # Find the agent call and verify question_type was passed
        decompose_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_decompose_question"
        ]
        assert len(decompose_calls) == 1
        kwargs = decompose_calls[0][1]
        assert kwargs.get("question_type") == "verificatory"

    async def test_uses_clarified_question_when_available(self, repo, fake_runner):
        """If clarified_question exists, the agent receives that rather than description."""
        obj = Objective(
            description="raw question",
            clarified_question="clarified version of the question",
            question_type="verificatory",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        await op.execute(
            OperationInput(
                entity_id=obj.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        decompose_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_decompose_question"
        ]
        assert decompose_calls[0][1]["question"] == "clarified version of the question"

    async def test_does_not_call_agent_in_seed_claim_mode(self, repo, fake_runner):
        """Idempotence-flavoured: seed_claim mode bypass leaves the agent untouched."""
        # This complements test_seed_claim_mode_bypasses_decomposition by
        # asserting on the runner state across two calls.
        obj1 = await _make_objective(
            repo,
            "Q1",
            question_type="verificatory",
            claim_to_verify="seed claim 1",
        )
        obj2 = await _make_objective(repo, "Q2", question_type="verificatory")

        op = DecomposeQuestionOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        # First call: bypassed (seed_claim mode)
        await op.execute(
            OperationInput(
                entity_id=obj1.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )
        # Second call: not bypassed
        await op.execute(
            OperationInput(
                entity_id=obj2.entity_id,
                entity_type="objective",
                operation="decompose_question",
            )
        )

        decompose_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_decompose_question"
        ]
        # Only the non-seed-claim case fired the agent.
        assert len(decompose_calls) == 1
