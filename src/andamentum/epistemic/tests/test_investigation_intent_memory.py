"""Regression test: InvestigateClaimOperation passes prior intents
back into the agent's prompt.

The pre-rewrite ``epistemic_investigate_claim`` agent had zero memory
of prior rounds — each call saw the same claim + scrutiny issues and
generated essentially the same lexicon-permutation queries each time.

The rewrite passes ``previous_intents`` to the agent as an explicit
input, and ``Claim.investigation_intents`` is the storage layer for
that memory. The agent is instructed to propose a fundamentally
different angle from anything in the list, but enforcement happens
at the prompt level. This test pins the *plumbing*: every prior
round's intent reaches the agent on the next call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.investigation import (
    InvestigateClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


class _RecordingRunner:
    """Agent runner with both protocols. Records each call so the test
    can assert what the gap-analysis agent saw, and routes the dispatch
    agent to immediate abstain so the routing layer is a no-op."""

    def __init__(self, *, intents_per_round: list[list[str]]):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "stub"
        self._intents_per_round = list(intents_per_round)
        self._round = 0
        self.core_runner = _AbstainCoreRunner(parent=self)

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        if agent_name == "epistemic_investigate_claim":
            intents = (
                self._intents_per_round[self._round]
                if self._round < len(self._intents_per_round)
                else []
            )
            self._round += 1
            return SimpleNamespace(intents=intents, reasoning="stub")
        return SimpleNamespace()

    def investigate_call_kwargs(self) -> list[dict[str, Any]]:
        return [k for name, k in self.calls if name == "epistemic_investigate_claim"]


class _AbstainCoreRunner:
    def __init__(self, *, parent: _RecordingRunner):
        self._parent = parent
        self.model = parent.model

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        self._parent.calls.append((getattr(defn, "name", str(defn)), kwargs))
        return SimpleNamespace(queries=[], reasoning="abstain", confidence=0.5)


class _NoOpProvider:
    description = "Stub provider used only to satisfy the providers-dict guard."
    query_guidance = "n/a"
    query_examples: list[tuple[str, str | None]] = []
    output_kind = "assertion_evidence"
    independence_group = "stub"
    provider_contract_version = 1

    async def gather(self, query: str) -> list[Any]:
        return []


class TestInvestigationIntentMemory:
    async def _make_claim(
        self, tmp_path: Path, name: str
    ) -> tuple[Claim, EpistemicRepository]:
        store = DocumentStore.for_database(name, db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="X holds in Y conditions",
            scope="laboratory replication",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
        )
        await repo.save(claim)
        return claim, repo

    async def test_first_round_sees_empty_prior_intents(
        self, tmp_path: Path
    ) -> None:
        """Round 1: previous_intents is the (none) placeholder."""
        claim, repo = await self._make_claim(tmp_path, "intent_mem_round1")
        runner = _RecordingRunner(intents_per_round=[["round-1 intent"]])

        op = InvestigateClaimOperation(
            repo,
            runner,
            embedding_model="t",
            providers={"stub": _NoOpProvider()},
        )
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )

        kwargs = runner.investigate_call_kwargs()
        assert len(kwargs) == 1
        prev = kwargs[0]["previous_intents"]
        assert "first investigation round" in prev.lower()

    async def test_second_round_sees_first_round_intent(
        self, tmp_path: Path
    ) -> None:
        """Round 2: previous_intents contains the intent from round 1."""
        claim, repo = await self._make_claim(tmp_path, "intent_mem_round2")
        runner = _RecordingRunner(
            intents_per_round=[
                ["mechanistic angle from round 1"],
                ["adversarial angle from round 2"],
            ]
        )

        op = InvestigateClaimOperation(
            repo,
            runner,
            embedding_model="t",
            providers={"stub": _NoOpProvider()},
        )

        # Round 1
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )
        # Round 2
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )

        kwargs = runner.investigate_call_kwargs()
        assert len(kwargs) == 2

        # Round 1 sees the placeholder.
        assert "first investigation round" in kwargs[0]["previous_intents"].lower()

        # Round 2 sees round 1's intent.
        round2_prev = kwargs[1]["previous_intents"]
        assert "mechanistic angle from round 1" in round2_prev
        assert "adversarial angle from round 2" not in round2_prev  # not yet posted

    async def test_claim_records_intents_across_rounds(
        self, tmp_path: Path
    ) -> None:
        """Persistence: ``Claim.investigation_intents`` accumulates one
        entry per intent the agent proposed each round."""
        claim, repo = await self._make_claim(tmp_path, "intent_mem_persist")
        runner = _RecordingRunner(
            intents_per_round=[
                ["intent A1", "intent A2"],
                ["intent B1"],
                ["intent C1"],
            ]
        )

        op = InvestigateClaimOperation(
            repo,
            runner,
            embedding_model="t",
            providers={"stub": _NoOpProvider()},
        )

        for _ in range(3):
            await op.execute(
                OperationInput(
                    entity_id=claim.entity_id,
                    entity_type="claim",
                    operation="investigate_claim",
                )
            )

        updated = await repo.get("claim", claim.entity_id)
        assert updated.investigation_intents == [
            "intent A1",
            "intent A2",
            "intent B1",
            "intent C1",
        ]
        # And each round's input grew accordingly.
        kwargs = runner.investigate_call_kwargs()
        assert "intent A1" not in kwargs[0]["previous_intents"]
        assert "intent A1" in kwargs[1]["previous_intents"]
        assert "intent A2" in kwargs[1]["previous_intents"]
        assert "intent B1" in kwargs[2]["previous_intents"]
