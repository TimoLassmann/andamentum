"""Regression tests for the gap-analysis agent's memory.

``Claim.investigation_intents`` is a yield-annotated list: each round
the gap-analysis agent proposes appends a ``IntentRecord`` carrying the
intent text and the number of Evidence entities the routing layer
persisted for it.

These tests pin:

* The plumbing — every prior round's intent reaches the next round's
  agent, with the yield annotation in the prompt.
* The yield count — ``IntentRecord.evidence_count`` is populated from
  what ``dispatch_and_persist_for_text`` returned.
* The graceful-exit path — when the agent returns zero intents, the
  operation succeeds, persists no Evidence, and emits a "suspending
  further inquiry" message; the cap remains the floor.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.intent_record import IntentRecord
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.investigation import (
    InvestigateClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


class _RecordingRunner:
    """Agent runner with both protocols. Records each call so tests
    can assert what the gap-analysis agent saw. The dispatch core
    runner is configurable — by default it abstains so no Evidence is
    persisted; tests that need yield > 0 can swap in a commit stub."""

    def __init__(
        self,
        *,
        intents_per_round: list[list[str]],
        dispatch_committer: Any | None = None,
    ):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "stub"
        self._intents_per_round = list(intents_per_round)
        self._round = 0
        self.core_runner = dispatch_committer or _AbstainCoreRunner(parent=self)

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
        if agent_name == "epistemic_assess_evidence_quality":
            return SimpleNamespace(
                source_credibility=0.7,
                relevance=0.7,
                specificity=0.7,
                recency_appropriate=0.7,
                justification="stub",
            )
        return SimpleNamespace()

    def investigate_call_kwargs(self) -> list[dict[str, Any]]:
        return [k for name, k in self.calls if name == "epistemic_investigate_claim"]


class _AbstainCoreRunner:
    """Definition-based dispatch stub — every provider abstains."""

    def __init__(self, *, parent: _RecordingRunner):
        self._parent = parent
        self.model = parent.model

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        self._parent.calls.append((getattr(defn, "name", str(defn)), kwargs))
        return SimpleNamespace(queries=[], reasoning="abstain", confidence=0.5)


class _CommitCoreRunner:
    """Definition-based dispatch stub — every provider commits one query."""

    def __init__(self, *, parent: _RecordingRunner):
        self._parent = parent
        self.model = parent.model

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        self._parent.calls.append((getattr(defn, "name", str(defn)), kwargs))
        return SimpleNamespace(
            queries=["committed query"], reasoning="fits", confidence=0.8
        )


class _NoOpProvider:
    description = "Stub provider used only to satisfy the providers-dict guard."
    query_guidance = "n/a"
    query_examples: list[tuple[str, str | None]] = []
    output_kind = "assertion_evidence"
    independence_group = "stub"
    provider_contract_version = 1

    async def gather(self, query: str) -> list[Any]:
        return []


class _OneItemProvider:
    description = "Stub provider that returns one gathered item per gather()."
    query_guidance = "n/a"
    query_examples: list[tuple[str, str | None]] = []
    output_kind = "assertion_evidence"
    independence_group = "stub"
    provider_contract_version = 1

    def __init__(self) -> None:
        self.gather_calls: list[str] = []

    async def gather(self, query: str) -> list[Any]:
        from andamentum.epistemic.operations.base import GatheredEvidence

        self.gather_calls.append(query)
        return [
            GatheredEvidence(
                content="stub evidence content",
                source_ref=f"stub-{len(self.gather_calls)}",
                source_type="stub_one",
                evidence_kind="literature",
                quality_score=0.5,
            )
        ]


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

    async def test_yield_annotation_in_prompt(
        self, tmp_path: Path
    ) -> None:
        """Round 2 sees round 1's intent with a yielded-N-items annotation
        — and the count matches what the routing layer actually persisted."""
        claim, repo = await self._make_claim(tmp_path, "intent_mem_yield")

        runner = _RecordingRunner(
            intents_per_round=[
                ["mechanistic angle"],
                ["adversarial angle"],
            ]
        )
        # Swap in a committing core runner so the routing layer
        # produces non-zero yield on the first round.
        runner.core_runner = _CommitCoreRunner(parent=runner)

        op = InvestigateClaimOperation(
            repo,
            runner,
            embedding_model="t",
            providers={"stub": _OneItemProvider()},
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

        round2_prev = kwargs[1]["previous_intents"]
        # Each provider commits, each commit produces one evidence — so
        # round 1's intent yielded 1 item against the one stub provider.
        assert "(yielded 1 items) mechanistic angle" in round2_prev

    async def test_zero_yield_dead_end_annotated(
        self, tmp_path: Path
    ) -> None:
        """An intent the routing layer abstained on (no committed queries
        anywhere) is recorded with evidence_count=0 so the next round's
        agent can recognise the dead end."""
        claim, repo = await self._make_claim(tmp_path, "intent_mem_dead_end")
        # Default _AbstainCoreRunner → all providers abstain → 0 yield.
        runner = _RecordingRunner(intents_per_round=[["dead-end angle"]])

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

        updated = await repo.get("claim", claim.entity_id)
        assert len(updated.investigation_intents) == 1
        record = updated.investigation_intents[0]
        assert isinstance(record, IntentRecord)
        assert record.text == "dead-end angle"
        assert record.evidence_count == 0

    async def test_claim_records_intents_across_rounds(
        self, tmp_path: Path
    ) -> None:
        """Persistence: ``Claim.investigation_intents`` accumulates one
        IntentRecord per intent the agent proposed each round, with
        yield annotation in the prompt for subsequent rounds."""
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
        intent_texts = [r.text for r in updated.investigation_intents]
        assert intent_texts == [
            "intent A1",
            "intent A2",
            "intent B1",
            "intent C1",
        ]
        # All persistent records are IntentRecord instances.
        for record in updated.investigation_intents:
            assert isinstance(record, IntentRecord)

        # Each round's input shows the prior rounds' intents with annotations.
        kwargs = runner.investigate_call_kwargs()
        assert "intent A1" not in kwargs[0]["previous_intents"]
        assert "(yielded 0 items) intent A1" in kwargs[1]["previous_intents"]
        assert "(yielded 0 items) intent A2" in kwargs[1]["previous_intents"]
        assert "(yielded 0 items) intent B1" in kwargs[2]["previous_intents"]


class TestClaimGroundedDispatch:
    """Investigation passes the actual claim as the dispatch agent's
    ``claim`` and the intent as ``angle`` — keeping the claim's subject
    matter present in the dispatch context across rounds.

    The earlier shape (passing the intent as the claim) abstracted the
    claim away and produced no_bearing-dominated evidence pools — the
    v8 calibration regression. This test pins that the new shape is in
    place.
    """

    async def test_dispatch_receives_claim_and_intent_separately(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from andamentum.epistemic.operations import investigation as inv_mod

        store = DocumentStore.for_database("claim_grounded", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="Aspirin reduces the risk of colorectal cancer.",
            scope="adults at average risk",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
        )
        await repo.save(claim)

        observed: list[dict[str, Any]] = []

        async def fake_helper(
            op, c, *, objective_id, providers, core_runner,
            angle=None, sub_investigation_id=None,
            depends_on_claim_id=None, created_by="dispatch",
        ):
            observed.append({"claim": c, "angle": angle})
            return []

        monkeypatch.setattr(inv_mod, "dispatch_and_persist_for_text", fake_helper)

        runner = _RecordingRunner(intents_per_round=[["adversarial replication"]])

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

        assert len(observed) == 1
        # The dispatch helper receives the claim's statement as the
        # subject and the intent as the angle — NOT the intent as the
        # claim. This is the fix for v8's no_bearing-dominated pattern.
        assert observed[0]["claim"] == "Aspirin reduces the risk of colorectal cancer."
        assert observed[0]["angle"] == "adversarial replication"


class TestZeroIntentGracefulExit:
    """The agent may legitimately return zero intents — rational
    suspension of judgment when the search space looks exhausted
    (Peirce). The operation must accept this without padding."""

    async def test_zero_intent_return_persists_no_evidence(
        self, tmp_path: Path
    ) -> None:
        store = DocumentStore.for_database("inv_zero_intent", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(description="parent", question_type="verificatory")
        obj.objective_id = obj.entity_id
        await repo.save(obj)
        claim = Claim(
            objective_id=obj.entity_id,
            statement="X holds",
            scope="lab",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
        )
        await repo.save(claim)

        runner = _RecordingRunner(intents_per_round=[[]])  # zero intents

        op = InvestigateClaimOperation(
            repo,
            runner,
            embedding_model="t",
            providers={"stub": _NoOpProvider()},
        )
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )

        assert result.success
        assert "suspending further inquiry" in result.message.lower()
        # No Evidence persisted.
        all_evidence = await repo.query("evidence", objective_id=obj.entity_id)
        assert all_evidence == []
        # investigation_count still increments (the round happened).
        updated = await repo.get("claim", claim.entity_id)
        assert updated.investigation_count == 1
        # No IntentRecord appended (the agent proposed nothing).
        assert updated.investigation_intents == []
