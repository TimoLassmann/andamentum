"""Regression test: InvestigateClaimOperation filters resolved uncertainties.

Fix #3 from the investigation-cycling analysis (2026-05-12). Prior to
this fix, ``InvestigateClaimOperation`` queried every uncertainty
affecting the claim — resolved or not — and concatenated the
descriptions into the agent's ``scrutiny_issues`` prompt input. So an
issue that had already been closed by ``ResolveUncertaintyOperation``
kept being re-targeted by subsequent investigation rounds, both wasting
LLM cycles and crowding out the genuinely-open issues.

The fix is a one-line ``if u.is_resolved: continue`` in the loop that
builds ``scrutiny_issues``. This test pins the regression by stamping
``resolved_at`` on one of two uncertainties and asserting that only the
unresolved description reaches the agent.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import (
    Claim,
    Objective,
    Uncertainty,
)
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.uncertainty import UncertaintyType
from andamentum.epistemic.operations.base import OperationInput
from andamentum.epistemic.operations.investigation import (
    InvestigateClaimOperation,
)
from andamentum.epistemic.repository import EpistemicRepository


class _CapturingRunner:
    """Stub runner that records what scrutiny_issues was passed to
    ``epistemic_investigate_claim`` and returns a minimal valid output."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "stub-model"

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        if agent_name == "epistemic_investigate_claim":
            return SimpleNamespace(
                evidence_queries=[
                    SimpleNamespace(
                        source_type="web_search",
                        query="follow-up",
                    )
                ],
                reasoning="stub",
            )
        # rank_providers path may also fire; provide a permissive default.
        return SimpleNamespace(chosen_provider="web_search")

    def last_call_kwargs(self, agent_name: str) -> dict[str, Any] | None:
        for name, kwargs in reversed(self.calls):
            if name == agent_name:
                return kwargs
        return None


class TestInvestigationFiltersResolvedUncertainties:
    async def test_resolved_uncertainty_excluded_from_agent_input(
        self, tmp_path: Path
    ) -> None:
        store = DocumentStore.for_database("inv_resolved_filter", db_dir=tmp_path)
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

        # Unresolved uncertainty — should reach the agent.
        open_unc = Uncertainty(
            objective_id=obj.entity_id,
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="OPEN: need a mechanistic study under condition Y",
            affected_claim_ids=[claim.entity_id],
        )
        await repo.save(open_unc)

        # Resolved uncertainty — should NOT reach the agent. We stamp
        # ``resolved_at`` (which is what ``is_resolved`` actually checks)
        # to a real datetime so the property returns True.
        resolved_unc = Uncertainty(
            objective_id=obj.entity_id,
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="CLOSED: this gap was already filled in round 1",
            affected_claim_ids=[claim.entity_id],
            resolution="filled by prior evidence",
            resolved_at=datetime.now(),
        )
        await repo.save(resolved_unc)

        runner = _CapturingRunner()
        op = InvestigateClaimOperation(
            repo=repo, agent_runner=runner, embedding_model="t"
        )
        result = await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )

        assert result.success

        invoke_kwargs = runner.last_call_kwargs("epistemic_investigate_claim")
        assert invoke_kwargs is not None, "investigate_claim agent never fired"

        scrutiny_issues_text = invoke_kwargs["scrutiny_issues"]
        assert "OPEN: need a mechanistic study" in scrutiny_issues_text
        assert "CLOSED:" not in scrutiny_issues_text, (
            "Resolved uncertainty description leaked into the agent prompt — "
            "Fix #3 regressed."
        )

    async def test_no_unresolved_yields_placeholder_string(
        self, tmp_path: Path
    ) -> None:
        """When every uncertainty is resolved, scrutiny_issues should be
        the empty-list placeholder, not a string accidentally built from
        the resolved descriptions."""
        store = DocumentStore.for_database(
            "inv_all_resolved", db_dir=tmp_path
        )
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

        for i in range(3):
            unc = Uncertainty(
                objective_id=obj.entity_id,
                uncertainty_type=UncertaintyType.EVIDENCE_GAP,
                description=f"resolved issue #{i}",
                affected_claim_ids=[claim.entity_id],
                resolution="closed",
                resolved_at=datetime.now(),
            )
            await repo.save(unc)

        runner = _CapturingRunner()
        op = InvestigateClaimOperation(
            repo=repo, agent_runner=runner, embedding_model="t"
        )
        await op.execute(
            OperationInput(
                entity_id=claim.entity_id,
                entity_type="claim",
                operation="investigate_claim",
            )
        )

        invoke_kwargs = runner.last_call_kwargs("epistemic_investigate_claim")
        assert invoke_kwargs is not None
        assert invoke_kwargs["scrutiny_issues"] == "No specific issues recorded"
