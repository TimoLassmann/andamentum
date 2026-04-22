"""Tests for adversarial counterargument storage as Evidence entities."""

import pytest
from ..entities.claim import Claim, ClaimStage
from ..entities.evidence import Evidence
from ..operations.verification import AdversarialSearchOperation
from ..operations.base import OperationInput


class TestAdversarialEvidenceStorage:
    """Quality-passing adversarial counterarguments stored as Evidence."""

    @pytest.mark.asyncio
    async def test_quality_counterarguments_stored_as_evidence(self, repo, fake_runner):
        """Quality-passing counterarguments become Evidence with support_judgment='contradicts'."""
        claim = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Homeopathy cures infections",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-1"],
        )
        await repo.save(claim)

        ev = Evidence(
            entity_id="e-1",
            objective_id="obj-1",
            source_type="paper",
            source_ref="https://example.com/study1",
            extracted_content="RCT showing positive result",
            extracted=True,
            support_judgment="supports",
            quality_score=0.7,
            cluster_status="representative",
        )
        await repo.save(ev)

        # Mock evidence gatherer that returns search hits
        class MockEvidenceGatherer:
            async def gather(self, provider, query):
                from types import SimpleNamespace

                return [
                    SimpleNamespace(
                        content="Cochrane review finds no evidence for homeopathy",
                        source_ref="https://example.com/cochrane",
                    ),
                ]

        op = AdversarialSearchOperation(repo=repo, agent_runner=fake_runner)
        op.evidence_gatherer = MockEvidenceGatherer()  # type: ignore[assignment]

        work = OperationInput(
            entity_id="c-1",
            entity_type="claim",
            operation="adversarial_search",
        )
        result = await op.execute(work)
        assert result.success

        # Check that adversarial evidence was stored
        all_evidence = await repo.get_evidence_for_objective("obj-1")
        adversarial_evidence = [
            e
            for e in all_evidence
            if e.support_judgment == "contradicts"
            and "adversarial" in (e.judgment_reasoning or "").lower()
        ]

        # Should have at least one adversarial evidence entity
        assert len(adversarial_evidence) >= 1

        # Check the evidence fields
        ae = adversarial_evidence[0]
        assert ae.source_type == "web_search"
        assert ae.extracted is True
        assert ae.cluster_status == "representative"

        # Check it's linked to the claim
        claim_after = await repo.get("claim", "c-1")
        assert any(
            ae.entity_id in claim_after.evidence_ids for ae in adversarial_evidence
        )
