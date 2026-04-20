"""Tests for the seed claim verification mode."""

from __future__ import annotations

import pytest

from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.evidence import Evidence
from andamentum.epistemic.entities.objective import Objective
from andamentum.epistemic.operations.seed_claim import SeedClaimOperation
from andamentum.epistemic.patterns import OperationInput
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.storage import InMemoryStorageBackend


@pytest.fixture
def repo():
    return EpistemicRepository(InMemoryStorageBackend())


class TestSeedClaimOperation:
    async def test_creates_claim_from_claim_to_verify(self, repo):
        """Seed claim uses the verbatim claim_to_verify text."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is the following claim true? Podocytes are motile.",
            claim_to_verify="Podocytes are motile and migrate in the presence of injury.",
            phase="planned",
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        result = await op.execute(work)

        assert result.success
        assert len(result.created_entities) == 1

        # Verify the claim entity
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims) == 1
        assert claims[0].statement == "Podocytes are motile and migrate in the presence of injury."
        assert claims[0].stage == ClaimStage.HYPOTHESIS

    async def test_links_all_evidence_to_seed_claim(self, repo):
        """The seed claim should be linked to all extracted evidence."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
        )
        await repo.save(obj)

        # Add some evidence
        for i in range(3):
            ev = Evidence(
                objective_id="obj-1",
                source_type="pubmed",
                source_ref=f"https://example.com/{i}",
                extracted=True,
                extracted_content=f"Evidence {i}",
            )
            await repo.save(ev)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        result = await op.execute(work)

        assert result.success
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims[0].evidence_ids) == 3

    async def test_sets_claims_proposed_and_phase(self, repo):
        """After seeding, objective.claims_proposed=True and phase='claims_proposed'."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        await op.execute(work)

        loaded = await repo.get_objective("obj-1")
        assert loaded.claims_proposed is True
        assert loaded.phase == "claims_proposed"

    async def test_defaults_question_type_to_verificatory(self, repo):
        """When question_type is None, seed_claim defaults it to 'verificatory'."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
            question_type=None,
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        await op.execute(work)

        loaded = await repo.get_objective("obj-1")
        assert loaded.question_type == "verificatory"

    async def test_preserves_existing_question_type(self, repo):
        """If question_type is already set, seed_claim doesn't overwrite it."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
            question_type="predictive",
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        await op.execute(work)

        loaded = await repo.get_objective("obj-1")
        assert loaded.question_type == "predictive"

    async def test_idempotent_second_call(self, repo):
        """Second call is a no-op when claims_proposed is already True."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        await op.execute(work)
        result2 = await op.execute(work)

        assert result2.success
        assert result2.message == "Seed claim already created"
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims) == 1  # not duplicated

    async def test_judges_evidence_against_seed_claim(self, repo):
        """With an agent_runner, evidence items get support_judgment set."""
        from types import SimpleNamespace

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
        )
        await repo.save(obj)

        # Add evidence with content
        for i in range(3):
            ev = Evidence(
                objective_id="obj-1",
                source_type="pubmed",
                source_ref=f"https://example.com/{i}",
                extracted=True,
                extracted_content=f"Study {i} shows X is true.",
            )
            await repo.save(ev)

        # Fake agent runner that returns "supports" for everything
        class FakeRunner:
            async def run(self, agent_name, **kwargs):
                return SimpleNamespace(
                    verdict="supports",
                    reasoning="Evidence directly supports the claim.",
                )

        op = SeedClaimOperation(repo, agent_runner=FakeRunner())
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        result = await op.execute(work)

        assert result.success
        assert "3 evidence items judged" in result.message

        # Verify support_judgment was set on all evidence
        all_ev = await repo.query("evidence", objective_id="obj-1")
        for ev in all_ev:
            assert ev.support_judgment == "supports"
            assert ev.judgment_reasoning == "Evidence directly supports the claim."

    async def test_skips_judging_without_agent_runner(self, repo):
        """Without an agent_runner, evidence is linked but not judged."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Is X true?",
            claim_to_verify="X is true.",
            phase="planned",
        )
        await repo.save(obj)

        ev = Evidence(
            objective_id="obj-1",
            source_type="pubmed",
            source_ref="https://example.com/1",
            extracted=True,
            extracted_content="Some content.",
        )
        await repo.save(ev)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        result = await op.execute(work)

        assert result.success
        # Evidence linked but not judged (no agent_runner)
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims[0].evidence_ids) == 1
        all_ev = await repo.query("evidence", objective_id="obj-1")
        assert all_ev[0].support_judgment is None

    async def test_fails_without_claim_to_verify(self, repo):
        """If claim_to_verify is None, the operation fails cleanly."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Just a question",
            phase="planned",
        )
        await repo.save(obj)

        op = SeedClaimOperation(repo, agent_runner=None)
        work = OperationInput(entity_id="obj-1", entity_type="objective", operation="seed_claim")
        result = await op.execute(work)

        assert not result.success
        assert "claim_to_verify" in result.message
