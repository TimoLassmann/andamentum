"""Tests for new verification tracks (contrastive evaluation, cross-claim consistency)."""

import pytest
from types import SimpleNamespace
from ..entities.claim import Claim
from ..primitives import ClaimStage
from ..agents import get_agent
from ..agents.output_models import ContrastiveEvaluationOutput, CrossClaimConsistencyOutput
from ..adapters import adapt_agent_output


class TestClaimNewFields:
    def test_contrastive_checked_defaults_false(self):
        claim = Claim(statement="test", objective_id="obj-1")
        assert claim.contrastive_checked is False

    def test_consistency_checked_defaults_false(self):
        claim = Claim(statement="test", objective_id="obj-1")
        assert claim.consistency_checked is False

    def test_can_set_contrastive_checked(self):
        claim = Claim(statement="test", objective_id="obj-1", contrastive_checked=True)
        assert claim.contrastive_checked is True

    def test_can_set_consistency_checked(self):
        claim = Claim(statement="test", objective_id="obj-1", consistency_checked=True)
        assert claim.consistency_checked is True

    def test_fields_in_metadata(self):
        claim = Claim(statement="test", objective_id="obj-1", contrastive_checked=True, consistency_checked=True)
        content, metadata = claim.to_document()
        assert metadata["contrastive_checked"] is True
        assert metadata["consistency_checked"] is True

    def test_fields_roundtrip(self):
        claim = Claim(statement="test", objective_id="obj-1", contrastive_checked=True, consistency_checked=True)
        content, metadata = claim.to_document()
        restored = Claim.from_document(content, metadata)
        assert restored.contrastive_checked is True
        assert restored.consistency_checked is True

    def test_fields_default_in_metadata(self):
        claim = Claim(statement="test", objective_id="obj-1")
        content, metadata = claim.to_document()
        assert metadata["contrastive_checked"] is False
        assert metadata["consistency_checked"] is False


class TestContrastiveEvaluationAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_contrastive_evaluation")
        assert agent.name == "epistemic_contrastive_evaluation"

    def test_output_model_fields(self):
        fields = ContrastiveEvaluationOutput.model_fields
        assert "better_claim" in fields
        assert "distinguishing_observation" in fields
        assert "confidence" in fields
        assert len(fields) == 3

    def test_adapter(self):
        raw = SimpleNamespace(better_claim="A", distinguishing_observation="test obs", confidence=0.8)
        result = adapt_agent_output("epistemic_contrastive_evaluation", raw)
        assert result.better_claim == "A"
        assert result.confidence == 0.8

    def test_adapter_normalizes_case(self):
        raw = SimpleNamespace(better_claim="a", distinguishing_observation="test", confidence=0.5)
        result = adapt_agent_output("epistemic_contrastive_evaluation", raw)
        assert result.better_claim == "A"


class TestCrossClaimConsistencyAgent:
    def test_agent_registered(self):
        agent = get_agent("epistemic_cross_claim_consistency")
        assert agent.name == "epistemic_cross_claim_consistency"

    def test_output_model_fields(self):
        fields = CrossClaimConsistencyOutput.model_fields
        assert "conflicts" in fields
        assert "tension_point" in fields
        assert len(fields) == 2

    def test_adapter_no_conflict(self):
        raw = SimpleNamespace(conflicts=False, tension_point="")
        result = adapt_agent_output("epistemic_cross_claim_consistency", raw)
        assert result.conflicts is False
        assert result.tension_point == ""

    def test_adapter_with_conflict(self):
        raw = SimpleNamespace(conflicts=True, tension_point="X contradicts Y")
        result = adapt_agent_output("epistemic_cross_claim_consistency", raw)
        assert result.conflicts is True
        assert result.tension_point == "X contradicts Y"


from ..storage import InMemoryStorageBackend
from ..repository import EpistemicRepository
from ..entities.objective import Objective
from ..entities.uncertainty import Uncertainty
from ..operations import ContrastiveEvaluationOperation, CrossClaimConsistencyOperation, OPERATION_CLASSES
from ..patterns import WorkItem


class TestContrastiveEvaluationOperation:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    def _make_runner(self, better_claim="A", confidence=0.8):
        class Runner:
            def __init__(self):
                self.calls = []
                self._agent_calls = []
            async def run(self, agent_name, **kwargs):
                self.calls.append((agent_name, kwargs))
                return SimpleNamespace(
                    better_claim=better_claim,
                    distinguishing_observation="Test observation",
                    confidence=confidence,
                )
        return Runner()

    @pytest.mark.asyncio
    async def test_sets_contrastive_checked(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(statement="Claim A", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim)

        runner = self._make_runner()
        op = ContrastiveEvaluationOperation(repo, runner)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="contrastive_evaluation")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert updated.contrastive_checked is True

    @pytest.mark.asyncio
    async def test_creates_uncertainty_when_inferior(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim_a = Claim(statement="Weak claim", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        claim_b = Claim(statement="Strong claim", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim_a)
        await repo.save(claim_b)

        # Agent says B is better — so claim_a (the target) is inferior
        runner = self._make_runner(better_claim="B", confidence=0.8)
        op = ContrastiveEvaluationOperation(repo, runner)
        work = WorkItem(entity_id=claim_a.entity_id, entity_type="claim", operation="contrastive_evaluation")
        result = await op.execute(work)

        assert result.success
        assert len(result.created_entities) == 1

    @pytest.mark.asyncio
    async def test_no_uncertainty_when_superior(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim_a = Claim(statement="Strong claim", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        claim_b = Claim(statement="Weak claim", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim_a)
        await repo.save(claim_b)

        # Agent says A is better — so claim_a (the target) is superior, no uncertainty
        runner = self._make_runner(better_claim="A", confidence=0.8)
        op = ContrastiveEvaluationOperation(repo, runner)
        work = WorkItem(entity_id=claim_a.entity_id, entity_type="claim", operation="contrastive_evaluation")
        result = await op.execute(work)

        assert result.success
        assert len(result.created_entities) == 0

    @pytest.mark.asyncio
    async def test_idempotent(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(statement="test", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED, contrastive_checked=True)
        await repo.save(claim)

        runner = self._make_runner()
        op = ContrastiveEvaluationOperation(repo, runner)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="contrastive_evaluation")
        result = await op.execute(work)

        assert result.success
        assert len(runner.calls) == 0

    @pytest.mark.asyncio
    async def test_registered_in_operation_classes(self):
        assert "contrastive_evaluation" in OPERATION_CLASSES


class TestCrossClaimConsistencyOperation:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    def _make_runner(self, conflicts=False, tension_point=""):
        class Runner:
            def __init__(self):
                self.calls = []
                self._agent_calls = []
            async def run(self, agent_name, **kwargs):
                self.calls.append((agent_name, kwargs))
                return SimpleNamespace(
                    conflicts=conflicts,
                    tension_point=tension_point,
                )
        return Runner()

    @pytest.mark.asyncio
    async def test_sets_consistency_checked(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(statement="Claim A", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim)

        runner = self._make_runner()
        op = CrossClaimConsistencyOperation(repo, runner)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="cross_claim_consistency")
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert updated.consistency_checked is True

    @pytest.mark.asyncio
    async def test_creates_uncertainty_on_conflict(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim_a = Claim(statement="X increases Y", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        claim_b = Claim(statement="X decreases Y", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim_a)
        await repo.save(claim_b)

        runner = self._make_runner(conflicts=True, tension_point="Direct contradiction about direction")
        op = CrossClaimConsistencyOperation(repo, runner)
        work = WorkItem(entity_id=claim_a.entity_id, entity_type="claim", operation="cross_claim_consistency")
        result = await op.execute(work)

        assert result.success
        assert len(result.created_entities) == 1

    @pytest.mark.asyncio
    async def test_no_uncertainty_when_consistent(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim_a = Claim(statement="X causes Y", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        claim_b = Claim(statement="Z also causes Y", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED)
        await repo.save(claim_a)
        await repo.save(claim_b)

        runner = self._make_runner(conflicts=False)
        op = CrossClaimConsistencyOperation(repo, runner)
        work = WorkItem(entity_id=claim_a.entity_id, entity_type="claim", operation="cross_claim_consistency")
        result = await op.execute(work)

        assert result.success
        assert len(result.created_entities) == 0

    @pytest.mark.asyncio
    async def test_idempotent(self, repo):
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)
        claim = Claim(statement="test", objective_id=obj.entity_id, stage=ClaimStage.SUPPORTED, consistency_checked=True)
        await repo.save(claim)

        runner = self._make_runner()
        op = CrossClaimConsistencyOperation(repo, runner)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="cross_claim_consistency")
        result = await op.execute(work)

        assert result.success
        assert len(runner.calls) == 0

    @pytest.mark.asyncio
    async def test_registered_in_operation_classes(self):
        assert "cross_claim_consistency" in OPERATION_CLASSES


from ..entities.evidence import Evidence as EvidenceEntity


class TestEvidenceClusterFields:
    def test_cluster_status_defaults_unclustered(self):
        ev = EvidenceEntity(objective_id="obj-1")
        assert ev.cluster_status == "unclustered"

    def test_corroboration_count_defaults_1(self):
        ev = EvidenceEntity(objective_id="obj-1")
        assert ev.corroboration_count == 1

    def test_cluster_fields_roundtrip(self):
        ev = EvidenceEntity(
            objective_id="obj-1",
            cluster_status="representative",
            cluster_id="cluster-abc",
            corroboration_count=5,
            corroborating_sources=["url1", "url2"],
        )
        content, metadata = ev.to_document()
        restored = EvidenceEntity.from_document(content, metadata)
        assert restored.cluster_status == "representative"
        assert restored.cluster_id == "cluster-abc"
        assert restored.corroboration_count == 5
        assert restored.corroborating_sources == ["url1", "url2"]


class TestClaimSaturatedField:
    def test_saturated_defaults_false(self):
        claim = Claim(statement="test", objective_id="obj-1")
        assert claim.saturated is False

    def test_saturated_roundtrip(self):
        claim = Claim(statement="test", objective_id="obj-1", saturated=True)
        content, metadata = claim.to_document()
        restored = Claim.from_document(content, metadata)
        assert restored.saturated is True
