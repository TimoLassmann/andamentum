"""Tests for epistemic operations using DocumentStore + MockAgentRunner."""

import pytest
from typing import Any
from unittest.mock import patch, AsyncMock

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Uncertainty,
    Snapshot,
)
from ..entities.uncertainty import UncertaintyType
from ..operations import (
    ExtractEvidenceOperation,
    GatheredEvidence,
    OPERATION_CLASSES,
    create_operations,
    ProposeClaimsOperation,
)
from ..patterns import OperationInput


class FakeEvidenceGatherer:
    """Stub evidence gatherer returning canned results."""

    def __init__(self, items: list[GatheredEvidence] | None = None):
        self.items = items or [
            GatheredEvidence(
                content="Spaced repetition improves retention by 40%",
                source_ref="https://example.com/sr",
                source_type="web_search",
                quality_score=0.7,
            ),
        ]
        self.calls: list[tuple[str, str]] = []

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        self.calls.append((source_type, query))
        return self.items


class TestOperationRegistry:
    def test_operations_registered(self):
        assert len(OPERATION_CLASSES) > 0
        assert "clarify_question" in OPERATION_CLASSES
        assert "propose_claims" in OPERATION_CLASSES
        assert "promote_claim" in OPERATION_CLASSES

    def test_create_operations(self, repo, fake_runner):
        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        assert "clarify_question" in ops
        assert "scrutinise_claim" in ops


class TestPreplanningChain:
    async def test_clarify_question(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="What is spaced repetition?",
        )
        await repo.save(obj)

        from andamentum.epistemic.alignment import AlignmentResult

        mock_validation = AsyncMock(
            return_value=AlignmentResult(aligned=True, issue="", suggestion="")
        )

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="clarify_question"
        )
        with patch(
            "andamentum.epistemic.alignment.validate_alignment", mock_validation
        ):
            result = await ops["clarify_question"].execute(work)

        assert result.success
        loaded = await repo.get_objective("obj-1")
        assert loaded.phase == "clarified"

    async def test_conceptual_analysis(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="What is spaced repetition?",
            phase="clarified",
        )
        await repo.save(obj)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="conceptual_analysis"
        )
        result = await ops["conceptual_analysis"].execute(work)

        assert result.success
        loaded = await repo.get_objective("obj-1")
        assert loaded.phase == "analyzed"

    async def test_plan_task(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="What is spaced repetition?",
            phase="analyzed",
        )
        await repo.save(obj)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="plan_task"
        )

        # PlanTaskOperation now uses semantic routing. Mock embed_texts so
        # the test does not require a live Ollama backend.
        from andamentum.epistemic import provider_routing

        provider_routing._clear_cache()
        fake_vec = [1.0, 0.0, 0.0]

        async def _fake_embed(texts, *, model):
            return [fake_vec] * len(texts)

        with patch(
            "andamentum.epistemic.provider_routing.embed_texts",
            side_effect=_fake_embed,
        ):
            result = await ops["plan_task"].execute(work)

        assert result.success
        loaded = await repo.get_objective("obj-1")
        assert loaded.phase == "planned"


class TestEvidenceExtraction:
    async def test_extract_evidence(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-1",
            objective_id="obj-1",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        gatherer = FakeEvidenceGatherer()
        ops = create_operations(repo, fake_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-1", entity_type="evidence", operation="extract_evidence"
        )
        result = await ops["extract_evidence"].execute(work)

        assert result.success
        loaded = await repo.get_evidence("e-1")
        assert loaded.extracted is True

    async def test_extract_multiple_creates_per_source_entities(
        self, repo, fake_runner
    ):
        """Multiple GatheredEvidence items create individual Evidence entities."""
        obj = Objective(
            entity_id="obj-m",
            objective_id="obj-m",
            description="Multi Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-m",
            objective_id="obj-m",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        gatherer = FakeEvidenceGatherer(
            items=[
                GatheredEvidence(
                    content="Source A content",
                    source_ref="https://a.com",
                    source_type="web_search",
                    quality_score=0.5,
                ),
                GatheredEvidence(
                    content="Source B content",
                    source_ref="https://b.com",
                    source_type="web_search",
                    quality_score=0.6,
                ),
                GatheredEvidence(
                    content="Source C content",
                    source_ref="https://c.com",
                    source_type="openalex",
                    quality_score=0.8,
                ),
            ]
        )
        ops = create_operations(repo, fake_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-m", entity_type="evidence", operation="extract_evidence"
        )
        result = await ops["extract_evidence"].execute(work)

        assert result.success
        assert len(result.created_entities) == 3

        # Verify all entities exist and are extracted
        all_evidence = await repo.query(
            "evidence", objective_id="obj-m", extracted=True
        )
        assert len(all_evidence) == 3

    async def test_new_entities_have_extracted_true(self, repo, fake_runner):
        """New Evidence entities from multi-source extraction must have extracted=True."""
        obj = Objective(
            entity_id="obj-ext",
            objective_id="obj-ext",
            description="Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-ext",
            objective_id="obj-ext",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        gatherer = FakeEvidenceGatherer(
            items=[
                GatheredEvidence(
                    content="First",
                    source_ref="https://1.com",
                    source_type="web_search",
                ),
                GatheredEvidence(
                    content="Second",
                    source_ref="https://2.com",
                    source_type="web_search",
                ),
            ]
        )
        ops = create_operations(repo, fake_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-ext", entity_type="evidence", operation="extract_evidence"
        )
        await ops["extract_evidence"].execute(work)

        all_evidence = await repo.query(
            "evidence", objective_id="obj-ext", extracted=True
        )
        assert len(all_evidence) == 2
        for ev in all_evidence:
            assert ev.extracted is True, (
                f"Evidence {ev.entity_id} should have extracted=True"
            )

    async def test_new_entities_inherit_objective_id(self, repo, fake_runner):
        """New Evidence entities must inherit objective_id from the original stub."""
        obj = Objective(
            entity_id="obj-inh",
            objective_id="obj-inh",
            description="Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-inh",
            objective_id="obj-inh",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        gatherer = FakeEvidenceGatherer(
            items=[
                GatheredEvidence(
                    content="A", source_ref="https://a.com", source_type="web_search"
                ),
                GatheredEvidence(
                    content="B", source_ref="https://b.com", source_type="openalex"
                ),
            ]
        )
        ops = create_operations(repo, fake_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-inh", entity_type="evidence", operation="extract_evidence"
        )
        await ops["extract_evidence"].execute(work)

        all_evidence = await repo.query("evidence", objective_id="obj-inh")
        assert len(all_evidence) == 2
        for ev in all_evidence:
            assert ev.objective_id == "obj-inh"

    async def test_quality_scores_per_entity(self, repo, fake_runner):
        """Each Evidence entity gets a quality score via agent assessment."""
        obj = Objective(
            entity_id="obj-qs", objective_id="obj-qs", description="Q", phase="planned"
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-qs",
            objective_id="obj-qs",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        gatherer = FakeEvidenceGatherer(
            items=[
                GatheredEvidence(
                    content="A", source_ref="https://a.com", source_type="web_search"
                ),
                GatheredEvidence(
                    content="B", source_ref="https://b.com", source_type="openalex"
                ),
                GatheredEvidence(
                    content="C", source_ref="https://c.com", source_type="web_search"
                ),
            ]
        )
        ops = create_operations(repo, fake_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-qs", entity_type="evidence", operation="extract_evidence"
        )
        result = await ops["extract_evidence"].execute(work)

        assert result.success
        all_evidence = await repo.query(
            "evidence", objective_id="obj-qs", extracted=True
        )
        # All evidence scored by agent assessment (no OpenAlex scorer injected)
        for ev in all_evidence:
            assert ev.quality_score is not None, (
                f"Evidence {ev.entity_id} should have a quality score"
            )
            assert ev.quality_metadata is not None
            assert ev.quality_metadata.get("source") == "agent"

    @pytest.mark.asyncio
    async def test_structured_data_passed_to_experimental_context(
        self, repo, fake_runner
    ):
        """When GatheredEvidence has structured_data, it should populate Evidence.experimental_context."""
        obj = Objective(
            entity_id="obj-sd", objective_id="obj-sd", description="Test objective"
        )
        await repo.save(obj)
        evidence = Evidence(
            entity_id="e-sd",
            objective_id="obj-sd",
            source_type="web_search",
            source_ref="http://example.com",
        )
        await repo.save(evidence)

        gathered = GatheredEvidence(
            content="Raw page text here.",
            source_ref="http://example.com",
            source_type="web_search",
            structured_data={
                "ai_summary": "AI thinks this page says X.",
                "key_points": ["Point 1", "Point 2"],
                "key_excerpts": ['"Verbatim quote"'],
            },
        )

        op = ExtractEvidenceOperation(repo, fake_runner, evidence_gatherer=None)
        op._fill_evidence_from_gathered(evidence, gathered)

        assert evidence.extracted_content == "Raw page text here."
        assert evidence.experimental_context is not None
        assert "AI thinks this page says X." in evidence.experimental_context
        assert "Point 1" in evidence.experimental_context
        assert "Verbatim quote" in evidence.experimental_context

    @pytest.mark.asyncio
    async def test_empty_structured_data_leaves_context_none(self, repo, fake_runner):
        """When structured_data is empty, experimental_context should stay None."""
        obj = Objective(
            entity_id="obj-nosd", objective_id="obj-nosd", description="Test objective"
        )
        await repo.save(obj)
        evidence = Evidence(
            entity_id="e-nosd",
            objective_id="obj-nosd",
            source_type="web_search",
            source_ref="http://example.com",
        )
        await repo.save(evidence)

        gathered = GatheredEvidence(
            content="Raw page text.",
            source_ref="http://example.com",
            source_type="web_search",
        )

        op = ExtractEvidenceOperation(repo, fake_runner, evidence_gatherer=None)
        op._fill_evidence_from_gathered(evidence, gathered)

        assert evidence.extracted_content == "Raw page text."
        assert evidence.experimental_context is None


class TestAgentOnlyExtraction:
    """Test Strategy 2: agent extraction without evidence_gatherer still scores."""

    async def test_agent_only_extraction_scores_evidence(self, repo, fake_runner):
        """When no evidence_gatherer is provided, agent extraction still scores quality."""
        obj = Objective(
            entity_id="obj-ao",
            objective_id="obj-ao",
            description="Test Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-ao",
            objective_id="obj-ao",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        # Create operations WITHOUT evidence_gatherer (model=None skips auto-creation)
        ops = create_operations(repo, fake_runner, evidence_gatherer=None)
        work = OperationInput(
            entity_id="e-ao", entity_type="evidence", operation="extract_evidence"
        )
        result = await ops["extract_evidence"].execute(work)

        assert result.success
        loaded = await repo.get_evidence("e-ao")
        assert loaded.extracted is True
        assert loaded.quality_score is not None, (
            "Agent-only extraction must score evidence"
        )
        assert loaded.quality_metadata is not None
        assert loaded.quality_metadata.get("source") == "agent"

    async def test_no_runner_no_gatherer_raises(self, repo, fake_runner):
        """When neither a runner nor a gatherer is wired up, extraction must raise
        rather than fabricate placeholder content."""
        obj = Objective(
            entity_id="obj-df",
            objective_id="obj-df",
            description="Test Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-df",
            objective_id="obj-df",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        # No runner, no gatherer — must raise, not silently fabricate content
        ops = create_operations(repo, agent_runner=None, evidence_gatherer=None)
        work = OperationInput(
            entity_id="e-df", entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="no extractor"):
            await ops["extract_evidence"].execute(work)

    async def test_gatherer_exception_propagates(self, repo, fake_runner):
        """When gatherer throws, the exception propagates out of the operation."""

        class FailingGatherer:
            async def gather(
                self, source_type: str, query: str
            ) -> list[GatheredEvidence]:
                raise ConnectionError("SearXNG not running")

        obj = Objective(
            entity_id="obj-gf",
            objective_id="obj-gf",
            description="Test Q",
            phase="planned",
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-gf", objective_id="obj-gf", source_type="all", extracted=False
        )
        await repo.save(e)

        ops = create_operations(repo, fake_runner, evidence_gatherer=FailingGatherer())
        work = OperationInput(
            entity_id="e-gf", entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(ConnectionError):
            await ops["extract_evidence"].execute(work)

    async def test_gatherer_agent_failure_propagates(self, repo):
        """When agent scoring fails, the exception propagates out of the operation."""

        # Runner that fails on quality assessment
        class FailingRunner:
            def __init__(self):
                self.calls: list[tuple[str, dict]] = []

            async def run(self, agent_name: str, **kwargs):
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_assess_evidence_quality":
                    raise RuntimeError("LLM unavailable")
                # Return a SimpleNamespace for extract_evidence
                from types import SimpleNamespace

                return SimpleNamespace(
                    relevant_quotes=["Test content"],
                    limitations=[],
                    experimental_context="test",
                )

        runner = FailingRunner()

        obj = Objective(
            entity_id="obj-fb",
            objective_id="obj-fb",
            description="Test Q",
            phase="planned",
        )
        await repo.save(obj)

        gatherer = FakeEvidenceGatherer(
            [
                GatheredEvidence(
                    content="Real evidence content",
                    source_ref="https://example.com",
                    source_type="web_search",
                    quality_score=0.6,
                ),
            ]
        )

        ops = create_operations(repo, runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id="e-fb", entity_type="evidence", operation="extract_evidence"
        )
        e = Evidence(
            entity_id="e-fb",
            objective_id="obj-fb",
            source_type="web_search",
            extracted=False,
        )
        await repo.save(e)

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            await ops["extract_evidence"].execute(work)


class TestScrutiny:
    async def test_scrutiny_pass(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="planned"
        )
        await repo.save(obj)
        c = Claim(entity_id="c-1", objective_id="obj-1", statement="X causes Y")
        await repo.save(c)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="c-1", entity_type="claim", operation="scrutinise_claim"
        )
        result = await ops["scrutinise_claim"].execute(work)

        assert result.success
        loaded = await repo.get_claim("c-1")
        assert loaded.scrutiny_verdict == "pass"

    async def test_scrutiny_fail(self, repo, fake_runner):
        # Override split agents with a failing scrutiny response
        fake_runner._overrides["epistemic_assess_evidence"] = {
            "claim_id": "c-1",
            "evidence_weight": "conflicting",
            "confidence_estimate": 0.25,
            "justification": "Evidence contradicts the claim",
        }
        fake_runner._overrides["epistemic_identify_issues"] = {
            "claim_id": "c-1",
            "issues": [
                {
                    "description": "Poorly scoped",
                    "issue_type": "scope_difference",
                    "reversal_test": False,
                },
            ],
        }
        # Also override legacy agent for backward compat testing
        fake_runner._overrides["epistemic_scrutinise_claim"] = {
            "passes_scrutiny": False,
            "recommendation": "demote",
            "issues_found": ["Poorly scoped"],
            "issue_types": ["scope"],
            "evidence_weight": 0.2,
            "confidence_estimate": 0.3,
        }
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="planned"
        )
        await repo.save(obj)
        c = Claim(
            entity_id="c-1", objective_id="obj-1", statement="Everything is related"
        )
        await repo.save(c)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="c-1", entity_type="claim", operation="scrutinise_claim"
        )
        result = await ops["scrutinise_claim"].execute(work)

        assert result.success
        loaded = await repo.get_claim("c-1")
        assert loaded.scrutiny_verdict == "fail"


class TestPromotion:
    async def test_promote_hypothesis_to_supported(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Q", phase="planned"
        )
        await repo.save(obj)
        e = Evidence(
            entity_id="e-1", objective_id="obj-1", quality_score=0.5, extracted=True
        )
        await repo.save(e)
        c = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="X",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="pass",
            evidence_ids=["e-1"],
        )
        await repo.save(c)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(entity_id="c-1", entity_type="claim", operation="promote_claim")
        result = await ops["promote_claim"].execute(work)

        assert result.success
        loaded = await repo.get_claim("c-1")
        assert loaded.stage == ClaimStage.SUPPORTED


class TestProposeClaims:
    async def test_propose_claims_creates_entities(self, repo, fake_runner):
        from andamentum.epistemic.alignment import AlignmentResult

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Does spaced repetition work?",
            phase="planned",
            claims_proposed=False,
        )
        await repo.save(obj)

        mock_validation = AsyncMock(
            return_value=AlignmentResult(aligned=True, issue="", suggestion="")
        )

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="propose_claims"
        )
        with patch(
            "andamentum.epistemic.alignment.validate_alignment", mock_validation
        ):
            result = await ops["propose_claims"].execute(work)

        assert result.success
        claims = await repo.get_claims_for_objective("obj-1")
        assert len(claims) >= 1
        obj_loaded = await repo.get_objective("obj-1")
        assert obj_loaded.claims_proposed is True


class TestFreezeSnapshot:
    async def test_freeze_snapshot(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)
        c = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="X",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(c)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        # A snapshot should have been created
        snapshots = await repo.query("snapshot", objective_id="obj-1")
        assert len(snapshots) >= 1
        assert snapshots[0].frozen is True

    async def test_freeze_snapshot_includes_hypothesis_claims(self, repo, fake_runner):
        """Hypothesis claims should be included in snapshots (not filtered out)."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)
        # A hypothesis claim (not yet promoted)
        h = Claim(
            entity_id="c-hyp",
            objective_id="obj-1",
            statement="Hypothesis X",
            stage=ClaimStage.HYPOTHESIS,
        )
        # A supported claim
        s = Claim(
            entity_id="c-sup",
            objective_id="obj-1",
            statement="Supported Y",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(h)
        await repo.save(s)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        snapshots = await repo.query("snapshot", objective_id="obj-1")
        assert len(snapshots) >= 1
        # Both hypothesis and supported claims should be in the snapshot
        assert "c-hyp" in snapshots[0].claim_ids
        assert "c-sup" in snapshots[0].claim_ids

    async def test_freeze_snapshot_excludes_abandoned_claims(self, repo, fake_runner):
        """Abandoned claims should be excluded from snapshots."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)
        # An abandoned claim
        a = Claim(
            entity_id="c-abn",
            objective_id="obj-1",
            statement="Abandoned Z",
            stage=ClaimStage.HYPOTHESIS,
            abandoned=True,
        )
        await repo.save(a)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        snapshots = await repo.query("snapshot", objective_id="obj-1")
        assert len(snapshots) >= 1
        assert "c-abn" not in snapshots[0].claim_ids


class TestCaveatDedup:
    """Test that duplicate caveats are deduped during freeze_snapshot."""

    @pytest.mark.asyncio
    async def test_duplicate_caveats_deduped(self, repo, fake_runner):
        """Near-duplicate non-blocking uncertainties are resolved during snapshot freeze."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)

        # Create 4 caveats: 3 near-duplicates + 1 distinct
        dup_emb = [1.0, 0.0, 0.0]
        dup_emb_2 = [0.98, 0.1, 0.0]  # cosine ≈ 0.995 to dup_emb
        dup_emb_3 = [0.96, 0.15, 0.0]  # cosine ≈ 0.988 to dup_emb
        distinct_emb = [0.0, 0.0, 1.0]  # cosine = 0.0 to dup_emb

        for i, (eid, desc) in enumerate(
            [
                ("u-dup1", "Evidence is about radiologists not comp bio"),
                ("u-dup2", "Source focuses on radiology not computational biology"),
                ("u-dup3", "Studies examined radiologists not bioinformatics"),
                ("u-distinct", "Sample sizes are too small"),
            ]
        ):
            u = Uncertainty(
                entity_id=eid,
                objective_id="obj-1",
                uncertainty_type=UncertaintyType.SCOPE_DIFFERENCE,
                description=desc,
                affected_claim_ids=["c-1"],
            )
            await repo.save(u)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        emb_map = {
            "Evidence is about radiologists not comp bio": dup_emb,
            "Source focuses on radiology not computational biology": dup_emb_2,
            "Studies examined radiologists not bioinformatics": dup_emb_3,
            "Sample sizes are too small": distinct_emb,
        }

        async def fake_embed(texts, **kwargs):
            return [emb_map.get(t, [0.5, 0.5, 0.0]) for t in texts]

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed
        ):
            work = OperationInput(
                entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
            )
            result = await ops["freeze_snapshot"].execute(work)

        assert result.success

        # Check uncertainties: 2 of the 3 duplicates should be resolved (deduped)
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        unresolved = [u for u in all_u if u.resolution is None]
        resolved = [u for u in all_u if u.resolution is not None]

        # Should have 2 unresolved (1 representative from dup group + 1 distinct)
        assert len(unresolved) == 2
        # Should have 2 resolved (the non-representative duplicates)
        assert len(resolved) == 2
        # Resolved ones should mention "Deduplicated"
        for u in resolved:
            assert "Deduplicated" in u.resolution

    @pytest.mark.asyncio
    async def test_no_dedup_with_single_caveat(self, repo, fake_runner):
        """A single caveat doesn't trigger dedup."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)

        u = Uncertainty(
            entity_id="u-1",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.SCOPE_DIFFERENCE,
            description="Only one caveat",
            affected_claim_ids=["c-1"],
        )
        await repo.save(u)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        assert all(u.resolution is None for u in all_u)

    @pytest.mark.asyncio
    async def test_blocking_uncertainties_not_deduped(self, repo, fake_runner):
        """Blocking uncertainties are never touched by caveat dedup."""
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)

        # Create 2 identical blocking uncertainties
        for eid in ["u-b1", "u-b2"]:
            u = Uncertainty(
                entity_id=eid,
                objective_id="obj-1",
                uncertainty_type=UncertaintyType.UNKNOWN,  # blocking type
                description="This is a blocking uncertainty",
                affected_claim_ids=["c-1"],
            )
            await repo.save(u)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="freeze_snapshot"
        )
        result = await ops["freeze_snapshot"].execute(work)

        assert result.success
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        # Both should remain unresolved — blocking uncertainties are not caveats
        assert all(u.resolution is None for u in all_u)


class TestSynthesizeReport:
    async def test_synthesize_report(self, repo, fake_runner):
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Q",
            phase="claims_done",
        )
        await repo.save(obj)
        snap = Snapshot(entity_id="snap-1", objective_id="obj-1", snapshot_type="final")
        await repo.save(snap)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="snap-1", entity_type="snapshot", operation="synthesize_report"
        )
        result = await ops["synthesize_report"].execute(work)

        assert result.success
        # An artefact should have been created
        artefacts = await repo.query("artefact", objective_id="obj-1")
        assert len(artefacts) >= 1


class TestResolveUncertaintySiblingGrouping:
    """Tests that similar unresolved siblings are resolved together."""

    async def test_similar_siblings_resolved_together(self, repo, fake_runner):
        """When resolving an uncertainty, similar siblings (cosine > 0.8) get the same resolution."""
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test objective"
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.HYPOTHESIS,
        )
        await repo.save(claim)

        # The uncertainty to resolve
        u_target = Uncertainty(
            entity_id="u-target",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Effect of caffeine on sleep duration",
            affected_claim_ids=["c-1"],
        )
        # Similar sibling — should be resolved too
        u_similar = Uncertainty(
            entity_id="u-similar",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="How caffeine affects sleep length",
            affected_claim_ids=["c-1"],
        )
        # Different sibling — should NOT be resolved
        u_different = Uncertainty(
            entity_id="u-different",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Effect of exercise on cardiovascular fitness",
            affected_claim_ids=["c-1"],
        )
        await repo.save(u_target)
        await repo.save(u_similar)
        await repo.save(u_different)

        # fake_runner default: can_resolve=True, resolution="Resolved through additional evidence"
        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        # Embeddings: target + similar have cosine > 0.8; target + different have cosine < 0.8
        # target=emb[0], similar=emb[1], different=emb[2]
        # We arrange embeddings so target·similar > 0.8 and target·different < 0.5
        target_emb = [1.0, 0.0, 0.0]
        similar_emb = [0.9, 0.1, 0.0]  # cosine with target ≈ 0.994
        different_emb = [0.0, 1.0, 0.0]  # cosine with target = 0.0

        # embed_texts is called with [target_desc] + [sibling_descs...]
        # order of siblings is non-deterministic; we handle it by returning a mock
        # that maps position in call order
        call_count = []

        async def fake_embed_texts(texts, **kwargs):
            call_count.append(len(texts))
            result = []
            for text in texts:
                if text == u_target.description:
                    result.append(target_emb)
                elif text == u_similar.description:
                    result.append(similar_emb)
                else:
                    result.append(different_emb)
            return result

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed_texts
        ):
            work = OperationInput(
                entity_id="u-target",
                entity_type="uncertainty",
                operation="resolve_uncertainty",
            )
            result = await ops["resolve_uncertainty"].execute(work)

        assert result.success

        # Target must be resolved
        loaded_target = await repo.get("uncertainty", "u-target")
        assert loaded_target.resolution is not None

        # Similar sibling must also be resolved with the same resolution
        loaded_similar = await repo.get("uncertainty", "u-similar")
        assert loaded_similar.resolution is not None
        assert loaded_similar.resolution == loaded_target.resolution

        # Different sibling must remain unresolved
        loaded_different = await repo.get("uncertainty", "u-different")
        assert loaded_different.resolution is None

    async def test_sibling_grouping_fails_when_embedding_unavailable(
        self, repo, fake_runner
    ):
        """When embed_texts raises, the error propagates — no silent fallbacks."""
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test objective"
        )
        await repo.save(obj)

        u_target = Uncertainty(
            entity_id="u-target",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Some uncertainty",
            affected_claim_ids=[],
        )
        u_sibling = Uncertainty(
            entity_id="u-sibling",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Some uncertainty similar",
            affected_claim_ids=[],
        )
        await repo.save(u_target)
        await repo.save(u_sibling)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        with patch(
            "andamentum.epistemic.embeddings.embed_texts",
            side_effect=RuntimeError("Ollama unavailable"),
        ):
            work = OperationInput(
                entity_id="u-target",
                entity_type="uncertainty",
                operation="resolve_uncertainty",
            )
            with pytest.raises(RuntimeError, match="Ollama unavailable"):
                await ops["resolve_uncertainty"].execute(work)

        # Sibling is NOT resolved (no embedding = no grouping)
        loaded_sibling = await repo.get("uncertainty", "u-sibling")
        assert loaded_sibling.resolution is None


class TestResolveUncertaintyDedup:
    """Tests that duplicate remaining concerns are not spawned."""

    async def test_duplicate_concern_not_spawned(self, repo, fake_runner):
        """A remaining concern similar to an existing uncertainty is skipped."""
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test objective"
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.HYPOTHESIS,
        )
        await repo.save(claim)

        # An already-existing uncertainty with a similar description to the concern
        existing = Uncertainty(
            entity_id="u-existing",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Lack of long-term studies on caffeine effects",
            affected_claim_ids=["c-1"],
            resolution="Resolved through additional evidence",
        )
        # Manually resolve it so it has a resolution set
        existing.resolve("Already resolved")
        await repo.save(existing)

        # New unresolved uncertainty to resolve
        u_new = Uncertainty(
            entity_id="u-new",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Does caffeine cause sleep problems?",
            affected_claim_ids=["c-1"],
        )
        await repo.save(u_new)

        # Configure fake_runner to return a concern similar to the existing uncertainty
        concern_text = (
            "Absence of long-term studies on caffeine"  # near-duplicate of existing
        )
        runner = fake_runner.__class__(
            overrides={
                "epistemic_resolve_uncertainty": {
                    "can_resolve": True,
                    "resolution": "Resolved",
                    "remaining_concerns": [concern_text],
                }
            }
        )

        ops = create_operations(repo, runner, embedding_model="test-model")

        existing_emb = [1.0, 0.0, 0.0]
        concern_emb = [0.95, 0.05, 0.0]  # cosine ≈ 0.998 — near-duplicate

        async def fake_embed_texts(texts, **kwargs):
            result = []
            for text in texts:
                if text == concern_text:
                    result.append(concern_emb)
                elif text == existing.description:
                    result.append(existing_emb)
                else:
                    # u_new description or anything else
                    result.append([0.5, 0.5, 0.0])
            return result

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed_texts
        ):
            work = OperationInput(
                entity_id="u-new",
                entity_type="uncertainty",
                operation="resolve_uncertainty",
            )
            result = await ops["resolve_uncertainty"].execute(work)

        assert result.success

        # Only the two original uncertainties should exist — no new child was spawned
        all_uncertainties = await repo.query("uncertainty", objective_id="obj-1")
        assert len(all_uncertainties) == 2
        ids = {u.entity_id for u in all_uncertainties}
        assert "u-existing" in ids
        assert "u-new" in ids

    async def test_unique_concern_is_spawned(self, repo, fake_runner):
        """A remaining concern NOT similar to any existing uncertainty IS spawned."""
        obj = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test objective"
        )
        await repo.save(obj)

        claim = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.HYPOTHESIS,
        )
        await repo.save(claim)

        existing = Uncertainty(
            entity_id="u-existing",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Lack of long-term studies on caffeine",
            affected_claim_ids=["c-1"],
        )
        existing.resolve("Already resolved")
        await repo.save(existing)

        u_new = Uncertainty(
            entity_id="u-new",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Does caffeine cause sleep problems?",
            affected_claim_ids=["c-1"],
        )
        await repo.save(u_new)

        concern_text = (
            "Impact of caffeine on cognitive performance"  # unrelated to existing
        )
        runner = fake_runner.__class__(
            overrides={
                "epistemic_resolve_uncertainty": {
                    "can_resolve": True,
                    "resolution": "Resolved",
                    "remaining_concerns": [concern_text],
                }
            }
        )

        ops = create_operations(repo, runner, embedding_model="test-model")

        existing_emb = [1.0, 0.0, 0.0]
        concern_emb = [0.0, 0.0, 1.0]  # cosine = 0.0 to all others

        async def fake_embed_texts(texts, **kwargs):
            result = []
            for text in texts:
                if text == concern_text:
                    result.append(concern_emb)
                elif text == existing.description:
                    result.append(existing_emb)
                else:
                    result.append([0.0, 1.0, 0.0])  # orthogonal to both
            return result

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed_texts
        ):
            work = OperationInput(
                entity_id="u-new",
                entity_type="uncertainty",
                operation="resolve_uncertainty",
            )
            result = await ops["resolve_uncertainty"].execute(work)

        assert result.success

        # Concern should be buffered on the objective, not created as entity yet
        obj_reloaded = await repo.get("objective", "obj-1")
        assert len(obj_reloaded.pending_concerns) == 1
        assert obj_reloaded.pending_concerns[0]["text"] == concern_text

        # Uncertainty count should still be 2 (no immediate creation)
        all_uncertainties = await repo.query("uncertainty", objective_id="obj-1")
        assert len(all_uncertainties) == 2


class TestDeduplicateConcerns:
    """Test batch dedup of buffered remaining concerns."""

    @pytest.mark.asyncio
    async def test_batch_dedup_collapses_duplicates(self, repo, fake_runner):
        """Near-duplicate buffered concerns collapse to one uncertainty."""

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test objective",
            phase="claims_done",
            pending_concerns=[
                {
                    "text": "Evidence is about radiologists not comp bio",
                    "parent_id": "u-1",
                    "affected_claim_ids": ["c-1"],
                    "depth": 1,
                },
                {
                    "text": "Source focuses on radiology not computational biology",
                    "parent_id": "u-2",
                    "affected_claim_ids": ["c-1"],
                    "depth": 1,
                },
                {
                    "text": "Studies examined radiologists not bioinformatics",
                    "parent_id": "u-3",
                    "affected_claim_ids": ["c-1"],
                    "depth": 1,
                },
                {
                    "text": "Sample sizes are too small for conclusions",
                    "parent_id": "u-4",
                    "affected_claim_ids": ["c-2"],
                    "depth": 1,
                },
            ],
        )
        await repo.save(obj)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        # Embeddings: first 3 are near-duplicates, 4th is distinct
        dup_emb = [1.0, 0.0, 0.0]
        dup_emb_2 = [0.98, 0.1, 0.0]
        dup_emb_3 = [0.96, 0.15, 0.0]
        distinct_emb = [0.0, 0.0, 1.0]

        emb_map = {
            "Evidence is about radiologists not comp bio": dup_emb,
            "Source focuses on radiology not computational biology": dup_emb_2,
            "Studies examined radiologists not bioinformatics": dup_emb_3,
            "Sample sizes are too small for conclusions": distinct_emb,
        }

        async def fake_embed(texts, **kwargs):
            return [emb_map.get(t, [0.5, 0.5, 0.0]) for t in texts]

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed
        ):
            work = OperationInput(
                entity_id="obj-1",
                entity_type="objective",
                operation="deduplicate_concerns",
            )
            result = await ops["deduplicate_concerns"].execute(work)

        assert result.success

        # Should create 2 uncertainties (1 per theme), not 4
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        assert len(all_u) == 2

        # Buffer should be cleared
        obj_reloaded = await repo.get("objective", "obj-1")
        assert len(obj_reloaded.pending_concerns) == 0

    @pytest.mark.asyncio
    async def test_filters_against_existing_uncertainties(self, repo, fake_runner):
        """Concerns that match existing uncertainties are dropped entirely."""

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test objective",
            phase="claims_done",
            pending_concerns=[
                {
                    "text": "This duplicates an existing uncertainty",
                    "parent_id": "u-1",
                    "affected_claim_ids": ["c-1"],
                    "depth": 1,
                },
            ],
        )
        await repo.save(obj)

        # Create an existing uncertainty with the same theme
        existing = Uncertainty(
            entity_id="u-existing",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="This is an existing concern about the same topic",
            affected_claim_ids=["c-1"],
        )
        await repo.save(existing)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        # Both texts get very similar embeddings
        async def fake_embed(texts, **kwargs):
            return [[1.0, 0.0, 0.0] for _ in texts]

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed
        ):
            work = OperationInput(
                entity_id="obj-1",
                entity_type="objective",
                operation="deduplicate_concerns",
            )
            result = await ops["deduplicate_concerns"].execute(work)

        assert result.success

        # No new uncertainties created (concern matched existing)
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        assert len(all_u) == 1  # only the pre-existing one
        assert all_u[0].entity_id == "u-existing"

    @pytest.mark.asyncio
    async def test_empty_buffer_noop(self, repo, fake_runner):
        """Empty pending_concerns is a no-op."""

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test objective",
            phase="claims_done",
        )
        await repo.save(obj)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id="obj-1", entity_type="objective", operation="deduplicate_concerns"
        )
        result = await ops["deduplicate_concerns"].execute(work)

        assert result.success
        assert "No pending" in result.message

    @pytest.mark.asyncio
    async def test_depth_demotion(self, repo, fake_runner):
        """Concerns at depth >= MAX_UNCERTAINTY_DEPTH become non-blocking."""
        from andamentum.epistemic.operations.base import MAX_UNCERTAINTY_DEPTH

        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Test objective",
            phase="claims_done",
            pending_concerns=[
                {
                    "text": "Deep concern",
                    "parent_id": "u-1",
                    "affected_claim_ids": ["c-1"],
                    "depth": MAX_UNCERTAINTY_DEPTH,
                },
            ],
        )
        await repo.save(obj)

        ops = create_operations(repo, fake_runner, embedding_model="test-model")

        async def fake_embed(texts, **kwargs):
            return [[float(i), 0.0, 0.0] for i, _ in enumerate(texts)]

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", side_effect=fake_embed
        ):
            work = OperationInput(
                entity_id="obj-1",
                entity_type="objective",
                operation="deduplicate_concerns",
            )
            result = await ops["deduplicate_concerns"].execute(work)

        assert result.success
        all_u = await repo.query("uncertainty", objective_id="obj-1")
        assert len(all_u) == 1
        # Should be demoted to non-blocking
        assert not all_u[0].is_blocking


class TestEvidenceRelevanceFiltering:
    """Test that irrelevant evidence is filtered before assertion extraction."""

    @pytest.mark.asyncio
    async def test_irrelevant_evidence_filtered(self, repo, fake_runner):
        """Evidence screened as irrelevant should not reach assertion extraction."""
        from types import SimpleNamespace

        obj = Objective(
            description="What should computational biologists do about AI?",
            phase="planned",
            claims_proposed=False,
        )
        obj.objective_id = obj.entity_id  # objectives are self-referential
        await repo.save(obj)

        ev_relevant = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref="doi:10.1234/relevant",
            extracted_content="AI is automating routine bioinformatics tasks, shifting demand toward validation roles.",
            extracted=True,
        )
        ev_irrelevant = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref="doi:10.1234/sensors",
            extracted_content="Flexible sensors were developed as an alternative to rigid sensors for wearable devices.",
            extracted=True,
        )
        await repo.save(ev_relevant)
        await repo.save(ev_irrelevant)

        # Track which agents are called and with what
        agent_calls: list[str] = []
        original_run = fake_runner.run

        async def tracking_run(agent_name: str, **kwargs: Any) -> Any:
            agent_calls.append(agent_name)
            if agent_name == "epistemic_screen_relevance":
                content = kwargs.get("evidence_content", "")
                if "sensors" in content.lower():
                    return SimpleNamespace(
                        is_relevant=False, reason="About sensors, not biology careers"
                    )
                return SimpleNamespace(
                    is_relevant=True, reason="Directly about AI in biology"
                )
            return await original_run(agent_name, **kwargs)

        fake_runner.run = tracking_run

        op = ProposeClaimsOperation(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id=obj.entity_id, entity_type="objective", operation="propose_claims"
        )

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.5] * 10]
            with patch(
                "andamentum.epistemic.similarity.group_by_similarity"
            ) as mock_cluster:
                mock_cluster.return_value = [[0]]
                result = await op.execute(work)

        assert result.success
        # Screen should have been called for both items
        screen_calls = [c for c in agent_calls if c == "epistemic_screen_relevance"]
        assert len(screen_calls) == 2
        # Assertion extraction should only have been called for the relevant one
        extract_calls = [c for c in agent_calls if c == "epistemic_extract_assertion"]
        assert len(extract_calls) == 1

    @pytest.mark.asyncio
    async def test_screening_failure_includes_evidence(self, repo, fake_runner):
        """If screening fails, evidence should be included (fail-open)."""
        obj = Objective(
            description="Test question",
            phase="planned",
            claims_proposed=False,
        )
        obj.objective_id = obj.entity_id  # objectives are self-referential
        await repo.save(obj)

        ev = Evidence(
            objective_id=obj.entity_id,
            source_type="openalex",
            source_ref="doi:10.1234/test",
            extracted_content="Some test content.",
            extracted=True,
        )
        await repo.save(ev)

        # Make screen_relevance raise an exception
        original_run = fake_runner.run

        async def failing_screen(agent_name: str, **kwargs: Any) -> Any:
            if agent_name == "epistemic_screen_relevance":
                raise RuntimeError("Screening failed")
            return await original_run(agent_name, **kwargs)

        fake_runner.run = failing_screen

        op = ProposeClaimsOperation(repo, fake_runner, embedding_model="test-model")
        work = OperationInput(
            entity_id=obj.entity_id, entity_type="objective", operation="propose_claims"
        )

        with patch(
            "andamentum.epistemic.embeddings.embed_texts", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.5] * 10]
            with patch(
                "andamentum.epistemic.similarity.group_by_similarity"
            ) as mock_cluster:
                mock_cluster.return_value = [[0]]
                result = await op.execute(work)

        # Should still succeed — screening failure doesn't block the pipeline
        assert result.success
