"""Tests for epistemic entity classes."""

import pytest

from ..entities import (
    EpistemicEntity,
    ENTITY_CLASSES,
    Objective,
    Evidence,
    Claim,
    ClaimStage,
    Uncertainty,
    UncertaintyType,
    BLOCKING_TYPES,
    Decision,
    Snapshot,
    Artefact,
)


class TestEntityCreation:
    def test_claim_auto_id(self):
        c = Claim(statement="X causes Y", objective_id="obj-1")
        assert c.entity_id  # auto-generated UUID
        assert c.entity_type == "claim"
        assert c.stage == ClaimStage.HYPOTHESIS

    def test_evidence_auto_id(self):
        e = Evidence(objective_id="obj-1", source_type="web_search")
        assert e.entity_id
        assert e.entity_type == "evidence"

    def test_objective_creation(self):
        o = Objective(
            entity_id="obj-1", objective_id="obj-1", description="Test question"
        )
        assert o.phase == "new"
        assert o.status == "active"

    def test_uncertainty_creation(self):
        u = Uncertainty(
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,
            description="Not enough data",
        )
        assert u.entity_type == "uncertainty"
        assert not u.is_blocking  # EVIDENCE_GAP is non-blocking

    def test_decision_creation(self):
        d = Decision(
            objective_id="obj-1",
            statement="Proceed with validation",
            justification="Evidence is sufficient",
        )
        assert not d.is_reversed
        assert d.reversible

    def test_snapshot_creation(self):
        s = Snapshot(
            objective_id="obj-1",
            snapshot_type="final",
            claim_ids=["c1", "c2"],
        )
        assert s.frozen is True
        assert s.artefact_id is None

    def test_artefact_creation(self):
        a = Artefact(
            objective_id="obj-1",
            snapshot_id="snap-1",
            content="# Summary\n\nResults here.",
        )
        assert a.artefact_type == "summary"


class TestEntityClasses:
    def test_entity_registry(self):
        assert set(ENTITY_CLASSES.keys()) == {
            "objective",
            "evidence",
            "claim",
            "uncertainty",
            "decision",
            "snapshot",
            "artefact",
        }

    def test_all_classes_inherit_from_base(self):
        for cls in ENTITY_CLASSES.values():
            assert issubclass(cls, EpistemicEntity)


class TestSerializationRoundTrip:
    def test_claim_roundtrip(self):
        original = Claim(
            entity_id="claim-001",
            objective_id="obj-1",
            statement="Caffeine improves alertness",
            scope="Adults",
            evidence_ids=["ev-1", "ev-2"],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        content, metadata = original.to_document()
        restored = Claim.from_document(content, metadata)
        assert restored.entity_id == "claim-001"
        assert restored.statement == "Caffeine improves alertness"
        assert restored.stage == ClaimStage.SUPPORTED
        assert restored.evidence_ids == ["ev-1", "ev-2"]
        assert restored.evidence_count == 2

    def test_evidence_roundtrip(self):
        original = Evidence(
            entity_id="ev-001",
            objective_id="obj-1",
            source_type="paper",
            source_ref="doi:10.1234/test",
            extracted_content="Key finding here",
            quality_score=0.85,
            extracted=True,
        )
        content, metadata = original.to_document()
        restored = Evidence.from_document(content, metadata)
        assert restored.source_ref == "doi:10.1234/test"
        assert restored.quality_score == 0.85
        assert restored.extracted is True

    def test_uncertainty_roundtrip(self):
        original = Uncertainty(
            entity_id="unc-001",
            objective_id="obj-1",
            uncertainty_type=UncertaintyType.CONTRADICTION,
            description="Sources disagree on dosage",
            affected_claim_ids=["c1"],
        )
        content, metadata = original.to_document()
        restored = Uncertainty.from_document(content, metadata)
        assert restored.uncertainty_type == UncertaintyType.CONTRADICTION
        assert restored.is_blocking is True
        assert restored.affected_claim_ids == ["c1"]

    def test_objective_roundtrip(self):
        original = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Does exercise reduce anxiety?",
            phase="planned",
            claims_proposed=True,
        )
        content, metadata = original.to_document()
        restored = Objective.from_document(content, metadata)
        assert restored.phase == "planned"
        assert restored.claims_proposed is True


class TestClaimStage:
    def test_stage_values(self):
        assert ClaimStage.HYPOTHESIS.value == "hypothesis"
        assert ClaimStage.SUPPORTED.value == "supported"
        assert ClaimStage.PROVISIONAL.value == "provisional"
        assert ClaimStage.ROBUST.value == "robust"
        assert ClaimStage.ACTIONABLE.value == "actionable"

    def test_stage_from_string(self):
        assert ClaimStage("hypothesis") == ClaimStage.HYPOTHESIS
        assert ClaimStage("actionable") == ClaimStage.ACTIONABLE


class TestUncertaintyBlocking:
    def test_blocking_types(self):
        for ut in BLOCKING_TYPES:
            u = Uncertainty(objective_id="o", uncertainty_type=ut, description="test")
            assert u.is_blocking, f"{ut} should be blocking"

    def test_non_blocking_types(self):
        non_blocking = set(UncertaintyType) - BLOCKING_TYPES
        for ut in non_blocking:
            u = Uncertainty(objective_id="o", uncertainty_type=ut, description="test")
            assert not u.is_blocking, f"{ut} should be non-blocking"


class TestClaimMethods:
    def test_record_modification(self):
        c = Claim(statement="X", objective_id="o")
        assert c.modification_count == 0
        c.record_modification()
        assert c.modification_count == 1
        assert len(c.modification_timestamps) == 1

    def test_record_promotion(self):
        c = Claim(statement="X", objective_id="o", stage=ClaimStage.HYPOTHESIS)
        c.record_promotion(
            ClaimStage.HYPOTHESIS, ClaimStage.SUPPORTED, "Evidence found"
        )
        assert c.stage == ClaimStage.SUPPORTED
        assert len(c.promotion_history) == 1
        assert c.promotion_history[0].from_stage == ClaimStage.HYPOTHESIS
        assert c.promotion_history[0].to_stage == ClaimStage.SUPPORTED

    def test_model_post_init_evidence_count(self):
        c = Claim(statement="X", objective_id="o", evidence_ids=["e1", "e2", "e3"])
        assert c.evidence_count == 3

    def test_evidence_count_updates_on_reinit(self):
        c = Claim(statement="X", objective_id="o", evidence_ids=["e1"])
        assert c.evidence_count == 1
        c.evidence_ids.append("e2")
        c.model_post_init(None)
        assert c.evidence_count == 2


class TestDecisionReverse:
    def test_reverse(self):
        d = Decision(objective_id="o", statement="Go", justification="Why not")
        d.reverse("Changed mind")
        assert d.is_reversed
        assert d.reversal_reason == "Changed mind"

    def test_irreversible(self):
        d = Decision(
            objective_id="o", statement="Go", justification="Why", reversible=False
        )
        with pytest.raises(ValueError, match="irreversible"):
            d.reverse("Nope")


class TestUncertaintyResolve:
    def test_resolve(self):
        u = Uncertainty(objective_id="o", description="Unknown")
        assert u.resolution is None
        assert not u.is_resolved
        u.resolve("Found the answer")
        assert u.is_resolved
        assert u.resolution == "Found the answer"
        assert u.resolved_at is not None
