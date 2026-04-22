"""Tests for epistemic operations under failure conditions.

Verifies that operations handle errors gracefully when:
- Agent runner throws exceptions
- Repository lookups fail
- Evidence scoring fallback chain is tested at each level
- Counterargument evaluation fails
- Writer-validator loop handles all edge cases

Each test triggers a specific error path and verifies the operation
either fails with a clear message or degrades gracefully.
"""

import pytest
from typing import Any

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
    UncertaintyType,
)
from ..operations import (
    create_operations,
    ExtractEvidenceOperation,
    ScrutiniseClaimOperation,
    AdversarialSearchOperation,
    SynthesizeReportOperation,
    GeneratePredictionOperation,
    InvestigateClaimOperation,
    ResolveUncertaintyOperation,
    GatheredEvidence,
    QualityScore,
)
from andamentum.document_store import DocumentStore
from ..operations.base import OperationInput
from ..repository import EpistemicRepository

import sys
import pathlib

_test_dir = str(pathlib.Path(__file__).parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

from conftest import (  # noqa: E402  # type: ignore[import-not-found]
    FakeAgentRunner,
    PartiallyFailingRunner,
    FailingRepo,
    _to_namespace,
    _FAKE_DEFAULTS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


class MockQualityScorer:
    """Quality scorer that can be configured to succeed or fail."""

    def __init__(self, score: QualityScore | None = None, fail: bool = False):
        self._score = score
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    async def score(self, source_ref: str, source_type: str) -> QualityScore:
        self.calls.append((source_ref, source_type))
        if self._fail:
            raise RuntimeError("Scorer failed")
        if self._score is not None:
            return self._score
        # Return a "needs_assessment" marker so Path 1 falls through
        return QualityScore(score=0.0, source="needs_assessment")


class MockGatherer:
    """Evidence gatherer that can be configured to succeed or fail."""

    def __init__(
        self,
        results: list[GatheredEvidence] | None = None,
        fail: bool = False,
    ):
        self._results = results or []
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        self.calls.append((source_type, query))
        if self._fail:
            raise RuntimeError("Gatherer failed")
        return self._results


async def _make_store(tmp_path) -> DocumentStore:
    """Create a fresh DocumentStore for test use."""
    s = DocumentStore.for_database("test", db_dir=tmp_path)
    await s.initialize()
    return s


async def _make_repo(tmp_path) -> EpistemicRepository:
    """Create a fresh repo backed by DocumentStore."""
    store = await _make_store(tmp_path)
    return EpistemicRepository(store)


async def _save_objective(
    repo: EpistemicRepository, description: str = "Test question"
) -> Objective:
    """Create and save an objective."""
    obj = Objective(description=description, phase="planned")
    # Objectives are self-referential: objective_id == entity_id
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    return obj


async def _save_evidence(
    repo: EpistemicRepository,
    objective_id: str,
    *,
    extracted: bool = False,
    source_ref: str = "test-source",
    source_type: str = "web_search",
    content: str = "",
    quality_score: float | None = None,
) -> Evidence:
    """Create and save an evidence entity."""
    ev = Evidence(
        objective_id=objective_id,
        source_ref=source_ref,
        source_type=source_type,
        extracted=extracted,
        extracted_content=content,
        quality_score=quality_score,
    )
    await repo.save(ev)
    return ev


async def _save_claim(
    repo: EpistemicRepository,
    objective_id: str,
    *,
    statement: str = "Test claim",
    stage: ClaimStage = ClaimStage.HYPOTHESIS,
    evidence_ids: list[str] | None = None,
    scrutiny_verdict: str | None = None,
) -> Claim:
    """Create and save a claim entity."""
    claim = Claim(
        objective_id=objective_id,
        statement=statement,
        stage=stage,
        evidence_ids=evidence_ids or [],
        scrutiny_verdict=scrutiny_verdict,
    )
    await repo.save(claim)
    return claim


async def _save_uncertainty(
    repo: EpistemicRepository,
    objective_id: str,
    *,
    description: str = "Test uncertainty",
    uncertainty_type: UncertaintyType = UncertaintyType.UNKNOWN,
    affected_claim_ids: list[str] | None = None,
) -> Uncertainty:
    """Create and save an uncertainty entity."""
    u = Uncertainty(
        objective_id=objective_id,
        description=description,
        uncertainty_type=uncertainty_type,
        affected_claim_ids=affected_claim_ids or [],
    )
    await repo.save(u)
    return u


# ── 1. Evidence Scoring Fallback Chain ───────────────────────────────────────


class TestEvidenceScoringFallbackChain:
    """Test the 4-path fallback chain in _score_evidence."""

    @pytest.mark.asyncio
    async def test_path1_openalex_success(self, tmp_path):
        """Path 1: OpenAlex scorer succeeds -> use its score."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="10.1234/test")

        scorer = MockQualityScorer(
            score=QualityScore(
                score=0.85, source="openalex", raw_metadata={"cited_by": 100}
            )
        )
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="Test content",
                    source_ref="10.1234/test",
                    source_type="openalex",
                )
            ]
        )
        runner = FakeAgentRunner()

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=scorer
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("evidence", ev.entity_id)
        assert isinstance(updated, Evidence)
        assert updated.quality_score == 0.85
        assert updated.quality_metadata is not None
        assert updated.quality_metadata.get("cited_by") == 100

    @pytest.mark.asyncio
    async def test_path1_fails_raises_on_failure(self, tmp_path):
        """Path 1 scorer raises -> exception propagates."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="http://example.com")

        # Scorer that throws
        scorer = MockQualityScorer(fail=True)
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="Test content",
                    source_ref="http://example.com",
                    source_type="web_search",
                )
            ]
        )
        runner = FakeAgentRunner()

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=scorer
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="Scorer failed"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_paths_1_and_2_fail_raises_on_failure(self, tmp_path):
        """Paths 1+2 both configured to fail -> exception from path 1 propagates."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="http://example.com")

        scorer = MockQualityScorer(fail=True)
        # Gatherer provides a pre-computed quality_score (never reached)
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="Test content",
                    source_ref="http://example.com",
                    source_type="web_search",
                    quality_score=0.65,
                )
            ]
        )
        # Runner that fails on quality assessment agent
        runner = PartiallyFailingRunner(
            fail_on={"epistemic_assess_evidence_quality"},
        )

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=scorer
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="Scorer failed"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_all_paths_fail_raises_on_failure(self, tmp_path):
        """All paths configured to fail -> exception from path 1 propagates."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="http://example.com")

        scorer = MockQualityScorer(fail=True)
        # Gatherer returns content but no quality_score (never reached)
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="Some content here",
                    source_ref="http://example.com",
                    source_type="web_search",
                    quality_score=None,
                )
            ]
        )
        # Runner that fails on quality assessment
        runner = PartiallyFailingRunner(
            fail_on={"epistemic_assess_evidence_quality"},
        )

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=scorer
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="Scorer failed"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_no_content_raises_on_agent_failure(self, tmp_path):
        """Agent scorer raises -> exception propagates even with no content."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="http://example.com")

        # Gatherer returns empty content with no quality_score
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="",
                    source_ref="http://example.com",
                    source_type="web_search",
                    quality_score=None,
                )
            ]
        )
        # No scorer, failing agent
        runner = PartiallyFailingRunner(
            fail_on={"epistemic_assess_evidence_quality"},
        )

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=None
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="Simulated agent failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_gatherer_fails_raises_on_failure(self, tmp_path):
        """When gatherer throws, the exception propagates."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(repo, obj.entity_id, source_ref="http://example.com")

        # Failing gatherer
        gatherer = MockGatherer(fail=True)
        runner = FakeAgentRunner()

        op = ExtractEvidenceOperation(
            repo, runner, evidence_gatherer=gatherer, quality_scorer=None
        )
        work = OperationInput(
            entity_id=ev.entity_id, entity_type="evidence", operation="extract_evidence"
        )
        with pytest.raises(RuntimeError, match="Gatherer failed"):
            await op.execute(work)


# ── 2. Scrutiny Operation Failure ────────────────────────────────────────────


class TestScrutinyOperationFailure:
    """Test ScrutiniseClaimOperation under failure."""

    @pytest.mark.asyncio
    async def test_evidence_loading_raises_on_failure(self, tmp_path):
        """Evidence loading raises RuntimeError when repo fails on a bad ID."""
        store = await _make_store(tmp_path)
        failing_repo = FailingRepo(store, fail_on={"bad-evidence-id"})
        obj = await _save_objective(failing_repo)

        # Create evidence that exists
        good_ev = await _save_evidence(
            failing_repo, obj.entity_id, extracted=True, content="Good evidence"
        )

        # Create claim with one good ID and one bad ID
        claim = await _save_claim(
            failing_repo,
            obj.entity_id,
            evidence_ids=[good_ev.entity_id, "bad-evidence-id"],
        )

        runner = FakeAgentRunner()
        op = ScrutiniseClaimOperation(failing_repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id, entity_type="claim", operation="scrutinise_claim"
        )
        with pytest.raises(RuntimeError, match="Simulated repo failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_agent_failure_propagates(self, tmp_path):
        """If scrutiny agent throws, operation should fail.

        Tests the split path (epistemic_assess_evidence) since it is preferred
        when the split agents are registered.
        """
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        claim = await _save_claim(repo, obj.entity_id)

        runner = PartiallyFailingRunner(fail_on={"epistemic_assess_evidence"})
        op = ScrutiniseClaimOperation(repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id, entity_type="claim", operation="scrutinise_claim"
        )

        with pytest.raises(RuntimeError, match="Simulated agent failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_no_agent_runner_defaults_to_pass(self, tmp_path):
        """Without agent runner, scrutiny verdict defaults to pass."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        claim = await _save_claim(repo, obj.entity_id)

        op = ScrutiniseClaimOperation(repo, agent_runner=None)
        work = OperationInput(
            entity_id=claim.entity_id, entity_type="claim", operation="scrutinise_claim"
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert isinstance(updated, Claim)
        assert updated.scrutiny_verdict == "pass"


# ── 3. Adversarial Search Failure ────────────────────────────────────────────


class TestAdversarialSearchFailure:
    """Test AdversarialSearchOperation under failure."""

    @pytest.mark.asyncio
    async def test_counterargument_evaluation_fails_raises(self, tmp_path):
        """If evaluate_counterargument agent fails, the operation raises — no silent fallback."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(
            repo,
            obj.entity_id,
            extracted=True,
            content="Evidence text",
            quality_score=0.7,
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            stage=ClaimStage.SUPPORTED,
            evidence_ids=[ev.entity_id],
        )

        failing_runner = PartiallyFailingRunner(
            fail_on={"epistemic_evaluate_counterargument"},
            fallback_runner=FakeAgentRunner(),
        )
        # Supply a gatherer that returns one hit so the evaluator is actually invoked.
        gatherer = MockGatherer(
            results=[
                GatheredEvidence(
                    content="Contradicting evidence",
                    source_ref="http://counter.example.com",
                    source_type="web_search",
                )
            ]
        )

        op = AdversarialSearchOperation(repo, failing_runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="adversarial_search",
        )
        with pytest.raises(RuntimeError, match="Simulated agent failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_gatherer_search_fails_per_query(self, tmp_path):
        """Individual search query failures should not crash the operation."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(
            repo,
            obj.entity_id,
            extracted=True,
            content="Evidence text",
            quality_score=0.7,
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            stage=ClaimStage.SUPPORTED,
            evidence_ids=[ev.entity_id],
        )

        gatherer = MockGatherer(fail=True)
        runner = FakeAgentRunner()

        op = AdversarialSearchOperation(repo, runner, evidence_gatherer=gatherer)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="adversarial_search",
        )
        result = await op.execute(work)

        # Should succeed — search failures are caught per-query
        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert isinstance(updated, Claim)
        assert updated.adversarial_checked is True

    @pytest.mark.asyncio
    async def test_no_evidence_gatherer(self, tmp_path):
        """Without gatherer, adversarial search uses world knowledge only."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        claim = await _save_claim(repo, obj.entity_id, stage=ClaimStage.SUPPORTED)

        runner = FakeAgentRunner()
        op = AdversarialSearchOperation(repo, runner, evidence_gatherer=None)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="adversarial_search",
        )
        result = await op.execute(work)

        assert result.success


# ── 4. Writer-Validator Loop ─────────────────────────────────────────────────


class TestWriterValidatorLoop:
    """Test the writer-validator loop in SynthesizeReportOperation."""

    async def _setup_synthesis(
        self,
        runner: Any,
        tmp_path,
    ) -> tuple[EpistemicRepository, Snapshot, OperationInput]:
        """Create the full entity chain needed for SynthesizeReportOperation."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo, description="What is spaced repetition?")
        obj.phase = "claims_done"
        await repo.save(obj)

        ev = await _save_evidence(
            repo,
            obj.entity_id,
            extracted=True,
            content="Spaced repetition improves memory retention.",
            quality_score=0.7,
            source_ref="study.pdf",
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            statement="Spaced repetition improves retention",
            stage=ClaimStage.SUPPORTED,
            evidence_ids=[ev.entity_id],
            scrutiny_verdict="pass",
        )

        snapshot = Snapshot(
            objective_id=obj.entity_id,
            claim_ids=[claim.entity_id],
            evidence_ids=[ev.entity_id],
            uncertainty_ids=[],
            snapshot_type="final",
        )
        await repo.save(snapshot)

        work = OperationInput(
            entity_id=snapshot.entity_id,
            entity_type="snapshot",
            operation="synthesize_report",
        )
        return repo, snapshot, work

    @pytest.mark.asyncio
    async def test_validator_rejects_then_accepts(self, tmp_path):
        """Validator rejects first round, accepts second."""
        call_count = {"validate": 0}

        class RoundAwareRunner(FakeAgentRunner):
            async def run(self, agent_name: str, **kwargs: Any) -> Any:
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_validate_answer":
                    call_count["validate"] += 1
                    if call_count["validate"] == 1:
                        return _to_namespace(
                            {
                                "approved": False,
                                "feedback": ["Missing evidence citation"],
                            }
                        )
                    return _to_namespace({"approved": True, "feedback": []})
                raw = _FAKE_DEFAULTS.get(agent_name, {})
                return _to_namespace(raw)

        runner = RoundAwareRunner()
        repo, snapshot, work = await self._setup_synthesis(runner, tmp_path)

        op = SynthesizeReportOperation(repo, runner)
        result = await op.execute(work)

        assert result.success
        assert call_count["validate"] == 2
        # Writer should have been called twice (once fresh, once with feedback)
        write_calls = [c for c in runner.calls if c[0] == "epistemic_write_answer"]
        assert len(write_calls) == 2
        # Second call should include prior feedback
        assert "validator_feedback" in write_calls[1][1]

    @pytest.mark.asyncio
    async def test_validator_rejects_all_rounds(self, tmp_path):
        """Validator rejects all rounds -> best effort answer used."""

        class AlwaysRejectRunner(FakeAgentRunner):
            async def run(self, agent_name: str, **kwargs: Any) -> Any:
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_validate_answer":
                    return _to_namespace(
                        {"approved": False, "feedback": ["Still not good enough"]}
                    )
                raw = _FAKE_DEFAULTS.get(agent_name, {})
                return _to_namespace(raw)

        runner = AlwaysRejectRunner()
        repo, snapshot, work = await self._setup_synthesis(runner, tmp_path)

        op = SynthesizeReportOperation(repo, runner)
        result = await op.execute(work)

        # Should still succeed with best-effort answer
        assert result.success
        # Verify all 10 rounds were attempted
        validate_calls = [
            c for c in runner.calls if c[0] == "epistemic_validate_answer"
        ]
        assert len(validate_calls) == SynthesizeReportOperation.MAX_VALIDATION_ROUNDS

    @pytest.mark.asyncio
    async def test_writer_returns_empty_answer(self, tmp_path):
        """Writer returning empty answer -> loop terminates early."""

        class EmptyWriterRunner(FakeAgentRunner):
            async def run(self, agent_name: str, **kwargs: Any) -> Any:
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_write_answer":
                    return _to_namespace({"title": "Empty Report", "answer": ""})
                raw = _FAKE_DEFAULTS.get(agent_name, {})
                return _to_namespace(raw)

        runner = EmptyWriterRunner()
        repo, snapshot, work = await self._setup_synthesis(runner, tmp_path)

        op = SynthesizeReportOperation(repo, runner)
        result = await op.execute(work)

        # Should still succeed (empty answer is valid, report has deterministic sections)
        assert result.success
        # Validator should NOT have been called since answer was empty
        validate_calls = [
            c for c in runner.calls if c[0] == "epistemic_validate_answer"
        ]
        assert len(validate_calls) == 0

    @pytest.mark.asyncio
    async def test_validator_throws(self, tmp_path):
        """If validator agent throws, operation propagates the error."""
        runner = PartiallyFailingRunner(fail_on={"epistemic_validate_answer"})
        repo, snapshot, work = await self._setup_synthesis(runner, tmp_path)

        op = SynthesizeReportOperation(repo, runner)
        # The validator throw propagates since there's no try/except around it
        with pytest.raises(RuntimeError, match="Simulated agent failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_validator_approves_immediately(self, tmp_path):
        """Validator approves on first round -> single iteration."""
        runner = FakeAgentRunner(
            overrides={
                "epistemic_validate_answer": {"approved": True, "feedback": []},
            }
        )
        repo, snapshot, work = await self._setup_synthesis(runner, tmp_path)

        op = SynthesizeReportOperation(repo, runner)
        result = await op.execute(work)

        assert result.success
        validate_calls = [
            c for c in runner.calls if c[0] == "epistemic_validate_answer"
        ]
        assert len(validate_calls) == 1


# ── 5. Prediction Classification Failure ─────────────────────────────────────


class TestPredictionClassificationFailure:
    """Test GeneratePredictionOperation when classify_prediction agent fails.

    The decomposed flow (identify → specify → falsify → classify) catches
    failures per-aspect. If classify_prediction fails, that aspect is skipped
    entirely (no partial prediction stored) but the operation still succeeds.
    """

    @pytest.mark.asyncio
    async def test_classification_raises_on_failure(self, tmp_path):
        """If classify_prediction throws, the exception propagates."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        ev = await _save_evidence(
            repo,
            obj.entity_id,
            extracted=True,
            content="Evidence text",
            quality_score=0.7,
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            stage=ClaimStage.ROBUST,
            evidence_ids=[ev.entity_id],
        )

        runner = PartiallyFailingRunner(fail_on={"epistemic_classify_prediction"})
        op = GeneratePredictionOperation(repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="generate_prediction",
        )
        with pytest.raises(RuntimeError, match="Simulated agent failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_evidence_loading_raises_on_failure(self, tmp_path):
        """Evidence loading in prediction generation raises when repo fails."""
        store = await _make_store(tmp_path)
        failing_repo = FailingRepo(store, fail_on={"bad-evidence-id"})
        obj = await _save_objective(failing_repo)
        claim = await _save_claim(
            failing_repo,
            obj.entity_id,
            stage=ClaimStage.ROBUST,
            evidence_ids=["bad-evidence-id"],
        )

        runner = FakeAgentRunner()
        op = GeneratePredictionOperation(failing_repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="generate_prediction",
        )
        with pytest.raises(RuntimeError, match="Simulated repo failure"):
            await op.execute(work)


# ── 6. Investigate Claim Failure ─────────────────────────────────────────────


class TestInvestigateClaimFailure:
    """Test InvestigateClaimOperation under failure."""

    @pytest.mark.asyncio
    async def test_evidence_loading_raises_on_failure(self, tmp_path):
        """Evidence loading in investigate raises when repo fails."""
        store = await _make_store(tmp_path)
        failing_repo = FailingRepo(store, fail_on={"bad-evidence-id"})
        obj = await _save_objective(failing_repo)
        claim = await _save_claim(
            failing_repo,
            obj.entity_id,
            evidence_ids=["bad-evidence-id"],
            scrutiny_verdict="needs_resolution",
        )

        runner = FakeAgentRunner()
        op = InvestigateClaimOperation(failing_repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        with pytest.raises(RuntimeError, match="Simulated repo failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_uncertainty_loading_raises_on_failure(self, tmp_path):
        """Uncertainty query failure in investigate raises the error."""
        store = await _make_store(tmp_path)
        failing_repo = FailingRepo(store, fail_on_query={"uncertainty"})
        obj = await _save_objective(failing_repo)
        claim = await _save_claim(
            failing_repo,
            obj.entity_id,
            scrutiny_verdict="needs_resolution",
        )

        runner = FakeAgentRunner()
        op = InvestigateClaimOperation(failing_repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        with pytest.raises(RuntimeError, match="Simulated query failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_investigation_exhausted_abandons_claim(self, tmp_path):
        """After MAX_INVESTIGATION_ATTEMPTS, claim is abandoned."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        claim = await _save_claim(
            repo,
            obj.entity_id,
            scrutiny_verdict="needs_resolution",
        )
        # Set investigation_count to max
        claim.investigation_count = 3  # MAX_INVESTIGATION_ATTEMPTS
        await repo.save(claim)

        runner = FakeAgentRunner()
        op = InvestigateClaimOperation(repo, runner)
        work = OperationInput(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="investigate_claim",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("claim", claim.entity_id)
        assert isinstance(updated, Claim)
        assert updated.abandoned is True
        assert updated.scrutiny_verdict == "fail"


# ── 7. Resolve Uncertainty Failure ───────────────────────────────────────────


class TestResolveUncertaintyFailure:
    """Test ResolveUncertaintyOperation under failure."""

    @pytest.mark.asyncio
    async def test_parent_entity_loading_raises_on_failure(self, tmp_path):
        """Loading claims raises RuntimeError when repo fails on a bad ID."""
        store = await _make_store(tmp_path)
        # Create a real objective first so _maybe_advance_phase doesn't crash
        repo = EpistemicRepository(store)
        obj = Objective(description="Test", phase="claims_proposed")
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # Now create a failing repo that fails on claim lookups
        failing_repo = FailingRepo(store, fail_on={"bad-claim-id"})

        u = Uncertainty(
            objective_id=obj.entity_id,
            description="Test uncertainty",
            uncertainty_type=UncertaintyType.UNKNOWN,
            affected_claim_ids=["bad-claim-id"],
        )
        await failing_repo.save(u)

        runner = FakeAgentRunner()
        op = ResolveUncertaintyOperation(failing_repo, runner)
        work = OperationInput(
            entity_id=u.entity_id,
            entity_type="uncertainty",
            operation="resolve_uncertainty",
        )
        with pytest.raises(RuntimeError, match="Simulated repo failure"):
            await op.execute(work)

    @pytest.mark.asyncio
    async def test_unresolvable_uncertainty(self, tmp_path):
        """When agent says can't resolve, uncertainty is marked as acknowledged limitation."""
        repo = await _make_repo(tmp_path)
        obj = await _save_objective(repo)
        u = await _save_uncertainty(repo, obj.entity_id)

        runner = FakeAgentRunner(
            overrides={
                "epistemic_resolve_uncertainty": {
                    "can_resolve": False,
                    "resolution": "",
                    "remaining_concerns": [],
                },
            }
        )
        op = ResolveUncertaintyOperation(repo, runner)
        work = OperationInput(
            entity_id=u.entity_id,
            entity_type="uncertainty",
            operation="resolve_uncertainty",
        )
        result = await op.execute(work)

        assert result.success
        updated = await repo.get("uncertainty", u.entity_id)
        assert isinstance(updated, Uncertainty)
        assert updated.resolution is not None
        assert "Unresolvable" in updated.resolution


# ── 8. create_operations Registry ────────────────────────────────────────────


class TestCreateOperationsErrorHandling:
    """Test create_operations() initialization."""

    @pytest.mark.asyncio
    async def test_creates_all_operations(self, tmp_path):
        """create_operations should return all registered operation types."""
        store = await _make_store(tmp_path)
        repo = EpistemicRepository(store)
        runner = FakeAgentRunner()

        ops = create_operations(repo, runner)

        expected = {
            "clarify_question",
            "conceptual_analysis",
            "plan_task",
            "propose_claims",
            "extract_evidence",
            "scrutinise_claim",
            "promote_claim",
            "demote_claim",
            "adversarial_search",
            "assess_convergence",
            "validate_deductively",
            "verify_computationally",
            "resolve_uncertainty",
            "freeze_snapshot",
            "synthesize_report",
            "analyze_argument",
            "generate_prediction",
            "record_decision",
            "investigate_claim",
            "invalidate_evidence",
            "revalidate_claim",
        }
        assert expected.issubset(set(ops.keys())), (
            f"Missing: {expected - set(ops.keys())}"
        )

    @pytest.mark.asyncio
    async def test_operations_share_repo_and_runner(self, tmp_path):
        """All operations should reference the same repo and runner."""
        store = await _make_store(tmp_path)
        repo = EpistemicRepository(store)
        runner = FakeAgentRunner()

        ops = create_operations(repo, runner)

        for name, op in ops.items():
            assert op.repo is repo, f"{name} has wrong repo"
            assert op.agent_runner is runner, f"{name} has wrong runner"

    @pytest.mark.asyncio
    async def test_no_runner_creates_operations(self, tmp_path):
        """create_operations works without agent runner."""
        store = await _make_store(tmp_path)
        repo = EpistemicRepository(store)

        ops = create_operations(repo, agent_runner=None)
        assert len(ops) > 0
        for op in ops.values():
            assert op.agent_runner is None
