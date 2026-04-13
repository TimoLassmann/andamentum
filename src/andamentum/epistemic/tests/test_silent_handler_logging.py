"""Tests verifying that previously-silent exception handlers now log warnings.

These are regression tests for Phase 6 of the error path testing PRD.
They ensure that exception handlers that were previously completely silent
now produce warning-level log messages.
"""

import logging
import sys
import pathlib

import pytest

from epistemic.entities import Claim, ClaimStage, Evidence, Objective
from epistemic.operations import (
    AdversarialSearchOperation,
    GeneratePredictionOperation,
    GatheredEvidence,
    create_operations,
)
from epistemic.patterns import PatternScheduler, WorkItem
from epistemic.storage import InMemoryStorageBackend
from epistemic.repository import EpistemicRepository

_test_dir = str(pathlib.Path(__file__).parent)
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)

from conftest import (  # noqa: E402
    FakeAgentRunner,
    FailingRepo,
    PartiallyFailingRunner,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _make_repo() -> EpistemicRepository:
    backend = InMemoryStorageBackend()
    return EpistemicRepository(backend)


async def _save_objective(repo: EpistemicRepository, description: str = "Test question") -> Objective:
    obj = Objective(description=description, phase="planned")
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
    claim = Claim(
        objective_id=objective_id,
        statement=statement,
        stage=stage,
        evidence_ids=evidence_ids or [],
        scrutiny_verdict=scrutiny_verdict,
    )
    await repo.save(claim)
    return claim


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPatternSchedulerLogsQueryFailure:
    """PatternScheduler.get_pending_work() should log when a pattern query fails."""

    @pytest.mark.asyncio
    async def test_pattern_scheduler_logs_query_failure(self, caplog):
        backend = InMemoryStorageBackend()
        # FailingRepo raises on query for "claim" entity type
        repo = FailingRepo(backend, fail_on_query={"claim"})

        # Save an objective so the scheduler has some work to try
        obj = Objective(entity_id="obj-1", objective_id="obj-1", description="Q", phase="new")
        await repo.save(obj)

        scheduler = PatternScheduler(repo)

        with caplog.at_level(logging.WARNING, logger="epistemic.patterns"):
            await scheduler.get_pending_work(objective_id="obj-1")

        # At least one pattern targets "claim"; its query should have failed and been logged
        assert any("Pattern query failed" in record.message for record in caplog.records), (
            f"Expected 'Pattern query failed' warning, got: {[r.message for r in caplog.records]}"
        )


class _SimpleGatherer:
    """Evidence gatherer returning fixed results for test setup."""

    def __init__(self, results: list[GatheredEvidence]):
        self._results = results

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        return self._results


class TestCounterargEvalLogsOnFailure:
    """AdversarialSearchOperation should log when evaluate_counterargument agent fails."""

    @pytest.mark.asyncio
    async def test_counterarg_eval_logs_on_failure(self, caplog):
        repo = await _make_repo()
        obj = await _save_objective(repo)
        ev = await _save_evidence(
            repo, obj.entity_id, extracted=True, content="Evidence text", quality_score=0.7
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            stage=ClaimStage.SUPPORTED,
            evidence_ids=[ev.entity_id],
        )

        # Evidence gatherer returns a search result that will be evaluated as a counterargument
        gatherer = _SimpleGatherer(
            results=[GatheredEvidence(content="This contradicts the claim", source_ref="http://counter.example.com", source_type="web_search")]
        )

        # Runner that succeeds on generate_counterquery but fails on evaluate_counterargument
        runner = FakeAgentRunner()
        failing_runner = PartiallyFailingRunner(
            fail_on={"epistemic_evaluate_counterargument"},
            fallback_runner=runner,
        )

        op = AdversarialSearchOperation(repo, failing_runner, evidence_gatherer=gatherer)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="adversarial_search")

        with caplog.at_level(logging.WARNING, logger="epistemic.operations"):
            result = await op.execute(work)

        assert result.success
        assert any("Counterargument evaluation failed" in record.message for record in caplog.records), (
            f"Expected 'Counterargument evaluation failed' warning, got: {[r.message for r in caplog.records]}"
        )


class TestPredictionClassificationLogsOnFailure:
    """GeneratePredictionOperation should log when a per-aspect step fails."""

    @pytest.mark.asyncio
    async def test_prediction_generation_logs_on_failure(self, caplog):
        repo = await _make_repo()
        obj = await _save_objective(repo)
        ev = await _save_evidence(
            repo, obj.entity_id, extracted=True, content="Evidence text", quality_score=0.7
        )
        claim = await _save_claim(
            repo,
            obj.entity_id,
            stage=ClaimStage.ROBUST,
            evidence_ids=[ev.entity_id],
        )

        # Runner that succeeds on identify/specify/define but fails on classify_prediction
        failing_runner = PartiallyFailingRunner(fail_on={"epistemic_classify_prediction"})

        op = GeneratePredictionOperation(repo, failing_runner)
        work = WorkItem(entity_id=claim.entity_id, entity_type="claim", operation="generate_prediction")

        with caplog.at_level(logging.WARNING, logger="epistemic.operations"):
            result = await op.execute(work)

        assert result.success
        assert any("Prediction generation failed" in record.message for record in caplog.records), (
            f"Expected 'Prediction generation failed' warning, got: {[r.message for r in caplog.records]}"
        )
