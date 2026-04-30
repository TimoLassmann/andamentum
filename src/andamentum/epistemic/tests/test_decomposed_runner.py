"""Tests for the decomposed orchestrator (Phase 3).

Two layers of coverage:

1. Pure-logic tests for ``combine_sub_verdicts`` and the
   ``DecomposedPipelineResult`` aggregations — unit tests with no I/O.
2. Orchestrator tests for ``run_research_question_decomposed`` that stub
   the inner graph runner so we exercise the coordinator without running
   the full pipeline. End-to-end pipeline behaviour is already covered
   by the per-graph and per-operation tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import PosteriorReport
from andamentum.epistemic.decomposed_runner import (
    DecomposedPipelineResult,
    combine_sub_verdicts,
    run_research_question_decomposed,
)
from andamentum.epistemic.entities import Objective
from andamentum.epistemic.graph.quarantine import QuarantineRecord
from andamentum.epistemic.operations_runner import PipelineResult
from andamentum.epistemic.repository import EpistemicRepository


async def _seed_parent(
    db_dir, database_name: str, *, description: str
) -> EpistemicRepository:
    """Pre-create a parent objective in phase=analyzed.

    The orchestrator's resume path picks it up, which makes the LLM-driven
    preplanning operations (clarify / classify / analyze) all
    early-return. Those operations are covered by their own unit tests;
    these orchestrator tests focus on decomposition + spawning + dispatch.
    """
    store = DocumentStore.for_database(database_name, db_dir=db_dir)
    await store.initialize()
    repo = EpistemicRepository(store)
    parent = Objective(
        description=description,
        clarified_question=description,
        phase="analyzed",
        question_type="verificatory",
    )
    parent.objective_id = parent.entity_id
    await repo.save(parent)
    return repo


# ── Helpers ───────────────────────────────────────────────────────────


def _posterior(p: float, terminal_state: str = "completed") -> PosteriorReport:
    return PosteriorReport(
        posterior=p,
        log_odds=0,
        supporting_count=0,
        contradicting_count=0,
        counting_posterior=p,
        integration_verdict="supports" if p > 0.5 else "contradicts",
        integration_confidence=0.8,
        mode="abductive",
        terminal_state=terminal_state,  # type: ignore[arg-type]
        objective_id="child",
        question_type="verificatory",
        explanation="stub",
    )


def _result(
    p: float | None,
    *,
    successful: int = 5,
    failed: int = 0,
    terminal_state: str = "completed",
) -> PipelineResult:
    return PipelineResult(
        objective_id="child",
        iterations=successful,
        successful=successful,
        failed=failed,
        status="ok",
        errors=[],
        posterior=_posterior(p, terminal_state) if p is not None else None,
        quarantined=[],
    )


# ══════════════════════════════════════════════════════════════════════════════
# combine_sub_verdicts
# ══════════════════════════════════════════════════════════════════════════════


class TestCombineSubVerdicts:
    def test_and_returns_min(self):
        results = [_result(0.9), _result(0.7), _result(0.4)]
        c = combine_sub_verdicts(results, "AND")
        assert c.posterior == pytest.approx(0.4)
        assert c.combination_rule == "AND"
        # 0.4 falls in the "insufficient" band (0.34 < p < 0.66).
        assert c.verdict == "insufficient"
        assert c.child_posteriors == [0.9, 0.7, 0.4]

    def test_and_supports_when_min_is_high(self):
        results = [_result(0.9), _result(0.85), _result(0.7)]
        c = combine_sub_verdicts(results, "AND")
        assert c.verdict == "supports"
        assert c.posterior == pytest.approx(0.7)

    def test_and_contradicts_when_min_is_low(self):
        results = [_result(0.5), _result(0.6), _result(0.1)]
        c = combine_sub_verdicts(results, "AND")
        assert c.verdict == "contradicts"
        assert c.posterior == pytest.approx(0.1)

    def test_or_returns_max(self):
        results = [_result(0.2), _result(0.6), _result(0.85)]
        c = combine_sub_verdicts(results, "OR")
        assert c.posterior == pytest.approx(0.85)
        assert c.combination_rule == "OR"
        assert c.verdict == "supports"

    def test_weighted_and_returns_mean(self):
        results = [_result(0.8), _result(0.6), _result(0.4)]
        c = combine_sub_verdicts(results, "WEIGHTED_AND")
        assert c.posterior == pytest.approx(0.6)
        assert c.verdict == "insufficient"

    def test_union_returns_none_posterior(self):
        results = [_result(0.8), _result(0.6)]
        c = combine_sub_verdicts(results, "UNION")
        assert c.posterior is None
        assert c.verdict == "union"
        assert c.combination_rule == "UNION"
        assert c.child_posteriors == [0.8, 0.6]

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="Unknown combination_rule"):
            combine_sub_verdicts([_result(0.8)], "MAJORITY")

    def test_no_numeric_data_returns_no_data(self):
        # Both children lack posteriors (e.g. ineligible question type).
        c = combine_sub_verdicts([_result(None), _result(None)], "AND")
        assert c.posterior is None
        assert c.verdict == "no_data"
        assert c.child_posteriors == [None, None]

    def test_mixed_none_and_numeric_uses_only_numeric(self):
        c = combine_sub_verdicts(
            [_result(0.9), _result(None), _result(0.5)], "AND"
        )
        assert c.posterior == pytest.approx(0.5)
        # The None is preserved in the diagnostic.
        assert c.child_posteriors == [0.9, None, 0.5]

    def test_retrieval_failed_propagates_from_any_child(self):
        results = [
            _result(0.7, terminal_state="completed"),
            _result(0.6, terminal_state="retrieval_failed"),
        ]
        c = combine_sub_verdicts(results, "AND")
        assert c.terminal_state == "retrieval_failed"

    def test_case_insensitive_rule(self):
        c = combine_sub_verdicts([_result(0.7), _result(0.5)], "and")
        assert c.combination_rule == "AND"
        assert c.posterior == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# DecomposedPipelineResult aggregations
# ══════════════════════════════════════════════════════════════════════════════


class TestDecomposedPipelineResult:
    def test_aggregates_counters(self):
        sub_results = [
            _result(0.8, successful=10, failed=2),
            _result(0.7, successful=8, failed=1),
        ]
        c = combine_sub_verdicts(sub_results, "AND")
        d = DecomposedPipelineResult(
            parent_objective_id="parent", sub_results=sub_results, combined=c
        )
        assert d.successful == 18
        assert d.failed == 3
        assert d.success is True

    def test_aggregates_errors_and_quarantined(self):
        r1 = PipelineResult(
            "child1",
            5,
            5,
            0,
            "ok",
            errors=["err1"],
            posterior=_posterior(0.7),
            quarantined=[
                QuarantineRecord("e1", "claim", "investigate", "X", "boom")
            ],
        )
        r2 = PipelineResult(
            "child2",
            3,
            3,
            1,
            "ok",
            errors=["err2", "err3"],
            posterior=_posterior(0.6),
            quarantined=[],
        )
        c = combine_sub_verdicts([r1, r2], "AND")
        d = DecomposedPipelineResult("parent", [r1, r2], c)
        assert d.errors == ["err1", "err2", "err3"]
        assert len(d.quarantined) == 1

    def test_synthesized_posterior_inherits_question_type(self):
        sub_results = [_result(0.9), _result(0.6)]
        c = combine_sub_verdicts(sub_results, "AND")
        d = DecomposedPipelineResult("parent", sub_results, c)
        post = d.posterior
        assert post is not None
        assert post.posterior == pytest.approx(0.6)
        assert post.question_type == "verificatory"
        assert post.mode == "decomposed"
        assert post.objective_id == "parent"

    def test_synthesized_posterior_is_none_for_union(self):
        sub_results = [_result(0.9), _result(0.6)]
        c = combine_sub_verdicts(sub_results, "UNION")
        d = DecomposedPipelineResult("parent", sub_results, c)
        assert d.posterior is None

    def test_status_reflects_combined_verdict(self):
        sub_results = [_result(0.85), _result(0.8)]
        c = combine_sub_verdicts(sub_results, "AND")
        d = DecomposedPipelineResult("parent", sub_results, c)
        assert d.status == "supports"


# ══════════════════════════════════════════════════════════════════════════════
# run_research_question_decomposed (with stubbed inner runner)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def repo(tmp_path):
    store = DocumentStore.for_database("test_decomposed", db_dir=tmp_path)
    await store.initialize()
    return EpistemicRepository(store)


class _StubInnerRunner:
    """Stub for run_epistemic_graph that records calls and returns canned results.

    Lets us exercise the orchestrator's decomposition coordination without
    running the full graph. Each call returns a PipelineResult whose
    posterior matches the configured posteriors_by_objective_id mapping.
    """

    def __init__(self, posteriors_by_id: dict[str, float | None]):
        self.posteriors_by_id = posteriors_by_id
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **call_kwargs: Any) -> PipelineResult:
        self.calls.append(call_kwargs)
        target_id = call_kwargs.get("objective_id") or "default"
        # Use the configured posterior for this objective, or 0.7 as default.
        p = self.posteriors_by_id.get(target_id, 0.7)
        return PipelineResult(
            objective_id=target_id,
            iterations=5,
            successful=5,
            failed=0,
            status="ok",
            errors=[],
            posterior=_posterior(p) if p is not None else None,
            quarantined=[],
        )


class TestDecomposedRunner:
    async def test_decompose_false_delegates_to_inner_runner(self, tmp_path):
        """When decompose=False, no preplanning / decomposition runs;
        the inner runner is called once with the original question."""
        stub = _StubInnerRunner({"default": 0.85})
        result = await run_research_question_decomposed(
            "test question",
            database_name="bypass_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=False,
            _inner_runner=stub,  # type: ignore[arg-type]
        )
        # decompose=False returns a plain PipelineResult, not a
        # DecomposedPipelineResult.
        assert isinstance(result, PipelineResult)
        assert len(stub.calls) == 1
        assert stub.calls[0]["question"] == "test question"
        # No objective_id is passed in the bypass path.
        assert stub.calls[0].get("objective_id") is None

    async def test_full_decomposed_path(self, tmp_path, fake_runner, monkeypatch):
        """Decomposition produces 3 children; runner is called for each
        and combines the results per the combination_rule."""
        await _seed_parent(
            tmp_path, "full_test", description="Are podocytes motile in injury?"
        )

        # Force the orchestrator to use fake_runner instead of building its
        # own DefaultAgentRunner.
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        # Skip embedding_model resolution which would call out to ollama.
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )

        # Three children with default posterior 0.7. AND → min = 0.7 → supports.
        stub = _StubInnerRunner({})
        result = await run_research_question_decomposed(
            "Are podocytes motile in injury?",
            database_name="full_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            _inner_runner=stub,  # type: ignore[arg-type]
        )
        assert isinstance(result, DecomposedPipelineResult)
        # The conftest fake decompose-question returns 3 sub-investigations.
        assert len(result.sub_results) == 3
        # AND with three 0.7 children → min = 0.7 → supports.
        assert result.combined.combination_rule == "AND"
        assert result.combined.posterior == pytest.approx(0.7)
        assert result.combined.verdict == "supports"
        # Inner runner was called once per child; each call targets a
        # different objective_id with skip_preplanning=True.
        assert len(stub.calls) == 3
        for call in stub.calls:
            assert call["skip_preplanning"] is True
            assert call["objective_id"] is not None

    async def test_combination_rule_propagates_from_decomposition(
        self, tmp_path, fake_runner, monkeypatch
    ):
        """The orchestrator honors the combination_rule the agent emits.
        Override the fake to return OR, and verify the combiner uses OR (max)."""
        await _seed_parent(tmp_path, "or_test", description="Alpha or beta?")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )
        # Override the fake's decompose-question response to use OR.
        fake_runner._overrides["epistemic_decompose_question"] = {
            "sub_investigations": [
                {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                {"id": "B", "seed_claim": "beta", "rationale": "rb"},
            ],
            "combination_rule": "OR",
            "rationale": "either suffices",
        }

        # Two children: posteriors 0.4 and 0.85. OR → max → 0.85 → supports.
        # Child IDs are unknown ahead of spawn; dispatch by call order.

        class _IndexStub:
            def __init__(self, posts):
                self.posts = posts
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **kwargs: Any) -> PipelineResult:
                idx = len(self.calls)
                self.calls.append(kwargs)
                p = self.posts[idx]
                return PipelineResult(
                    objective_id=kwargs.get("objective_id", f"c{idx}"),
                    iterations=5,
                    successful=5,
                    failed=0,
                    status="ok",
                    errors=[],
                    posterior=_posterior(p),
                    quarantined=[],
                )

        ix = _IndexStub([0.4, 0.85])
        result = await run_research_question_decomposed(
            "Alpha or beta?",
            database_name="or_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            _inner_runner=ix,  # type: ignore[arg-type]
        )
        assert isinstance(result, DecomposedPipelineResult)
        assert result.combined.combination_rule == "OR"
        assert result.combined.posterior == pytest.approx(0.85)
        assert result.combined.verdict == "supports"

    async def test_no_model_raises(self, tmp_path):
        """Decomposed runner requires a model; absence is loud."""
        with pytest.raises(ValueError, match="model is required"):
            await run_research_question_decomposed(
                "test",
                database_name="x",
                db_dir=str(tmp_path),
                decompose=True,
                model=None,
            )
