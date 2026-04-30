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

    def test_weighted_and_returns_mean_with_no_weights(self):
        results = [_result(0.8), _result(0.6), _result(0.4)]
        c = combine_sub_verdicts(results, "WEIGHTED_AND")
        assert c.posterior == pytest.approx(0.6)
        assert c.verdict == "insufficient"

    def test_weighted_and_with_weights_pulls_toward_heavy_child(self):
        # Children at 0.9 and 0.3. Weight 0.9 at 3.0, 0.3 at 1.0:
        # weighted mean = (0.9*3 + 0.3*1) / (3+1) = 3.0 / 4 = 0.75
        results = [_result(0.9), _result(0.3)]
        c = combine_sub_verdicts(results, "WEIGHTED_AND", weights=[3.0, 1.0])
        assert c.posterior == pytest.approx(0.75)
        assert c.verdict == "supports"

    def test_weighted_and_equal_weights_equals_simple_mean(self):
        results = [_result(0.8), _result(0.6), _result(0.4)]
        c_weighted = combine_sub_verdicts(
            results, "WEIGHTED_AND", weights=[1.0, 1.0, 1.0]
        )
        c_unweighted = combine_sub_verdicts(results, "WEIGHTED_AND")
        assert c_weighted.posterior == pytest.approx(c_unweighted.posterior)

    def test_weighted_and_zero_weight_drops_child_via_normalization(self):
        # Child at 0.1 with weight 0.0 contributes nothing. Other child at 0.8
        # with weight 1.0 dominates.
        results = [_result(0.8), _result(0.1)]
        c = combine_sub_verdicts(results, "WEIGHTED_AND", weights=[1.0, 0.0])
        assert c.posterior == pytest.approx(0.8)

    def test_weighted_and_all_zero_weights_falls_back_to_mean(self):
        results = [_result(0.8), _result(0.4)]
        c = combine_sub_verdicts(results, "WEIGHTED_AND", weights=[0.0, 0.0])
        assert c.posterior == pytest.approx(0.6)  # simple mean
        assert "all weights zero" in c.explanation

    def test_weighted_and_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            combine_sub_verdicts(
                [_result(0.8), _result(0.6)], "WEIGHTED_AND", weights=[1.0, -1.0]
            )

    def test_weighted_and_mismatched_weights_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            combine_sub_verdicts(
                [_result(0.8), _result(0.6)], "WEIGHTED_AND", weights=[1.0]
            )

    def test_weighted_and_drops_none_children_and_their_weights(self):
        # Child indices 0 and 2 are numeric; index 1 is None. Weights
        # should be sliced to match: (0.9*1 + 0.5*3) / (1+3) = 2.4/4 = 0.6.
        results = [_result(0.9), _result(None), _result(0.5)]
        c = combine_sub_verdicts(
            results, "WEIGHTED_AND", weights=[1.0, 100.0, 3.0]
        )
        assert c.posterior == pytest.approx(0.6)

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

    async def test_reflection_skipped_when_verdict_is_decisive(
        self, tmp_path, fake_runner, monkeypatch
    ):
        """Combined verdict 'supports' → no reflection, agent untouched."""
        await _seed_parent(tmp_path, "skip_test", description="Q")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )
        # 3 children at 0.85 → AND → 0.85 → supports → no reflection.
        stub = _StubInnerRunner({})  # default 0.7 isn't enough for >0.66; bump
        # Actually default 0.7 gives "supports" already (>0.66). Use that.

        await run_research_question_decomposed(
            "Q",
            database_name="skip_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            max_reflection_rounds=1,
            _inner_runner=stub,  # type: ignore[arg-type]
        )
        # Reflection agent should not have been called.
        reflect_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_reflect_on_gaps"
        ]
        assert reflect_calls == []

    async def test_reflection_triggers_on_insufficient_verdict(
        self, tmp_path, fake_runner, monkeypatch
    ):
        """Combined verdict 'insufficient' → reflection runs once."""
        await _seed_parent(tmp_path, "trigger_test", description="Q")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )
        # 3 children at 0.5 → AND → 0.5 → "insufficient" → reflection fires.
        # Default fake says sufficient=True so no new sub-investigations
        # are added; the loop exits cleanly after one reflection call.

        class _AllInsufficientStub:
            def __init__(self):
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **call_kwargs: Any) -> PipelineResult:
                self.calls.append(call_kwargs)
                return PipelineResult(
                    objective_id=call_kwargs.get("objective_id", "x"),
                    iterations=5,
                    successful=5,
                    failed=0,
                    status="ok",
                    posterior=_posterior(0.5),
                )

        stub = _AllInsufficientStub()
        result = await run_research_question_decomposed(
            "Q",
            database_name="trigger_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            max_reflection_rounds=1,
            _inner_runner=stub,  # type: ignore[arg-type]
        )
        assert isinstance(result, DecomposedPipelineResult)
        # Reflection agent fired exactly once.
        reflect_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_reflect_on_gaps"
        ]
        assert len(reflect_calls) == 1
        # No new children spawned (default fake says sufficient).
        assert len(stub.calls) == 3

    async def test_max_reflection_rounds_zero_disables_reflection(
        self, tmp_path, fake_runner, monkeypatch
    ):
        await _seed_parent(tmp_path, "disable_test", description="Q")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )

        class _Stub:
            def __init__(self):
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **call_kwargs: Any) -> PipelineResult:
                self.calls.append(call_kwargs)
                return PipelineResult(
                    objective_id=call_kwargs.get("objective_id", "x"),
                    iterations=5,
                    successful=5,
                    failed=0,
                    status="ok",
                    posterior=_posterior(0.5),  # would normally trigger reflection
                )

        stub = _Stub()
        await run_research_question_decomposed(
            "Q",
            database_name="disable_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            max_reflection_rounds=0,
            _inner_runner=stub,  # type: ignore[arg-type]
        )
        reflect_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_reflect_on_gaps"
        ]
        assert reflect_calls == []

    async def test_reflection_adds_children_runs_them_recombines(
        self, tmp_path, fake_runner, monkeypatch
    ):
        """End-to-end: insufficient verdict → reflection → new
        sub-investigation spawned → new child run → re-combined."""
        await _seed_parent(tmp_path, "add_test", description="Q")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )
        # Initial decomposition: 2 children. Override fake to use 2-child
        # AND so we can predict the verdict.
        fake_runner._overrides["epistemic_decompose_question"] = {
            "sub_investigations": [
                {"id": "A", "seed_claim": "alpha", "rationale": "ra", "weight": 1.0},
                {"id": "B", "seed_claim": "beta", "rationale": "rb", "weight": 1.0},
            ],
            "combination_rule": "AND",
            "rationale": "two pillars",
        }
        # First two children → 0.5 each (insufficient). New child from
        # reflection → 0.9. AND with 3 children at [0.5, 0.5, 0.9] → min = 0.5.
        # Reflection adds C with high weight; new combined still 0.5.
        # The test verifies the loop ran, not that the verdict improved.
        fake_runner._overrides["epistemic_reflect_on_gaps"] = {
            "sufficient": False,
            "gap_description": "Confounder check missing.",
            "additional_sub_investigations": [
                {
                    "id": "?",
                    "seed_claim": "Confounders ruled out.",
                    "rationale": "Causality requires this.",
                    "weight": 1.0,
                },
            ],
            "rationale": "Adding the confounder check.",
        }

        class _IndexStub:
            def __init__(self, posts):
                self.posts = posts
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **call_kwargs: Any) -> PipelineResult:
                idx = len(self.calls)
                self.calls.append(call_kwargs)
                p = self.posts[idx]
                return PipelineResult(
                    objective_id=call_kwargs.get("objective_id", f"c{idx}"),
                    iterations=5,
                    successful=5,
                    failed=0,
                    status="ok",
                    posterior=_posterior(p),
                )

        ix = _IndexStub([0.5, 0.5, 0.9])  # third call is the reflection-added child
        result = await run_research_question_decomposed(
            "Q",
            database_name="add_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            max_reflection_rounds=1,
            _inner_runner=ix,  # type: ignore[arg-type]
        )
        assert isinstance(result, DecomposedPipelineResult)
        # Three children total (2 initial + 1 from reflection).
        assert len(result.sub_results) == 3
        assert len(ix.calls) == 3
        # The reflection-added child is the last call, with a fresh objective_id.
        assert ix.calls[2]["objective_id"] is not None
        assert ix.calls[2]["objective_id"] != ix.calls[0]["objective_id"]
        assert ix.calls[2]["objective_id"] != ix.calls[1]["objective_id"]
        # Reflection agent fired once.
        reflect_calls = [
            c for c in fake_runner.calls if c[0] == "epistemic_reflect_on_gaps"
        ]
        assert len(reflect_calls) == 1

    async def test_weights_from_decomposition_flow_through(
        self, tmp_path, fake_runner, monkeypatch
    ):
        """Decomposer-emitted weights are persisted on the parent and
        applied by the orchestrator when combining."""
        await _seed_parent(tmp_path, "weighted_test", description="Q")
        monkeypatch.setattr(
            "andamentum.epistemic.runner.DefaultAgentRunner",
            lambda **_kwargs: fake_runner,
        )
        monkeypatch.setattr(
            "andamentum.core.models.resolve_embedding_model_from_args",
            lambda: "test-embed",
        )
        # Decomposer emits two sub-investigations with weights 3 and 1
        # under WEIGHTED_AND. Children's posteriors are 0.9 and 0.3 (set
        # by the index stub below) → weighted mean = 0.75.
        fake_runner._overrides["epistemic_decompose_question"] = {
            "sub_investigations": [
                {
                    "id": "A",
                    "seed_claim": "alpha",
                    "rationale": "ra",
                    "weight": 3.0,
                },
                {
                    "id": "B",
                    "seed_claim": "beta",
                    "rationale": "rb",
                    "weight": 1.0,
                },
            ],
            "combination_rule": "WEIGHTED_AND",
            "rationale": "alpha is more critical",
        }

        class _IndexStub:
            def __init__(self, posts):
                self.posts = posts
                self.calls: list[dict[str, Any]] = []

            async def __call__(self, **call_kwargs: Any) -> PipelineResult:
                idx = len(self.calls)
                self.calls.append(call_kwargs)
                p = self.posts[idx]
                return PipelineResult(
                    objective_id=call_kwargs.get("objective_id", f"c{idx}"),
                    iterations=5,
                    successful=5,
                    failed=0,
                    status="ok",
                    errors=[],
                    posterior=_posterior(p),
                    quarantined=[],
                )

        ix = _IndexStub([0.9, 0.3])
        result = await run_research_question_decomposed(
            "Q",
            database_name="weighted_test",
            model="test:stub",
            embedding_model="t",
            db_dir=str(tmp_path),
            decompose=True,
            _inner_runner=ix,  # type: ignore[arg-type]
        )
        assert isinstance(result, DecomposedPipelineResult)
        assert result.combined.combination_rule == "WEIGHTED_AND"
        assert result.combined.posterior == pytest.approx(0.75)
        # Weights surface in the explanation for diagnostics.
        assert "weights=" in result.combined.explanation
