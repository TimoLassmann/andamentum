"""Unit tests for semantic provider routing (example-query max-match).

All tests mock ``embed_texts`` so they run without Ollama. The goal is
to verify the max-sim ranking, top-K/min-score policies, the cache,
and the fallback contract.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from andamentum.epistemic import provider_routing
from andamentum.epistemic.provider_routing import (
    _clear_cache,
    rank_providers,
    select_providers,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unit_vec(axis: int, dim: int = 8) -> list[float]:
    """Unit vector along one axis."""
    v = [0.0] * dim
    v[axis % dim] = 1.0
    return v


@pytest.fixture(autouse=True)
def _reset_cache():
    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture
def fake_catalogue(monkeypatch):
    """Replace registries with a synthetic 3-provider setup.

    Each provider has 2 example queries identified by sentinel strings.
    """
    fake_registry = {
        "alpha": object,
        "beta": object,
        "gamma": object,
    }
    fake_descriptions = {
        "alpha": "Description for alpha.",
        "beta": "Description for beta.",
        "gamma": "Description for gamma.",
    }
    fake_examples = {
        "alpha": ["alpha example one __alpha_1__", "alpha example two __alpha_2__"],
        "beta": ["beta example one __beta_1__", "beta example two __beta_2__"],
        "gamma": ["gamma example one __gamma_1__", "gamma example two __gamma_2__"],
    }
    monkeypatch.setattr(provider_routing, "PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr(provider_routing, "PROVIDER_DESCRIPTIONS", fake_descriptions)
    monkeypatch.setattr(provider_routing, "PROVIDER_EXAMPLES", fake_examples)
    yield


def _make_embed_mock(
    query_vec: list[float],
    example_vecs: dict[str, list[float]],
) -> AsyncMock:
    """Build a mock embed_texts that returns canned vectors.

    Maps sentinel strings in text to predefined vectors.
    """

    def _side_effect(texts, *, model):
        result = []
        for text in texts:
            if text.startswith("__query__"):
                result.append(query_vec)
            else:
                matched = None
                for sentinel, vec in example_vecs.items():
                    if sentinel in text:
                        matched = vec
                        break
                if matched is None:
                    result.append([0.0] * len(query_vec))
                else:
                    result.append(matched)
        return result

    return AsyncMock(side_effect=_side_effect)


# ── rank_providers: max-sim correctness ──────────────────────────────────────


class TestRankProvidersMath:
    async def test_ranks_by_best_example_match(self, fake_catalogue):
        """Provider whose example is closest to the query wins."""
        mock = _make_embed_mock(
            query_vec=_unit_vec(1),
            example_vecs={
                "__alpha_1__": _unit_vec(0),
                "__alpha_2__": _unit_vec(0),
                "__beta_1__": _unit_vec(1),  # best match
                "__beta_2__": _unit_vec(2),
                "__gamma_1__": _unit_vec(3),
                "__gamma_2__": _unit_vec(4),
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            scores = await rank_providers("__query__", embedding_model="test")

        assert scores[0].name == "beta"
        assert scores[0].score == pytest.approx(1.0)
        assert "__beta_1__" in scores[0].matched_example

    async def test_returns_all_registered_providers(self, fake_catalogue):
        mock = _make_embed_mock(
            query_vec=_unit_vec(0),
            example_vecs={
                "__alpha_1__": _unit_vec(0), "__alpha_2__": _unit_vec(1),
                "__beta_1__": _unit_vec(2), "__beta_2__": _unit_vec(3),
                "__gamma_1__": _unit_vec(4), "__gamma_2__": _unit_vec(5),
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            scores = await rank_providers("__query__", embedding_model="test")

        assert len(scores) == 3
        assert {s.name for s in scores} == {"alpha", "beta", "gamma"}

    async def test_max_sim_not_average(self, fake_catalogue):
        """Score is the MAX across examples, not average."""
        mock = _make_embed_mock(
            query_vec=_unit_vec(0),
            example_vecs={
                "__alpha_1__": _unit_vec(0),  # perfect match
                "__alpha_2__": _unit_vec(7),  # terrible match
                "__beta_1__": [0.5] * 8,     # moderate match
                "__beta_2__": [0.5] * 8,     # moderate match
                "__gamma_1__": _unit_vec(1), "__gamma_2__": _unit_vec(2),
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            scores = await rank_providers("__query__", embedding_model="test")

        # Alpha wins because its BEST example is perfect (1.0),
        # even though its other example is terrible (0.0).
        assert scores[0].name == "alpha"
        assert scores[0].score == pytest.approx(1.0)


# ── select_providers: policy ─────────────────────────────────────────────────


class TestSelectProvidersPolicy:
    async def test_top_k_applied(self, fake_catalogue):
        mock = _make_embed_mock(
            query_vec=[1.0, 1.0, 1.0, 0, 0, 0, 0, 0],
            example_vecs={
                "__alpha_1__": [0.9, 0.1, 0.1, 0, 0, 0, 0, 0],
                "__alpha_2__": [0.0] * 8,
                "__beta_1__": [1.0, 1.0, 1.0, 0, 0, 0, 0, 0],
                "__beta_2__": [0.0] * 8,
                "__gamma_1__": [0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
                "__gamma_2__": [0.0] * 8,
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            result = await select_providers(
                "__query__", embedding_model="test", top_k=1, min_score=0.0,
            )
        assert result == ["beta", "web_search"]

    async def test_min_score_gate(self, fake_catalogue):
        mock = _make_embed_mock(
            query_vec=_unit_vec(7),
            example_vecs={
                "__alpha_1__": _unit_vec(0), "__alpha_2__": _unit_vec(1),
                "__beta_1__": _unit_vec(2), "__beta_2__": _unit_vec(3),
                "__gamma_1__": _unit_vec(4), "__gamma_2__": _unit_vec(5),
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            result = await select_providers(
                "__query__", embedding_model="test", min_score=0.5,
            )
        assert result == ["web_search"]

    async def test_web_search_always_appended(self, fake_catalogue):
        mock = _make_embed_mock(
            query_vec=_unit_vec(0),
            example_vecs={
                "__alpha_1__": _unit_vec(0), "__alpha_2__": _unit_vec(1),
                "__beta_1__": _unit_vec(2), "__beta_2__": _unit_vec(3),
                "__gamma_1__": _unit_vec(4), "__gamma_2__": _unit_vec(5),
            },
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            result = await select_providers(
                "__query__", embedding_model="test", min_score=0.0,
            )
        assert result[-1] == "web_search"
        assert result.count("web_search") == 1


# ── Cache ────────────────────────────────────────────────────────────────────


class TestEmbeddingCache:
    async def test_examples_cached_across_calls(self, fake_catalogue):
        call_count = {"example_batches": 0}

        def _side_effect(texts, *, model):
            if not any(t.startswith("__query__") for t in texts):
                call_count["example_batches"] += 1
            return [[1.0, 0.0]] * len(texts)

        mock = AsyncMock(side_effect=_side_effect)
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            await rank_providers("__query__ first", embedding_model="test")
            await rank_providers("__query__ second", embedding_model="test")

        assert call_count["example_batches"] == 1


# ── Failure modes ────────────────────────────────────────────────────────────


class TestFailureModes:
    async def test_ollama_unreachable_propagates(self, fake_catalogue):
        mock = AsyncMock(side_effect=RuntimeError("Ollama unreachable"))
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            with pytest.raises(RuntimeError, match="Ollama unreachable"):
                await select_providers("any", embedding_model="test")

    async def test_provider_without_examples_skipped(self, monkeypatch):
        monkeypatch.setattr(
            provider_routing, "PROVIDER_REGISTRY", {"good": object, "orphan": object},
        )
        monkeypatch.setattr(
            provider_routing, "PROVIDER_DESCRIPTIONS", {"good": "desc", "orphan": "desc"},
        )
        monkeypatch.setattr(
            provider_routing, "PROVIDER_EXAMPLES", {"good": ["example __good__"]},
        )

        mock = _make_embed_mock(
            query_vec=[1.0, 0.0],
            example_vecs={"__good__": [1.0, 0.0]},
        )
        with patch("andamentum.epistemic.provider_routing.embed_texts", mock):
            scores = await rank_providers("__query__", embedding_model="test")

        assert [s.name for s in scores] == ["good"]


# ── PlanTaskOperation contract ───────────────────────────────────────────────


class TestPlanTaskOperationContract:
    async def test_plan_without_agent_runner_uses_web_search_fallback(self):
        """Without an agent_runner, LLM routing can't run, so only web_search is selected."""
        from andamentum.epistemic.entities.objective import Objective
        from andamentum.epistemic.operations.preplanning import PlanTaskOperation
        from andamentum.epistemic.patterns import WorkItem
        from andamentum.epistemic.repository import EpistemicRepository
        from andamentum.epistemic.storage import InMemoryStorageBackend

        repo = EpistemicRepository(InMemoryStorageBackend())
        obj = Objective(
            entity_id="obj-1",
            objective_id="obj-1",
            description="Does spaced repetition work?",
            phase="analyzed",
            clarified_question="Does spaced repetition work for long-term memory?",
        )
        await repo.save(obj)

        op = PlanTaskOperation(repo, agent_runner=None)
        work = WorkItem(
            entity_id="obj-1", entity_type="objective", operation="plan_task"
        )
        result = await op.execute(work)

        assert result.success is True
        assert "web_search" in result.message
