"""Unit tests for semantic provider routing.

All tests here mock ``embed_texts`` so they run without Ollama. The goal is
to verify the ranking math, the top-K and min-score policies, the cache,
and the fallback contract — not the embedding backend itself.

The real-embedding accuracy benchmark lives in
``test_provider_routing_benchmark.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from andamentum.epistemic import provider_routing
from andamentum.epistemic.provider_routing import (
    ProviderScore,
    _clear_cache,
    rank_providers,
    select_providers,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unit_vec(axis: int, dim: int = 8) -> list[float]:
    """Unit vector along one axis — used to build orthogonal test vectors."""
    v = [0.0] * dim
    v[axis % dim] = 1.0
    return v


def _make_embed_mock(
    query_vec: list[float],
    provider_vecs: dict[str, list[float]],
) -> AsyncMock:
    """Build a mock ``embed_texts`` that returns canned vectors.

    First call: embeds the provider descriptions (list of N texts) →
    returns N vectors in registration order.
    Second call: embeds a single query → returns ``[query_vec]``.

    The mock tracks call count so tests can assert cache behavior.
    """
    # The real call order is: provider embeddings first (cached), then query.
    # But _embed_providers_cached only embeds missing providers, so if the
    # cache is warm it skips straight to the query.

    def _side_effect(texts, *, model):
        # Match texts to either provider descriptions or a query.
        if len(texts) == 1 and texts[0].startswith("__query__"):
            return [query_vec]
        # Otherwise assume this is a batch of provider descriptions; return
        # the vector for each by looking up which provider it belongs to.
        result = []
        for text in texts:
            # Find the provider whose description this is. We identify it by
            # a sentinel inserted into the description.
            matched = None
            for name, vec in provider_vecs.items():
                if f"__sentinel_{name}__" in text:
                    matched = vec
                    break
            if matched is None:
                raise AssertionError(
                    f"Mock embed_texts got an unrecognized text: {text[:80]!r}"
                )
            result.append(matched)
        return result

    mock = AsyncMock(side_effect=_side_effect)
    return mock


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure every test starts with an empty provider-embedding cache."""
    _clear_cache()
    yield
    _clear_cache()


@pytest.fixture
def fake_catalogue(monkeypatch):
    """Replace PROVIDER_REGISTRY and PROVIDER_DESCRIPTIONS with a fixed set.

    Every test in this file uses a synthetic catalogue so we don't depend
    on the real descriptions (which may be tuned independently). Each fake
    provider has a sentinel string in its description that the mock embed
    function uses to identify it.
    """
    fake_registry = {
        "alpha": object,
        "beta": object,
        "gamma": object,
    }
    fake_descriptions = {
        "alpha": "Description for alpha. __sentinel_alpha__",
        "beta": "Description for beta. __sentinel_beta__",
        "gamma": "Description for gamma. __sentinel_gamma__",
    }
    monkeypatch.setattr(provider_routing, "PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr(
        provider_routing, "PROVIDER_DESCRIPTIONS", fake_descriptions
    )
    yield


# ── rank_providers: math correctness ─────────────────────────────────────────


class TestRankProvidersMath:
    async def test_ranks_by_cosine_similarity(self, fake_catalogue):
        """Provider whose vector is closest to the query wins."""
        # Query aligns exactly with beta.
        mock = _make_embed_mock(
            query_vec=_unit_vec(1),
            provider_vecs={
                "alpha": _unit_vec(0),
                "beta": _unit_vec(1),
                "gamma": _unit_vec(2),
            },
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            # Tag the query so the mock can distinguish it from descriptions.
            scores = await rank_providers(
                "__query__ about beta", embedding_model="test-model"
            )

        assert [s.name for s in scores] == ["beta", "alpha", "gamma"]
        assert scores[0].score == pytest.approx(1.0)
        assert scores[1].score == pytest.approx(0.0, abs=1e-6)
        assert scores[2].score == pytest.approx(0.0, abs=1e-6)

    async def test_returns_all_registered_providers(self, fake_catalogue):
        """rank_providers returns one ProviderScore per registered provider."""
        mock = _make_embed_mock(
            query_vec=_unit_vec(0),
            provider_vecs={
                "alpha": _unit_vec(0),
                "beta": _unit_vec(1),
                "gamma": _unit_vec(2),
            },
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            scores = await rank_providers(
                "__query__ anything", embedding_model="test-model"
            )

        assert len(scores) == 3
        assert {s.name for s in scores} == {"alpha", "beta", "gamma"}
        assert all(isinstance(s, ProviderScore) for s in scores)


# ── select_providers: policy (top-K + min_score + fallback) ──────────────────


class TestSelectProvidersPolicy:
    async def test_top_k_applied(self, fake_catalogue):
        """Only top_k matches are returned, plus web_search."""
        # Query aligns with beta; all three have non-zero similarity.
        mock = _make_embed_mock(
            query_vec=[1.0, 1.0, 1.0, 0, 0, 0, 0, 0],
            provider_vecs={
                "alpha": [0.9, 0.1, 0.1, 0, 0, 0, 0, 0],
                "beta": [1.0, 1.0, 1.0, 0, 0, 0, 0, 0],
                "gamma": [0.5, 0.5, 0.5, 0, 0, 0, 0, 0],
            },
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            result = await select_providers(
                "__query__ anything",
                embedding_model="test-model",
                top_k=1,
                min_score=0.0,
            )

        # top_k=1 → one semantic match + web_search appended
        assert result == ["beta", "web_search"]

    async def test_min_score_gate_excludes_low_scores(self, fake_catalogue):
        """Providers below min_score are dropped."""
        # All providers have similarity below 0.5 to the query.
        mock = _make_embed_mock(
            query_vec=_unit_vec(7),
            provider_vecs={
                "alpha": _unit_vec(0),
                "beta": _unit_vec(1),
                "gamma": _unit_vec(2),
            },
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            result = await select_providers(
                "__query__ off-topic",
                embedding_model="test-model",
                min_score=0.5,
            )

        # No semantic match clears the gate → only web_search fallback.
        assert result == ["web_search"]

    async def test_web_search_always_appended(self, fake_catalogue):
        """web_search is always the last element regardless of matches."""
        mock = _make_embed_mock(
            query_vec=_unit_vec(0),
            provider_vecs={
                "alpha": _unit_vec(0),
                "beta": _unit_vec(1),
                "gamma": _unit_vec(2),
            },
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            result = await select_providers(
                "__query__ anything",
                embedding_model="test-model",
                min_score=0.0,
            )

        assert result[-1] == "web_search"
        assert result.count("web_search") == 1  # not duplicated

    async def test_empty_catalogue_returns_only_web_search(self, monkeypatch):
        """With no registered providers, the result is just web_search."""
        monkeypatch.setattr(provider_routing, "PROVIDER_REGISTRY", {})
        monkeypatch.setattr(provider_routing, "PROVIDER_DESCRIPTIONS", {})

        mock = AsyncMock(return_value=[[1.0, 0.0]])
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            result = await select_providers(
                "anything",
                embedding_model="test-model",
            )

        assert result == ["web_search"]


# ── Cache correctness ────────────────────────────────────────────────────────


class TestEmbeddingCache:
    async def test_provider_embeddings_cached_across_calls(self, fake_catalogue):
        """Second rank_providers call reuses cached provider embeddings."""
        call_count = {"total": 0, "provider_batches": 0}

        def _side_effect(texts, *, model):
            call_count["total"] += 1
            # If the batch contains the sentinel, it's provider descriptions.
            if any("__sentinel_" in t for t in texts):
                call_count["provider_batches"] += 1
                return [
                    [1.0, 0.0, 0.0] if "alpha" in t
                    else [0.0, 1.0, 0.0] if "beta" in t
                    else [0.0, 0.0, 1.0]
                    for t in texts
                ]
            # Otherwise it's a query embedding.
            return [[1.0, 0.0, 0.0]]

        mock = AsyncMock(side_effect=_side_effect)
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            await rank_providers("first query", embedding_model="test-model")
            await rank_providers("second query", embedding_model="test-model")
            await rank_providers("third query", embedding_model="test-model")

        # Provider batch embedding should happen exactly once.
        assert call_count["provider_batches"] == 1
        # But query embedding should happen three times.
        assert call_count["total"] == 4  # 1 provider batch + 3 query calls

    async def test_cache_invalidated_by_model_change(self, fake_catalogue):
        """Switching embedding_model triggers a fresh provider embed batch."""
        call_count = {"provider_batches": 0}

        def _side_effect(texts, *, model):
            if any("__sentinel_" in t for t in texts):
                call_count["provider_batches"] += 1
                return [[1.0, 0.0]] * len(texts)
            return [[1.0, 0.0]]

        mock = AsyncMock(side_effect=_side_effect)
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            await rank_providers("q", embedding_model="model-a")
            await rank_providers("q", embedding_model="model-b")
            await rank_providers("q", embedding_model="model-a")

        # model-a: batch once; model-b: batch once; model-a again: cache hit.
        assert call_count["provider_batches"] == 2


# ── Failure modes ────────────────────────────────────────────────────────────


class TestFailureModes:
    async def test_ollama_unreachable_propagates_error(self, fake_catalogue):
        """If embed_texts raises RuntimeError, the router does not swallow it."""
        mock = AsyncMock(side_effect=RuntimeError("Ollama unreachable"))
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            with pytest.raises(RuntimeError, match="Ollama unreachable"):
                await select_providers(
                    "any question",
                    embedding_model="test-model",
                )

    async def test_provider_with_no_description_is_skipped(self, monkeypatch):
        """A provider registered without a description is excluded."""
        monkeypatch.setattr(
            provider_routing,
            "PROVIDER_REGISTRY",
            {"good": object, "orphan": object},
        )
        monkeypatch.setattr(
            provider_routing,
            "PROVIDER_DESCRIPTIONS",
            {"good": "has __sentinel_good__ description"},  # orphan missing
        )

        mock = _make_embed_mock(
            query_vec=[1.0, 0.0],
            provider_vecs={"good": [1.0, 0.0]},
        )
        with patch(
            "andamentum.epistemic.provider_routing.embed_texts", mock
        ):
            scores = await rank_providers(
                "__query__ anything", embedding_model="test-model"
            )

        # Orphan is silently excluded; only good is ranked.
        assert [s.name for s in scores] == ["good"]


# ── Integration with PlanTaskOperation contract ──────────────────────────────


class TestPlanTaskOperationContract:
    """The operation reads self.embedding_model and expects it to be set."""

    async def test_missing_embedding_model_in_operation(self):
        """PlanTaskOperation returns a failure result when embedding_model is None."""
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

        op = PlanTaskOperation(repo, agent_runner=None, embedding_model=None)
        work = WorkItem(
            entity_id="obj-1", entity_type="objective", operation="plan_task"
        )
        result = await op.execute(work)

        assert result.success is False
        assert "embedding_model" in result.message
