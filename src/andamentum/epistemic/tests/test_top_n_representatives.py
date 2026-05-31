"""Tests for ``top_n_representatives`` — claim-relevance ranking.

The function picks the top N evidence pieces for an LLM panel. Old
behaviour ranked by ``quality_score`` (source reliability and
extraction completeness). New behaviour ranks by cosine similarity
between each piece's content embedding and the claim text embedding,
which is the relevance axis the IBE chain actually needs. The legacy
quality-score ranking is preserved as a fallback when ``claim_text``
or ``embedding_model`` is not provided, so test paths and other
callers without embedding infrastructure continue to work.

These tests pin both behaviours plus the failure-mode fallback (when
the embedding endpoint is unreachable, fall back to quality_score
rather than crash the IBE chain).
"""

from __future__ import annotations

from andamentum.epistemic.entities import Evidence
from andamentum.epistemic.operations.claims import top_n_representatives


def _ev(
    eid: str,
    content: str,
    quality: float | None = None,
    source_ref: str = "",
) -> Evidence:
    return Evidence(
        entity_id=eid,
        objective_id="obj-1",
        source_type="pubmed",
        source_ref=source_ref or eid,
        extracted_content=content,
        quality_score=quality,
    )


class TestLegacyQualityScoreFallback:
    """When ``claim_text`` or ``embedding_model`` is not provided, the
    function falls back to the legacy quality-score ranking. This
    preserves existing callers that don't have embedding infrastructure
    wired up (notably some test paths)."""

    async def test_no_claim_text_falls_back_to_quality(self) -> None:
        evidence = [
            _ev("a", "content a", quality=0.3),
            _ev("b", "content b", quality=0.9),
            _ev("c", "content c", quality=0.5),
        ]
        result = await top_n_representatives(evidence, n=2)
        assert [e.entity_id for e in result] == ["b", "c"]

    async def test_no_embedding_model_falls_back_to_quality(self) -> None:
        evidence = [
            _ev("a", "x", quality=0.1),
            _ev("b", "y", quality=0.7),
        ]
        result = await top_n_representatives(evidence, n=2, claim_text="some claim")
        # No embedding_model → quality fallback.
        assert [e.entity_id for e in result] == ["b", "a"]

    async def test_none_quality_sorts_lowest(self) -> None:
        evidence = [
            _ev("a", "x", quality=None),
            _ev("b", "y", quality=0.4),
            _ev("c", "z", quality=None),
        ]
        result = await top_n_representatives(evidence, n=3)
        assert result[0].entity_id == "b"
        # Stable ordering for the tied None entries (alphabetical by source_ref).
        assert {e.entity_id for e in result[1:]} == {"a", "c"}

    async def test_empty_input_returns_empty(self) -> None:
        result = await top_n_representatives([], n=10)
        assert result == []


class TestClaimRelevanceRanking:
    """When both ``claim_text`` and ``embedding_model`` are provided,
    the function ranks by cosine similarity between each piece's
    content embedding and the claim embedding. Higher similarity
    (more claim-relevant) ranks higher.
    """

    async def test_ranks_by_cosine_similarity(self, monkeypatch) -> None:
        """The most-similar piece (by embedding cosine to the claim)
        comes first; the least-similar comes last."""
        from andamentum.epistemic import embeddings as ep_embeddings

        # Synthetic embeddings: claim is [1, 0, 0]; pieces' similarity
        # to claim depends on their first dimension.
        def fake_embed_texts(
            texts, *, model, base_url=None, max_chars=None, timeout=None
        ):
            mapping = {
                "the claim": [1.0, 0.0, 0.0],
                "very relevant content": [0.95, 0.3, 0.0],
                "moderately relevant content": [0.5, 0.5, 0.5],
                "barely relevant content": [0.1, 0.7, 0.7],
            }

            async def _async():
                return [mapping[t] for t in texts]

            return _async()

        monkeypatch.setattr(ep_embeddings, "embed_texts", fake_embed_texts)

        evidence = [
            _ev("low", "barely relevant content", quality=0.99, source_ref="z-low"),
            _ev("mid", "moderately relevant content", quality=0.5, source_ref="m-mid"),
            _ev("high", "very relevant content", quality=0.1, source_ref="a-high"),
        ]
        result = await top_n_representatives(
            evidence,
            n=3,
            claim_text="the claim",
            embedding_model="test:embed",
        )
        # Ranked by cosine similarity (descending), NOT quality_score.
        # Note: the high-similarity piece has the LOWEST quality_score
        # — pinning the new behaviour against the old.
        assert [e.entity_id for e in result] == ["high", "mid", "low"]

    async def test_caps_at_n(self, monkeypatch) -> None:
        from andamentum.epistemic import embeddings as ep_embeddings

        def fake_embed_texts(
            texts, *, model, base_url=None, max_chars=None, timeout=None
        ):
            mapping = {
                "claim": [1.0, 0.0],
                "ev1": [0.9, 0.4],
                "ev2": [0.7, 0.5],
                "ev3": [0.5, 0.7],
                "ev4": [0.3, 0.9],
                "ev5": [0.1, 1.0],
            }

            async def _async():
                return [mapping[t] for t in texts]

            return _async()

        monkeypatch.setattr(ep_embeddings, "embed_texts", fake_embed_texts)

        evidence = [_ev(f"e{i}", f"ev{i}") for i in range(1, 6)]
        result = await top_n_representatives(
            evidence,
            n=2,
            claim_text="claim",
            embedding_model="test:embed",
        )
        # Only top 2 by similarity returned.
        assert len(result) == 2
        assert [e.entity_id for e in result] == ["e1", "e2"]

    async def test_deterministic_tiebreak_by_source_ref(self, monkeypatch) -> None:
        """When two pieces have identical similarity scores, the
        tiebreaker is ``source_ref`` (alphabetical) for stable output
        across re-runs on the same evidence base."""
        from andamentum.epistemic import embeddings as ep_embeddings

        def fake_embed_texts(
            texts, *, model, base_url=None, max_chars=None, timeout=None
        ):
            # All evidence has IDENTICAL embedding to the claim.
            mapping = {t: [1.0, 0.0] for t in texts}

            async def _async():
                return [mapping[t] for t in texts]

            return _async()

        monkeypatch.setattr(ep_embeddings, "embed_texts", fake_embed_texts)

        evidence = [
            _ev("z", "same content", source_ref="zzz"),
            _ev("a", "same content", source_ref="aaa"),
            _ev("m", "same content", source_ref="mmm"),
        ]
        result = await top_n_representatives(
            evidence,
            n=3,
            claim_text="claim",
            embedding_model="test:embed",
        )
        assert [e.source_ref for e in result] == ["aaa", "mmm", "zzz"]


class TestEmbeddingFailureFallback:
    """When the embedding endpoint is unreachable, the function falls
    back to ``quality_score`` ranking rather than crashing the IBE
    chain. The downstream consumer still gets a sensible candidate
    set, just less relevance-optimal."""

    async def test_runtime_error_falls_back_to_quality(self, monkeypatch) -> None:
        from andamentum.epistemic import embeddings as ep_embeddings

        def fake_embed_texts(
            texts, *, model, base_url=None, max_chars=None, timeout=None
        ):
            async def _async():
                raise RuntimeError("ollama unreachable")

            return _async()

        monkeypatch.setattr(ep_embeddings, "embed_texts", fake_embed_texts)

        evidence = [
            _ev("a", "x", quality=0.2),
            _ev("b", "y", quality=0.9),
        ]
        result = await top_n_representatives(
            evidence,
            n=2,
            claim_text="claim",
            embedding_model="test:embed",
        )
        # Endpoint failed → quality_score fallback.
        assert [e.entity_id for e in result] == ["b", "a"]
