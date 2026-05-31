"""Tests for description-driven evidence-provider dispatch.

Phase 2 deliverables: the dispatch agent triages and constructs queries,
the embedding pre-filter is a pass-through at current scale, and the
``gather_evidence_new`` orchestrator runs end-to-end producing
``GatheredEvidence`` in the same shape as the legacy pipeline.

Tests use ``FakeAgentRunner`` (in conftest.py) for agent calls and a
hand-rolled stub provider for the gather() layer — no live network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from andamentum.epistemic.dispatch import (
    DispatchResult,
    _render_examples,
    formulate_provider_query,
    gather_evidence_new,
    select_candidates_by_embedding,
)
from andamentum.epistemic.operations import GatheredEvidence


# ──────────────────────────────────────────────────────────────────────────────
# Stub providers and runners
# ──────────────────────────────────────────────────────────────────────────────


class _StubProvider:
    """Minimal provider satisfying the Phase 1 contract for tests.

    Records ``gather()`` calls so tests can assert which providers got
    invoked with which queries.
    """

    description = "Stub provider for tests. Returns one piece of evidence."
    query_guidance = "Plain query. Boolean optional. ID lookup supported."
    query_examples: list[tuple[str, str | None]] = [
        ("test claim that this provider helps with", "test query"),
    ]
    output_kind = "assertion_evidence"
    independence_group = "test_stub"
    provider_contract_version = 1

    def __init__(
        self, name: str = "stub", returns: list[GatheredEvidence] | None = None
    ):
        self.name = name
        self._returns = returns if returns is not None else []
        self.gather_calls: list[str] = []

    async def gather(self, query: str) -> list[GatheredEvidence]:
        self.gather_calls.append(query)
        return list(self._returns)


def _make_evidence(content: str, source_type: str = "stub") -> GatheredEvidence:
    return GatheredEvidence(
        content=content,
        source_ref=f"stub:{content[:20]}",
        source_type=source_type,
        evidence_kind="literature",
        identifiers={},
        structured_data={},
        quality_score=None,
        quality_metadata={},
        limitations=[],
    )


class _DispatchAgentRunner:
    """Specialised stub runner that returns canned dispatch decisions.

    Maps (provider_name) → DispatchProviderOutput-shaped dict. Any
    other agent name falls back to a default.
    """

    def __init__(
        self,
        *,
        by_provider: dict[str, dict[str, Any]] | None = None,
        raise_on: set[str] | None = None,
    ):
        self._by_provider = by_provider or {}
        self._raise_on = raise_on or set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "test-model"

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        agent_name = getattr(defn, "name", str(defn))
        self.calls.append((agent_name, kwargs))
        provider_name = kwargs.get("provider_name", "")
        if provider_name in self._raise_on:
            raise RuntimeError(f"Simulated dispatch failure on {provider_name}")
        spec = self._by_provider.get(
            provider_name,
            {"queries": [], "reasoning": "default abstain", "confidence": 0.5},
        )
        return SimpleNamespace(
            queries=list(spec.get("queries", [])),
            reasoning=spec.get("reasoning", ""),
            confidence=float(spec.get("confidence", 0.5)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — DispatchResult shape
# ──────────────────────────────────────────────────────────────────────────────


class TestDispatchResult:
    def test_abstained_empty_queries(self) -> None:
        r = DispatchResult(queries=[], reasoning="x", confidence=0.9)
        assert r.abstained is True

    def test_not_abstained_one_query(self) -> None:
        r = DispatchResult(queries=["q"], reasoning="x", confidence=0.7)
        assert r.abstained is False

    def test_not_abstained_two_queries(self) -> None:
        r = DispatchResult(queries=["q1", "q2"], reasoning="x", confidence=0.7)
        assert r.abstained is False


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — _render_examples helper
# ──────────────────────────────────────────────────────────────────────────────


class TestRenderExamples:
    def test_empty_examples_returns_placeholder(self) -> None:
        out = _render_examples([])
        assert "no examples" in out.lower()

    def test_commit_example_shows_query(self) -> None:
        out = _render_examples([("claim a", "query a")])
        assert "claim a" in out
        assert "query a" in out
        # The query is presented as a Query: line
        assert "Query:" in out

    def test_abstain_example_marked_explicitly(self) -> None:
        out = _render_examples([("claim b", None)])
        assert "claim b" in out
        assert "ABSTAIN" in out
        # An abstain example must not look like a commit example
        assert "Query:" not in out

    def test_mixed_examples_both_styles_visible(self) -> None:
        out = _render_examples(
            [
                ("a in-domain", "qa"),
                ("a out-of-domain", None),
            ]
        )
        assert "qa" in out
        assert "ABSTAIN" in out


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — formulate_provider_query
# ──────────────────────────────────────────────────────────────────────────────


class TestFormulateProviderQuery:
    async def test_returns_committed_query_when_agent_commits(self) -> None:
        runner = _DispatchAgentRunner(
            by_provider={
                "stub": {
                    "queries": ["test query"],
                    "reasoning": "fits",
                    "confidence": 0.8,
                },
            }
        )
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="some claim",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert result.queries == ["test query"]
        assert result.confidence == pytest.approx(0.8)
        assert result.abstained is False

    async def test_returns_empty_queries_when_agent_abstains(self) -> None:
        runner = _DispatchAgentRunner(
            by_provider={
                "stub": {
                    "queries": [],
                    "reasoning": "out of scope",
                    "confidence": 0.95,
                },
            }
        )
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="some claim",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert result.queries == []
        assert result.abstained is True
        # High abstain confidence is preserved.
        assert result.confidence == pytest.approx(0.95)

    async def test_dispatch_agent_failure_treated_as_abstain(self) -> None:
        runner = _DispatchAgentRunner(raise_on={"stub"})
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="some claim",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        # On hard failure we get an abstain result, not an exception.
        assert result.abstained
        assert result.confidence == 0.0
        assert "Dispatch failed" in result.reasoning

    async def test_more_than_two_queries_truncated_to_two(self) -> None:
        runner = _DispatchAgentRunner(
            by_provider={
                "stub": {
                    "queries": ["q1", "q2", "q3", "q4"],
                    "reasoning": "small model misbehaved",
                    "confidence": 0.6,
                },
            }
        )
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="x",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        # Defensive truncation — prompt asks for ≤ 2 but we clamp.
        assert result.queries == ["q1", "q2"]

    async def test_empty_string_queries_filtered_out(self) -> None:
        runner = _DispatchAgentRunner(
            by_provider={
                "stub": {
                    "queries": ["", "valid query", "   "],
                    "reasoning": "ok",
                    "confidence": 0.7,
                },
            }
        )
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="x",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert result.queries == ["valid query"]

    async def test_confidence_clamped_to_unit_interval(self) -> None:
        runner = _DispatchAgentRunner(
            by_provider={
                "stub": {"queries": ["q"], "reasoning": "ok", "confidence": 1.7},
            }
        )
        provider = _StubProvider("stub")
        result = await formulate_provider_query(
            claim="x",
            provider_name="stub",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert 0.0 <= result.confidence <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — select_candidates_by_embedding
# ──────────────────────────────────────────────────────────────────────────────


class TestSelectCandidatesByEmbedding:
    async def test_top_k_none_passes_through(self) -> None:
        """Phase 2 default: no pre-filter — return all providers."""
        providers = {
            "a": _StubProvider("a"),
            "b": _StubProvider("b"),
            "c": _StubProvider("c"),
        }
        result = await select_candidates_by_embedding(
            claim="anything",
            providers=providers,
            top_k=None,
        )
        assert set(result.keys()) == {"a", "b", "c"}

    async def test_top_k_at_catalogue_size_passes_through(self) -> None:
        providers = {"a": _StubProvider("a"), "b": _StubProvider("b")}
        result = await select_candidates_by_embedding(
            claim="anything",
            providers=providers,
            top_k=2,
        )
        assert set(result.keys()) == {"a", "b"}

    async def test_top_k_larger_than_catalogue_passes_through(self) -> None:
        providers = {"a": _StubProvider("a")}
        result = await select_candidates_by_embedding(
            claim="anything",
            providers=providers,
            top_k=100,
        )
        assert set(result.keys()) == {"a"}


# ──────────────────────────────────────────────────────────────────────────────
# Integration test — gather_evidence_new end-to-end
# ──────────────────────────────────────────────────────────────────────────────


class TestGatherEvidenceNew:
    async def test_committed_provider_gets_called_and_evidence_aggregated(self) -> None:
        """A provider whose dispatch commits gets its gather called and
        its evidence aggregated into the output."""
        ev_a = _make_evidence("evidence from A", source_type="provider_a")
        ev_b = _make_evidence("evidence from B", source_type="provider_b")
        provider_a = _StubProvider("provider_a", returns=[ev_a])
        provider_b = _StubProvider("provider_b", returns=[ev_b])

        runner = _DispatchAgentRunner(
            by_provider={
                "provider_a": {
                    "queries": ["q-for-a"],
                    "reasoning": "fits a",
                    "confidence": 0.8,
                },
                "provider_b": {
                    "queries": ["q-for-b"],
                    "reasoning": "fits b",
                    "confidence": 0.7,
                },
            }
        )

        evidence = await gather_evidence_new(
            claim="some claim",
            providers={"provider_a": provider_a, "provider_b": provider_b},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        # Both providers got called with their respective committed queries.
        assert provider_a.gather_calls == ["q-for-a"]
        assert provider_b.gather_calls == ["q-for-b"]
        # All evidence is in the aggregated result.
        assert len(evidence) == 2
        assert {e.content for e in evidence} == {ev_a.content, ev_b.content}

    async def test_abstained_provider_never_gets_gathered(self) -> None:
        """A provider whose dispatch abstains gets zero gather calls
        — its HTTP layer is never reached."""
        provider_a = _StubProvider("provider_a", returns=[_make_evidence("from a")])
        provider_b = _StubProvider("provider_b", returns=[_make_evidence("from b")])

        runner = _DispatchAgentRunner(
            by_provider={
                "provider_a": {
                    "queries": ["q"],
                    "reasoning": "fits",
                    "confidence": 0.8,
                },
                "provider_b": {
                    "queries": [],
                    "reasoning": "out of scope",
                    "confidence": 0.95,
                },
            }
        )

        evidence = await gather_evidence_new(
            claim="some claim",
            providers={"provider_a": provider_a, "provider_b": provider_b},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        assert provider_a.gather_calls == ["q"]
        assert provider_b.gather_calls == []
        assert len(evidence) == 1

    async def test_all_abstaining_returns_empty(self) -> None:
        """If every provider abstains, the orchestrator returns an
        empty list without calling any gather."""
        provider = _StubProvider("p", returns=[_make_evidence("never returned")])

        runner = _DispatchAgentRunner(
            by_provider={
                "p": {"queries": [], "reasoning": "out of scope", "confidence": 0.9},
            }
        )

        evidence = await gather_evidence_new(
            claim="x",
            providers={"p": provider},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        assert evidence == []
        assert provider.gather_calls == []

    async def test_multiple_queries_for_one_provider_all_run(self) -> None:
        """When dispatch returns 2 queries for one provider, gather is
        called twice for that provider and both results aggregate."""
        ev1 = _make_evidence("from query 1", source_type="p")
        ev2 = _make_evidence("from query 2", source_type="p")

        class _MultiReturnProvider(_StubProvider):
            def __init__(self) -> None:
                super().__init__("p")
                self._iter = iter([[ev1], [ev2]])

            async def gather(self, query: str) -> list[GatheredEvidence]:
                self.gather_calls.append(query)
                return next(self._iter, [])

        provider = _MultiReturnProvider()
        runner = _DispatchAgentRunner(
            by_provider={
                "p": {
                    "queries": ["q1", "q2"],
                    "reasoning": "complementary",
                    "confidence": 0.6,
                },
            }
        )

        evidence = await gather_evidence_new(
            claim="x",
            providers={"p": provider},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        assert provider.gather_calls == ["q1", "q2"]
        assert {e.content for e in evidence} == {ev1.content, ev2.content}

    async def test_provider_gather_failure_logged_but_other_results_kept(self) -> None:
        """If one provider's gather() raises, the orchestrator logs the
        failure and returns the other providers' results — never
        propagates the exception."""
        ev_good = _make_evidence("from good", source_type="good")

        class _FailingProvider(_StubProvider):
            async def gather(self, query: str) -> list[GatheredEvidence]:
                raise RuntimeError("simulated provider HTTP failure")

        good = _StubProvider("good", returns=[ev_good])
        bad = _FailingProvider("bad")
        bad.description = "bad provider description, long enough to pass checks"

        runner = _DispatchAgentRunner(
            by_provider={
                "good": {"queries": ["q"], "reasoning": "x", "confidence": 0.7},
                "bad": {"queries": ["q"], "reasoning": "x", "confidence": 0.7},
            }
        )

        evidence = await gather_evidence_new(
            claim="x",
            providers={"good": good, "bad": bad},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        # The good provider's evidence survives the bad provider's failure.
        assert len(evidence) == 1
        assert evidence[0].content == ev_good.content

    async def test_dispatch_agent_failure_treats_provider_as_abstaining(self) -> None:
        """If the dispatch agent itself errors for one provider, that
        provider's gather is skipped — other providers proceed."""
        ev_good = _make_evidence("from good")
        good = _StubProvider("good", returns=[ev_good])
        bad = _StubProvider("bad", returns=[_make_evidence("never used")])

        runner = _DispatchAgentRunner(
            by_provider={
                "good": {"queries": ["q"], "reasoning": "x", "confidence": 0.7},
            },
            raise_on={"bad"},
        )

        evidence = await gather_evidence_new(
            claim="x",
            providers={"good": good, "bad": bad},
            agent_runner=runner,  # type: ignore[arg-type]
        )

        assert good.gather_calls == ["q"]
        assert bad.gather_calls == []
        assert len(evidence) == 1
