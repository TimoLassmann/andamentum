"""Tests for the K8 Bug #1 fix — provider tournament helper.

Phase 1 of docs/superpowers/plans/2026-05-05-k8-bug1-provider-tournament.md.
The helper is a pure function added to ``operations/preplanning.py``;
this file pins its behaviour without yet wiring it into PlanTask.

The tournament's job: pick the top K providers for an objective via
iterative ranker calls (pick → remove → pick again). The structural
guarantee we want: K different providers (no collapse), regardless
of what the LLM picks on each call.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.epistemic.operations.preplanning import (
    RESEARCH_MODE_PROVIDER_K,
    _run_provider_tournament,
)


class _FakeRanker:
    """Minimal AgentRunner double — returns the next provider in
    ``picks`` each time ``run`` is called. Records all calls."""

    def __init__(self, picks: list[str]) -> None:
        self.picks = list(picks)
        self.calls: list[dict[str, Any]] = []

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        assert agent_name == "epistemic_rank_providers", (
            f"unexpected agent: {agent_name}"
        )
        self.calls.append(kwargs)
        if not self.picks:
            raise AssertionError("FakeRanker exhausted — too many calls")
        chosen = self.picks.pop(0)

        # Mimic RankProvidersOutput shape (the real agent returns a
        # pydantic model with .chosen_provider).
        class _Result:
            def __init__(self, provider: str) -> None:
                self.chosen_provider = provider

        return _Result(chosen)


# Realistic candidate set for the tests below — same set shipped in
# providers/__init__.py, minus a couple to keep tests compact.
_CANDIDATES = ["pubmed", "europepmc", "cochrane", "openalex", "biorxiv"]
_DESCRIPTIONS = {
    "pubmed": "Biomedical literature (NCBI). Strong on clinical research.",
    "europepmc": "Open biomedical/life-sciences literature.",
    "cochrane": "Systematic reviews of clinical interventions.",
    "openalex": "Generalist scholarly index across all fields.",
    "biorxiv": "Preprints in biology and life sciences.",
}


class TestHappyPath:
    async def test_tournament_picks_k_distinct_providers(self) -> None:
        """The structural guarantee: K rounds, K distinct providers.
        Even if the LLM tried to pick the same provider twice, it
        couldn't — once picked, it's removed from the pool."""
        ranker = _FakeRanker(picks=["pubmed", "cochrane"])
        result = await _run_provider_tournament(
            agent_runner=ranker,
            question="Does X reduce Y in adults?",
            candidates=_CANDIDATES,
            candidate_descriptions=_DESCRIPTIONS,
            k=2,
        )
        assert result == ["pubmed", "cochrane"]
        assert len(ranker.calls) == 2

        # First call sees all 5 candidates.
        first_candidates = ranker.calls[0]["candidates"]
        for name in _CANDIDATES:
            assert name in first_candidates

        # Second call must NOT include pubmed (already picked).
        second_candidates = ranker.calls[1]["candidates"]
        assert "pubmed" not in second_candidates
        assert "cochrane" in second_candidates  # still in pool to be picked


class TestEdgeCases:
    async def test_k_exceeds_candidates_returns_all(self) -> None:
        """K > len(candidates): tournament returns all candidates in
        ranker order, no infinite loop."""
        ranker = _FakeRanker(picks=["pubmed", "cochrane"])
        result = await _run_provider_tournament(
            agent_runner=ranker,
            question="q",
            candidates=["pubmed", "cochrane"],
            candidate_descriptions={"pubmed": "p", "cochrane": "c"},
            k=5,
        )
        assert result == ["pubmed", "cochrane"]
        assert len(ranker.calls) == 2  # capped at len(candidates), not k

    async def test_k_equals_one_returns_single(self) -> None:
        """K=1 degrades cleanly to the existing per-objective single-pick
        behaviour. Useful as a back-compat path if a future caller
        wants the old shape."""
        ranker = _FakeRanker(picks=["pubmed"])
        result = await _run_provider_tournament(
            agent_runner=ranker,
            question="q",
            candidates=_CANDIDATES,
            candidate_descriptions=_DESCRIPTIONS,
            k=1,
        )
        assert result == ["pubmed"]
        assert len(ranker.calls) == 1

    async def test_k_zero_raises(self) -> None:
        ranker = _FakeRanker(picks=[])
        with pytest.raises(ValueError, match="k must be"):
            await _run_provider_tournament(
                agent_runner=ranker,
                question="q",
                candidates=_CANDIDATES,
                candidate_descriptions=_DESCRIPTIONS,
                k=0,
            )

    async def test_empty_candidates_raises(self) -> None:
        ranker = _FakeRanker(picks=[])
        with pytest.raises(ValueError, match="candidates is empty"):
            await _run_provider_tournament(
                agent_runner=ranker,
                question="q",
                candidates=[],
                candidate_descriptions={},
                k=2,
            )


class TestDefensiveFallback:
    async def test_ranker_picks_unknown_provider_falls_back(self) -> None:
        """Defensive: small models can hallucinate a provider name not
        in the pool. The tournament must still progress — fall back to
        the first remaining candidate."""
        # First call: ranker hallucinates "pubmd" (typo); fall back.
        # Second call: ranker picks legitimately.
        ranker = _FakeRanker(picks=["pubmd", "cochrane"])
        result = await _run_provider_tournament(
            agent_runner=ranker,
            question="q",
            candidates=_CANDIDATES,
            candidate_descriptions=_DESCRIPTIONS,
            k=2,
        )
        # Round 1: hallucination → fallback to remaining[0] = pubmed.
        # Round 2: ranker picks cochrane legitimately.
        assert result == ["pubmed", "cochrane"]
        assert len(ranker.calls) == 2

    async def test_caller_candidates_list_not_mutated(self) -> None:
        """The function works on a copy of ``candidates``; the caller's
        list must be unchanged after the call. (Important when the
        same candidate list is reused across tournaments — e.g. for
        repeated test setup or for hypothetical future per-sub-claim
        tournaments.)"""
        caller_list = list(_CANDIDATES)
        snapshot = list(caller_list)
        ranker = _FakeRanker(picks=["pubmed", "cochrane"])
        await _run_provider_tournament(
            agent_runner=ranker,
            question="q",
            candidates=caller_list,
            candidate_descriptions=_DESCRIPTIONS,
            k=2,
        )
        assert caller_list == snapshot


class TestConstantValue:
    def test_research_mode_provider_k_is_two(self) -> None:
        """Pin the K=2 default. If we change to K=3 in the future,
        this test will surface it explicitly so the manuscript
        narrative stays in sync."""
        assert RESEARCH_MODE_PROVIDER_K == 2
