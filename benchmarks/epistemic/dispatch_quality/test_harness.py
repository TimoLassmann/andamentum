"""Tests for the dispatch-quality benchmark harness.

The harness drives the dispatch agent against per-provider claim sets.
Tests use a stub agent runner with canned routing decisions so we
verify the metric math without spending real LLM tokens.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from .fixtures import (
    all_examples,
    get_in_domain_claims,
    get_out_of_domain_claims,
)
from .harness import (
    TierOneClaimOutcome,
    _ProviderWithMaskedExamples,
    render_tier_one_failures,
    render_tier_one_summary,
    run_tier_one_for_provider,
)


# asyncio_mode = "auto" is set in pyproject.toml — async tests don't need a mark.


class _CannedRunner:
    """Stub agent runner with a predetermined response per claim.

    Maps a claim string to ``{queries, reasoning, confidence}``. Any
    claim not in the map yields an abstain (empty queries).
    """

    def __init__(self, *, by_claim: dict[str, dict[str, Any]] | None = None):
        self._by_claim = by_claim or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "test-model"

    async def run(self, defn: Any, **kwargs: Any) -> Any:
        agent_name = getattr(defn, "name", str(defn))
        self.calls.append((agent_name, kwargs))
        claim = kwargs.get("claim", "")
        spec = self._by_claim.get(
            claim,
            {"queries": [], "reasoning": "default abstain", "confidence": 0.5},
        )
        return SimpleNamespace(
            queries=list(spec["queries"]),
            reasoning=spec["reasoning"],
            confidence=float(spec["confidence"]),
        )


class _MinimalProvider:
    description = (
        "Test provider with enough description to satisfy the contract. "
        "Strong for some imaginary in-domain claims. Weak for everything else. "
        "Returns mock evidence on gather()."
    )
    query_guidance = (
        "Plain text. Boolean operators AND/OR/NOT. Phrase quoting. "
        "ID lookup via 'id:' prefix."
    )
    output_kind = "assertion_evidence"
    independence_group = "test"
    provider_contract_version = 1

    def __init__(self, examples: list[tuple[str, str | None]]):
        self.query_examples = examples


# ──────────────────────────────────────────────────────────────────────────────
# fixtures helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestFixtures:
    def test_all_examples_covers_every_registered_provider(self) -> None:
        from andamentum.epistemic.providers import PROVIDER_REGISTRY

        examples = all_examples()
        assert set(examples.keys()) == set(PROVIDER_REGISTRY.keys())

    def test_every_provider_has_in_domain_and_out_of_domain_examples(self) -> None:
        from andamentum.epistemic.providers import PROVIDER_REGISTRY

        for name in PROVIDER_REGISTRY:
            in_d = get_in_domain_claims(name)
            out_d = get_out_of_domain_claims(name)
            assert len(in_d) >= 1, f"{name}: no in-domain examples"
            assert len(out_d) >= 1, f"{name}: no out-of-domain examples"

    def test_in_domain_and_out_of_domain_are_disjoint(self) -> None:
        from andamentum.epistemic.providers import PROVIDER_REGISTRY

        for name in PROVIDER_REGISTRY:
            in_d = set(get_in_domain_claims(name))
            out_d = set(get_out_of_domain_claims(name))
            assert not (in_d & out_d), f"{name}: overlapping in/out-of-domain"


# ──────────────────────────────────────────────────────────────────────────────
# Tier 1 harness — single provider
# ──────────────────────────────────────────────────────────────────────────────


class TestTierOneForProvider:
    async def test_all_correct_gives_perfect_scores(self) -> None:
        """When the dispatch agent triages perfectly, hit_rate = 1.0,
        abstention_accuracy = 1.0, triage_accuracy = 1.0."""
        provider = _MinimalProvider(
            [
                ("in-domain claim A", "q-A"),
                ("in-domain claim B", "q-B"),
                ("out-of-domain claim X", None),
            ]
        )
        runner = _CannedRunner(
            by_claim={
                "in-domain claim A": {
                    "queries": ["q-A"],
                    "reasoning": "fits",
                    "confidence": 0.8,
                },
                "in-domain claim B": {
                    "queries": ["q-B"],
                    "reasoning": "fits",
                    "confidence": 0.7,
                },
                "out-of-domain claim X": {
                    "queries": [],
                    "reasoning": "abstain",
                    "confidence": 0.9,
                },
            }
        )

        result = await run_tier_one_for_provider(
            provider_name="testp",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )

        assert result.provider == "testp"
        assert result.in_domain_count == 2
        assert result.out_of_domain_count == 1
        assert result.hit_rate == pytest.approx(1.0)
        assert result.abstention_accuracy == pytest.approx(1.0)
        assert result.triage_accuracy == pytest.approx(1.0)
        assert all(o.correct for o in result.outcomes)

    async def test_in_domain_abstain_is_a_miss(self) -> None:
        """When the agent abstains on an in-domain claim, hit_rate drops."""
        provider = _MinimalProvider(
            [
                ("in-domain claim", "expected query"),
                ("out-of-domain claim", None),
            ]
        )
        runner = _CannedRunner(
            by_claim={
                "in-domain claim": {
                    "queries": [],
                    "reasoning": "wrongly abstained",
                    "confidence": 0.5,
                },
                "out-of-domain claim": {
                    "queries": [],
                    "reasoning": "correctly abstained",
                    "confidence": 0.9,
                },
            }
        )

        result = await run_tier_one_for_provider(
            provider_name="testp",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert result.hit_rate == pytest.approx(0.0)
        # Abstention accuracy is unaffected — out-of-domain still got abstained correctly.
        assert result.abstention_accuracy == pytest.approx(1.0)
        # Triage accuracy: 1 correct (out-of-domain) of 2 total
        assert result.triage_accuracy == pytest.approx(0.5)

    async def test_out_of_domain_commit_is_a_miss(self) -> None:
        """When the agent commits on an out-of-domain claim, abstention
        accuracy drops."""
        provider = _MinimalProvider(
            [
                ("in-domain claim", "qid"),
                ("out-of-domain claim", None),
            ]
        )
        runner = _CannedRunner(
            by_claim={
                "in-domain claim": {
                    "queries": ["qid"],
                    "reasoning": "fits",
                    "confidence": 0.8,
                },
                "out-of-domain claim": {
                    "queries": ["wrong"],
                    "reasoning": "should have abstained",
                    "confidence": 0.6,
                },
            }
        )

        result = await run_tier_one_for_provider(
            provider_name="testp",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert result.hit_rate == pytest.approx(1.0)
        assert result.abstention_accuracy == pytest.approx(0.0)
        assert result.triage_accuracy == pytest.approx(0.5)

    async def test_outcomes_record_each_claim(self) -> None:
        """The result has one TierOneClaimOutcome per example tested."""
        provider = _MinimalProvider(
            [
                ("a", "qa"),
                ("b", None),
                ("c", "qc"),
            ]
        )
        runner = _CannedRunner(
            by_claim={
                "a": {"queries": ["qa"], "reasoning": "x", "confidence": 0.7},
                "c": {"queries": ["qc"], "reasoning": "y", "confidence": 0.7},
                # "b" not in map → defaults to abstain
            }
        )

        result = await run_tier_one_for_provider(
            provider_name="testp",
            provider=provider,
            agent_runner=runner,  # type: ignore[arg-type]
        )
        assert len(result.outcomes) == 3
        assert [o.claim for o in result.outcomes] == ["a", "b", "c"]

    async def test_held_out_evaluation_hides_test_examples(self) -> None:
        """When examples_under_test is provided, the dispatch agent's
        in-context teaching block excludes those examples — verifying
        the leave-one-out evaluation shape works."""
        examples = [
            ("claim 1", "q1"),
            ("claim 2", "q2"),
            ("claim 3", None),
        ]
        provider = _MinimalProvider(examples)
        runner = _CannedRunner()  # all responses default to abstain
        captured_provider_examples: list[list[tuple[str, str | None]]] = []

        # Wrap the runner to capture what query_examples got rendered into
        # the agent's context. We trap it by intercepting the dispatch
        # agent's invocation — the harness calls formulate_provider_query
        # which calls runner.run(...).
        original_run = runner.run

        async def capturing_run(defn, **kwargs):
            captured_provider_examples.append(
                # The query_examples kwarg is the rendered string; recover the
                # claim list by checking which claims appear in it.
                [(c, q) for c, q in examples if c in kwargs.get("query_examples", "")]
            )
            return await original_run(defn, **kwargs)

        runner.run = capturing_run  # type: ignore[assignment]

        await run_tier_one_for_provider(
            provider_name="testp",
            provider=provider,
            examples_under_test=examples,
            agent_runner=runner,  # type: ignore[arg-type]
        )

        # For each tested claim, that claim should NOT appear in the
        # query_examples block shown to the dispatch agent. The other
        # examples should appear.
        # (Three calls = three test claims, each masking itself.)
        assert len(captured_provider_examples) == 3
        # In call i, claim i is masked but the others appear.
        # Call 0 evaluates "claim 1", so "claim 1" should NOT appear in
        # the examples shown for that call.
        for i, (test_claim, _) in enumerate(examples):
            shown_claims = {c for c, _ in captured_provider_examples[i]}
            assert test_claim not in shown_claims, (
                f"Test claim {test_claim!r} leaked into in-context examples "
                f"for its own evaluation"
            )

    async def test_provider_proxy_masks_correctly(self) -> None:
        """Unit-test the _ProviderWithMaskedExamples proxy directly."""
        inner = _MinimalProvider(
            [
                ("a", "qa"),
                ("b", "qb"),
                ("c", None),
            ]
        )
        proxy = _ProviderWithMaskedExamples(
            inner=inner,
            held_out_claims={"a", "c"},
        )
        # Only "b" survives.
        assert proxy.query_examples == [("b", "qb")]
        # Description and query_guidance pass through unchanged.
        assert proxy.description == inner.description
        assert proxy.query_guidance == inner.query_guidance


# ──────────────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────────────


class TestReportRendering:
    def _make_result(
        self,
        provider: str = "p",
        in_d: int = 2,
        out_d: int = 1,
        outcomes: list[TierOneClaimOutcome] | None = None,
    ):
        from .harness import TierOneProviderResult

        return TierOneProviderResult(
            provider=provider,
            in_domain_count=in_d,
            out_of_domain_count=out_d,
            hit_rate=0.5,
            abstention_accuracy=1.0,
            triage_accuracy=0.667,
            outcomes=outcomes or [],
        )

    def test_summary_includes_every_provider(self) -> None:
        results = [self._make_result("a"), self._make_result("b")]
        out = render_tier_one_summary(results)
        assert "| a |" in out
        assert "| b |" in out
        assert "| n in-domain |" in out

    def test_failures_lists_per_provider_misses(self) -> None:
        bad_outcome = TierOneClaimOutcome(
            claim="bad claim",
            expected_in_domain=True,
            queries=[],
            confidence=0.3,
            reasoning="agent wrongly abstained",
        )
        good_outcome = TierOneClaimOutcome(
            claim="good claim",
            expected_in_domain=True,
            queries=["query"],
            confidence=0.8,
            reasoning="fits",
        )
        r = self._make_result(outcomes=[bad_outcome, good_outcome])
        out = render_tier_one_failures([r])
        assert "bad claim" in out
        # Good outcomes don't appear.
        assert "good claim" not in out
        assert "expected: commit" in out
        assert "actual: abstain" in out

    def test_failures_shows_no_failures_message_when_clean(self) -> None:
        # All-correct outcomes
        outcomes = [
            TierOneClaimOutcome(
                claim="ok claim",
                expected_in_domain=True,
                queries=["q"],
                confidence=0.8,
                reasoning="fits",
            ),
        ]
        r = self._make_result(outcomes=outcomes)
        out = render_tier_one_failures([r])
        assert "no triage failures" in out.lower()
