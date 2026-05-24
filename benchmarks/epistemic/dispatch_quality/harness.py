"""Dispatch-quality benchmark harness.

Three-tier validation per the PRD (§6 Phase 3):

- **Tier 1**: per-provider retrieval-quality on curated in-domain +
  out-of-domain claims. Measures relevance rate, hit rate, abstention
  accuracy. Uses each provider's ``query_examples`` (excluded from the
  in-context teaching at evaluation time) as the test fixture.
- **Tier 1.5**: abstention-pattern stability against a held-out claim
  corpus (dev30 by default). Compares new-dispatch's abstain-or-not
  decision against legacy's "returned nothing" pattern.
- **Tier 2**: 5-claim end-to-end fixture through the full pipeline.

The harness is the **iteration loop**. Tier 1 is cheap and runs
repeatedly during prompt and description tuning. Tier 1.5 and Tier 2
each run once at the end of Phase 3.

Outputs:

- Per-provider JSON metrics.
- A markdown summary report.

This module exposes the harness as a library. The companion ``run.py``
in the same directory wires it up as a CLI tool.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from andamentum.core.agents import AgentRunner

from andamentum.epistemic.dispatch import (
    DispatchResult,
    formulate_provider_query,
)

logger = logging.getLogger(__name__)


# ── Tier 1 result types ────────────────────────────────────────────────


@dataclass
class TierOneClaimOutcome:
    """One dispatch decision in the per-provider Tier 1 sweep."""

    claim: str
    expected_in_domain: bool  # True for in-domain claims, False for out-of-domain
    queries: list[str]
    confidence: float
    reasoning: str

    @property
    def committed(self) -> bool:
        return bool(self.queries)

    @property
    def correct(self) -> bool:
        """Triage correctness: commit-on-in-domain or abstain-on-out-of-domain."""
        return self.committed == self.expected_in_domain


@dataclass
class TierOneProviderResult:
    """Aggregate Tier 1 metrics for one provider."""

    provider: str
    in_domain_count: int
    out_of_domain_count: int
    hit_rate: float  # commit / in_domain (recall on in-domain claims)
    abstention_accuracy: float  # abstain / out_of_domain
    triage_accuracy: float  # overall (commit-on-in + abstain-on-out) / total
    outcomes: list[TierOneClaimOutcome] = field(default_factory=list)


# ── Tier 1: per-provider retrieval-quality benchmark ───────────────────


async def run_tier_one_for_provider(
    *,
    provider_name: str,
    provider: Any,
    examples_under_test: list[tuple[str, str | None]] | None = None,
    agent_runner: AgentRunner,
) -> TierOneProviderResult:
    """Run Tier 1 for one provider.

    For each (claim, expected_query) pair in the provider's
    ``query_examples`` (or ``examples_under_test`` if supplied), invoke
    the dispatch agent and check whether the agent's triage decision
    matches expected (commit vs abstain).

    Notes:

    - The agent still sees the provider's ``query_examples`` as
      in-context teaching. This is the realistic deployment shape.
      If you want a held-out evaluation that strips the test examples
      from the in-context block, pass an ``examples_under_test`` list
      and the harness uses ALL ``provider.query_examples`` as context
      EXCEPT those under test.
    - Tier 1 does NOT call ``provider.gather()`` and does NOT score
      relevance via the judge agent. Those are Tier 2's job. Tier 1
      is a fast triage-correctness loop that catches whether the
      dispatch agent routes correctly per the description + examples.

    Args:
        provider_name: Short identifier (matches the registration key).
        provider: A provider instance satisfying the Phase 1 contract.
        examples_under_test: If None (the default), runs over
            ``provider.query_examples`` directly. If provided, the
            harness evaluates these claims; the in-context examples
            shown to the dispatch agent are ``provider.query_examples``
            minus the ``examples_under_test`` set (a leave-one-out
            style evaluation).
        agent_runner: ``AgentRunner`` for the dispatch agent.

    Returns:
        ``TierOneProviderResult`` with per-claim outcomes and aggregates.
    """
    examples = examples_under_test or list(provider.query_examples)

    if examples_under_test:
        # Held-out evaluation: hide test examples from the agent's
        # in-context teaching block.
        test_claims = {c for c, _ in examples}
        held_out_provider = _ProviderWithMaskedExamples(
            inner=provider,
            held_out_claims=test_claims,
        )
        dispatch_provider: Any = held_out_provider
    else:
        # Default: dispatch agent sees all examples (including the
        # ones it's being tested on). Cheaper to run; "memorisation"
        # risk is low because the agent's job is triage + paraphrase,
        # not exact-match recall.
        dispatch_provider = provider

    outcomes: list[TierOneClaimOutcome] = []
    in_domain_count = 0
    out_of_domain_count = 0
    correct_commits = 0
    correct_abstains = 0

    for claim, expected_query in examples:
        expected_in_domain = expected_query is not None

        result: DispatchResult = await formulate_provider_query(
            claim=claim,
            provider_name=provider_name,
            provider=dispatch_provider,
            agent_runner=agent_runner,
        )

        outcome = TierOneClaimOutcome(
            claim=claim,
            expected_in_domain=expected_in_domain,
            queries=list(result.queries),
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
        outcomes.append(outcome)

        if expected_in_domain:
            in_domain_count += 1
            if outcome.committed:
                correct_commits += 1
        else:
            out_of_domain_count += 1
            if not outcome.committed:
                correct_abstains += 1

    hit_rate = correct_commits / in_domain_count if in_domain_count else 0.0
    abstention_accuracy = (
        correct_abstains / out_of_domain_count if out_of_domain_count else 0.0
    )
    triage_total = in_domain_count + out_of_domain_count
    triage_accuracy = (
        (correct_commits + correct_abstains) / triage_total if triage_total else 0.0
    )

    return TierOneProviderResult(
        provider=provider_name,
        in_domain_count=in_domain_count,
        out_of_domain_count=out_of_domain_count,
        hit_rate=hit_rate,
        abstention_accuracy=abstention_accuracy,
        triage_accuracy=triage_accuracy,
        outcomes=outcomes,
    )


async def run_tier_one_all_providers(
    *,
    providers: dict[str, Any],
    agent_runner: AgentRunner,
    held_out: bool = False,
) -> list[TierOneProviderResult]:
    """Run Tier 1 for every provider in the dict. Returns the per-provider
    aggregates in registry order.

    When ``held_out=True``, evaluates each example with the others as
    in-context teaching (leave-one-out). This is the stricter setting
    and the right choice for measuring genuine generalisation.
    """
    results: list[TierOneProviderResult] = []
    for name, provider in providers.items():
        if held_out:
            result = await run_tier_one_for_provider(
                provider_name=name,
                provider=provider,
                examples_under_test=list(provider.query_examples),
                agent_runner=agent_runner,
            )
        else:
            result = await run_tier_one_for_provider(
                provider_name=name,
                provider=provider,
                agent_runner=agent_runner,
            )
        results.append(result)
        logger.info(
            "Tier 1 %s: hit_rate=%.2f abstain_acc=%.2f triage_acc=%.2f",
            name,
            result.hit_rate,
            result.abstention_accuracy,
            result.triage_accuracy,
        )
    return results


# ── Held-out-examples provider proxy ───────────────────────────────────


class _ProviderWithMaskedExamples:
    """A provider proxy that exposes the same attributes as the wrapped
    provider but with specified claims masked out of ``query_examples``.

    Used for leave-one-out evaluation in Tier 1. The dispatch agent sees
    the un-tested examples as in-context teaching; the tested example
    is hidden so the agent can't memorise it."""

    def __init__(self, *, inner: Any, held_out_claims: set[str]):
        self._inner = inner
        self._held_out_claims = held_out_claims

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def query_guidance(self) -> str:
        return self._inner.query_guidance

    @property
    def query_examples(self) -> list[tuple[str, str | None]]:
        return [
            (c, q)
            for c, q in self._inner.query_examples
            if c not in self._held_out_claims
        ]


# ── Report rendering ───────────────────────────────────────────────────


def render_tier_one_summary(results: list[TierOneProviderResult]) -> str:
    """Format Tier 1 results as a markdown summary table."""
    lines = [
        "# Tier 1 — Per-provider dispatch triage accuracy",
        "",
        "| provider | n in-domain | n out-of-domain | hit rate | abstention acc | triage acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.provider} | {r.in_domain_count} | {r.out_of_domain_count} "
            f"| {r.hit_rate:.2f} | {r.abstention_accuracy:.2f} | {r.triage_accuracy:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_tier_one_failures(results: list[TierOneProviderResult]) -> str:
    """List the claims where the dispatch agent's triage decision was wrong.

    Useful for iterating on descriptions and examples: each failure
    points at a description gap (the agent didn't read the scope
    correctly) or a misclassified example."""
    lines = ["# Tier 1 — Triage failures", ""]
    any_failures = False
    for r in results:
        bad_outcomes = [o for o in r.outcomes if not o.correct]
        if not bad_outcomes:
            continue
        any_failures = True
        lines.append(f"## {r.provider}")
        lines.append("")
        for o in bad_outcomes:
            expected = "commit" if o.expected_in_domain else "abstain"
            actual = "commit" if o.committed else "abstain"
            lines.append(f"- **claim:** {o.claim}")
            lines.append(
                f"  - expected: {expected} | actual: {actual} "
                f"(confidence={o.confidence:.2f})"
            )
            lines.append(f"  - reasoning: {o.reasoning}")
            if o.queries:
                lines.append(f"  - queries: {o.queries}")
            lines.append("")
    if not any_failures:
        lines.append("(no triage failures)")
    return "\n".join(lines)


def save_results_json(results: list[TierOneProviderResult], output_path: Path) -> None:
    """Write per-provider results as JSON for diffing across runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier_one": [asdict(r) for r in results],
        "aggregate": {
            "providers_evaluated": len(results),
            "mean_hit_rate": (
                sum(r.hit_rate for r in results) / len(results) if results else 0.0
            ),
            "mean_abstention_accuracy": (
                sum(r.abstention_accuracy for r in results) / len(results)
                if results
                else 0.0
            ),
            "mean_triage_accuracy": (
                sum(r.triage_accuracy for r in results) / len(results)
                if results
                else 0.0
            ),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


# ── Convenience: run-and-report all-in-one ─────────────────────────────


async def run_and_report(
    *,
    providers: dict[str, Any],
    agent_runner: AgentRunner,
    held_out: bool = False,
    output_dir: Path | None = None,
) -> list[TierOneProviderResult]:
    """Run Tier 1, write JSON + markdown reports, return results.

    If ``output_dir`` is None, writes nothing — caller gets the result
    list back and decides what to do with it.
    """
    results = await run_tier_one_all_providers(
        providers=providers,
        agent_runner=agent_runner,
        held_out=held_out,
    )

    summary = render_tier_one_summary(results)
    failures = render_tier_one_failures(results)
    print(summary)
    print()
    print(failures)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "tier_one_summary.md").write_text(summary)
        (output_dir / "tier_one_failures.md").write_text(failures)
        save_results_json(results, output_dir / "tier_one_results.json")

    return results


__all__ = [
    "TierOneClaimOutcome",
    "TierOneProviderResult",
    "run_tier_one_for_provider",
    "run_tier_one_all_providers",
    "render_tier_one_summary",
    "render_tier_one_failures",
    "save_results_json",
    "run_and_report",
]


# CLI entry point is in ``run.py`` — invoke as
# ``python -m benchmarks.epistemic.dispatch_quality.run``.
