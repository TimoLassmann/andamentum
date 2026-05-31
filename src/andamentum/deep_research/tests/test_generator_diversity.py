"""Surface 2 — generator diversity calibration (cloud LLM).

Real ``query_generator`` agent against ``openai:gpt-5.4-nano``. The
verifier is not invoked here — we simulate the loop's rejection path by
calling ``GenerateOne`` directly with feedback as if every prior attempt
had been rejected.

Two questions this surfaces:

1. Given only generic feedback ("not specific enough"), does the
   generator produce distinct queries across attempts, or does it loop on
   the same wording?
2. Given specific feedback ("doesn't mention metformin"), does the
   subsequent query incorporate the missing element?

Failure modes these expose:
- Generator loops on identical wording (prompt needs "produce a different
  query each time" emphasis or temperature bump)
- Generator drifts farther off goal each retry (feedback design issue)
- Generator ignores the feedback string entirely (prompt thread broken)
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from pydantic_graph import GraphRunContext

from andamentum.core.models import resolve_model
from andamentum.deep_research.nodes import (
    GenerateOne,
    NodeDeps,
    Verify,
)
from andamentum.deep_research.state import ResearchState

load_dotenv()
pytestmark = pytest.mark.cloud


CLOUD_MODEL = "openai:gpt-5.4-nano"


class _NoBackend:
    """Search/fetch backend stub — never called in this test."""

    async def search(self, query: str, max_results: int = 10):
        raise AssertionError("search must not be called in diversity tests")

    async def fetch_page(self, url: str):
        raise AssertionError("fetch_page must not be called")


def _make_ctx(initial_query: str, target_count: int = 1):
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    state = ResearchState(query=initial_query)
    state.cycle.mode = "initial"
    state.cycle.target_count = target_count
    deps = NodeDeps(
        backend=_NoBackend(),  # type: ignore[arg-type]
        model=resolve_model(CLOUD_MODEL),
        correlation_id="diversity-test",
    )
    return GraphRunContext(state=state, deps=deps)


def _bigram_set(s: str) -> set[str]:
    tokens = [t.lower() for t in s.split() if t.strip()]
    return {f"{a} {b}" for a, b in zip(tokens, tokens[1:])}


def _max_bigram_overlap(queries: list[str]) -> float:
    """Highest pairwise bigram-overlap fraction across all query pairs."""
    bigram_sets = [_bigram_set(q) for q in queries]
    worst = 0.0
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            a, b = bigram_sets[i], bigram_sets[j]
            if not a or not b:
                continue
            overlap = len(a & b) / max(len(a), len(b))
            worst = max(worst, overlap)
    return worst


async def test_generator_produces_distinct_queries_under_generic_rejection():
    """10 attempts simulating verifier rejection in the production loop.

    Mirrors what ``Verify`` does on rejection: appends the query to
    ``slot_rejected_queries`` and feeds back to ``GenerateOne``. This is
    the actual production code path — calling ``GenerateOne`` with only
    ``feedback`` (without ``slot_rejected_queries``) would mis-test the
    real loop behaviour.
    """
    ctx = _make_ctx(
        "What is the typical elimination half-life of metformin in healthy adults?"
    )
    captured: list[str] = []
    feedback: str | None = None

    for _ in range(10):
        node = GenerateOne(feedback=feedback)
        nxt = await node.run(ctx)
        assert isinstance(nxt, Verify)
        captured.append(nxt.query)
        # Mirror what Verify does on rejection.
        ctx.state.cycle.slot_rejected_queries.append(nxt.query)
        feedback = "not specific enough — try a different angle"

    distinct = set(captured)
    print(
        f"\n=== captured queries ({len(captured)} attempts, "
        f"{len(distinct)} distinct) ==="
    )
    for i, q in enumerate(captured, 1):
        print(f"  {i:2d}. {q}")

    assert len(distinct) >= 6, (
        f"Generator looped — only {len(distinct)} distinct queries in "
        f"{len(captured)} attempts: {distinct}"
    )


async def test_generator_does_not_cluster_into_paraphrases():
    """Even when distinct, queries shouldn't be near-bigram-duplicates."""
    ctx = _make_ctx("What causes muscle weakness in patients on statin therapy?")
    captured: list[str] = []
    feedback: str | None = None

    for _ in range(8):
        node = GenerateOne(feedback=feedback)
        nxt = await node.run(ctx)
        assert isinstance(nxt, Verify)
        captured.append(nxt.query)
        # Mirror Verify's rejection path so the generator sees prior attempts.
        ctx.state.cycle.slot_rejected_queries.append(nxt.query)
        feedback = "explore a different aspect"

    overlap = _max_bigram_overlap(captured)
    print(f"\n=== max pairwise bigram overlap: {overlap:.0%} ===")
    for i, q in enumerate(captured, 1):
        print(f"  {i:2d}. {q}")

    # Threshold tuned at 75%: queries on the same topic *should* share a
    # core backbone (e.g. "statin myopathy") while exploring different
    # angles. Two queries sharing >75% of bigrams are near-paraphrases;
    # 60-75% is normal topic overlap with angle differentiation.
    assert overlap < 0.75, (
        f"Generator clustering: max bigram overlap {overlap:.0%} "
        f"between two of {len(captured)} queries"
    )


async def test_generator_responds_to_specific_feedback():
    """When feedback names a missing element, next query should include it."""
    ctx = _make_ctx("What is metformin's pharmacokinetic profile?")

    # First attempt: no feedback.
    n1 = await GenerateOne().run(ctx)
    assert isinstance(n1, Verify)
    q1 = n1.query

    # Reject with explicit feedback naming a specific missing constraint.
    n2 = await GenerateOne(
        feedback="missing 'healthy adults' qualifier — the goal is about "
        "the healthy-adult population specifically"
    ).run(ctx)
    assert isinstance(n2, Verify)
    q2 = n2.query

    print("\n=== feedback responsiveness ===")
    print(f"  q1 (no feedback): {q1}")
    print(f"  q2 (feedback: needs 'healthy adults'): {q2}")

    # The corrected query must mention the missing constraint in some form.
    q2_lower = q2.lower()
    assert any(token in q2_lower for token in ("healthy", "adult", "volunteer")), (
        f"Generator did not address feedback about 'healthy adults' — "
        f"second query was: {q2!r}"
    )


async def test_generator_avoids_validated_queries():
    """When validated_queries already has items, next query should differ."""
    ctx = _make_ctx(
        "What is the typical elimination half-life of metformin in healthy adults?",
        target_count=5,
    )
    # Pre-seed validated_queries to simulate slot 1 + 2 having succeeded.
    ctx.state.cycle.validated_queries = [
        "metformin elimination half-life",
        "metformin pharmacokinetics healthy adults",
    ]

    n = await GenerateOne().run(ctx)
    assert isinstance(n, Verify)
    print("\n=== avoidance test ===")
    print(f"  validated already: {ctx.state.cycle.validated_queries}")
    print(f"  next query:        {n.query}")

    # The next query shouldn't be a near-duplicate of either validated one.
    new_bg = _bigram_set(n.query)
    for prior in ctx.state.cycle.validated_queries:
        prior_bg = _bigram_set(prior)
        if not new_bg or not prior_bg:
            continue
        overlap = len(new_bg & prior_bg) / max(len(new_bg), len(prior_bg))
        assert overlap < 0.70, (
            f"New query {n.query!r} overlaps {overlap:.0%} bigrams "
            f"with already-validated {prior!r}"
        )
