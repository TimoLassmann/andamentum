"""Quick start examples for andamentum.deep_research.

All examples run without external services (no LLM, no SearXNG).
"""

import asyncio
from types import SimpleNamespace

from andamentum.deep_research import (
    ResearchState,
    CircuitBreaker,
    CircuitOpenError,
    verify_sources,
)
from andamentum.deep_research.novelty import check_novelty, NoveltyAssessment


def demo_source_verification():
    """Verify which cited sources were actually accessed during research."""
    print("=== Source Verification ===\n")

    cited = [
        "https://example.com/paper1",
        "https://example.com/paper2",
        "https://example.com/hallucinated",
    ]
    searched = {"https://example.com/paper1", "https://example.com/extra"}
    fetched = {"https://example.com/paper2"}

    result = verify_sources(cited, searched, fetched)

    print(f"Total cited:       {result['total_cited']}")
    print(f"Verified:          {result['verified_count']}")
    print(f"Verification rate: {result['verification_rate']:.0%}")
    print(f"Hallucinated:      {result['unverified']}")
    print(f"Accessed not cited:{result['accessed_not_cited']}")
    print()


def demo_circuit_breaker():
    """Show circuit breaker state transitions."""
    print("=== Circuit Breaker ===\n")

    cb = CircuitBreaker(name="demo_service", failure_threshold=3, recovery_timeout=5.0)

    print(f"Initial state: {cb.state.value}")
    print(f"Allow request: {cb.allow_request()}")

    # Simulate failures
    for i in range(3):
        cb.record_failure()
        print(f"After failure {i + 1}: state={cb.state.value}, allow={cb.allow_request()}")

    # Circuit is now open
    try:
        if not cb.allow_request():
            raise CircuitOpenError(cb.name)
    except CircuitOpenError as e:
        print(f"Caught: {e}")

    # Reset for demo
    cb.reset()
    print(f"After reset: state={cb.state.value}")
    print()


def demo_research_state():
    """Construct and inspect a ResearchState."""
    print("=== Research State ===\n")

    state = ResearchState(query="What is spaced repetition?", max_iterations=3)

    print(f"Query:           {state.query}")
    print(f"Max iterations:  {state.max_iterations}")
    print(f"Current phase:   {state.current_phase}")
    print(f"Is complete:     {state.is_complete}")
    print(f"Iteration count: {state.iteration_count}")
    print()


async def demo_novelty_checking():
    """Check novelty with mock research and assessment functions."""
    print("=== Novelty Checking ===\n")

    async def mock_research(**kwargs):
        return {
            "output": SimpleNamespace(
                evidence_summary="Smith et al. (2023) explored this exact claim in Nature.",
                key_findings=["Prior work by Smith (2023)", "Replicated by Jones (2024)"],
                sources=["https://example.com/smith2023", "https://example.com/jones2024"],
            )
        }

    async def mock_assess(claim, evidence_summary, key_findings, sources):
        return NoveltyAssessment(
            is_novel=False,
            confidence=0.9,
            assessment="This claim has been explored in at least two prior publications.",
            similar_works=[
                {
                    "title": "Smith et al. 2023",
                    "url": "https://example.com/smith2023",
                    "relevance": "direct",
                    "summary": "Addresses the same claim directly",
                },
            ],
        )

    report = await check_novelty("My novel research claim", mock_research, mock_assess)

    print(f"Claim:      {report.claim}")
    print(f"Novel:      {report.is_novel}")
    print(f"Confidence: {report.confidence:.0%}")
    print(f"Assessment: {report.assessment}")
    if report.similar_work:
        print(f"Similar:    {report.similar_work[0].title} ({report.similar_work[0].relevance.value})")
    print()


if __name__ == "__main__":
    demo_source_verification()
    demo_circuit_breaker()
    demo_research_state()
    asyncio.run(demo_novelty_checking())
    print("All demos completed successfully.")
