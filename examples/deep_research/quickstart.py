"""Quick start examples for andamentum.deep_research.

All examples run without external services (no LLM, no SearXNG).

Note: novelty checking (``run_novelty_check``) is intentionally not demoed
here — it drives a live search + LLM pipeline and so cannot run offline.
See the ``andamentum-research`` CLI / ``run_novelty_check`` docstring for
that path.
"""

from andamentum.deep_research import (
    ResearchState,
    CircuitBreaker,
    CircuitOpenError,
    verify_sources,
)


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
        print(
            f"After failure {i + 1}: state={cb.state.value}, allow={cb.allow_request()}"
        )

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


if __name__ == "__main__":
    demo_source_verification()
    demo_circuit_breaker()
    demo_research_state()
    print("All demos completed successfully.")
