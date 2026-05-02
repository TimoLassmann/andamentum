"""Surface 1 — plumbing tests for the per-slot generate→verify→search loop.

No LLM, no network. Stubs both agents (``query_generator``,
``topic_verifier``) and the search backend, then hand-walks the graph
nodes from ``PrepareSearchCycle`` through ``ParallelSearch`` to verify:

- happy path (3 queries pass first attempt)
- single retry within a slot (verifier feedback path)
- slot retry exhaustion → skip-and-tighten
- total collapse (all rejected → target_count drops to 0)
- gap mode initial state (target_count=2, gaps populated)

Each test runs in milliseconds and exercises one transition in isolation.
"""

from __future__ import annotations

import pytest
from pydantic_graph import GraphRunContext

from andamentum.deep_research.models import (
    GeneratorOutput,
    SearchResult,
    VerifierOutput,
)
from andamentum.deep_research.nodes import (
    GenerateOne,
    NodeDeps,
    ParallelSearch,
    PrepareSearchCycle,
    Verify,
)
from andamentum.deep_research.state import ResearchState


# ── Stubs ──────────────────────────────────────────────────────────────


class _StubResult:
    """Mimics the ``RunResult`` shape pydantic-ai's Agent.run returns."""

    def __init__(self, output):
        self.output = output


class StubAgent:
    """Drop-in replacement for a pydantic-ai Agent in tests.

    Returns scripted outputs in order; raises if exhausted so test failures
    surface obviously instead of falling through with stale data.
    """

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls: list[str] = []

    async def run(self, prompt, **kwargs):
        if not self._outputs:
            raise AssertionError(
                f"StubAgent exhausted; received call: {prompt[:80]!r}"
            )
        self.calls.append(prompt)
        return _StubResult(self._outputs.pop(0))


class StubBackend:
    """Minimal SearchBackend — returns one fake SearchResult per query."""

    def __init__(self):
        self.searched: list[str] = []

    async def search(self, query: str, max_results: int = 10):
        self.searched.append(query)
        return [
            SearchResult(
                link_id=0,
                title=f"Result for {query}",
                url=f"https://example.test/{len(self.searched)}",
                snippet="...",
                domain="example.test",
                relevance_score=0.5,
            )
        ]

    async def fetch_page(self, url: str):
        raise AssertionError(
            "fetch_page should not be called in plumbing tests"
        )


def _make_ctx(
    *,
    generator_outputs: list[GeneratorOutput],
    verifier_outputs: list[VerifierOutput],
    initial_query: str = "What is metformin's elimination half-life in healthy adults?",
    iteration_count: int = 0,
    identified_gaps: list[str] | None = None,
):
    state = ResearchState(query=initial_query)
    state.iteration_count = iteration_count
    if identified_gaps:
        state.identified_gaps = identified_gaps
    deps = NodeDeps(
        backend=StubBackend(),
        model="stub",
        correlation_id="test",
        agent_overrides={
            "query_generator": StubAgent(generator_outputs),
            "topic_verifier": StubAgent(verifier_outputs),
        },
    )
    return GraphRunContext(state=state, deps=deps)


async def _walk_cycle(ctx) -> ResearchState:
    """Drive the graph from PrepareSearchCycle until ParallelSearch returns.

    Returns the final state. Asserts we exit through ParallelSearch (or
    End for the max-iterations case — which the plumbing tests don't
    exercise; they always start with iteration_count==0).
    """
    node = PrepareSearchCycle()
    # PrepareSearchCycle → GenerateOne. Verify it.
    nxt = await node.run(ctx)
    assert isinstance(nxt, GenerateOne)
    node = nxt

    # Bound the inner loop so a buggy node graph can't hang the test.
    for _ in range(50):
        nxt = await node.run(ctx)
        if isinstance(nxt, ParallelSearch):
            await nxt.run(ctx)
            return ctx.state
        node = nxt

    raise AssertionError("Loop did not terminate within 50 hops")


# ── Tests ──────────────────────────────────────────────────────────────


async def test_initial_mode_target_count_three_and_gaps_empty():
    """PrepareSearchCycle in initial mode: target=3, gaps=[]."""
    ctx = _make_ctx(generator_outputs=[], verifier_outputs=[])
    await PrepareSearchCycle().run(ctx)
    c = ctx.state.cycle
    assert c.mode == "initial"
    assert c.target_count == 3
    assert c.gaps == []
    assert c.validated_queries == []
    assert c.slot_attempts == 0
    assert ctx.state.iteration_count == 1


async def test_gap_mode_target_count_two_and_gaps_populated():
    """PrepareSearchCycle in gap mode: target=2, gaps from state."""
    ctx = _make_ctx(
        generator_outputs=[],
        verifier_outputs=[],
        iteration_count=1,
        identified_gaps=["gap A", "gap B"],
    )
    await PrepareSearchCycle().run(ctx)
    c = ctx.state.cycle
    assert c.mode == "gap"
    assert c.target_count == 2
    assert c.gaps == ["gap A", "gap B"]
    assert ctx.state.iteration_count == 2


async def test_happy_path_three_queries_all_accepted():
    """All three slots filled on first attempt; ParallelSearch gets all 3."""
    ctx = _make_ctx(
        generator_outputs=[
            GeneratorOutput(query="q1", rationale="r1"),
            GeneratorOutput(query="q2", rationale="r2"),
            GeneratorOutput(query="q3", rationale="r3"),
        ],
        verifier_outputs=[
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=True, reason="ok"),
        ],
    )
    state = await _walk_cycle(ctx)
    assert state.cycle.validated_queries == ["q1", "q2", "q3"]
    assert state.cycle.slot_attempts == 0
    assert state.total_searches == 3
    backend = ctx.deps.backend
    assert isinstance(backend, StubBackend)
    assert backend.searched == ["q1", "q2", "q3"]


async def test_one_retry_on_slot_two_then_accept():
    """Slot 2 rejected once, second attempt accepted. slot_attempts resets."""
    ctx = _make_ctx(
        generator_outputs=[
            GeneratorOutput(query="q1", rationale="r1"),
            GeneratorOutput(query="q2-bad", rationale="r2a"),
            GeneratorOutput(query="q2-good", rationale="r2b"),
            GeneratorOutput(query="q3", rationale="r3"),
        ],
        verifier_outputs=[
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=False, reason="too vague"),
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=True, reason="ok"),
        ],
    )
    state = await _walk_cycle(ctx)
    assert state.cycle.validated_queries == ["q1", "q2-good", "q3"]
    assert state.cycle.slot_attempts == 0  # reset after q2-good was accepted


async def test_slot_exhaustion_triggers_skip_and_tighten():
    """Slot 2 rejected 3× → skip and tighten. target_count drops 3→2.

    After the skip, only one more slot is needed (already have q1; target
    is now 2). q3 gets accepted as that slot.
    """
    ctx = _make_ctx(
        generator_outputs=[
            GeneratorOutput(query="q1", rationale="r1"),
            GeneratorOutput(query="bad-a", rationale="ra"),
            GeneratorOutput(query="bad-b", rationale="rb"),
            GeneratorOutput(query="bad-c", rationale="rc"),
            GeneratorOutput(query="q3", rationale="r3"),
        ],
        verifier_outputs=[
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=False, reason="bad-1"),
            VerifierOutput(on_topic=False, reason="bad-2"),
            VerifierOutput(on_topic=False, reason="bad-3"),
            VerifierOutput(on_topic=True, reason="ok"),
        ],
    )
    state = await _walk_cycle(ctx)
    assert state.cycle.validated_queries == ["q1", "q3"]
    assert state.cycle.target_count == 2  # tightened from 3
    assert state.cycle.slot_attempts == 0  # reset after q3 accept


async def test_total_collapse_all_rejected_proceeds_with_empty():
    """Verifier rejects everything → target tightens to 0 → empty search."""
    ctx = _make_ctx(
        # 3 slots × 3 retries = 9 generator calls expected before tighten-to-0
        generator_outputs=[
            GeneratorOutput(query=f"q{i}", rationale="r") for i in range(9)
        ],
        verifier_outputs=[
            VerifierOutput(on_topic=False, reason="nope") for _ in range(9)
        ],
    )
    state = await _walk_cycle(ctx)
    assert state.cycle.validated_queries == []
    assert state.cycle.target_count == 0
    assert state.total_searches == 0
    # Backend was not asked to search anything.
    backend = ctx.deps.backend
    assert isinstance(backend, StubBackend)
    assert backend.searched == []


async def test_skip_then_tighten_to_zero_when_slot_one_exhausts():
    """If slot 1 itself exhausts before any acceptance, target drops by 1."""
    ctx = _make_ctx(
        generator_outputs=[
            GeneratorOutput(query="bad1", rationale="r"),
            GeneratorOutput(query="bad2", rationale="r"),
            GeneratorOutput(query="bad3", rationale="r"),
            GeneratorOutput(query="q1", rationale="r"),
            GeneratorOutput(query="q2", rationale="r"),
        ],
        verifier_outputs=[
            VerifierOutput(on_topic=False, reason="r1"),
            VerifierOutput(on_topic=False, reason="r2"),
            VerifierOutput(on_topic=False, reason="r3"),
            VerifierOutput(on_topic=True, reason="ok"),
            VerifierOutput(on_topic=True, reason="ok"),
        ],
    )
    state = await _walk_cycle(ctx)
    # First slot exhausted → target 3→2; then q1, q2 accepted.
    assert state.cycle.validated_queries == ["q1", "q2"]
    assert state.cycle.target_count == 2


async def test_feedback_threaded_to_generator_on_retry():
    """The verifier's reject-reason should reach GenerateOne via feedback."""
    captured_prompts: list[str] = []

    class CapturingStub(StubAgent):
        async def run(self, prompt, **kwargs):
            captured_prompts.append(prompt)
            return await super().run(prompt, **kwargs)

    state = ResearchState(query="metformin half-life")
    deps = NodeDeps(
        backend=StubBackend(),
        model="stub",
        correlation_id="test",
        agent_overrides={
            "query_generator": CapturingStub(
                [
                    GeneratorOutput(query="bad", rationale="r"),
                    GeneratorOutput(query="good", rationale="r"),
                ]
            ),
            "topic_verifier": StubAgent(
                [
                    VerifierOutput(
                        on_topic=False, reason="needs more keywords"
                    ),
                    VerifierOutput(on_topic=True, reason="ok"),
                ]
            ),
        },
    )
    ctx = GraphRunContext(state=state, deps=deps)
    # Force target_count to 1 so the cycle ends after one accept.
    state.cycle.target_count = 1
    state.cycle.mode = "initial"

    node = GenerateOne()
    nxt = await node.run(ctx)
    assert isinstance(nxt, Verify)
    nxt2 = await nxt.run(ctx)
    assert isinstance(nxt2, GenerateOne)
    assert nxt2.feedback == "needs more keywords"
    nxt3 = await nxt2.run(ctx)
    assert isinstance(nxt3, Verify)
    final = await nxt3.run(ctx)
    assert isinstance(final, ParallelSearch)

    # Two prompts captured for generator: first without feedback, second with.
    assert len(captured_prompts) == 2
    assert "feedback:" not in captured_prompts[0]
    assert "feedback: needs more keywords" in captured_prompts[1]


async def test_generator_prompt_includes_validated_queries_for_diversity():
    """After slot 1 accepts q1, the slot-2 generator prompt must include q1."""
    captured_prompts: list[str] = []

    class CapturingStub(StubAgent):
        async def run(self, prompt, **kwargs):
            captured_prompts.append(prompt)
            return await super().run(prompt, **kwargs)

    state = ResearchState(query="goal")
    state.cycle.target_count = 2
    state.cycle.mode = "initial"
    deps = NodeDeps(
        backend=StubBackend(),
        model="stub",
        correlation_id="test",
        agent_overrides={
            "query_generator": CapturingStub(
                [
                    GeneratorOutput(query="q1", rationale="r"),
                    GeneratorOutput(query="q2", rationale="r"),
                ]
            ),
            "topic_verifier": StubAgent(
                [
                    VerifierOutput(on_topic=True, reason="ok"),
                    VerifierOutput(on_topic=True, reason="ok"),
                ]
            ),
        },
    )
    ctx = GraphRunContext(state=state, deps=deps)

    node = GenerateOne()
    while True:
        nxt = await node.run(ctx)
        if isinstance(nxt, ParallelSearch):
            break
        node = nxt

    # Slot 2's prompt should mention validated_queries: q1
    assert "validated_queries: (none yet)" in captured_prompts[0]
    assert "validated_queries: q1" in captured_prompts[1]


@pytest.mark.parametrize("rejection_count", [1, 2])
async def test_slot_attempts_increments_on_rejection_below_budget(
    rejection_count: int,
):
    """slot_attempts counts up per rejection until budget exhausted."""
    ctx = _make_ctx(
        generator_outputs=[
            GeneratorOutput(query=f"bad{i}", rationale="r")
            for i in range(rejection_count)
        ]
        + [GeneratorOutput(query="good", rationale="r")],
        verifier_outputs=[
            VerifierOutput(on_topic=False, reason=f"reject {i}")
            for i in range(rejection_count)
        ]
        + [VerifierOutput(on_topic=True, reason="ok")],
    )
    state = ResearchState(query="goal")
    state.cycle.target_count = 1
    state.cycle.mode = "initial"
    ctx.state = state

    node = GenerateOne()
    while True:
        nxt = await node.run(ctx)
        if isinstance(nxt, ParallelSearch):
            break
        node = nxt

    # After accepting "good", slot_attempts is reset to 0.
    assert state.cycle.slot_attempts == 0
    assert state.cycle.validated_queries == ["good"]
