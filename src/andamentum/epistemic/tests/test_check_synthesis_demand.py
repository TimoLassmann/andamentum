"""Tests for CheckSynthesisDemand — deterministic gates.

These tests exercise the cheap deterministic gates of the satisfaction
check that fire without an LLM call. Phase 4's loop-back behavior is
tested separately in ``test_phase4_synthesis_loop_back.py``; here we
only pin:

  1. The deterministic gates produce the correct Demand shape.
  2. Their justification text is structured so future audits can
     reconstruct what the system thought.

The tests below have no claims saved so the loop-back falls through
to Synthesize; they don't probe the Phase 4 routing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.decomposition import (
    CombinedVerdictData,
    Decomposition,
    SubInvestigation,
)
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import CheckSynthesisDemand, Synthesize
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


# ── Deterministic gate: open-research mode (no decomposition) ────────


async def test_no_decomposition_gates_to_satisfied(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When the objective has no decomposition (open-research mode),
    the synthesis demand check doesn't apply. Should return satisfied
    via deterministic gate (no LLM call) and continue to Synthesize."""
    store = DocumentStore.for_database("no_decomp", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="open research")
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # One claim so we'd reach this node in production
    claim = Claim(
        objective_id=obj.entity_id,
        statement="some claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment="supports",
        integrated_confidence=0.7,
    )
    await repo.save(claim)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Always returns Synthesize in Phase 1 (logging-only mode).
    assert isinstance(next_node, Synthesize)
    # Demand was logged with needs_more=False (satisfied).
    demand_logs = [r for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert len(demand_logs) == 1
    assert "needs_more=False" in demand_logs[0].getMessage()


# ── Deterministic gate: no combined_verdict ──────────────────────────


async def test_no_combined_verdict_gates_to_needs_more(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When decomposition is set but no combined_verdict was produced
    (every claim was abandoned/cycle-capped/no-verdict), the headline
    is the no-data fallback. Should log needs_more=True via
    deterministic gate."""
    store = DocumentStore.for_database("no_cv", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
            # NO combined_verdict (None)
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # No claims saved → no eligible claims → loop-back falls through
    # to Synthesize with the "synthesizing anyway" safety log.
    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    # Find the gate's demand log (the first one), separate from any
    # loop-back safety log.
    gate_msgs = [m for m in msgs if "no combined verdict" in m.lower() or "abandoned" in m.lower()]
    assert gate_msgs, f"Expected gate to log a 'no combined verdict' demand. Got: {msgs}"
    assert "needs_more=True" in gate_msgs[0]


# ── Deterministic gate: stranded claims (n_no_verdict > 0) ───────────


async def test_stranded_claims_gates_to_needs_more(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When the combined verdict reports n_no_verdict > 0, IBE was
    bypassed for some claims — defensive deterministic check should
    log needs_more=True."""
    store = DocumentStore.for_database("stranded", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.5,
                verdict="insufficient",
                combination_rule="AND",
                claim_posteriors=[0.5, None],
                n_no_verdict=1,  # stranded!
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # No claims → loop-back falls through to Synthesize.
    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    gate_msgs = [m for m in msgs if "without an integration verdict" in m]
    assert gate_msgs, f"Expected stranded-claims gate log. Got: {msgs}"
    assert "needs_more=True" in gate_msgs[0]
    # Justification should specifically mention the stranded claim count.
    assert "1" in gate_msgs[0]


# ── Deterministic gate: decisive posterior ───────────────────────────


async def test_decisive_supports_posterior_gates_to_satisfied(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When posterior >= 0.85, the verdict direction is decisive.
    Deterministic gate skips the LLM."""
    store = DocumentStore.for_database("decisive_supp", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r"),
                SubInvestigation(id="B", seed_claim="b", rationale="r"),
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.92,
                verdict="supports",
                combination_rule="AND",
                claim_posteriors=[0.92, 0.95],
                n_no_verdict=0,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert len(msgs) == 1
    assert "needs_more=False" in msgs[0]
    assert "supports" in msgs[0].lower() and "decisive" in msgs[0].lower()


async def test_decisive_contradicts_posterior_gates_to_satisfied(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """Mirror: posterior <= 0.15 is decisive contradicts."""
    store = DocumentStore.for_database("decisive_con", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.10,
                verdict="contradicts",
                combination_rule="AND",
                claim_posteriors=[0.10],
                n_no_verdict=0,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert "needs_more=False" in msgs[0]
    assert "contradicts" in msgs[0].lower()


# ── No agent runner ──────────────────────────────────────────────────


async def test_no_agent_runner_falls_through_to_satisfied(
    tmp_path: Path, caplog
) -> None:
    """When deps.agent_runner is None and deterministic gates didn't
    decide, default to satisfied (don't block synthesis on missing
    LLM). Justification names the cause."""
    store = DocumentStore.for_database("no_runner", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.5,
                verdict="insufficient",
                combination_rule="AND",
                claim_posteriors=[0.5],
                n_no_verdict=0,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert "needs_more=False" in msgs[0]
    assert "no agent runner" in msgs[0].lower()


# ── Phase 4: satisfaction-default returns Synthesize ─────────────────


async def test_default_satisfied_path_returns_synthesize(
    tmp_path: Path, fake_runner
) -> None:
    """When the satisfaction LLM returns ``needs_more=False`` (the
    fake_runner default), the node continues to Synthesize. This is
    the common case: the deterministic gates passed and the LLM
    confirmed the verdict is good enough."""
    store = DocumentStore.for_database("phase_4_default", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id="A", seed_claim="a", rationale="r")
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.5,  # ambiguous — triggers LLM judgment
                verdict="insufficient",
                combination_rule="AND",
                claim_posteriors=[0.5],
                n_no_verdict=0,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(
        repo=repo, agent_runner=fake_runner, embedding_model="t"
    )

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, Synthesize)
