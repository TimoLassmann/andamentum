"""Tests for CheckSynthesisDemand — Phase 1 of the lazy-escalation plan.

Phase 1 ships the satisfaction check in **logging-only** mode: the
node computes a Demand and logs it, but always returns Synthesize
regardless of the Demand's value. The tests pin:

  1. The deterministic gates fire correctly without an LLM call.
  2. The node ALWAYS returns Synthesize (Phase 1 contract).
  3. The Demand is logged with the right shape so future audit can
     reconstruct what the system thought.

Phase 4 will activate loop-back; tests for that go elsewhere.
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

    assert isinstance(next_node, Synthesize)  # Phase 1 always continues
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert len(msgs) == 1
    assert "needs_more=True" in msgs[0]
    assert "no combined verdict" in msgs[0].lower() or "abandoned" in msgs[0].lower()


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

    assert isinstance(next_node, Synthesize)
    msgs = [r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert len(msgs) == 1
    assert "needs_more=True" in msgs[0]
    # Justification should specifically mention the stranded claim count.
    assert "1" in msgs[0] and "without an integration verdict" in msgs[0]


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


# ── Phase 1 always returns Synthesize ────────────────────────────────


async def test_always_returns_synthesize_phase_1(
    tmp_path: Path, fake_runner
) -> None:
    """Phase 1 contract: the node ALWAYS returns Synthesize, regardless
    of whether the demand says needs_more or not. Phase 4 will activate
    the loop-back; in Phase 1 we only LOG.

    This test pins the contract so a future PR that prematurely
    activates loop-back gets caught before merge."""
    store = DocumentStore.for_database("phase_1_contract", db_dir=tmp_path)
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
                posterior=0.5,  # ambiguous — would trigger LLM
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

    # No matter what the agent returned (fake_runner gives a
    # configurable demand), Phase 1 always continues to Synthesize.
    assert isinstance(next_node, Synthesize), (
        "Phase 1 of the lazy-escalation plan is logging-only — the node "
        "MUST always return Synthesize. If this test fails, someone has "
        "prematurely activated Phase 4's loop-back without a separate "
        "design review."
    )
