"""Tests for CheckSynthesisDemand — deterministic gates.

These tests exercise the cheap deterministic gates of the satisfaction
check that fire without an LLM call. Phase 4's loop-back behavior is
tested separately in ``test_phase4_synthesis_loop_back.py``; here we
only pin:

  1. The deterministic gates produce the correct Demand shape.
  2. Their justification text is structured so future audits can
     reconstruct what the system thought.

BEHAVIOUR CHANGE (2026-08-09, the verdict-object collapse): the gates
read ``state.verdict`` — the run's single ``EpistemicVerdict`` — instead
of ``objective.decomposition.combined_verdict``. The fixtures therefore
seed that state by running the real predecessor, ``CombineClaimVerdicts``,
over a claim table that actually implies the numbers under test. The
previous fixtures hand-wrote a combined verdict that the claim table did
not support, so they could assert gate behaviour on states the pipeline
could never produce.
"""

from __future__ import annotations

import logging
from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.decomposition import (
    Decomposition,
    SubInvestigation,
)
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    CheckSynthesisDemand,
    CombineClaimVerdicts,
    Scrutinize,
    Synthesize,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


class _FlippingRunner:
    """Paraphrases a claim, then re-judges its evidence to a flipped verdict —
    simulating a confident-but-non-reproducible judgment for the tripwire."""

    async def run(self, agent_name: str, **kwargs):
        if agent_name == "epistemic_paraphrase_claim":
            return type("P", (), {"paraphrase": "reworded claim"})()
        if agent_name == "epistemic_judge_evidence":
            return type("J", (), {"verdict": "contradicts"})()
        raise AssertionError(f"unexpected agent {agent_name}")


async def _seed_verdict(state: EpistemicGraphState, repo: EpistemicRepository) -> None:
    """Run the real sole predecessor so ``state.verdict`` is present.

    CheckSynthesisDemand is only ever reached via CheckCompletion, which
    is only ever reached from CombineClaimVerdicts. Driving that node
    here means the gates are tested against a verdict the pipeline could
    actually produce.
    """
    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await CombineClaimVerdicts().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]


def _decomposition(*sub_ids: str) -> Decomposition:
    return Decomposition(
        sub_investigations=[
            SubInvestigation(id=sid, seed_claim=sid.lower(), rationale="r")
            for sid in sub_ids
        ],
        combination_rule="AND",
    )


# ── Deterministic gate: open-research mode (no combination rule) ─────


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
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Always returns Synthesize in Phase 1 (logging-only mode).
    assert isinstance(next_node, Synthesize)
    # Demand was logged with needs_more=False (satisfied).
    demand_logs = [r for r in caplog.records if "[synthesis_demand]" in r.getMessage()]
    assert len(demand_logs) == 1
    assert "needs_more=False" in demand_logs[0].getMessage()


# ── Deterministic gate: nothing adjudicated ──────────────────────────


async def test_no_combined_verdict_gates_to_needs_more(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When a combination rule is set but nothing was adjudicated
    (every claim abandoned/cycle-capped/no-verdict), the headline
    is the no-data fallback. Should log needs_more=True via
    deterministic gate."""
    store = DocumentStore.for_database("no_cv", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="x", decomposition=_decomposition("A"))
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    # No claims at all → n_adjudicated == 0.

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # No claims saved → no eligible claims → routes to the
    # structurally-explicit insufficient terminal (Maximal B). The
    # graph refuses to invent a directional verdict from no-data
    # state — Peirce's fallibilism encoded in the topology.
    assert isinstance(next_node, SynthesizeInsufficient)
    msgs = [
        r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()
    ]
    # Find the gate's demand log (the first one), separate from any
    # loop-back safety log.
    gate_msgs = [
        m
        for m in msgs
        if "no combined verdict" in m.lower() or "abandoned" in m.lower()
    ]
    assert gate_msgs, (
        f"Expected gate to log a 'no combined verdict' demand. Got: {msgs}"
    )
    assert "needs_more=True" in gate_msgs[0]


# ── Deterministic gate: stranded claims (n_no_verdict > 0) ───────────


async def test_stranded_claims_gates_to_needs_more(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When the verdict reports n_no_verdict > 0, IBE was bypassed for
    some claims — defensive deterministic check should log
    needs_more=True.

    One adjudicated claim (so Gate 2 doesn't fire first) plus one
    stranded claim. Both are cycle-capped, so no loop-back is possible
    and the routing assertion below stays the same as before.
    """
    store = DocumentStore.for_database("stranded", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="x", decomposition=_decomposition("A", "B"))
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    adjudicated = Claim(
        objective_id=obj.entity_id,
        statement="claim A",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        sub_investigation_id="A",
        cycle_capped=True,
        integrated_assessment="insufficient",
        integrated_confidence=0.5,
    )
    stranded = Claim(
        objective_id=obj.entity_id,
        statement="claim B",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        sub_investigation_id="B",
        cycle_capped=True,
        integrated_assessment=None,
    )
    await repo.save(adjudicated)
    await repo.save(stranded)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # Stranded-claim demands are needs_more, and with no claim able to
    # make progress the graph suspends judgment rather than fabricating
    # a verdict.
    assert isinstance(next_node, SynthesizeInsufficient)
    msgs = [
        r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()
    ]
    gate_msgs = [m for m in msgs if "without an integration verdict" in m]
    assert gate_msgs, f"Expected stranded-claims gate log. Got: {msgs}"
    assert "needs_more=True" in gate_msgs[0]
    # Justification should specifically mention the stranded claim count.
    assert "1" in gate_msgs[0]


# ── Deterministic gate: decisive posterior ───────────────────────────


async def _decisive_setup(
    tmp_path: Path,
    db_name: str,
    *,
    assessment: str,
    confidence: float,
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="x", decomposition=_decomposition("A"))
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    claim = Claim(
        objective_id=obj.entity_id,
        statement="claim A",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        sub_investigation_id="A",
        integrated_assessment=assessment,
        integrated_confidence=confidence,
    )
    await repo.save(claim)
    return obj, repo


async def test_decisive_supports_posterior_gates_to_satisfied(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """When posterior >= 0.85, the verdict direction is decisive.
    Deterministic gate skips the LLM."""
    obj, repo = await _decisive_setup(
        tmp_path, "decisive_supp", assessment="supports", confidence=0.92
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [
        r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()
    ]
    assert len(msgs) == 1
    assert "needs_more=False" in msgs[0]
    assert "supports" in msgs[0].lower() and "decisive" in msgs[0].lower()


async def test_decisive_contradicts_posterior_gates_to_satisfied(
    tmp_path: Path, fake_runner, caplog
) -> None:
    """Mirror: posterior <= 0.15 is decisive contradicts."""
    obj, repo = await _decisive_setup(
        tmp_path, "decisive_con", assessment="contradicts", confidence=0.80
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [
        r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()
    ]
    assert "needs_more=False" in msgs[0]
    assert "contradicts" in msgs[0].lower()


# ── Reproducibility tripwire on the decisive gate (item 2a) ──────────


async def _repro_setup(
    tmp_path: Path, db_name: str
) -> tuple[Objective, EpistemicRepository]:
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(description="x", decomposition=_decomposition("A"))
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    ev = Evidence(
        entity_id="e1",
        objective_id=obj.entity_id,
        extracted=True,
        extracted_content="A confident finding",
        support_judgment="supports",
        judgment_distribution=[1.0, 0.0, 0.0],  # one-hot → entropy-blind
    )
    await repo.save(ev)
    claim = Claim(
        entity_id="c1",
        objective_id=obj.entity_id,
        statement="X causes Y",
        stage=ClaimStage.SUPPORTED,
        sub_investigation_id="A",
        evidence_ids=["e1"],
        integrated_assessment="supports",
        integrated_confidence=0.92,
    )
    await repo.save(claim)
    return obj, repo


async def test_decisive_posterior_reproducibility_flip_loops_back(
    tmp_path: Path, caplog
) -> None:
    """A decisive posterior whose supporting judgment is one-hot but flips
    under paraphrase must NOT finalize — it loops back to Scrutinize."""
    obj, repo = await _repro_setup(tmp_path, "repro_flip")

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=_FlippingRunner(), embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # The flip converts the decisive-satisfied shortcut into needs_more, and the
    # eligible claim loops back for another round of scrutiny.
    assert isinstance(next_node, Scrutinize)
    assert "c1" in state.claims_needing_rescrutiny


async def test_decisive_posterior_reproducible_judgment_finalizes(
    tmp_path: Path,
) -> None:
    """A one-hot judgment that HOLDS under paraphrase does not block synthesis."""

    class _HoldingRunner:
        async def run(self, agent_name: str, **kwargs):
            if agent_name == "epistemic_paraphrase_claim":
                return type("P", (), {"paraphrase": "reworded"})()
            if agent_name == "epistemic_judge_evidence":
                return type("J", (), {"verdict": "supports"})()
            raise AssertionError(f"unexpected agent {agent_name}")

    obj, repo = await _repro_setup(tmp_path, "repro_hold")

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=_HoldingRunner(), embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, Synthesize)


# ── No agent runner ──────────────────────────────────────────────────


async def test_no_agent_runner_falls_through_to_satisfied(
    tmp_path: Path, caplog
) -> None:
    """When deps.agent_runner is None and deterministic gates didn't
    decide, default to satisfied (don't block synthesis on missing
    LLM). Justification names the cause."""
    obj, repo = await _decisive_setup(
        tmp_path, "no_runner", assessment="insufficient", confidence=0.5
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    with caplog.at_level(logging.INFO, logger="andamentum.epistemic.graph.nodes"):
        next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Synthesize)
    msgs = [
        r.getMessage() for r in caplog.records if "[synthesis_demand]" in r.getMessage()
    ]
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
    obj, repo = await _decisive_setup(
        tmp_path, "phase_4_default", assessment="insufficient", confidence=0.5
    )

    state = EpistemicGraphState(objective_id=obj.entity_id)
    await _seed_verdict(state, repo)
    deps = EpistemicDeps(repo=repo, agent_runner=fake_runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, Synthesize)
