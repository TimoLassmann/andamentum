"""Tests for Phase 4 of the lazy-escalation plan: synthesis demand
activates loop-back to Scrutinize.

Phase 1 shipped CheckSynthesisDemand in logging-only mode. Phase 4
activates the loop-back: when the satisfaction LLM (or a deterministic
gate) returns ``needs_more=True`` AND there are eligible claims that
could benefit from more investigation, the node adds them to
``state.claims_needing_rescrutiny`` and routes to Scrutinize.

The load-bearing safety is the per-claim cap, not a global budget:
when all non-abandoned claims have hit ``SCRUTINY_RESOLVE_CYCLE_CAP``
or are cycle-capped, no claims are added to rescrutiny and the node
routes to ``SynthesizeInsufficient`` (Maximal B of the K3 fix). The
fallibilism terminal — distinct from a directional verdict — is the
correct outcome when more work is needed but no work can be done.

These tests pin:
  1. needs_more + eligible claims → routes to Scrutinize, adds claims
     to rescrutiny.
  2. needs_more + all claims at per-claim cap → routes to
     SynthesizeInsufficient (cap-driven fallibilism).
  3. needs_more + all claims cycle-capped → routes to
     SynthesizeInsufficient (terminal-state fallibilism).
  4. needs_more + all claims abandoned → routes to
     SynthesizeInsufficient.
  5. The deterministic-needs_more gate (n_no_verdict > 0) also
     triggers loop-back when eligible claims exist.
"""

from __future__ import annotations

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
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    CheckSynthesisDemand,
    Scrutinize,
    SynthesizeInsufficient,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.tests.conftest import FakeAgentRunner


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def _setup_unsatisfying_run(
    tmp_path: Path,
    n_claims: int = 2,
    cycle_capped: list[int] | None = None,
    at_cap: list[int] | None = None,
    abandoned: list[int] | None = None,
) -> tuple[Objective, list[Claim], EpistemicGraphState, EpistemicRepository]:
    """Build a run state where the synthesis demand will fire (because
    posterior is in the ambiguous middle and the LLM-judgment path
    will run). Each claim's terminal state can be configured via
    ``cycle_capped``, ``at_cap``, ``abandoned`` index lists."""
    cycle_capped = cycle_capped or []
    at_cap = at_cap or []
    abandoned = abandoned or []

    store = DocumentStore.for_database("phase4_loopback", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    obj = Objective(
        description="x",
        decomposition=Decomposition(
            sub_investigations=[
                SubInvestigation(id=chr(ord("A") + i), seed_claim="a", rationale="r")
                for i in range(n_claims)
            ],
            combination_rule="AND",
            combined_verdict=CombinedVerdictData(
                posterior=0.5,  # ambiguous — triggers LLM
                verdict="insufficient",
                combination_rule="AND",
                claim_posteriors=[0.5] * n_claims,
                n_no_verdict=0,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    claims: list[Claim] = []
    for i in range(n_claims):
        c = Claim(
            objective_id=obj.entity_id,
            statement=f"claim {chr(ord('A') + i)}",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="insufficient",
            integrated_confidence=0.5,
            sub_investigation_id=chr(ord("A") + i),
            cycle_capped=(i in cycle_capped),
            abandoned=(i in abandoned),
        )
        await repo.save(c)
        claims.append(c)
        if i in at_cap:
            state.scrutiny_resolve_cycles[c.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP

    return obj, claims, state, repo


# ── needs_more + eligible claims → loop back ─────────────────────────


async def test_needs_more_with_eligible_claims_loops_back_to_scrutinize(
    tmp_path: Path,
) -> None:
    """The defining behavior of Phase 4: when the satisfaction LLM
    says needs_more AND at least one claim is eligible for more
    investigation, route to Scrutinize and add the claim to
    rescrutiny."""
    _obj, claims, state, repo = await _setup_unsatisfying_run(tmp_path, n_claims=2)
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Need more evidence on mortality outcomes.",
                "target_hint": "RCT registries",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    # Both eligible claims added to rescrutiny.
    for c in claims:
        assert c.entity_id in state.claims_needing_rescrutiny


# ── all claims at per-claim cap → route to SynthesizeInsufficient ────


async def test_needs_more_with_all_claims_at_cap_routes_insufficient(
    tmp_path: Path,
) -> None:
    """When all non-abandoned, non-cycle-capped claims have hit
    ``SCRUTINY_RESOLVE_CYCLE_CAP``, no claims can make progress. With
    Maximal B, the node routes to ``SynthesizeInsufficient`` rather
    than falling through to ``Synthesize`` — the system suspends
    judgment instead of asking the writer to invent a verdict.
    Carries the gate's justification onto state for the deterministic
    body to surface."""
    _obj, _claims, state, repo = await _setup_unsatisfying_run(
        tmp_path, n_claims=2, at_cap=[0, 1]
    )
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Need more evidence (but caps fired).",
                "target_hint": "",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient), (
        "When all claims have hit per-claim cap, the structural "
        "outcome is fallibilism (SynthesizeInsufficient), not a "
        "directional verdict. The cap-driven termination still "
        "holds — it just routes to the correct terminal."
    )
    # No claims should have been added to rescrutiny — they're all
    # at cap.
    assert state.claims_needing_rescrutiny == set()
    # The gate's justification was carried onto state for the
    # deterministic body to surface.
    assert state.synthesis_insufficient_reason is not None
    assert "caps fired" in state.synthesis_insufficient_reason


async def test_needs_more_with_all_claims_cycle_capped_routes_insufficient(
    tmp_path: Path,
) -> None:
    """Cycle-capped claims are also terminal — same fallibilism path
    as the per-claim cap, but a different state field on the claim."""
    _obj, _claims, state, repo = await _setup_unsatisfying_run(
        tmp_path, n_claims=2, cycle_capped=[0, 1]
    )
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Need more evidence (but cycle-capped).",
                "target_hint": "",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.claims_needing_rescrutiny == set()


async def test_needs_more_with_all_claims_abandoned_routes_insufficient(
    tmp_path: Path,
) -> None:
    """Abandoned claims are filtered out by ``active_claims`` upstream;
    if all claims are abandoned, the eligible list is empty and the
    node routes to SynthesizeInsufficient — the structurally correct
    "no-data" terminal."""
    # We can't use _setup_unsatisfying_run directly because it builds
    # the objective with a combined_verdict assuming claims contribute.
    # Build a minimal abandoned-only state inline.
    store = DocumentStore.for_database("phase4_abandoned", db_dir=tmp_path)
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
                claim_posteriors=[None],
                n_no_verdict=0,
                n_abandoned=1,
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    c = Claim(
        objective_id=obj.entity_id,
        statement="abandoned claim",
        scope="x",
        stage=ClaimStage.HYPOTHESIS,
        sub_investigation_id="A",
        abandoned=True,
    )
    await repo.save(c)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Want more, but everything's abandoned.",
                "target_hint": "",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, SynthesizeInsufficient)


# ── Mixed eligibility: some claims at cap, some not ──────────────────


async def test_partial_eligibility_loops_back_only_eligible_claims(
    tmp_path: Path,
) -> None:
    """When some claims are eligible and some are at cap, only the
    eligible ones get added to rescrutiny. The loop fires for the
    subset, not for terminal claims."""
    _obj, claims, state, repo = await _setup_unsatisfying_run(
        tmp_path, n_claims=3, at_cap=[0], cycle_capped=[1]
    )
    runner = FakeAgentRunner(
        overrides={
            "epistemic_check_synthesis_demand": {
                "needs_more": True,
                "justification": "Mixed eligibility.",
                "target_hint": "",
            }
        }
    )
    deps = EpistemicDeps(repo=repo, agent_runner=runner, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    # At least one claim is eligible (claim C, index 2), so we loop back.
    assert isinstance(next_node, Scrutinize)
    # Only the eligible one was added.
    assert claims[2].entity_id in state.claims_needing_rescrutiny
    assert claims[0].entity_id not in state.claims_needing_rescrutiny  # at cap
    assert claims[1].entity_id not in state.claims_needing_rescrutiny  # cycle-capped


# ── Deterministic-gate path: stranded claims gate triggers loop-back ─


async def test_stranded_claims_gate_loops_back_when_eligible(
    tmp_path: Path,
) -> None:
    """The deterministic ``n_no_verdict > 0`` gate produces
    ``needs_more=True`` without an LLM call. Phase 4's loop-back
    should fire here too, not just on LLM-emitted demands."""
    store = DocumentStore.for_database("phase4_stranded", db_dir=tmp_path)
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
                claim_posteriors=[None],
                n_no_verdict=1,  # stranded! deterministic gate fires
            ),
        ),
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    # An eligible claim so loop-back has somewhere to go.
    c = Claim(
        objective_id=obj.entity_id,
        statement="eligible claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        integrated_assessment=None,  # stranded
        sub_investigation_id="A",
    )
    await repo.save(c)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(next_node, Scrutinize)
    assert c.entity_id in state.claims_needing_rescrutiny
