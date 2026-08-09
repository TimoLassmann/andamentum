"""Gate 0 of ``CheckSynthesisDemand``: measured judgment doubt buys ONE more
round of inquiry.

The pipeline makes many sampled LLM judgments per claim and feeds them to
threshold-based gates, so small judgment noise becomes categorical output
flips. ``Claim.scrutiny_doubt`` measures that noise; this gate turns a high
value into the system's existing "do more work" primitive — a ``Demand``
routed through the lazy-escalation loop (P7) — rather than a new mechanism.

Every test here is offline (``agent_runner=None`` or a recording fake) and
drives the node directly via the ``_FakeRunContext`` duck type.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.entities.judge_opinion import JudgeOpinion
from andamentum.epistemic.graph.deps import EpistemicDeps
from andamentum.epistemic.graph.nodes import (
    SCRUTINY_RESOLVE_CYCLE_CAP,
    CheckSynthesisDemand,
    CombineClaimVerdicts,
    Scrutinize,
    Synthesize,
    SynthesizeInsufficient,
    _loopback_eligible,
)
from andamentum.epistemic.graph.state import EpistemicGraphState
from andamentum.epistemic.graph.topology import topology
from andamentum.epistemic.repository import EpistemicRepository
from andamentum.epistemic.tests.conftest import FakeAgentRunner
from andamentum.epistemic.thresholds import JUDGMENT_DOUBT_ESCALATION_THRESHOLD


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


def _opinion(*, dissents: bool, distribution: list[float]) -> JudgeOpinion:
    """A claim-level second opinion, as ``ScrutiniseClaimOperation`` records it.

    ``dissents`` and the vector are set independently on purpose: the two
    signals Gate 0 reads from an opinion are orthogonal. A CONFIDENT dissent
    has near-zero entropy (so ``needs_review`` is False and only ``dissents``
    can speak for it), and a doubtful AGREEMENT has high entropy (so only
    ``needs_review`` can). Folding either into the other is the exact
    scale-forking mistake the doubt-scale rule forbids.
    """
    return JudgeOpinion(
        subject="does this body of evidence establish the claim as stated?",
        reference="moderate",
        distribution=distribution,
        overall="fails" if dissents else "meets",
        expected_score=0.5,
        summary="",
        dissents=dissents,
    )


async def _setup(
    tmp_path: Path,
    *,
    doubts: list[float | None],
    cycle_capped: list[int] | None = None,
    at_cap: list[int] | None = None,
    with_decomposition: bool = True,
    name: str = "gate0",
    dissent_counts: list[int] | None = None,
    opinions: list[JudgeOpinion | None] | None = None,
) -> tuple[list[Claim], EpistemicGraphState, EpistemicRepository]:
    """One objective, one claim per entry in ``doubts``.

    Each claim carries ``integrated_assessment='supports'`` at a middling
    confidence, so the DOWNSTREAM gates would land in the ambiguous-LLM
    branch. Gate 0, if it fires, must pre-empt all of them.

    ``dissent_counts`` and ``opinions`` seed Gate 0's two other inputs. Both
    default to absent, so every pre-existing caller describes a claim on
    which only ``scrutiny_doubt`` can speak.
    """
    cycle_capped = cycle_capped or []
    at_cap = at_cap or []
    counts: list[int] = dissent_counts or [0] * len(doubts)
    reads: list[JudgeOpinion | None] = opinions or [None] * len(doubts)

    store = DocumentStore.for_database(name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)

    decomposition = None
    if with_decomposition:
        from andamentum.epistemic.entities.decomposition import (
            Decomposition,
            SubInvestigation,
        )

        decomposition = Decomposition(
            sub_investigations=[
                SubInvestigation(id=chr(ord("A") + i), seed_claim="a", rationale="r")
                for i in range(len(doubts))
            ],
            combination_rule="AND",
        )

    obj = Objective(description="x", decomposition=decomposition)
    obj.objective_id = obj.entity_id
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=obj.entity_id)
    claims: list[Claim] = []
    for i, doubt in enumerate(doubts):
        c = Claim(
            objective_id=obj.entity_id,
            statement=f"claim {chr(ord('A') + i)}",
            scope="x",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.55,
            sub_investigation_id=chr(ord("A") + i) if with_decomposition else None,
            cycle_capped=(i in cycle_capped),
            scrutiny_verdict="pass",
            scrutiny_doubt=doubt,
            judgment_dissent_count=counts[i],
            scrutiny_opinion=reads[i],
        )
        await repo.save(c)
        claims.append(c)
        if i in at_cap:
            state.scrutiny_resolve_cycles[c.entity_id] = SCRUTINY_RESOLVE_CYCLE_CAP

    deps = EpistemicDeps(repo=repo, agent_runner=None)
    await CombineClaimVerdicts().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    return claims, state, repo


# ── The mechanism ────────────────────────────────────────────────────


async def test_high_doubt_eligible_claim_loops_back_to_scrutinize(
    tmp_path: Path,
) -> None:
    """End to end: a measured, above-threshold doubt on an eligible claim
    routes back to Scrutinize and marks the objective as having spent its
    one escalation."""
    claims, state, repo = await _setup(tmp_path, doubts=[0.8])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert claims[0].entity_id in state.claims_needing_rescrutiny
    assert state.judgment_doubt_escalated is True


async def test_escalation_fires_at_most_once_per_objective(tmp_path: Path) -> None:
    """TERMINATION.

    Doubt buys one extra round, not a budget. Without this bound a
    persistently-unsure claim would burn its whole scrutiny-resolve budget
    on noise before any other gate got a look in.

    PINS: the second visit falls through to the normal gates.
    """
    _claims, state, repo = await _setup(tmp_path, doubts=[0.8])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    first = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(first, Scrutinize)

    second = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert not isinstance(second, Scrutinize), (
        "the doubt gate must not fire twice — it is the bound that makes "
        "the loop-back terminate"
    )


# ── The other two inputs ─────────────────────────────────────────────
#
# Gate 0 is ONE gate over THREE inputs, and each covers a blind spot the
# others cannot see. Every test below sets ``scrutiny_doubt=None`` so the
# ORIGINAL predicate provably cannot fire: if the new branch were dropped,
# these fail rather than passing on the old signal's back.


async def test_a_dissenting_evidence_confirmation_escalates_on_its_own(
    tmp_path: Path,
) -> None:
    """SITE A's route to Gate 0, and the reason the gate needed widening.

    ``scrutiny_doubt`` is a mean of judgment ENTROPIES, and a one-hot
    judgment has none — so on the 53-82% of small-model judgments that come
    back one-hot the aggregate is structurally silent. That is precisely
    where an independent judge contradicting the primary direction speaks,
    and a confident contradiction has near-zero entropy, so it cannot be
    smuggled through an entropy-shaped number.
    """
    _claims, state, repo = await _setup(tmp_path, doubts=[None], dissent_counts=[1])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is True


async def test_a_confidently_dissenting_claim_opinion_escalates_on_its_own(
    tmp_path: Path,
) -> None:
    """SITE B's route. The opinion is one-hot on 'fails', so its own
    ``doubt`` is ~0 and ``needs_review`` is False — only ``dissents`` can
    carry it. Pins that the gate reads the dissent flag and not merely the
    opinion's entropy."""
    opinion = _opinion(dissents=True, distribution=[0.01, 0.01, 0.98])
    assert opinion.needs_review is False, (
        "fixture must isolate the dissent flag from the entropy branch"
    )

    _claims, state, repo = await _setup(tmp_path, doubts=[None], opinions=[opinion])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is True


async def test_a_doubtful_but_agreeing_claim_opinion_escalates_on_its_own(
    tmp_path: Path,
) -> None:
    """The complement of the test above: the judge AGREED, so ``dissents``
    is False, but it barely committed. Only ``needs_review`` can carry
    this one — the two opinion branches are genuinely independent."""
    opinion = _opinion(dissents=False, distribution=[0.34, 0.33, 0.33])
    assert opinion.dissents is False
    assert opinion.needs_review is True, (
        "fixture must isolate the entropy branch from the dissent flag"
    )

    _claims, state, repo = await _setup(tmp_path, doubts=[None], opinions=[opinion])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is True


async def test_a_confident_agreeing_opinion_never_escalates(tmp_path: Path) -> None:
    """The happy path stays free. An independent judge that confidently
    AGREED is the one outcome that must buy nothing — otherwise every
    borderline reading would pay for an extra round regardless of what the
    second opinion said."""
    opinion = _opinion(dissents=False, distribution=[0.98, 0.01, 0.01])
    assert opinion.needs_review is False

    _claims, state, repo = await _setup(tmp_path, doubts=[None], opinions=[opinion])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert not isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is False


async def test_the_three_inputs_together_still_spend_only_one_round(
    tmp_path: Path,
) -> None:
    """COST BOUND UNDER THE WIDENED PREDICATE.

    Three claims, each firing a DIFFERENT one of the three inputs. However
    many signals fire, ``state.judgment_doubt_escalated`` still allows
    exactly one extra scrutiny round per run — which is what keeps the
    whole change's routing cost bounded regardless of how wide the
    predicate got.
    """
    _claims, state, repo = await _setup(
        tmp_path,
        doubts=[0.8, None, None],
        dissent_counts=[0, 2, 0],
        opinions=[None, None, _opinion(dissents=True, distribution=[0.01, 0.01, 0.98])],
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    first = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert isinstance(first, Scrutinize)

    second = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]
    assert not isinstance(second, Scrutinize)


async def test_a_cycle_capped_claim_does_not_escalate_on_the_new_inputs_either(
    tmp_path: Path,
) -> None:
    """The eligibility predicate is shared by all three inputs, not just the
    original one. A capped claim is not loop-back eligible, so a dissent on
    it must not route — otherwise the gate could hand ``_maybe_loop_back``
    an empty eligible set, the exact shape that falls through to
    ``SynthesizeInsufficient``."""
    _claims, state, repo = await _setup(
        tmp_path,
        doubts=[None],
        dissent_counts=[3],
        opinions=[_opinion(dissents=True, distribution=[0.01, 0.01, 0.98])],
        cycle_capped=[0],
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert not isinstance(next_node, Scrutinize)
    assert not isinstance(next_node, SynthesizeInsufficient)
    assert state.judgment_doubt_escalated is False


# ── Inertness ────────────────────────────────────────────────────────


async def test_unmeasured_doubt_never_escalates(tmp_path: Path) -> None:
    """PARTIAL-INSTRUMENTATION GUARD.

    ``None`` is "not measured", not "measured as zero" and not "measured
    as maximal". Every pre-existing fixture in the suite has
    ``scrutiny_doubt=None``, and none of them may change behaviour.
    """
    _claims, state, repo = await _setup(tmp_path, doubts=[None, None])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert not isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is False


async def test_doubt_below_the_threshold_never_escalates(tmp_path: Path) -> None:
    """The named constant really is the cut."""
    _claims, state, repo = await _setup(
        tmp_path, doubts=[JUDGMENT_DOUBT_ESCALATION_THRESHOLD - 0.01]
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert not isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is False


# ── Non-collision with the cycle-cap terminal-verdict change ─────────


async def test_cycle_capped_high_doubt_claim_does_not_escalate_and_does_not_route_to_insufficient(
    tmp_path: Path,
) -> None:
    """THE COLLISION, CLOSED BY CONSTRUCTION.

    ``_maybe_loop_back`` routes to ``SynthesizeInsufficient`` precisely when
    its eligible set is empty. If Gate 0 could fire on an INELIGIBLE claim it
    would hand ``_maybe_loop_back`` an empty set and manufacture a false
    "insufficient" — exactly the defect the cycle-cap terminal verdict was
    added to remove.

    Gate 0 and ``_maybe_loop_back`` share ``_loopback_eligible``, so this is
    structurally impossible.

    PINS: a cycle-capped, maximally-doubtful claim neither escalates nor
    suspends judgment.
    """
    _claims, state, repo = await _setup(tmp_path, doubts=[1.0], cycle_capped=[0])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert state.judgment_doubt_escalated is False
    assert not isinstance(next_node, SynthesizeInsufficient)
    assert isinstance(next_node, Synthesize)


async def test_claim_at_the_scrutiny_resolve_cap_does_not_escalate(
    tmp_path: Path,
) -> None:
    """The second half of the eligibility predicate."""
    _claims, state, repo = await _setup(tmp_path, doubts=[0.9], at_cap=[0])
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert state.judgment_doubt_escalated is False
    assert not isinstance(next_node, Scrutinize)


# ── Ordering, cost, and contract ─────────────────────────────────────


async def test_gate_zero_fires_in_verify_mode_with_no_decomposition(
    tmp_path: Path,
) -> None:
    """THE ORDERING DECISION, PINNED.

    Gate 1 short-circuits to "satisfied" whenever there is no combination
    rule — i.e. every verify-mode and open-research run, which is exactly
    the arm this signal exists for. A Gate 0 placed after it is dead code
    on the whole SciFact arm.

    PINS: moving Gate 0 below Gate 1 fails here.
    """
    _claims, state, repo = await _setup(
        tmp_path, doubts=[0.8], with_decomposition=False, name="gate0_verify"
    )
    assert state.verdict is not None
    assert state.verdict.provenance.combination_rule is None, (
        "fixture sanity: this must be the run shape Gate 1 short-circuits"
    )
    deps = EpistemicDeps(repo=repo, agent_runner=None)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert state.judgment_doubt_escalated is True


async def test_gate_zero_makes_no_llm_call(tmp_path: Path) -> None:
    """The deterministic doubt gate is free, and pre-empts the expensive
    reproducibility tripwire rather than running after it.

    All three inputs are seeded: widening the predicate must not have turned
    a free gate into an eliciting one. The two new signals are produced by
    the scrutiny OPERATION and merely read here — the gate never asks a
    judge anything itself.
    """
    _claims, state, repo = await _setup(
        tmp_path,
        doubts=[0.8],
        dissent_counts=[1],
        opinions=[_opinion(dissents=True, distribution=[0.01, 0.01, 0.98])],
    )
    runner = FakeAgentRunner()
    deps = EpistemicDeps(repo=repo, agent_runner=runner)

    next_node = await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, Scrutinize)
    assert runner.calls == []


async def test_gate_zero_and_maybe_loop_back_share_one_eligibility_predicate(
    tmp_path: Path,
) -> None:
    """The two must not drift.

    Over a matrix of (normal, cycle_capped, at-cap) claims, Gate 0 fires if
    and only if ``_maybe_loop_back`` would have a non-empty eligible set —
    since the only doubtful claim is the one under test.

    PINS: a future edit to one predicate that forgets the other fails here.
    """
    matrix: list[tuple[str, list[int], list[int]]] = [
        ("normal", [], []),
        ("cycle_capped", [0], []),
        ("at_cap", [], [0]),
    ]
    for i, (label, capped, at_cap) in enumerate(matrix):
        claims, state, repo = await _setup(
            tmp_path,
            doubts=[0.9],
            name=f"gate0_matrix_{i}",
            cycle_capped=capped,
            at_cap=at_cap,
        )
        deps = EpistemicDeps(repo=repo, agent_runner=None)
        eligible = [c for c in claims if _loopback_eligible(c, state)]

        await CheckSynthesisDemand().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

        assert state.judgment_doubt_escalated is bool(eligible), (
            f"[{label}] Gate 0 fired={state.judgment_doubt_escalated} but "
            f"_loopback_eligible said {bool(eligible)} — the two predicates "
            f"have drifted apart"
        )


def test_check_synthesis_demand_contract_declares_the_new_state_field() -> None:
    """P6 contract update. ``test_node_contracts.py`` only checks the
    ClassVars exist, so drift in their CONTENTS is otherwise invisible."""
    assert "judgment_doubt_escalated" in CheckSynthesisDemand.reads
    assert "judgment_doubt_escalated" in CheckSynthesisDemand.writes


def test_topology_is_unchanged() -> None:
    """This change adds no edge and no node — no new silent dead zone."""
    # Keyed by name rather than by class object: pydantic-graph's BaseNode
    # generics make the class-object subscript a pyright variance error (the
    # same one test_topology.py carries), and the name is what this test is
    # actually asserting about.
    topo = {node.__name__: succs for node, succs in topology().items()}
    successors = {n.__name__ for n in topo["CheckSynthesisDemand"]}
    assert successors == {"Synthesize", "Scrutinize", "SynthesizeInsufficient"}
