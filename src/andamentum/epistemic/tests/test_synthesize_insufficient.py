"""Tests for the structurally-insufficient synthesis path (Maximal B
of the K3 fix in the 2026-05-03 freeze sheet).

The architectural contract these tests pin:

* ``SynthesizeInsufficientReportOperation`` produces a deterministic
  artefact stamped ``artefact_type="insufficient"``, with a fixed
  verdict string ("Insufficient evidence to answer."). No LLM call.
* The body surfaces structural counts (claims, evidence, abandoned,
  capped, no-verdict, blocking uncertainties) — the system's audit
  trail for *why* it suspended judgment.
* The diagnosis from ``CheckSynthesisDemand`` is carried through via
  ``OperationInput.metadata["synthesis_insufficient_reason"]`` and
  appears in the body — closing the loop between the gate that
  detected the no-data state and the artefact that records it.
* The objective and snapshot are stamped (``artefact_id`` set, phase
  ``"complete"``) so the same downstream consumers (CLI, stage runner
  invariant) work identically for both terminals.
"""

from __future__ import annotations

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
    UncertaintyType,
)
from ..operations import SynthesizeInsufficientReportOperation
from ..operations.base import OperationInput


async def _seed_no_verdict_db(repo) -> str:
    """Build a minimal DB representing the K3 failure mode: claims
    exist, evidence exists, but no claim reached an integration verdict.
    Returns the snapshot id."""
    obj = Objective(
        entity_id="obj-insuf",
        objective_id="obj-insuf",
        description="Does aspirin prevent first heart attacks in healthy adults?",
        phase="claims_done",
    )
    await repo.save(obj)

    e1 = Evidence(
        entity_id="e-insuf-1",
        objective_id="obj-insuf",
        quality_score=0.4,
        extracted=True,
        extracted_content="An evidence snippet of moderate quality.",
    )
    await repo.save(e1)

    # Two claims: one cycle-capped (couldn't make progress), one
    # abandoned (nothing actionable found). Neither has an integration
    # verdict — the K3 case.
    c_capped = Claim(
        entity_id="c-capped",
        objective_id="obj-insuf",
        statement="Aspirin reduces first MI in healthy adults.",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="needs_resolution",
        cycle_capped=True,
        evidence_ids=["e-insuf-1"],
    )
    await repo.save(c_capped)

    c_abandoned = Claim(
        entity_id="c-abandoned",
        objective_id="obj-insuf",
        statement="Aspirin's bleeding risk dominates in healthy adults.",
        stage=ClaimStage.HYPOTHESIS,
        abandoned=True,
        evidence_ids=[],
    )
    await repo.save(c_abandoned)

    u = Uncertainty(
        entity_id="u-insuf-1",
        objective_id="obj-insuf",
        description="Population stratification by baseline risk is missing.",
        uncertainty_type=UncertaintyType.UNKNOWN,  # blocking
    )
    await repo.save(u)

    snap = Snapshot(
        entity_id="snap-insuf",
        objective_id="obj-insuf",
        snapshot_type="final",
        claim_ids=["c-capped", "c-abandoned"],
        evidence_ids=["e-insuf-1"],
        uncertainty_ids=["u-insuf-1"],
    )
    await repo.save(snap)
    return snap.entity_id


async def test_artefact_is_typed_insufficient(repo) -> None:
    """The artefact's ``artefact_type`` is the typed signal that
    distinguishes a fallibilism terminal from a directional verdict.
    Downstream consumers must be able to read this field rather than
    parsing the prose."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
    )
    result = await op.execute(work)
    assert result.success

    artefacts = await repo.query("artefact", objective_id="obj-insuf")
    assert len(artefacts) == 1
    assert artefacts[0].artefact_type == "insufficient", (
        "artefact_type is the load-bearing typed signal that this "
        "terminal is structurally distinct from a directional verdict."
    )


async def test_verdict_is_fixed_string_no_llm(repo) -> None:
    """The verdict is encoded in the operation, not produced by an LLM.
    This is the Peircean fallibilism property: the system's "I don't
    know" state is a structural property, not a prompt outcome."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
    )
    await op.execute(work)

    artefact = (await repo.query("artefact", objective_id="obj-insuf"))[0]
    assert "Insufficient evidence to answer." in artefact.content
    assert "Verdict:" in artefact.content
    # The body must NOT begin with a directional verdict.
    assert "**Verdict:** No." not in artefact.content
    assert "**Verdict:** Yes." not in artefact.content


async def test_body_surfaces_structural_counts(repo) -> None:
    """The artefact must surface what the system attempted: claim
    counts, evidence counts, abandoned, capped, no-verdict,
    uncertainties. This is the audit trail readers need to evaluate
    whether the suspended judgment is well-founded."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
    )
    await op.execute(work)

    artefact = (await repo.query("artefact", objective_id="obj-insuf"))[0]
    body = artefact.content
    # Claim counts (1 active + 1 abandoned = 1 in claims_total per
    # quality_signals, which excludes abandoned)
    assert "claim(s) investigated" in body
    assert "1 claim(s) abandoned" in body
    assert "1 claim(s) reached the per-claim investigation cap" in body
    # Evidence count
    assert "1 evidence item(s) gathered" in body
    # Blocking uncertainty
    assert "1 blocking uncertainty(ies) identified" in body


async def test_demand_reason_carried_through_metadata(repo) -> None:
    """The structural diagnosis the gate produced flows from
    ``CheckSynthesisDemand`` to the artefact via the operation's
    metadata. This is the closed loop: the system surfaces the same
    reason it used to route here."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    reason = (
        "No combined verdict produced (every claim was abandoned, "
        "cycle-capped, or had no integration verdict)."
    )
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
        metadata={"synthesis_insufficient_reason": reason},
    )
    await op.execute(work)

    artefact = (await repo.query("artefact", objective_id="obj-insuf"))[0]
    assert reason in artefact.content
    assert "Why no directional verdict is offered" in artefact.content


async def test_objective_and_snapshot_stamped(repo) -> None:
    """Both terminals (Synthesize, SynthesizeInsufficient) must leave
    the objective and snapshot in the same shape so the stage runner's
    synthesis invariant (``obj.artefact_id is not None``) and other
    downstream consumers don't need to special-case the path."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
    )
    await op.execute(work)

    obj = await repo.get("objective", "obj-insuf")
    snap = await repo.get("snapshot", snap_id)
    assert obj.artefact_id is not None
    assert obj.phase == "complete"
    assert snap.artefact_id is not None


async def test_idempotent_when_artefact_exists(repo) -> None:
    """Re-running the operation against a snapshot whose artefact has
    already been compiled is a no-op. Same idempotence contract as
    ``SynthesizeReportOperation``."""
    snap_id = await _seed_no_verdict_db(repo)
    op = SynthesizeInsufficientReportOperation(repo=repo)
    work = OperationInput(
        entity_id=snap_id,
        entity_type="snapshot",
        operation="synthesize_insufficient",
    )
    r1 = await op.execute(work)
    assert r1.success

    r2 = await op.execute(work)
    assert r2.success
    assert "already compiled" in r2.message
    artefacts = await repo.query("artefact", objective_id="obj-insuf")
    assert len(artefacts) == 1


# ── CheckCompletion routes all "system can't conclude" paths ─────────
#
# Maximal B (extended): every End() bypass that used to exit without
# producing an artefact now routes through SynthesizeInsufficient.
# These tests pin the three previously-bypassed paths:
#
#   1. retrieval_failed — evidence extraction kept returning empty.
#   2. no_claims — decomposition / claim-creation produced nothing.
#   3. partial — claims existed but were all abandoned during scrutiny.
#
# All three must now produce an artefact and route to
# SynthesizeInsufficient (not End directly), with a reason text that
# names the specific cause.


async def _setup_check_completion_state(repo, objective_id: str = "obj-cc"):
    """Build the minimal entities CheckCompletion needs to make its
    routing decision."""
    from andamentum.epistemic.entities import Objective
    from andamentum.epistemic.graph.deps import EpistemicDeps
    from andamentum.epistemic.graph.state import EpistemicGraphState

    obj = Objective(
        entity_id=objective_id,
        objective_id=objective_id,
        description="test question",
        question_type="verificatory",
    )
    await repo.save(obj)

    state = EpistemicGraphState(objective_id=objective_id)
    deps = EpistemicDeps(repo=repo, agent_runner=None, embedding_model="t")
    return state, deps


class _FakeRunContext:
    def __init__(self, state, deps):
        self.state = state
        self.deps = deps


async def test_check_completion_routes_retrieval_failed_to_insufficient(repo) -> None:
    """The retrieval_failed short-circuit no longer exits to End; it
    routes through SynthesizeInsufficient with a reason that names
    the retrieval failure. The retrieval_failed BOOL is still
    available downstream via state."""
    from andamentum.epistemic.graph.nodes import (
        CheckCompletion,
        SynthesizeInsufficient,
    )

    state, deps = await _setup_check_completion_state(repo)
    state.retrieval_failed = True

    next_node = await CheckCompletion().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient), (
        "retrieval_failed must route to SynthesizeInsufficient (Maximal "
        "B extended), not exit to End. Without an artefact the user "
        "gets ambiguous output that looks like a research report but "
        "isn't."
    )
    assert state.synthesis_insufficient_reason is not None
    assert "retrieval failed" in state.synthesis_insufficient_reason.lower()
    # The retrieval_failed BOOL is preserved for downstream consumers
    # (confidence.compute_posterior, audit_report) that branch on it.
    assert state.retrieval_failed is True


async def test_check_completion_routes_no_claims_to_insufficient(repo) -> None:
    """No claims at all: the decomposition / claim-creation step
    produced nothing investigable. Route through SynthesizeInsufficient
    so the user gets a structurally honest "no claims" artefact."""
    from andamentum.epistemic.graph.nodes import (
        CheckCompletion,
        SynthesizeInsufficient,
    )

    state, deps = await _setup_check_completion_state(repo)
    # No claims saved.

    next_node = await CheckCompletion().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    assert "no claims" in state.synthesis_insufficient_reason.lower()


async def test_check_completion_routes_all_abandoned_to_insufficient(repo) -> None:
    """All claims abandoned during scrutiny: structurally a "the
    system tried but couldn't" outcome. Route through
    SynthesizeInsufficient so the user gets an artefact rather than
    a silent exit."""
    from andamentum.epistemic.entities import Claim, ClaimStage
    from andamentum.epistemic.graph.nodes import (
        CheckCompletion,
        SynthesizeInsufficient,
    )

    state, deps = await _setup_check_completion_state(repo, objective_id="obj-aban")

    # Two claims, both abandoned.
    for i in range(2):
        c = Claim(
            objective_id="obj-aban",
            statement=f"claim {i}",
            scope="x",
            stage=ClaimStage.HYPOTHESIS,
            abandoned=True,
        )
        await repo.save(c)

    next_node = await CheckCompletion().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, SynthesizeInsufficient)
    assert state.synthesis_insufficient_reason is not None
    reason = state.synthesis_insufficient_reason.lower()
    assert "abandoned" in reason
    # The reason should mention how many claims were abandoned (the
    # operator-level diagnostic the user reads off the artefact).
    assert "2" in state.synthesis_insufficient_reason


async def test_check_completion_routes_active_claims_to_synthesis_demand(
    repo,
) -> None:
    """Sanity: when there ARE non-abandoned claims, CheckCompletion
    still routes to CheckSynthesisDemand (the existing happy path).
    Maximal B's extension only changes the bypass paths."""
    from andamentum.epistemic.entities import Claim, ClaimStage
    from andamentum.epistemic.graph.nodes import (
        CheckCompletion,
        CheckSynthesisDemand,
    )

    state, deps = await _setup_check_completion_state(repo, objective_id="obj-active")

    # One active claim with an integrated_assessment — this is the
    # "happy path" where the IBE chain certified a verdict. Routing
    # must continue through CheckSynthesisDemand so the existing
    # gate-and-loop-back logic handles it.
    #
    # Note: a SUPPORTED claim WITHOUT integrated_assessment now
    # routes to SynthesizeInsufficient under the no_certified_verdict
    # exit (added 2026-05-05). That exit is tested separately in
    # test_no_certified_verdict_exit.py; this test pins the IA-present
    # branch.
    c = Claim(
        objective_id="obj-active",
        statement="active claim",
        scope="x",
        stage=ClaimStage.SUPPORTED,
        scrutiny_verdict="pass",
        abandoned=False,
        integrated_assessment="supports",
        integrated_confidence=0.7,
    )
    await repo.save(c)

    next_node = await CheckCompletion().run(_FakeRunContext(state, deps))  # type: ignore[arg-type]

    assert isinstance(next_node, CheckSynthesisDemand), (
        "When there are active claims with integrated_assessment, the "
        "existing routing must hold; Maximal B's extension catches "
        "uncertified claims."
    )
    # CheckCompletion didn't write a synthesis_insufficient_reason on
    # this path — the demand check might still ask for it later.
    assert state.synthesis_insufficient_reason is None
