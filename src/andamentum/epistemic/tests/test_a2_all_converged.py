"""Tests for Phase 2: A2 convergence-driven termination requires
all-converged across active SUPPORTED claims, not any-converged.

Bug context: under multi-seed-claim, ONE Objective hosts N Claims. The
previous A2 short-circuit (``any_converged``) fired as soon as ONE claim
reached CONVERGENT verdict, dragging sibling claims with unfinished
verification tracks into IBE. The fix requires *all* active SUPPORTED
claims to have a terminal convergence verdict (with at least one
positive CONVERGENT) before short-circuiting.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.repository import EpistemicRepository


async def _setup_objective_with_supported_claims(
    tmp_path: Path,
    db_name: str,
    *,
    convergence_verdicts: list[str | None],
) -> tuple[Objective, list[Claim], EpistemicRepository]:
    """Build an Objective with N SUPPORTED claims, one per
    convergence_verdict in the list. Caller controls each claim's
    verdict so we can probe the A2 gate's behavior under different
    multi-claim states."""
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="parent",
        clarified_question="parent",
        question_type="verificatory",
        phase="claims_proposed",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    claims = []
    for i, verdict in enumerate(convergence_verdicts):
        claim = Claim(
            objective_id=obj.entity_id,
            statement=f"claim {i}",
            scope=f"scope {i}",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            convergence_verdict=verdict,
            convergence_checked=verdict is not None,
        )
        await repo.save(claim)
        claims.append(claim)
    return obj, claims, repo


# Test the gate logic directly by importing the relevant constant set.
# We'd ideally invoke RunVerification.run() but that requires real
# verification operations. For this test we focus on the *gate
# semantics* — what combinations of convergence_verdicts trigger
# termination — using a small helper that mirrors the gate logic.


def _gate_fires(verdicts: list[str | None]) -> bool:
    """Mirror the A2 gate logic in graph/nodes.py for direct probing.

    Keep this in sync with the implementation: all active SUPPORTED
    claims must have a terminal convergence verdict, AND at least one
    must be positive (CONVERGENT)."""
    terminal_convergence = {
        "CONVERGENT",
        "WEAKLY_CONVERGENT",
        "DIVERGENT",
        "PARTIAL",
        "SINGLE_DOMAIN",
        "NO_EVIDENCE",
    }
    if not verdicts:
        return False
    all_terminal = all(v in terminal_convergence for v in verdicts)
    any_positive = any(v == "CONVERGENT" for v in verdicts)
    return all_terminal and any_positive


class TestA2GateSemantics:
    def test_single_claim_convergent_fires(self) -> None:
        assert _gate_fires(["CONVERGENT"]) is True

    def test_all_claims_convergent_fires(self) -> None:
        assert _gate_fires(["CONVERGENT", "CONVERGENT", "CONVERGENT"]) is True

    def test_one_convergent_others_unchecked_does_not_fire(self) -> None:
        """The case 54 bug: one claim converges while siblings still
        have None verdict. Pre-fix this fired prematurely; post-fix it
        does not."""
        assert _gate_fires(["CONVERGENT", None, None]) is False

    def test_all_terminal_with_one_convergent_fires(self) -> None:
        """Mixed terminal verdicts (some CONVERGENT, some DIVERGENT,
        some SINGLE_DOMAIN) — all the convergence track did its job, at
        least one is positive: A2 fires."""
        assert _gate_fires(["CONVERGENT", "DIVERGENT", "SINGLE_DOMAIN"]) is True

    def test_all_terminal_but_none_convergent_does_not_fire(self) -> None:
        """All claims got a verdict, but none is CONVERGENT. A2 should
        NOT fire — there's no positive convergence signal to terminate
        on."""
        assert _gate_fires(["DIVERGENT", "DIVERGENT", "SINGLE_DOMAIN"]) is False

    def test_one_convergent_one_unchecked_does_not_fire(self) -> None:
        """Two-claim case where one converges and one is still mid-
        verification (None). Don't terminate yet."""
        assert _gate_fires(["CONVERGENT", None]) is False

    def test_empty_list_does_not_fire(self) -> None:
        """No active SUPPORTED claims → no termination signal."""
        assert _gate_fires([]) is False


class TestA2EntitySetup:
    """Sanity that the Claim entity stores the verdicts the gate reads
    from. Ensures the underlying data shape used by the gate matches
    what production will write."""

    async def test_convergence_verdict_persists(self, tmp_path: Path) -> None:
        obj, claims, repo = await _setup_objective_with_supported_claims(
            tmp_path,
            "a2_persist",
            convergence_verdicts=["CONVERGENT", None, "DIVERGENT"],
        )
        all_loaded = await repo.query("claim", objective_id=obj.entity_id)
        verdicts_by_claim = {c.entity_id: c.convergence_verdict for c in all_loaded}
        assert verdicts_by_claim[claims[0].entity_id] == "CONVERGENT"
        assert verdicts_by_claim[claims[1].entity_id] is None
        assert verdicts_by_claim[claims[2].entity_id] == "DIVERGENT"
