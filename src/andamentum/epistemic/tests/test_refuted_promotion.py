"""Tests for the refuted claim promotion path."""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.gates import (
    count_support_contradict,
    is_refuted_by_evidence,
)
from andamentum.epistemic.repository import EpistemicRepository


async def _setup_claim_with_evidence(
    tmp_path: Path,
    n_supports: int,
    n_contradicts: int,
    n_unjudged: int = 0,
) -> tuple[Claim, EpistemicRepository]:
    """Create an objective + claim linked to N supporting + M contradicting evidence."""
    store = DocumentStore.for_database("test", db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(description="test objective", question_type="predictive")
    await repo.save(obj)
    claim = Claim(
        statement="Test claim.", scope="test scope", objective_id=obj.entity_id,
        stage=ClaimStage.HYPOTHESIS,
    )
    await repo.save(claim)
    for i in range(n_supports):
        ev = Evidence(
            source_type="web", source_ref=f"https://ex.com/s{i}",
            extracted_content="supports", objective_id=obj.entity_id,
            support_judgment="supports",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    for i in range(n_contradicts):
        ev = Evidence(
            source_type="web", source_ref=f"https://ex.com/c{i}",
            extracted_content="contradicts", objective_id=obj.entity_id,
            support_judgment="contradicts",
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    for i in range(n_unjudged):
        ev = Evidence(
            source_type="web", source_ref=f"https://ex.com/u{i}",
            extracted_content="x", objective_id=obj.entity_id,
        )
        await repo.save(ev)
        claim.evidence_ids.append(ev.entity_id)
    claim.evidence_count = len(claim.evidence_ids)
    await repo.save(claim)
    return claim, repo


class TestCountSupportContradict:
    async def test_counts_directional_judgments(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 5)
        n_sup, n_con = await count_support_contradict(claim, repo)
        assert n_sup == 2
        assert n_con == 5

    async def test_ignores_unjudged(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 1, n_unjudged=3)
        n_sup, n_con = await count_support_contradict(claim, repo)
        assert (n_sup, n_con) == (1, 1)

    async def test_empty_evidence(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 0)
        assert await count_support_contradict(claim, repo) == (0, 0)

    async def test_skips_invalidated_evidence(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 2)
        # Invalidate one supporting and one contradicting evidence item
        evs = []
        for eid in claim.evidence_ids:
            ev = await repo.get("evidence", eid)
            evs.append(ev)
        sup_ev = next(e for e in evs if e.support_judgment == "supports")
        con_ev = next(e for e in evs if e.support_judgment == "contradicts")
        sup_ev.invalidated = True
        con_ev.invalidated = True
        await repo.save(sup_ev)
        await repo.save(con_ev)

        n_sup, n_con = await count_support_contradict(claim, repo)
        assert (n_sup, n_con) == (1, 1)


class TestIsRefutedByEvidence:
    async def test_refuted_when_contradicts_dominate(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 9)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_not_refuted_when_balanced(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 4, 5)
        assert await is_refuted_by_evidence(claim, repo) is False

    async def test_not_refuted_when_too_few_contradicts(self, tmp_path: Path) -> None:
        # Threshold requires at least 3 contradicts.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 2)
        assert await is_refuted_by_evidence(claim, repo) is False

    async def test_refuted_with_zero_supports(self, tmp_path: Path) -> None:
        claim, repo = await _setup_claim_with_evidence(tmp_path, 0, 3)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_refuted_at_minimum_ratio_edge(self, tmp_path: Path) -> None:
        # (n_con=3, n_sup=1) sits on the tightest passing edge:
        # 3 >= 3 AND 3 >= 2 * max(1, 1) = 2. Must be True.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 1, 3)
        assert await is_refuted_by_evidence(claim, repo) is True

    async def test_not_refuted_just_below_ratio(self, tmp_path: Path) -> None:
        # (n_con=3, n_sup=2) is the just-below-ratio case:
        # 3 >= 3 passes but 3 >= 2 * 2 = 4 fails. Must be False.
        claim, repo = await _setup_claim_with_evidence(tmp_path, 2, 3)
        assert await is_refuted_by_evidence(claim, repo) is False
