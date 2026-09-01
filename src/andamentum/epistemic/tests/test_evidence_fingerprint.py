"""Tests for the evidence fingerprint and the contraction re-promotion rule.

Certify-before-contract plan (docs/epistemic/CERTIFY_BEFORE_CONTRACT_PLAN.md),
Phase A, D3. Covers the PURE part of S3: no claim runs the verification
tracks twice on an unchanged evidence set.
"""

from ..entities import Claim, ClaimStage, Evidence
from ..gates import (
    claim_evidence_fingerprint,
    evidence_fingerprint,
    promotion_blocked_by_contraction,
)


class TestEvidenceFingerprintPure:
    """``evidence_fingerprint`` — pure, no repository."""

    def test_same_rows_same_fingerprint(self):
        rows = [("e-1", "supports", False), ("e-2", "contradicts", False)]
        assert evidence_fingerprint(rows) == evidence_fingerprint(rows)

    def test_order_independent(self):
        rows_a = [("e-1", "supports", False), ("e-2", "contradicts", False)]
        rows_b = [("e-2", "contradicts", False), ("e-1", "supports", False)]
        assert evidence_fingerprint(rows_a) == evidence_fingerprint(rows_b)

    def test_different_judgment_changes_fingerprint(self):
        rows_a = [("e-1", "supports", False)]
        rows_b = [("e-1", "contradicts", False)]
        assert evidence_fingerprint(rows_a) != evidence_fingerprint(rows_b)

    def test_invalidation_changes_fingerprint(self):
        rows_a = [("e-1", "supports", False)]
        rows_b = [("e-1", "supports", True)]
        assert evidence_fingerprint(rows_a) != evidence_fingerprint(rows_b)

    def test_added_evidence_changes_fingerprint(self):
        rows_a = [("e-1", "supports", False)]
        rows_b = [("e-1", "supports", False), ("e-2", "supports", False)]
        assert evidence_fingerprint(rows_a) != evidence_fingerprint(rows_b)

    def test_none_judgment_normalizes_to_empty_string(self):
        rows_a = [("e-1", None, False)]
        rows_b = [("e-1", "", False)]
        assert evidence_fingerprint(rows_a) == evidence_fingerprint(rows_b)

    def test_empty_rows_is_stable(self):
        assert evidence_fingerprint([]) == evidence_fingerprint([])

    def test_returns_hex_digest(self):
        fp = evidence_fingerprint([("e-1", "supports", False)])
        assert isinstance(fp, str)
        int(fp, 16)  # raises ValueError if not valid hex


class TestClaimEvidenceFingerprintAsync:
    """``claim_evidence_fingerprint`` — loads evidence via the repository."""

    async def test_matches_pure_helper_for_loaded_evidence(self, repo):
        e1 = Evidence(
            entity_id="e-1",
            objective_id="o",
            support_judgment="supports",
            invalidated=False,
        )
        e2 = Evidence(
            entity_id="e-2",
            objective_id="o",
            support_judgment="contradicts",
            invalidated=False,
        )
        await repo.save(e1)
        await repo.save(e2)
        claim = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            evidence_ids=["e-1", "e-2"],
        )
        await repo.save(claim)

        fp = await claim_evidence_fingerprint(claim, repo)
        expected = evidence_fingerprint(
            [("e-1", "supports", False), ("e-2", "contradicts", False)]
        )
        assert fp == expected

    async def test_changes_when_evidence_invalidated(self, repo):
        e1 = Evidence(
            entity_id="e-1",
            objective_id="o",
            support_judgment="supports",
            invalidated=False,
        )
        await repo.save(e1)
        claim = Claim(
            entity_id="c-1", objective_id="o", statement="X", evidence_ids=["e-1"]
        )
        await repo.save(claim)

        fp_before = await claim_evidence_fingerprint(claim, repo)

        e1.invalidated = True
        await repo.save(e1)
        fp_after = await claim_evidence_fingerprint(claim, repo)

        assert fp_before != fp_after

    async def test_no_evidence_is_stable_empty_fingerprint(self, repo):
        claim = Claim(entity_id="c-1", objective_id="o", statement="X")
        await repo.save(claim)
        assert await claim_evidence_fingerprint(
            claim, repo
        ) == await claim_evidence_fingerprint(claim, repo)


class TestPromotionBlockedByContraction:
    """``promotion_blocked_by_contraction`` — the re-promotion refusal."""

    async def test_never_contracted_is_not_blocked(self, repo):
        e1 = Evidence(entity_id="e-1", objective_id="o", support_judgment="supports")
        await repo.save(e1)
        claim = Claim(
            entity_id="c-1", objective_id="o", statement="X", evidence_ids=["e-1"]
        )
        await repo.save(claim)
        assert await promotion_blocked_by_contraction(claim, repo) is None

    async def test_blocked_when_fingerprint_unchanged_since_contraction(self, repo):
        e1 = Evidence(entity_id="e-1", objective_id="o", support_judgment="supports")
        await repo.save(e1)
        claim = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            stage=ClaimStage.SUPPORTED,
            evidence_ids=["e-1"],
        )
        await repo.save(claim)

        current_fp = await claim_evidence_fingerprint(claim, repo)
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="contracted", fingerprint=current_fp
        )
        await repo.save(claim)

        reason = await promotion_blocked_by_contraction(claim, repo)
        assert reason is not None
        assert "unchanged" in reason.lower()

    async def test_not_blocked_once_evidence_changes(self, repo):
        e1 = Evidence(entity_id="e-1", objective_id="o", support_judgment="supports")
        await repo.save(e1)
        claim = Claim(
            entity_id="c-1",
            objective_id="o",
            statement="X",
            stage=ClaimStage.SUPPORTED,
            evidence_ids=["e-1"],
        )
        await repo.save(claim)

        stale_fp = await claim_evidence_fingerprint(claim, repo)
        claim.contract(
            ClaimStage.HYPOTHESIS, justification="contracted", fingerprint=stale_fp
        )
        await repo.save(claim)

        # New evidence arrives.
        e2 = Evidence(entity_id="e-2", objective_id="o", support_judgment="supports")
        await repo.save(e2)
        claim.evidence_ids.append("e-2")
        await repo.save(claim)

        assert await promotion_blocked_by_contraction(claim, repo) is None
