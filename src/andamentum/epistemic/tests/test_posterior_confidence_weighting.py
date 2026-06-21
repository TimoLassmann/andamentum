"""Tier 1 — verbalized judgment confidence informs the counting posterior.

The counting path used to treat every supports/contradicts as a hard
one-vote-per-item count. With a verbalized belief distribution captured at
judgment time (Tier 0), each item now contributes its soft mass instead — a
near-tie barely moves the posterior, a confident judgment moves it fully.

Critical property under test: the change is *exactly* backward-compatible in the
limiting case. A one-hot distribution (the small-model degeneracy mode) and
evidence with no captured distribution (adversarial / legacy) both reduce to the
original hard vote, so runs without the verbalized signal are unaffected.
"""

import math

import pytest

from ..confidence import _evidence_counting_vote, compute_posterior
from ..entities import Claim, Evidence, Objective

OBJ_ID = "tier1-posterior-obj"


def _ev(
    *,
    dist: list[float] | None = None,
    sj: str | None = "supports",
    corro: int = 1,
    eid: str = "e",
) -> Evidence:
    return Evidence(
        entity_id=eid,
        objective_id=OBJ_ID,
        source_type="web_search",
        source_ref="https://example.com",
        extracted_content="content",
        extracted=True,
        support_judgment=sj,
        cluster_status="representative",
        corroboration_count=corro,
        judgment_distribution=dist,
    )


class TestEvidenceCountingVote:
    def test_one_hot_supports_equals_hard_vote(self) -> None:
        assert _evidence_counting_vote(_ev(dist=[1.0, 0.0, 0.0])) == (1.0, 0.0)

    def test_one_hot_contradicts_equals_hard_vote(self) -> None:
        assert _evidence_counting_vote(_ev(dist=[0.0, 1.0, 0.0])) == (0.0, 1.0)

    def test_no_distribution_falls_back_to_hard_vote(self) -> None:
        assert _evidence_counting_vote(_ev(dist=None, sj="supports")) == (1.0, 0.0)
        assert _evidence_counting_vote(_ev(dist=None, sj="contradicts")) == (0.0, 1.0)
        assert _evidence_counting_vote(_ev(dist=None, sj="no_bearing")) == (0.0, 0.0)
        assert _evidence_counting_vote(_ev(dist=None, sj=None)) == (0.0, 0.0)

    def test_graded_contributes_soft_mass(self) -> None:
        s, c = _evidence_counting_vote(_ev(dist=[0.6, 0.3, 0.1]))
        assert s == pytest.approx(0.6)
        assert c == pytest.approx(0.3)

    def test_near_tie_nets_near_zero(self) -> None:
        s, c = _evidence_counting_vote(_ev(dist=[0.5, 0.45, 0.05]))
        assert (s - c) == pytest.approx(0.05)

    def test_corroboration_scales_weight(self) -> None:
        weight = 1.0 + math.log(3)
        s, c = _evidence_counting_vote(_ev(dist=[1.0, 0.0, 0.0], corro=3))
        assert s == pytest.approx(weight)
        assert c == pytest.approx(0.0)

    def test_corroboration_scales_soft_mass(self) -> None:
        weight = 1.0 + math.log(4)
        s, c = _evidence_counting_vote(_ev(dist=[0.7, 0.2, 0.1], corro=4))
        assert s == pytest.approx(0.7 * weight)
        assert c == pytest.approx(0.2 * weight)


async def _setup(repo, evidence: list[Evidence]) -> None:
    obj = Objective(
        entity_id=OBJ_ID,
        objective_id=OBJ_ID,
        description="q",
        question_type="verificatory",
    )
    await repo.save(obj)
    for ev in evidence:
        await repo.save(ev)
    # integrated_assessment set so the no-certified-verdict gate doesn't fire;
    # we assert on the counting diagnostic, which is where Tier 1 acts.
    claim = Claim(
        objective_id=OBJ_ID,
        statement="claim",
        evidence_ids=[ev.entity_id for ev in evidence],
        integrated_assessment="supports",
        integrated_confidence=0.8,
    )
    await repo.save(claim)


class TestPosteriorSoftWeighting:
    async def test_one_hot_reproduces_hard_counting(self, repo) -> None:
        """Three one-hot supports → counting identical to the pre-Tier-1 hard
        count (supporting_count == 3.0)."""
        await _setup(
            repo,
            [_ev(dist=[1.0, 0.0, 0.0], eid=f"oh{i}") for i in range(3)],
        )
        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.supporting_count == pytest.approx(3.0)
        assert report.contradicting_count == pytest.approx(0.0)
        assert report.counting_posterior == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))

    async def test_graded_is_less_decisive_than_one_hot(self, repo) -> None:
        """Three graded supports [0.7, 0.2, 0.1] → softer counts and a
        counting_posterior pulled toward 0.5 vs the one-hot case."""
        await _setup(
            repo,
            [_ev(dist=[0.7, 0.2, 0.1], eid=f"g{i}") for i in range(3)],
        )
        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.supporting_count == pytest.approx(2.1)
        assert report.contradicting_count == pytest.approx(0.6)
        expected = 1.0 / (1.0 + math.exp(-(2.1 - 0.6)))
        assert report.counting_posterior == pytest.approx(expected)
        # Less decisive than the one-hot equivalent (sigmoid(3.0) ≈ 0.953).
        assert report.counting_posterior < 1.0 / (1.0 + math.exp(-3.0))

    async def test_none_distribution_matches_legacy(self, repo) -> None:
        """Evidence without a captured distribution counts exactly as before."""
        await _setup(
            repo,
            [
                _ev(dist=None, sj="supports", eid="n1"),
                _ev(dist=None, sj="supports", eid="n2"),
                _ev(dist=None, sj="contradicts", eid="n3"),
            ],
        )
        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.supporting_count == pytest.approx(2.0)
        assert report.contradicting_count == pytest.approx(1.0)
