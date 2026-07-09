"""Tests for the promotion-gate judgment-confidence floor (harness-backstops 2b).

``count_supporting_sources`` counts discrete supporting sources for a stage gate,
but a supporting judgment must clear ``GATE_MIN_JUDGMENT_CONFIDENCE`` to count —
a high-entropy support should not ratchet a claim up a stage. Evidence with no
captured distribution is exempt (backward-compat).
"""

from __future__ import annotations

from ..entities import Claim, Evidence
from ..gates import count_supporting_sources
from ..thresholds import GATE_MIN_JUDGMENT_CONFIDENCE


async def _ev(repo, eid: str, *, judgment=None, support="supports") -> None:
    await repo.save(
        Evidence(
            entity_id=eid,
            objective_id="o",
            source_type="web_search",
            source_ref="s",
            extracted_content="finding",
            extracted=True,
            support_judgment=support,
            judgment_distribution=judgment,
        )
    )


class TestGateConfidenceFloor:
    async def test_one_hot_support_counts(self, repo) -> None:
        await _ev(repo, "e1", judgment=[1.0, 0.0, 0.0])
        c = Claim(entity_id="c1", objective_id="o", statement="x", evidence_ids=["e1"])
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 1

    async def test_confident_graded_support_counts(self, repo) -> None:
        # confidence 0.7 ≥ 0.6 floor.
        await _ev(repo, "e1", judgment=[0.7, 0.2, 0.1])
        c = Claim(entity_id="c1", objective_id="o", statement="x", evidence_ids=["e1"])
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 1

    async def test_high_entropy_support_excluded(self, repo) -> None:
        # confidence 0.5 < 0.6 floor — too shaky to count toward promotion.
        assert GATE_MIN_JUDGMENT_CONFIDENCE > 0.5
        await _ev(repo, "e1", judgment=[0.5, 0.3, 0.2])
        c = Claim(entity_id="c1", objective_id="o", statement="x", evidence_ids=["e1"])
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 0

    async def test_no_distribution_is_exempt(self, repo) -> None:
        # Adversarial / pre-Tier-0 evidence: no measured confidence → counts.
        await _ev(repo, "e1", judgment=None)
        c = Claim(entity_id="c1", objective_id="o", statement="x", evidence_ids=["e1"])
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 1

    async def test_contradicting_never_counts(self, repo) -> None:
        await _ev(repo, "e1", judgment=[0.05, 0.9, 0.05], support="contradicts")
        c = Claim(entity_id="c1", objective_id="o", statement="x", evidence_ids=["e1"])
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 0

    async def test_mixed_counts_only_confident_supports(self, repo) -> None:
        await _ev(repo, "e1", judgment=[1.0, 0.0, 0.0])  # counts
        await _ev(repo, "e2", judgment=[0.5, 0.3, 0.2])  # excluded (shaky)
        await _ev(repo, "e3", judgment=None)  # exempt → counts
        c = Claim(
            entity_id="c1",
            objective_id="o",
            statement="x",
            evidence_ids=["e1", "e2", "e3"],
        )
        await repo.save(c)
        assert await count_supporting_sources(c, repo) == 2
