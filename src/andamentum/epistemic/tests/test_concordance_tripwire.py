"""Tests for the concordance tripwire (harness-backstops item 1).

The posterior's direction rests on the IBE verdict. When that direction is
contradicted by BOTH independent legs — the confidence-weighted counting mass
AND the adversarial band — the claim's directional verdict is suspended (treated
as insufficient). Suspend-only: it never flips the direction. Covers the pure
predicate and its effect on ``compute_posterior`` (rule-blind path) plus the
combiner's ``suspended_ids``.
"""

from __future__ import annotations

from ..confidence import (
    compute_posterior,
    ibe_contradicted_by_independent_signals,
)
from ..entities import Claim, Evidence, Objective
from ..graph.combination import combine_claim_verdicts

OBJ_ID = "test-concordance-obj"


class TestPredicate:
    def test_supports_refuted_and_mass_contradicts_fires(self) -> None:
        assert ibe_contradicted_by_independent_signals("supports", 0.2, 0.9, 0.1)

    def test_contradicts_survived_and_mass_supports_fires(self) -> None:
        assert ibe_contradicted_by_independent_signals("contradicts", 0.9, 0.2, 0.85)

    def test_only_one_leg_disagrees_does_not_fire(self) -> None:
        # Mass contradicts but adversarial survived (agrees with supports) → no.
        assert not ibe_contradicted_by_independent_signals("supports", 0.2, 0.9, 0.85)
        # Adversarial refuted but mass supports (agrees with supports) → no.
        assert not ibe_contradicted_by_independent_signals("supports", 0.9, 0.2, 0.1)

    def test_no_adversarial_leg_never_fires(self) -> None:
        assert not ibe_contradicted_by_independent_signals("supports", 0.2, 0.9, None)

    def test_insufficient_verdict_never_fires(self) -> None:
        assert not ibe_contradicted_by_independent_signals(
            "insufficient", 0.2, 0.9, 0.1
        )


class TestCombinerSuspension:
    def _claim(self, cid: str, verdict: str) -> Claim:
        return Claim(
            entity_id=cid,
            objective_id=OBJ_ID,
            statement="c",
            integrated_assessment=verdict,
            integrated_confidence=0.9,
        )

    def test_suspended_claim_contributes_neutral(self) -> None:
        claims = [self._claim("c1", "supports")]
        base = combine_claim_verdicts(claims, "AND")
        suspended = combine_claim_verdicts(claims, "AND", suspended_ids={"c1"})
        assert base.posterior is not None and base.posterior > 0.8
        assert suspended.posterior == 0.5

    def test_empty_suspended_is_inert(self) -> None:
        claims = [self._claim("c1", "supports")]
        a = combine_claim_verdicts(claims, "AND")
        b = combine_claim_verdicts(claims, "AND", suspended_ids=set())
        assert a.posterior == b.posterior


class TestPosteriorIntegration:
    async def test_wrong_direction_ibe_suspended(self, repo) -> None:
        """IBE says supports, but evidence nets contradicting AND adversarial is
        REFUTED → the decisive support is withheld (posterior → 0.5)."""
        await repo.save(
            Objective(
                entity_id=OBJ_ID,
                objective_id=OBJ_ID,
                description="q",
                question_type="verificatory",
            )
        )
        for i in range(3):
            await repo.save(
                Evidence(
                    entity_id=f"e{i}",
                    objective_id=OBJ_ID,
                    source_type="web_search",
                    source_ref="s",
                    extracted_content="contradicting finding",
                    extracted=True,
                    support_judgment="contradicts",
                )
            )
        claim = Claim(
            entity_id="c1",
            objective_id=OBJ_ID,
            statement="X",
            evidence_ids=["e0", "e1", "e2"],
            integrated_assessment="supports",
            integrated_confidence=0.9,
        )
        claim.adversarial_balance = 0.1  # REFUTED
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior == 0.5

    async def test_agreeing_adversarial_not_suspended(self, repo) -> None:
        """Same contradicting mass, but adversarial SURVIVED (agrees with the
        supports verdict) → only one leg disagrees → NOT suspended → directional."""
        await repo.save(
            Objective(
                entity_id=OBJ_ID,
                objective_id=OBJ_ID,
                description="q",
                question_type="verificatory",
            )
        )
        for i in range(3):
            await repo.save(
                Evidence(
                    entity_id=f"e{i}",
                    objective_id=OBJ_ID,
                    source_type="web_search",
                    source_ref="s",
                    extracted_content="contradicting finding",
                    extracted=True,
                    support_judgment="contradicts",
                )
            )
        claim = Claim(
            entity_id="c1",
            objective_id=OBJ_ID,
            statement="X",
            evidence_ids=["e0", "e1", "e2"],
            integrated_assessment="supports",
            integrated_confidence=0.9,
        )
        claim.adversarial_balance = 0.85  # SURVIVED — agrees with supports
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior > 0.8  # directional supports verdict stands
