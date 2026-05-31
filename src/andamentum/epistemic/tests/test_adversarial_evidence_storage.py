"""Tests for adversarial counterargument storage as Evidence entities.

K7 (probe B4 finding): the support_judgment on adversarial-found
Evidence is determined by the impartial ``epistemic_judge_evidence``
agent, NOT hard-coded "contradicts". The hard-code used to double-
count: papers found via adversarial search were stamped as
counter-evidence regardless of whether they actually contradicted the
claim. The metformin/HbA1c run was the canonical failure — the
canonical Cochrane review of metformin for T2DM was found via
adversarial search and stored as "contradicts" because the adversarial
agent harvested its limitations section. The fix routes every
adversarial-found item through the same judge the regular evidence
flow uses; whether it ends up "supports", "contradicts", or
"no_bearing" depends on what the judge says about its actual content.
"""

import pytest
from ..entities.claim import Claim, ClaimStage
from ..entities.evidence import Evidence
from ..operations.verification import AdversarialSearchOperation
from ..operations.base import OperationInput
from .conftest import FakeAgentRunner


class _MockGatherer:
    """Minimal evidence gatherer for adversarial tests — returns one
    canned hit per query so the per-source dedup keeps exactly one
    Evidence entity through to storage."""

    def __init__(self, content: str, source_ref: str) -> None:
        self._content = content
        self._source_ref = source_ref

    async def gather(self, source_type: str, query: str):
        from ..operations.base import GatheredEvidence

        del query  # canned response regardless of query text
        return [
            GatheredEvidence(
                content=self._content,
                source_ref=self._source_ref,
                source_type=source_type,
            ),
        ]


class TestAdversarialEvidenceStorage:
    """Adversarial-found evidence is judged by the impartial judge,
    not stamped hard-coded contradicts."""

    @pytest.mark.asyncio
    async def test_judge_says_contradicts_evidence_stored_contradicts(self, repo):
        """When the impartial judge labels the adversarial-found item
        as contradicts (the typical case for genuine counter-evidence),
        the stored Evidence has support_judgment='contradicts' and the
        adversarial provenance is preserved in the reasoning text."""
        runner = FakeAgentRunner(
            overrides={
                "epistemic_judge_evidence": {
                    "verdict": "contradicts",
                    "reasoning": "Genuinely contradicts the claim.",
                }
            }
        )

        claim = Claim(
            entity_id="c-1",
            objective_id="obj-1",
            statement="Homeopathy cures infections",
            scope="Adult patients with bacterial infections.",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=["e-1"],
        )
        await repo.save(claim)

        ev = Evidence(
            entity_id="e-1",
            objective_id="obj-1",
            source_type="paper",
            source_ref="https://example.com/study1",
            extracted_content="RCT showing positive result",
            extracted=True,
            support_judgment="supports",
            quality_score=0.7,
            cluster_status="representative",
        )
        await repo.save(ev)

        op = AdversarialSearchOperation(repo=repo, agent_runner=runner)
        op.evidence_gatherer = _MockGatherer(
            content="Cochrane review finds no evidence for homeopathy",
            source_ref="https://example.com/cochrane",
        )  # type: ignore[assignment]

        result = await op.execute(
            OperationInput(
                entity_id="c-1",
                entity_type="claim",
                operation="adversarial_search",
            )
        )
        assert result.success

        all_evidence = await repo.get_evidence_for_objective("obj-1")
        adv = [
            e for e in all_evidence if e.source_ref == "https://example.com/cochrane"
        ]
        assert len(adv) == 1, (
            f"Expected exactly one stored adversarial evidence; got {len(adv)}"
        )

        ae = adv[0]
        assert ae.source_type == "web_search"
        assert ae.support_judgment == "contradicts", (
            "Judge returned 'contradicts'; stored verdict must match what the "
            "impartial judge said, not the old hard-coded value."
        )
        # Adversarial provenance preserved in reasoning text:
        assert "adversarial" in (ae.judgment_reasoning or "").lower()
        # The judge's reasoning is also surfaced (combined into the same field):
        assert "judge" in (ae.judgment_reasoning or "").lower()

        claim_after = await repo.get("claim", "c-1")
        assert ae.entity_id in claim_after.evidence_ids

    @pytest.mark.asyncio
    async def test_judge_says_supports_evidence_stored_supports(self, repo):
        """K7's load-bearing case: when the adversarial agent finds a
        paper that the impartial judge classifies as SUPPORTING the
        claim (e.g. a Cochrane review whose findings actually back the
        claim, even though the adversarial agent harvested its
        limitations section), the stored Evidence must have
        support_judgment='supports' — NOT the old hard-coded
        'contradicts'. This is the metformin/HbA1c failure we're
        fixing: the canonical supporting review used to be labeled
        contradicts and counted against the claim."""
        runner = FakeAgentRunner(
            overrides={
                "epistemic_judge_evidence": {
                    "verdict": "supports",
                    "reasoning": (
                        "The cited review's overall finding supports the claim; "
                        "the limitations section the adversarial agent picked up "
                        "is methodological caveat, not refutation."
                    ),
                }
            }
        )

        claim = Claim(
            entity_id="c-met",
            objective_id="obj-met",
            statement="Metformin reduces HbA1c in adults with type 2 diabetes",
            scope="Adults with T2DM, HbA1c outcome.",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            evidence_ids=[],
        )
        await repo.save(claim)

        op = AdversarialSearchOperation(repo=repo, agent_runner=runner)
        op.evidence_gatherer = _MockGatherer(
            content=(
                "Cochrane review CD012906: 18 studies, 10,680 participants. "
                "Metformin compared with placebo, sulphonylureas, etc. "
                "Limitations: trials were small and of short duration."
            ),
            source_ref=(
                "https://www.cochrane.org/evidence/"
                "CD012906_metformin-effective-treatment-adults-type-2-diabetes"
            ),
        )  # type: ignore[assignment]

        result = await op.execute(
            OperationInput(
                entity_id="c-met",
                entity_type="claim",
                operation="adversarial_search",
            )
        )
        assert result.success

        all_evidence = await repo.get_evidence_for_objective("obj-met")
        adv = [e for e in all_evidence if "cochrane.org" in (e.source_ref or "")]
        assert len(adv) == 1
        ae = adv[0]

        assert ae.support_judgment == "supports", (
            "K7 contract: adversarial-found evidence whose IMPARTIAL judge "
            "verdict is 'supports' must be stored as 'supports', not "
            "hard-coded 'contradicts'. If this fires, the old hard-code has "
            "been re-introduced and the metformin/HbA1c failure mode is back."
        )
        # Provenance preserved — caller can still see it came via adversarial:
        assert "adversarial" in (ae.judgment_reasoning or "").lower()
