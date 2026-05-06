"""Tests for the 4-stage IBE abductive integration pipeline.

Verifies that running EnumerateCandidates → ScoreLoveliness →
ScoreLikeliness → SelectBestExplanation in sequence:
- Populates Claim.integration_candidates with the per-candidate trace
- Sets integrated_assessment / integrated_confidence /
  integrated_reasoning (the fields compute_posterior reads)
- Is idempotent at every stage
- Falls back to default candidates when enumeration produces too few
- Resets the candidate trace on demotion
"""

import pytest

from ..entities import Claim, ClaimStage, Evidence, Objective
from ..operations.base import OperationInput
from ..operations.integration import (
    EnumerateCandidatesOperation,
    ScoreLikelinessOperation,
    ScoreLovelinessOperation,
    SelectBestExplanationOperation,
)


def _make_objective(obj_id: str = "obj-1") -> Objective:
    return Objective(
        entity_id=obj_id,
        objective_id=obj_id,
        description="Test objective",
        phase="claims_done",
    )


async def _run_full_ibe(repo, fake_runner, claim_id: str = "cl-1") -> Claim:
    """Run all four IBE stages on a single claim and return the updated claim."""
    for op_cls, op_name in [
        (EnumerateCandidatesOperation, "enumerate_candidates"),
        (ScoreLovelinessOperation, "score_loveliness"),
        (ScoreLikelinessOperation, "score_likeliness"),
        (SelectBestExplanationOperation, "select_best_explanation"),
    ]:
        op = op_cls(repo=repo, agent_runner=fake_runner, embedding_model="test")
        work = OperationInput(entity_id=claim_id, entity_type="claim", operation=op_name)
        await op.execute(work)
    return await repo.get("claim", claim_id)


class TestIBEIntegrationPipeline:
    async def test_full_pipeline_sets_integrated_fields(self, repo, fake_runner):
        """End-to-end IBE produces verdict + confidence + reasoning + candidate trace."""
        await repo.save(_make_objective())
        ev = Evidence(
            entity_id="ev-1",
            objective_id="obj-1",
            extracted=True,
            extracted_content="Test evidence content",
            support_judgment="no_bearing",
        )
        await repo.save(ev)

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=["ev-1"],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.8,
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)

        # Integrated fields are populated by Stage 4
        assert updated.integrated_assessment in ("supports", "contradicts", "insufficient")
        assert updated.integrated_confidence is not None
        assert 0.0 <= updated.integrated_confidence <= 1.0
        assert updated.integrated_reasoning is not None

        # Per-candidate trace is preserved
        assert len(updated.integration_candidates) >= 2
        assert all(c.loveliness is not None for c in updated.integration_candidates)
        assert all(c.likeliness is not None for c in updated.integration_candidates)
        # Exactly one chosen, exactly one runner-up
        chosen = [c for c in updated.integration_candidates if c.chosen]
        runner_up = [c for c in updated.integration_candidates if c.runner_up]
        assert len(chosen) == 1
        assert len(runner_up) == 1
        # Gap fields populated on the chosen candidate
        assert chosen[0].gap_loveliness is not None
        assert chosen[0].gap_likeliness is not None

    async def test_select_is_idempotent(self, repo, fake_runner):
        """SelectBestExplanation skips a claim that already has a verdict."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
        )
        await repo.save(claim)

        op = SelectBestExplanationOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="select_best_explanation",
            )
        )
        assert result.success
        assert result.did_work is False
        assert "already" in result.message.lower()

    async def test_enumerate_seeds_defaults_without_agent_runner(self, repo):
        """No agent runner → enumerate falls back to default 3 candidates."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = EnumerateCandidatesOperation(
            repo=repo, agent_runner=None, embedding_model="test"
        )
        result = await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="enumerate_candidates",
            )
        )
        assert result.success

        updated = await repo.get("claim", "cl-1")
        # Default set has 3 verdicts: supports / contradicts / insufficient
        assert len(updated.integration_candidates) == 3
        verdicts = {c.verdict for c in updated.integration_candidates}
        assert verdicts == {"supports", "contradicts", "insufficient"}

    async def test_pipeline_handles_no_evidence(self, repo, fake_runner):
        """A claim with no evidence still completes IBE (defaults + scoring)."""
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            adversarial_checked=True,
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)
        assert updated.integrated_assessment is not None
        assert len(updated.integration_candidates) >= 2

    async def test_demotion_resets_integration_candidates(self, repo):
        """record_demotion clears integrated_* fields AND integration_candidates."""
        from ..entities.claim import CandidateRecord

        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="supports",
            integrated_confidence=0.8,
            integrated_reasoning="Test reasoning",
            integration_candidates=[
                CandidateRecord(
                    candidate_id="A",
                    verdict="supports",
                    description="test",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        claim.record_demotion(ClaimStage.HYPOTHESIS, "Test demotion")

        assert claim.integrated_assessment is None
        assert claim.integrated_confidence is None
        assert claim.integrated_reasoning is None
        assert claim.integration_candidates == []


class TestAdversarialConfidenceCap:
    """Unit tests for ``_adversarial_confidence_cap``.

    Three zones (Option A — soft tri-state). The cap is what
    ``SelectBestExplanationOperation`` enforces on
    ``integrated_confidence`` so the IBE chain's verdict cannot
    over-commit when adversarial search found non-trivial counter-
    evidence.
    """

    def test_no_adversarial_signal_no_cap(self) -> None:
        from ..operations.integration import _adversarial_confidence_cap

        assert _adversarial_confidence_cap(None) == 1.0

    def test_survived_zone_no_cap(self) -> None:
        from ..operations.integration import _adversarial_confidence_cap

        # At and above 0.7 → no cap.
        assert _adversarial_confidence_cap(0.7) == 1.0
        assert _adversarial_confidence_cap(0.85) == 1.0
        assert _adversarial_confidence_cap(1.0) == 1.0

    def test_refuted_zone_hard_cap(self) -> None:
        from ..operations.integration import _adversarial_confidence_cap

        # Below 0.3 → hard cap at 0.5. Refuted claims should normally
        # be demoted before reaching IBE; the cap is the safety net.
        assert _adversarial_confidence_cap(0.0) == 0.5
        assert _adversarial_confidence_cap(0.15) == 0.5
        assert _adversarial_confidence_cap(0.29) == 0.5

    def test_contested_zone_linear_interpolation(self) -> None:
        from ..operations.integration import _adversarial_confidence_cap

        # Linear from 0.5 (at refuted threshold 0.3) to 1.0 (at
        # survived threshold 0.7).
        assert _adversarial_confidence_cap(0.3) == 0.5
        # Midpoint of contested band: balance=0.5 → cap=0.75
        assert _adversarial_confidence_cap(0.5) == 0.75
        # 1/4 of the way through the band: balance=0.4 → cap=0.625
        assert _adversarial_confidence_cap(0.4) == 0.625
        # Just below survived: balance=0.69 → cap≈0.9875
        assert abs(_adversarial_confidence_cap(0.69) - 0.9875) < 1e-9


class TestSelectBestExplanationAppliesCap:
    """Integration tests: the caps actually clip ``integrated_confidence``.

    With Phase C added, two caps fire in sequence: the adversarial
    cap (Phase A — based on adversarial_balance) and the framing-tie
    cap (Phase C — based on chosen-vs-best-opposing-candidate
    loveliness gap). The fake runner produces all candidates with
    equal loveliness=0.78, so the framing-tie cap fires hard
    (gap=0 → cap=0.5) on every run regardless of adversarial_balance.
    These integration tests verify that both caps fire correctly and
    annotate the reasoning appropriately. Direct unit tests for each
    cap function are in TestAdversarialConfidenceCap and
    TestFramingTieCap below; those exercise the cap logic with
    crafted CandidateRecord inputs that don't suffer the fake runner's
    tied-loveliness limitation.
    """

    async def test_survived_balance_only_framing_tie_clips(
        self, repo, fake_runner
    ) -> None:
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.8,  # SURVIVED — adv cap=1.0
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)
        # Adv cap doesn't fire. Framing-tie cap fires (tied loveliness
        # in fake runner) → cap=0.5. min(1.0, 0.5) clips 0.75 → 0.5.
        assert updated.integrated_confidence == 0.5
        assert "Adversarial cap applied" not in (
            updated.integrated_reasoning or ""
        )
        assert "Framing-tie cap applied" in (updated.integrated_reasoning or "")

    async def test_contested_balance_both_caps_fire(
        self, repo, fake_runner
    ) -> None:
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.4,  # CONTESTED — adv cap=0.625
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)
        # adv_cap=0.625, ft_cap=0.5; min=0.5 clips 0.75 → 0.5.
        assert updated.integrated_confidence == 0.5
        assert "Adversarial cap applied" in (updated.integrated_reasoning or "")
        assert "Framing-tie cap applied" in (updated.integrated_reasoning or "")

    async def test_refuted_balance_hard_caps(
        self, repo, fake_runner
    ) -> None:
        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            adversarial_checked=True,
            adversarial_balance=0.15,  # REFUTED — adv cap=0.5
        )
        await repo.save(claim)

        updated = await _run_full_ibe(repo, fake_runner)
        # Both caps land at 0.5 here.
        assert updated.integrated_confidence == 0.5
        assert "Adversarial cap applied" in (updated.integrated_reasoning or "")


class TestFramingTieCap:
    """Unit tests for ``_framing_tie_cap``.

    Lipton's IBE: the strength of inference-to-best-explanation is
    bounded by the loveliness gap between the chosen explanation and
    the best opposing alternative. The cap is 1.0 when chosen
    dominates by ≥ FRAMING_TIE_SATURATION_GAP, 0.5 at perfect tie,
    linear in between. No threshold tuning beyond the saturation gap
    (which mirrors the adversarial CONTESTED band's 0.4 width).
    """

    def test_no_opposing_candidate_no_cap(self) -> None:
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description="x",
            loveliness=0.8,
            likeliness=0.9,
        )
        # All other candidates are "insufficient" (not opposing).
        others = [
            CandidateRecord(
                candidate_id="B",
                verdict="insufficient",
                description="y",
                loveliness=0.3,
                likeliness=0.5,
            )
        ]
        cap, opp, gap = _framing_tie_cap(chosen, [chosen, *others])
        assert cap == 1.0
        assert opp is None
        assert gap is None

    def test_chosen_dominates_no_cap(self) -> None:
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description="x",
            loveliness=0.9,
            likeliness=0.9,
        )
        opposing = CandidateRecord(
            candidate_id="B",
            verdict="contradicts",
            description="y",
            loveliness=0.3,  # gap = 0.6 > 0.4 saturation
            likeliness=0.7,
        )
        cap, opp, gap = _framing_tie_cap(chosen, [chosen, opposing])
        assert cap == 1.0
        assert opp is opposing
        assert gap == pytest.approx(0.6)

    def test_perfect_tie_caps_at_half(self) -> None:
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description="x",
            loveliness=0.8,
            likeliness=0.9,
        )
        opposing = CandidateRecord(
            candidate_id="B",
            verdict="contradicts",
            description="y",
            loveliness=0.8,  # perfect tie
            likeliness=0.7,
        )
        cap, opp, gap = _framing_tie_cap(chosen, [chosen, opposing])
        assert cap == 0.5
        assert opp is opposing
        assert gap == 0.0

    def test_partial_gap_linear_interp(self) -> None:
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description="x",
            loveliness=0.85,
            likeliness=0.9,
        )
        opposing = CandidateRecord(
            candidate_id="B",
            verdict="contradicts",
            description="y",
            loveliness=0.65,  # gap = 0.2 → cap = 0.5 + 0.2/0.4 * 0.5 = 0.75
            likeliness=0.7,
        )
        cap, opp, gap = _framing_tie_cap(chosen, [chosen, opposing])
        assert cap == pytest.approx(0.75)
        assert opp is opposing

    def test_picks_highest_loveliness_opposing_not_runner_up(self) -> None:
        """The cap looks at the FULL candidate set, not just the named
        runner-up. Case 847 v22 rep 3 showed the chain sometimes picks
        a same-direction runner-up while a strongly opposing candidate
        exists with high loveliness — that opposing candidate is the
        signal we care about for framing-tie detection."""
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description="x",
            loveliness=0.85,
            likeliness=0.9,
            chosen=True,
        )
        # Runner-up is same-direction (would not trigger naive
        # chosen-vs-runner-up gap detection).
        runner_up_same = CandidateRecord(
            candidate_id="B",
            verdict="supports_refined",
            description="y",
            loveliness=0.78,
            likeliness=0.85,
            runner_up=True,
        )
        # Strong opposing candidate that wasn't picked as runner-up.
        opposing = CandidateRecord(
            candidate_id="D",
            verdict="contradicts_refined",
            description="z",
            loveliness=0.7,  # gap = 0.15 → cap = 0.5 + 0.15/0.4 * 0.5 ≈ 0.6875
            likeliness=0.9,
        )
        cap, opp, gap = _framing_tie_cap(
            chosen, [chosen, runner_up_same, opposing]
        )
        # Cap is from the OPPOSING candidate, not the same-direction
        # runner-up.
        assert opp is opposing
        assert gap == pytest.approx(0.15)
        assert cap == pytest.approx(0.6875)

    def test_insufficient_chosen_no_cap(self) -> None:
        """When chosen is insufficient, there's no canonical-direction
        commitment to dampen — return cap=1.0."""
        from ..entities.claim import CandidateRecord
        from ..operations.integration import _framing_tie_cap

        chosen = CandidateRecord(
            candidate_id="A",
            verdict="insufficient",
            description="x",
            loveliness=0.5,
            likeliness=0.5,
        )
        opposing = CandidateRecord(
            candidate_id="B",
            verdict="supports",
            description="y",
            loveliness=0.6,
            likeliness=0.5,
        )
        cap, _, _ = _framing_tie_cap(chosen, [chosen, opposing])
        assert cap == 1.0


class TestBalancedEnumeration:
    """Phase D: balanced enumeration ensures IBE always considers rival
    canonical verdicts. Lipton's IBE works by COMPARATIVE selection
    across rival framings; that comparison is only meaningful if rival
    framings exist in the candidate set. The enumerator LLM has a
    confirmation-leaning bias on many claims (case 847 v22 trace: 4/5
    reps' candidate sets contained no contradicts-framed candidate).
    Without rival candidates, the framing-tie cap (Phase C) has nothing
    to grab onto. After this fix, the enumeration step always produces
    at least one candidate per canonical verdict
    (supports / contradicts / insufficient) — augmenting from defaults
    when the LLM's output skipped one.
    """

    async def test_llm_only_supports_balanced_adds_contradicts_and_insufficient(
        self, repo
    ) -> None:
        """Simulates the case 847 v22 pattern: the enumerator produces
        only supports-framed candidates. Balanced augmentation should
        add a contradicts and an insufficient candidate so the chain
        considers all rivals."""
        from .conftest import FakeAgentRunner

        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        # Override the proposer to always return a supports framing.
        # Loop runs 5 iterations (done=False) → 5 supports candidates.
        biased_runner = FakeAgentRunner(
            overrides={
                "epistemic_propose_one_candidate": {
                    "done": False,
                    "verdict": "supports",
                    "description": "Biased toward supports framing",
                }
            }
        )
        op = EnumerateCandidatesOperation(
            repo=repo, agent_runner=biased_runner, embedding_model="test"
        )
        await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="enumerate_candidates",
            )
        )

        from ..operations.integration import _verdict_to_canonical

        updated = await repo.get("claim", "cl-1")
        canonicals = {
            _verdict_to_canonical(c.verdict) for c in updated.integration_candidates
        }
        # All three canonical verdicts present after balanced augmentation.
        assert canonicals == {"supports", "contradicts", "insufficient"}
        # Should have 5 LLM-generated supports + 2 default rivals = 7 total.
        n_supports = sum(
            1
            for c in updated.integration_candidates
            if _verdict_to_canonical(c.verdict) == "supports"
        )
        n_contradicts = sum(
            1
            for c in updated.integration_candidates
            if _verdict_to_canonical(c.verdict) == "contradicts"
        )
        n_insufficient = sum(
            1
            for c in updated.integration_candidates
            if _verdict_to_canonical(c.verdict) == "insufficient"
        )
        assert n_supports == 5
        assert n_contradicts == 1
        assert n_insufficient == 1
        # The added defaults are tagged in their description so the
        # downstream trace is auditable.
        added = [
            c
            for c in updated.integration_candidates
            if "[Balanced enumeration:" in (c.description or "")
        ]
        assert len(added) == 2

    async def test_llm_balanced_no_augmentation_needed(self, repo) -> None:
        """When the enumerator's output already covers all canonical
        verdicts, balanced augmentation is a no-op (zero defaults
        added, descriptions don't get the [Balanced enumeration] tag)."""
        from .conftest import FakeAgentRunner

        await repo.save(_make_objective())
        claim = Claim(
            entity_id="cl-1",
            objective_id="obj-1",
            statement="Test claim",
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        # Sequence of responses: supports, contradicts, insufficient,
        # then done. Use a custom runner that cycles.
        responses = [
            {"done": False, "verdict": "supports", "description": "s"},
            {"done": False, "verdict": "contradicts", "description": "c"},
            {"done": False, "verdict": "insufficient", "description": "i"},
            {"done": True, "verdict": None, "description": None},
        ]
        call_idx = [0]

        class CyclingRunner(FakeAgentRunner):
            async def run(self, agent_name, **kwargs):
                self.calls.append((agent_name, kwargs))
                if agent_name == "epistemic_propose_one_candidate":
                    r = responses[call_idx[0]]
                    call_idx[0] += 1
                    from types import SimpleNamespace

                    return SimpleNamespace(**r)
                return await super().run(agent_name, **kwargs)

        op = EnumerateCandidatesOperation(
            repo=repo, agent_runner=CyclingRunner(), embedding_model="test"
        )
        await op.execute(
            OperationInput(
                entity_id="cl-1",
                entity_type="claim",
                operation="enumerate_candidates",
            )
        )

        from ..operations.integration import _verdict_to_canonical

        updated = await repo.get("claim", "cl-1")
        # 3 LLM-generated, balanced — no defaults added.
        assert len(updated.integration_candidates) == 3
        canonicals = {
            _verdict_to_canonical(c.verdict) for c in updated.integration_candidates
        }
        assert canonicals == {"supports", "contradicts", "insufficient"}
        # No augmentation tags.
        assert not any(
            "[Balanced enumeration:" in (c.description or "")
            for c in updated.integration_candidates
        )
