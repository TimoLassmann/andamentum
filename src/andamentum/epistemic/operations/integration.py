"""IBE-decomposed abductive integration (Peirce + Lipton + Kahneman).

Replaces the monolithic AbductiveIntegrationOperation with a four-stage
Inference-to-Best-Explanation pipeline. Each stage is its own operation
with a focused agent and small structured output, and writes its result
incrementally into ``Claim.integration_candidates``:

1. ``EnumerateCandidatesOperation`` (Peirce, generative). Iteratively
   calls ``epistemic_propose_one_candidate`` with the running list of
   already-proposed candidates as context. Stops when the agent signals
   ``done`` or a hard cap is reached. Falls back to a default three-
   candidate set if fewer than 2 distinct candidates were produced.

2. ``ScoreLovelinessOperation`` (Lipton, evaluative). For each candidate
   on the claim, calls ``epistemic_score_candidate_loveliness`` with
   only that candidate visible — Kahneman independence is enforced by
   not passing other candidates' scores or descriptions. Calls run in
   parallel via ``asyncio.gather``.

3. ``ScoreLikelinessOperation`` (Lipton, evaluative). Mirrors stage 2
   for the likeliness score.

4. ``SelectBestExplanationOperation`` (Lipton, comparative). Calls
   ``epistemic_select_best_explanation`` with all scored candidates
   visible. Writes ``integrated_assessment`` / ``integrated_confidence``
   / ``integrated_reasoning`` on the claim and tags the chosen and
   runner-up candidate records. Computes ``gap_loveliness`` and
   ``gap_likeliness`` from persisted scores in code; the model only
   chooses, it does not arithmetic.

Operates on: Claim entities (writes integration_candidates and
integrated_* fields).
"""

from __future__ import annotations

import asyncio

from .base import BaseOperation, OperationInput, OperationResult
from ..agents.output_models import LikelinessScore, LovelinessScore
from ..entities import Claim, Evidence, Uncertainty
from ..entities.claim import CandidateRecord
from ..thresholds import ADVERSARIAL_SURVIVED_THRESHOLD


# ── Shared brief-building helper ──────────────────────────────────────


async def _build_evidence_brief(op: BaseOperation, claim: Claim) -> dict[str, object]:
    """Build the evidence summary block shared by all IBE stages.

    Returns a dict with keys: supporting_evidence, contradicting_evidence,
    no_bearing_evidence, adversarial_outcome, open_uncertainties,
    evidence_count, supporting_count, contradicting_count,
    no_bearing_count. Same shape the old monolithic prompt consumed.

    Filters to representative evidence, caps at LLM_PANEL_CAP top-quality
    reps, and annotates each with ``cluster_size`` so the agent can read
    cluster size as a corroboration signal in its reasoning. (The
    aggregation, comparison, and difference-method-style reasoning all
    happen inside the agent prompt, not in this code; we provide the
    structured input.)
    """
    from .claims import LLM_PANEL_CAP, top_n_representatives

    candidates: list[Evidence] = []
    for eid in claim.evidence_ids:
        ev = await op.repo.get("evidence", eid)
        if not isinstance(ev, Evidence) or ev.invalidated:
            continue
        if ev.cluster_status in ("corroborative", "deferred"):
            continue
        candidates.append(ev)

    supports_items: list[str] = []
    contradicts_items: list[str] = []
    no_bearing_items: list[str] = []
    for ev in top_n_representatives(candidates, LLM_PANEL_CAP):
        content = ev.extracted_content or ""
        cluster_size = max(1, getattr(ev, "corroboration_count", 1) or 1)
        summary = f"[{ev.source_type}, cluster_size={cluster_size}] {content}"
        if ev.support_judgment == "supports":
            supports_items.append(summary)
        elif ev.support_judgment == "contradicts":
            contradicts_items.append(summary)
        else:
            no_bearing_items.append(summary)

    adversarial_text = "Adversarial search has NOT been conducted."
    if claim.adversarial_checked and claim.adversarial_balance is not None:
        if claim.adversarial_balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
            adversarial_text = (
                f"Adversarial search conducted: NO strong counterevidence "
                f"found (balance: {claim.adversarial_balance:.2f}). "
                f"The claim survived active refutation attempts."
            )
        else:
            adversarial_text = (
                f"Adversarial search found significant counterevidence "
                f"(balance: {claim.adversarial_balance:.2f})."
            )

    uncertainties = await op.repo.query("uncertainty", objective_id=claim.objective_id)
    open_blocking = [
        u
        for u in uncertainties
        if isinstance(u, Uncertainty)
        and claim.entity_id in u.affected_claim_ids
        and u.resolution is None
        and u.is_blocking
    ]
    unc_text = (
        "\n".join(f"- {u.description}" for u in open_blocking)
        if open_blocking
        else "No unresolved blocking uncertainties."
    )

    return {
        "supporting_evidence": "\n\n".join(supports_items) or "None found.",
        "contradicting_evidence": "\n\n".join(contradicts_items) or "None found.",
        "no_bearing_evidence": "\n\n".join(no_bearing_items) or "None.",
        "evidence_summary": "\n\n".join(
            supports_items + contradicts_items + no_bearing_items
        )
        or "No representative evidence available.",
        "adversarial_outcome": adversarial_text,
        "open_uncertainties": unc_text,
        "evidence_count": len(supports_items)
        + len(contradicts_items)
        + len(no_bearing_items),
        "supporting_count": len(supports_items),
        "contradicting_count": len(contradicts_items),
        "no_bearing_count": len(no_bearing_items),
    }


# ── Stage 1: Enumerate candidates (Peirce, generative) ───────────────


# IBE candidate budget. Five slots A-E, two minimum viable.
#
# A previous Phase-1-efficiency cut reduced this to three slots
# A-C to halve IBE chain cost (each candidate is scored on
# loveliness + likeliness, so 5 → 3 saves 4 LLM calls per claim).
# Reverted (2026-05-02) after benchmark runs showed convergence
# degradation: with fewer candidates the abductive search had less
# diversity, claims more often produced wishy-washy verdicts that
# couldn't carry the IBE chain to a decisive answer. Restored to
# A-E.
_CANDIDATE_IDS = ["A", "B", "C", "D", "E"]
_MIN_VIABLE_CANDIDATES = 2


def _default_candidates() -> list[CandidateRecord]:
    """Fallback set when the enumerator fails to produce >=2 candidates.

    Seeds the three default verdict directions so stages 2-4 always have
    something to score and compare. Keeps the pipeline alive on small or
    flaky local models.
    """
    return [
        CandidateRecord(
            candidate_id="A",
            verdict="supports",
            description=(
                "Default candidate (enumerator returned no usable output): "
                "the evidence pattern, taken at face value, makes the claim "
                "more likely true."
            ),
        ),
        CandidateRecord(
            candidate_id="B",
            verdict="contradicts",
            description=(
                "Default candidate (enumerator returned no usable output): "
                "the evidence pattern makes the claim more likely false."
            ),
        ),
        CandidateRecord(
            candidate_id="C",
            verdict="insufficient",
            description=(
                "Default candidate (enumerator returned no usable output): "
                "the evidence is too sparse, conflicted, or tangential to "
                "commit either way."
            ),
        ),
    ]


class EnumerateCandidatesOperation(BaseOperation):
    """Iteratively enumerate distinct candidate verdicts (Peirce).

    Each agent call sees the running list of already-proposed candidates
    so it can diversify. Generative role; priors REQUIRED. Stops on
    ``done=true`` or after ``_MAX_CANDIDATES`` calls. Falls back to the
    default three-verdict set when enumeration fails to surface >=2
    distinct candidates.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        if claim.integration_candidates:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message=f"Already enumerated ({len(claim.integration_candidates)} candidates)",
                did_work=False,
            )

        if not self.agent_runner:
            claim.integration_candidates = _default_candidates()
            await self.repo.save(claim)
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Enumeration skipped (no agent runner) — seeded defaults",
            )

        brief = await _build_evidence_brief(self, claim)

        proposed: list[CandidateRecord] = []
        for slot in _CANDIDATE_IDS:
            already_text = (
                "\n".join(
                    f"- {c.candidate_id} ({c.verdict}): {c.description}"
                    for c in proposed
                )
                if proposed
                else "(none yet — propose the first candidate)"
            )
            result = await self.run_agent(
                "epistemic_propose_one_candidate",
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                supporting_evidence=brief["supporting_evidence"],
                contradicting_evidence=brief["contradicting_evidence"],
                no_bearing_evidence=brief["no_bearing_evidence"],
                adversarial_outcome=brief["adversarial_outcome"],
                open_uncertainties=brief["open_uncertainties"],
                already_proposed=already_text,
            )

            if result.done or not result.verdict or not result.description:
                break

            proposed.append(
                CandidateRecord(
                    candidate_id=slot,
                    verdict=str(result.verdict),
                    description=str(result.description),
                )
            )

        if len(proposed) < _MIN_VIABLE_CANDIDATES:
            proposed = _default_candidates()
            message = (
                f"Enumeration produced <{_MIN_VIABLE_CANDIDATES} candidates; "
                f"seeded {len(proposed)} defaults"
            )
        else:
            message = f"Enumerated {len(proposed)} candidates"

        claim.integration_candidates = proposed
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=message,
        )


# ── Stage 2: Score loveliness (Lipton, evaluative) ────────────────────


class ScoreLovelinessOperation(BaseOperation):
    """Score each candidate's explanatory virtue (Lipton's loveliness).

    Each candidate is scored independently — the agent never sees other
    candidates' scores. Calls fan out in parallel via asyncio.gather.
    Idempotent: skips candidates already populated.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        if not claim.integration_candidates:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="No candidates to score (enumerate first)",
                did_work=False,
            )

        unscored = [c for c in claim.integration_candidates if c.loveliness is None]
        if not unscored:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="All candidates already scored on loveliness",
                did_work=False,
            )

        if not self.agent_runner:
            for c in unscored:
                c.loveliness = 0.5
                c.loveliness_reasoning = "No agent runner — neutral default."
            await self.repo.save(claim)
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message=f"Loveliness skipped (no agent runner) — defaulted {len(unscored)}",
            )

        brief = await _build_evidence_brief(self, claim)

        async def _score_one(
            cand: CandidateRecord,
        ) -> tuple[CandidateRecord, LovelinessScore]:
            result = await self.run_agent(
                "epistemic_score_candidate_loveliness",
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                candidate_verdict=cand.verdict,
                candidate_description=cand.description,
                evidence_summary=brief["evidence_summary"],
            )
            return cand, result

        results = await asyncio.gather(*[_score_one(c) for c in unscored])
        for cand, score in results:
            cand.loveliness = float(score.loveliness)
            cand.loveliness_reasoning = str(score.reasoning)

        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Scored loveliness for {len(unscored)} candidates",
        )


# ── Stage 3: Score likeliness (Lipton, evaluative) ────────────────────


class ScoreLikelinessOperation(BaseOperation):
    """Score each candidate's fit-with-evidence (Lipton's likeliness).

    Mirrors loveliness scoring but evaluates fit rather than virtue.
    Each candidate is scored independently (Kahneman). Parallel within
    a claim. Idempotent.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        if not claim.integration_candidates:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="No candidates to score (enumerate first)",
                did_work=False,
            )

        unscored = [c for c in claim.integration_candidates if c.likeliness is None]
        if not unscored:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="All candidates already scored on likeliness",
                did_work=False,
            )

        if not self.agent_runner:
            for c in unscored:
                c.likeliness = 0.5
                c.likeliness_reasoning = "No agent runner — neutral default."
            await self.repo.save(claim)
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message=f"Likeliness skipped (no agent runner) — defaulted {len(unscored)}",
            )

        brief = await _build_evidence_brief(self, claim)

        async def _score_one(
            cand: CandidateRecord,
        ) -> tuple[CandidateRecord, LikelinessScore]:
            result = await self.run_agent(
                "epistemic_score_candidate_likeliness",
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                candidate_verdict=cand.verdict,
                candidate_description=cand.description,
                supporting_evidence=brief["supporting_evidence"],
                contradicting_evidence=brief["contradicting_evidence"],
                no_bearing_evidence=brief["no_bearing_evidence"],
                adversarial_outcome=brief["adversarial_outcome"],
                open_uncertainties=brief["open_uncertainties"],
            )
            return cand, result

        results = await asyncio.gather(*[_score_one(c) for c in unscored])
        for cand, score in results:
            cand.likeliness = float(score.likeliness)
            cand.likeliness_reasoning = str(score.reasoning)

        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Scored likeliness for {len(unscored)} candidates",
        )


# ── Stage 4: Select best explanation (Lipton, comparative) ────────────


def _verdict_to_canonical(verdict: str) -> str:
    """Collapse refined verdicts to the canonical {supports, contradicts,
    insufficient} set for the integrated_assessment field. Refined
    detail survives in the candidate record's description."""
    if verdict.startswith("supports"):
        return "supports"
    if verdict.startswith("contradicts"):
        return "contradicts"
    return "insufficient"


class SelectBestExplanationOperation(BaseOperation):
    """Pick the best candidate by combined loveliness × likeliness, with
    gap-based confidence (Lipton's comparative selection).

    Writes ``integrated_assessment`` / ``integrated_confidence`` /
    ``integrated_reasoning`` on the claim and tags the chosen and
    runner-up candidate records. Computes ``gap_loveliness`` and
    ``gap_likeliness`` deterministically from persisted scores —
    the agent only chooses; arithmetic stays in code.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)
        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        if claim.integrated_assessment is not None:
            return OperationResult(
                success=True,
                entity_id=claim.entity_id,
                message="Already selected",
                did_work=False,
            )

        if not claim.integration_candidates:
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="No candidates to select from (enumerate first)",
                did_work=False,
            )

        # All candidates must be scored before selection
        if any(
            c.loveliness is None or c.likeliness is None
            for c in claim.integration_candidates
        ):
            return OperationResult(
                success=False,
                entity_id=claim.entity_id,
                message="Cannot select: not all candidates have loveliness + likeliness scores",
                did_work=False,
            )

        if not self.agent_runner:
            # Deterministic fallback: pick the candidate with highest
            # loveliness * likeliness; runner-up is second-highest.
            ranked = sorted(
                claim.integration_candidates,
                key=lambda c: (c.loveliness or 0.0) * (c.likeliness or 0.0),
                reverse=True,
            )
            chosen = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
            confidence = 0.5
            reasoning = "No agent runner — selected by deterministic loveliness × likeliness ranking."
        else:
            candidates_block = "\n\n".join(
                (
                    f"### Candidate {c.candidate_id} ({c.verdict})\n"
                    f"Description: {c.description}\n"
                    f"Loveliness: {c.loveliness:.2f} — {c.loveliness_reasoning or ''}\n"
                    f"Likeliness: {c.likeliness:.2f} — {c.likeliness_reasoning or ''}"
                )
                for c in claim.integration_candidates
            )
            result = await self.run_agent(
                "epistemic_select_best_explanation",
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                candidates=candidates_block,
            )

            chosen = next(
                (
                    c
                    for c in claim.integration_candidates
                    if c.candidate_id == result.chosen_candidate_id
                ),
                None,
            )
            runner_up = next(
                (
                    c
                    for c in claim.integration_candidates
                    if c.candidate_id == result.runner_up_candidate_id
                ),
                None,
            )

            if chosen is None:
                return OperationResult(
                    success=False,
                    entity_id=claim.entity_id,
                    message=(
                        f"Selection failed: agent named candidate "
                        f"'{result.chosen_candidate_id}' which is not in the candidate set"
                    ),
                    did_work=False,
                )
            if runner_up is None:
                # Fallback: pick second-best by score product if agent's
                # runner-up is invalid
                ranked = sorted(
                    [c for c in claim.integration_candidates if c is not chosen],
                    key=lambda c: (c.loveliness or 0.0) * (c.likeliness or 0.0),
                    reverse=True,
                )
                runner_up = ranked[0] if ranked else chosen

            confidence = float(result.confidence)
            reasoning = str(result.reasoning)

        # Tag candidates
        for c in claim.integration_candidates:
            c.chosen = c is chosen
            c.runner_up = c is runner_up

        # Code-side gap derivation. The agent calibrated its confidence
        # against the gap; we record the gap separately so a reader can
        # verify the alignment.
        if chosen is not runner_up:
            chosen.gap_loveliness = (chosen.loveliness or 0.0) - (
                runner_up.loveliness or 0.0
            )
            chosen.gap_likeliness = (chosen.likeliness or 0.0) - (
                runner_up.likeliness or 0.0
            )
        else:
            chosen.gap_loveliness = 0.0
            chosen.gap_likeliness = 0.0

        # Write to the integration fields compute_posterior reads
        claim.integrated_assessment = _verdict_to_canonical(chosen.verdict)
        claim.integrated_confidence = max(0.0, min(1.0, confidence))
        claim.integrated_reasoning = reasoning
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"Selected {chosen.candidate_id} ({chosen.verdict}) over "
                f"{runner_up.candidate_id}; verdict={claim.integrated_assessment}, "
                f"confidence={claim.integrated_confidence:.2f}"
            ),
        )
