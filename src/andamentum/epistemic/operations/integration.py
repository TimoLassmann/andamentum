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
from dataclasses import dataclass

from .base import BaseOperation, OperationInput, OperationResult
from ..agents.output_models import LikelinessScore, LovelinessScore
from ..entities import Claim, Evidence, Uncertainty
from ..entities.claim import CandidateRecord
from ..thresholds import (
    ADVERSARIAL_REFUTED_THRESHOLD,
    ADVERSARIAL_SURVIVED_THRESHOLD,
    FRAMING_TIE_SATURATION_GAP,
)


def _adversarial_confidence_cap(adversarial_balance: float | None) -> float:
    """Cap on ``integrated_confidence`` from the adversarial-balance signal.

    Three zones (Option A — soft tri-state on the existing thresholds):

      * ``balance >= ADVERSARIAL_SURVIVED_THRESHOLD`` (≥ 0.7): SURVIVED.
        No cap (returns 1.0).
      * ``balance < ADVERSARIAL_REFUTED_THRESHOLD`` (< 0.3): REFUTED.
        Hard cap at 0.5 — the TMS gate normally demotes refuted claims
        before they reach IBE; if one slips through, the cap prevents
        a confident directional certification on a refuted foundation.
      * In between (CONTESTED, Lakatosian): linear interpolation from
        0.5 (at the refuted threshold) to 1.0 (at the survived
        threshold). A balance of 0.5 caps confidence at 0.75; a balance
        of 0.31 caps at ~0.51.

    No cap when ``adversarial_balance`` is None (adversarial search not
    run, e.g. for question types where adversarial track is SKIP).
    """
    if adversarial_balance is None:
        return 1.0
    if adversarial_balance >= ADVERSARIAL_SURVIVED_THRESHOLD:
        return 1.0
    if adversarial_balance < ADVERSARIAL_REFUTED_THRESHOLD:
        return 0.5
    band_width = ADVERSARIAL_SURVIVED_THRESHOLD - ADVERSARIAL_REFUTED_THRESHOLD
    band_position = (adversarial_balance - ADVERSARIAL_REFUTED_THRESHOLD) / band_width
    return 0.5 + band_position * 0.5


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
    selected = await top_n_representatives(
        candidates,
        LLM_PANEL_CAP,
        claim_text=claim.statement,
        embedding_model=op.embedding_model,
    )
    for ev in selected:
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


# IBE candidate budget. The LLM enumerator iterates the first
# ``_MAX_LLM_ENUM_CANDIDATES`` (=5) slots; balanced enumeration (Phase D)
# reuses the remaining slots to seed missing canonical verdicts when
# the LLM's output didn't cover all three. Single source of truth so
# the two paths can't accidentally overlap on ID assignment.
#
# A previous Phase-1-efficiency cut reduced LLM slots to three
# (A-C) to halve IBE chain cost (each candidate is scored on
# loveliness + likeliness, so 5 → 3 saves 4 LLM calls per claim).
# Reverted (2026-05-02) after benchmark runs showed convergence
# degradation: with fewer candidates the abductive search had less
# diversity, claims more often produced wishy-washy verdicts that
# couldn't carry the IBE chain to a decisive answer.
_CANDIDATE_ID_POOL: list[str] = list("ABCDEFGHIJ")
_MAX_LLM_ENUM_CANDIDATES = 5
_CANDIDATE_IDS = _CANDIDATE_ID_POOL[:_MAX_LLM_ENUM_CANDIDATES]
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
            # Balanced enumeration (Phase D — Lipton). IBE selects the
            # best explanation by COMPARATIVE evaluation across rival
            # framings; that comparison is only meaningful if rival
            # framings exist in the candidate set. The enumerator LLM
            # has a confirmation-leaning bias on many claims (case 847
            # v22 trace: 4/5 reps' candidate sets contained no
            # contradicts-framed candidate; case 957 v20: framings
            # rolled all-supports or all-contradicts depending on the
            # run). Without rival candidates the framing-tie cap
            # (Phase C) has nothing to grab onto, and the chain
            # commits confidently to whichever direction the
            # enumerator happened to roll.
            #
            # Fix: ensure the final candidate set covers all three
            # canonical verdicts. If the enumerator skipped one
            # (canonical verdicts are derived via _verdict_to_canonical
            # so refined verdicts count as their canonical parent),
            # append a default candidate for the missing verdict. The
            # defaults go through loveliness/likeliness scoring like
            # any other candidate; if the literature genuinely doesn't
            # support that framing, the scorer will rate it low and
            # the framing-tie cap won't fire — but the chain has
            # _considered_ the rival, which is Lipton's requirement.
            canonical_present = {
                _verdict_to_canonical(c.verdict) for c in proposed
            }
            required_canonicals = {"supports", "contradicts", "insufficient"}
            missing = required_canonicals - canonical_present
            n_added = 0
            if missing:
                used_ids = {c.candidate_id for c in proposed}
                # ``_CANDIDATE_ID_POOL`` is a strict superset of
                # ``_CANDIDATE_IDS`` (the LLM-enumeration slots); if
                # the LLM filled all 5 slots, balanced augmentation
                # uses the remaining slots in the pool. We need at
                # most 3 (one per canonical verdict).
                available_ids = [
                    cid for cid in _CANDIDATE_ID_POOL if cid not in used_ids
                ]
                defaults_by_verdict = {
                    c.verdict: c for c in _default_candidates()
                }
                # Sort missing for deterministic ordering across runs.
                for verdict in sorted(missing):
                    if not available_ids:
                        break  # ran out of candidate IDs
                    default_c = defaults_by_verdict[verdict]
                    proposed.append(
                        CandidateRecord(
                            candidate_id=available_ids.pop(0),
                            verdict=verdict,
                            description=(
                                f"[Balanced enumeration: enumerator did not "
                                f"produce a {verdict}-framed candidate; "
                                f"seeded with the default framing to ensure "
                                f"the IBE chain considers all canonical "
                                f"verdicts (Lipton: comparative selection "
                                f"requires rivals).] "
                                + default_c.description
                            ),
                        )
                    )
                    n_added += 1

            if n_added:
                message = (
                    f"Enumerated {len(proposed) - n_added} candidates "
                    f"+ {n_added} balanced-enumeration default"
                    f"{'s' if n_added != 1 else ''} "
                    f"(missing canonical verdicts: "
                    f"{', '.join(sorted(missing))})"
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


def _framing_tie_cap(
    chosen: "CandidateRecord", candidates: "list[CandidateRecord]"
) -> tuple[float, "CandidateRecord | None", float | None]:
    """Cap on ``integrated_confidence`` from the framing-tie signal.

    Lipton's IBE says: the strength of inference-to-best-explanation is
    bounded by how decisively the chosen explanation dominates
    alternatives. When an *opposing* candidate (canonical verdict
    different from the chosen's) has a competitive loveliness score,
    the chain is in a frame-ambiguous state — both stories are coherent
    explanations of the evidence, and the chain's commitment to one
    over the other is essentially a coin flip dressed as a verdict.

    This function looks at the FULL candidate set (not just the named
    runner-up) because the v22 case 847 trace showed the chain
    sometimes picks a same-direction runner-up while a strongly
    opposing candidate exists with high loveliness. The richer signal
    asks: "is there ANY opposing candidate the chain found coherent?".

    Returns:
        (cap, best_opposing_or_None, gap_or_None)

        cap: 0.5 (perfect tie) → 1.0 (chosen dominates by ≥
             FRAMING_TIE_SATURATION_GAP). Linear in between.
        best_opposing: the highest-loveliness candidate whose canonical
             verdict opposes the chosen's. None if no such candidate.
        gap: chosen.loveliness - best_opposing.loveliness, or None.
    """
    chosen_canonical = _verdict_to_canonical(chosen.verdict)
    chosen_love = chosen.loveliness or 0.0
    if chosen_canonical == "insufficient":
        # Insufficient is not "opposing" anything in the supports/
        # contradicts sense — no framing-tie signal to compute.
        return 1.0, None, None

    opposing_canonical = (
        "contradicts" if chosen_canonical == "supports" else "supports"
    )

    best_opposing: "CandidateRecord | None" = None
    best_love = -1.0
    for c in candidates:
        if c is chosen:
            continue
        if _verdict_to_canonical(c.verdict) != opposing_canonical:
            continue
        love = c.loveliness or 0.0
        if love > best_love:
            best_love = love
            best_opposing = c

    if best_opposing is None or best_love <= 0.0:
        return 1.0, None, None

    gap = chosen_love - best_love
    if gap >= FRAMING_TIE_SATURATION_GAP:
        return 1.0, best_opposing, gap
    if gap <= 0.0:
        # Opposing matches or exceeds chosen on loveliness — perfect
        # tie or worse. Cap at 0.5; downstream consumers can decide
        # whether to surface as "contested".
        return 0.5, best_opposing, gap
    cap = 0.5 + (gap / FRAMING_TIE_SATURATION_GAP) * 0.5
    return cap, best_opposing, gap


# ── In-memory IBE chain helpers (for K-agreement runs) ────────────────
#
# These helpers replicate the LLM-driving logic of the four IBE
# operations but operate on a list of CandidateRecord passed by
# reference. They do not mutate the claim entity and do not save.
#
# Run 1 of the IBE chain executes through the graph nodes
# (Enumerate → ScoreLoveliness → ScoreLikeliness → SelectBest)
# and the per-stage Operation classes write results to the claim.
# Runs 2..K (when ``ibe_agreement_k > 1``) execute through these
# helpers from inside ``SelectBestExplanationOperation``.


async def _run_enumerate_inline(
    op: BaseOperation,
    claim: Claim,
    brief: dict[str, object],
) -> list[CandidateRecord]:
    """Generate a fresh candidate list. Mirrors EnumerateCandidatesOperation
    but does not mutate the claim or persist."""
    if not op.agent_runner:
        return _default_candidates()

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
        result = await op.run_agent(
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
        return _default_candidates()

    # Balanced enumeration — ensure all three canonical verdicts are
    # represented. See EnumerateCandidatesOperation for the full
    # rationale.
    canonical_present = {_verdict_to_canonical(c.verdict) for c in proposed}
    required = {"supports", "contradicts", "insufficient"}
    missing = required - canonical_present
    if missing:
        used_ids = {c.candidate_id for c in proposed}
        available_ids = [cid for cid in _CANDIDATE_ID_POOL if cid not in used_ids]
        defaults_by_verdict = {c.verdict: c for c in _default_candidates()}
        for verdict in sorted(missing):
            if not available_ids:
                break
            default_c = defaults_by_verdict[verdict]
            proposed.append(
                CandidateRecord(
                    candidate_id=available_ids.pop(0),
                    verdict=verdict,
                    description=(
                        f"[Balanced enumeration: enumerator did not "
                        f"produce a {verdict}-framed candidate; "
                        f"seeded with the default framing.] "
                        + default_c.description
                    ),
                )
            )
    return proposed


async def _run_loveliness_inline(
    op: BaseOperation,
    claim: Claim,
    brief: dict[str, object],
    candidates: list[CandidateRecord],
) -> None:
    """Score loveliness on each candidate in place."""
    unscored = [c for c in candidates if c.loveliness is None]
    if not unscored:
        return
    if not op.agent_runner:
        for c in unscored:
            c.loveliness = 0.5
            c.loveliness_reasoning = "No agent runner — neutral default."
        return

    async def _score_one(
        cand: CandidateRecord,
    ) -> tuple[CandidateRecord, LovelinessScore]:
        result = await op.run_agent(
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


async def _run_likeliness_inline(
    op: BaseOperation,
    claim: Claim,
    brief: dict[str, object],
    candidates: list[CandidateRecord],
) -> None:
    """Score likeliness on each candidate in place."""
    unscored = [c for c in candidates if c.likeliness is None]
    if not unscored:
        return
    if not op.agent_runner:
        for c in unscored:
            c.likeliness = 0.5
            c.likeliness_reasoning = "No agent runner — neutral default."
        return

    async def _score_one(
        cand: CandidateRecord,
    ) -> tuple[CandidateRecord, LikelinessScore]:
        result = await op.run_agent(
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


@dataclass
class _SelectionResult:
    """Bundle the outputs of a single IBE selection run for K-agreement
    aggregation. ``canonical_verdict`` is the agreement axis; the other
    fields preserve the per-run detail for diagnostic reporting."""

    canonical_verdict: str  # "supports" / "contradicts" / "insufficient"
    capped_confidence: float


async def _run_select_inline(
    op: BaseOperation,
    claim: Claim,
    candidates: list[CandidateRecord],
) -> _SelectionResult | None:
    """Run the LLM selection on a candidate set, apply both confidence
    caps, return the canonical verdict and capped confidence. Returns
    None on hard agent failure."""
    if not candidates or any(
        c.loveliness is None or c.likeliness is None for c in candidates
    ):
        return None

    if not op.agent_runner:
        ranked = sorted(
            candidates,
            key=lambda c: (c.loveliness or 0.0) * (c.likeliness or 0.0),
            reverse=True,
        )
        chosen = ranked[0]
        confidence = 0.5
    else:
        candidates_block = "\n\n".join(
            (
                f"### Candidate {c.candidate_id} ({c.verdict})\n"
                f"Description: {c.description}\n"
                f"Loveliness: {c.loveliness:.2f} — {c.loveliness_reasoning or ''}\n"
                f"Likeliness: {c.likeliness:.2f} — {c.likeliness_reasoning or ''}"
            )
            for c in candidates
        )
        result = await op.run_agent(
            "epistemic_select_best_explanation",
            claim_statement=claim.statement,
            claim_scope=claim.scope,
            candidates=candidates_block,
        )
        chosen = next(
            (c for c in candidates if c.candidate_id == result.chosen_candidate_id),
            None,
        )
        if chosen is None:
            return None
        confidence = float(result.confidence)

    adv_cap = _adversarial_confidence_cap(claim.adversarial_balance)
    ft_cap, _best_opposing, _ft_gap = _framing_tie_cap(chosen, list(candidates))
    cap = min(adv_cap, ft_cap)
    capped_confidence = max(0.0, min(1.0, min(confidence, cap)))

    return _SelectionResult(
        canonical_verdict=_verdict_to_canonical(chosen.verdict),
        capped_confidence=capped_confidence,
    )


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

        # Apply two confidence caps in sequence (Phase A: adversarial,
        # Phase C: framing-tie). Both attenuate ``integrated_confidence``
        # via principled signals from the IBE chain's existing trace —
        # neither introduces a tunable threshold beyond the existing
        # canonical thresholds.
        #
        # 1. Adversarial cap (Option A — soft tri-state on
        #    adversarial_balance). A claim in the CONTESTED zone (0.3 ≤
        #    balance < 0.7) should not certify at high confidence even
        #    if loveliness × likeliness say otherwise — a non-trivial
        #    counter-evidence signal exists.
        # 2. Framing-tie cap (Phase C — exposing IBE deliberation). When
        #    an opposing-verdict candidate has competitive loveliness,
        #    the chain's commitment to the chosen verdict is essentially
        #    a coin flip dressed as a verdict. Lipton's IBE explicitly
        #    bounds inference strength by the gap between best and
        #    second-best.
        adv_cap = _adversarial_confidence_cap(claim.adversarial_balance)
        ft_cap, best_opposing, ft_gap = _framing_tie_cap(
            chosen, list(claim.integration_candidates)
        )
        cap = min(adv_cap, ft_cap)
        capped_confidence = min(confidence, cap)
        adv_capped = adv_cap < confidence - 1e-9
        ft_capped = ft_cap < confidence - 1e-9

        # Write to the integration fields compute_posterior reads
        claim.integrated_assessment = _verdict_to_canonical(chosen.verdict)
        claim.integrated_confidence = max(0.0, min(1.0, capped_confidence))
        claim.integrated_reasoning = reasoning
        if adv_capped:
            claim.integrated_reasoning += (
                f"\n\n[Adversarial cap applied: IBE confidence "
                f"{confidence:.2f} reduced toward {adv_cap:.2f} because "
                f"adversarial_balance={claim.adversarial_balance:.2f} is in "
                f"the contested zone (< {ADVERSARIAL_SURVIVED_THRESHOLD}).]"
            )
        if ft_capped and best_opposing is not None and ft_gap is not None:
            claim.integrated_reasoning += (
                f"\n\n[Framing-tie cap applied: IBE confidence "
                f"{confidence:.2f} reduced toward {ft_cap:.2f} because the "
                f"chosen candidate's loveliness ({chosen.loveliness:.2f}) "
                f"barely dominates an opposing candidate "
                f"'{best_opposing.candidate_id}' "
                f"(verdict={best_opposing.verdict}, "
                f"loveliness={best_opposing.loveliness:.2f}, "
                f"gap={ft_gap:.2f}). The abductive chain's commitment to "
                f"this direction is contested — a coherent opposing "
                f"explanation has comparable explanatory force.]"
            )
        await self.repo.save(claim)

        cap_notes: list[str] = []
        if adv_capped:
            cap_notes.append(
                f"adv_balance={claim.adversarial_balance:.2f}, adv_cap={adv_cap:.2f}"
            )
        if ft_capped and ft_gap is not None:
            cap_notes.append(f"framing_tie_gap={ft_gap:.2f}, ft_cap={ft_cap:.2f}")
        cap_note = f" [capped from {confidence:.2f}: {', '.join(cap_notes)}]" if cap_notes else ""

        # ── K-agreement check (Reichenbach over LLM-stochastic IBE) ──
        # Run the IBE chain (Enumerate → Loveliness → Likeliness →
        # Select) K-1 more times in memory and require all K runs to
        # agree on canonical direction. K=1 disables this check; K=2
        # is the canonical default.
        ibe_k = int(work.metadata.get("ibe_agreement_k", 1) or 1)
        agreement_note = ""
        if ibe_k > 1:
            run1_canonical = claim.integrated_assessment
            confidences = [claim.integrated_confidence or 0.5]
            additional: list[str] = []
            brief = await _build_evidence_brief(self, claim)
            for _ in range(ibe_k - 1):
                cand_i = await _run_enumerate_inline(self, claim, brief)
                await _run_loveliness_inline(self, claim, brief, cand_i)
                await _run_likeliness_inline(self, claim, brief, cand_i)
                sel_i = await _run_select_inline(self, claim, cand_i)
                if sel_i is None:
                    # Treat hard failure as a missing run; if no other
                    # runs disagree, agreement is computed over the
                    # successful ones only.
                    continue
                additional.append(sel_i.canonical_verdict)
                confidences.append(sel_i.capped_confidence)

            all_verdicts = [run1_canonical, *additional]
            distinct = set(all_verdicts)
            if len(distinct) > 1:
                # Disagreement → fall back to insufficient. The
                # framing-tie cap dampens single-run confidence; the
                # K-agreement check is the discrete-direction analogue.
                claim.integrated_assessment = "insufficient"
                claim.integrated_confidence = 0.5
                tally = ", ".join(
                    f"{v}={all_verdicts.count(v)}" for v in sorted(distinct)
                )
                claim.integrated_reasoning = (claim.integrated_reasoning or "") + (
                    f"\n\n[K-agreement (K={ibe_k}): runs disagreed on direction "
                    f"({tally}). Independent IBE chains did not converge; "
                    f"committing insufficient rather than ratifying a single-run "
                    f"argmax.]"
                )
                await self.repo.save(claim)
                agreement_note = (
                    f" [K-agreement={ibe_k}: disagreed ({tally}) → insufficient]"
                )
            elif len(additional) > 0:
                # All K runs agreed on direction. Use min(confidence)
                # across runs as the conservative reduction. Always
                # record the agreement check so the trace shows it
                # ran, even when the confidences happen to match.
                new_conf = max(0.0, min(1.0, min(confidences)))
                claim.integrated_confidence = new_conf
                claim.integrated_reasoning = (
                    claim.integrated_reasoning or ""
                ) + (
                    f"\n\n[K-agreement (K={ibe_k}): all {len(all_verdicts)} "
                    f"runs agreed on '{run1_canonical}'; confidence is "
                    f"min across runs ({new_conf:.2f}).]"
                )
                await self.repo.save(claim)
                agreement_note = (
                    f" [K-agreement={ibe_k}: all agreed on {run1_canonical}, "
                    f"confidence={new_conf:.2f}]"
                )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"Selected {chosen.candidate_id} ({chosen.verdict}) over "
                f"{runner_up.candidate_id}; verdict={claim.integrated_assessment}, "
                f"confidence={claim.integrated_confidence:.2f}{cap_note}"
                f"{agreement_note}"
            ),
        )
