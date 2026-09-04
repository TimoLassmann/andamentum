"""Scrutiny operations.

Skeptic review of claims using a split-agent architecture: evidence
assessment + one-issue-at-a-time identification + deterministic
verdict combination. Creates Uncertainty entities for issues found.

Depends on: base (BaseOperation, OperationResult), claims (select_top_k_evidence)
Operates on: Claim, Evidence, Uncertainty, Objective entities
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from .base import BaseOperation, OperationInput, OperationResult
from .claims import LLM_PANEL_CAP, select_top_k_evidence, top_n_representatives

from ..entities import (
    Claim,
    Evidence,
    JudgeOpinion,
    UncertaintyType,
)
from ..scrutiny_weight import EvidenceWeight, compute_evidence_weight
from ..uncertainty_raising import raise_uncertainty

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceSummary:
    """One representative evidence item, and the text scrutiny shows a model.

    The pairing is load-bearing, not bookkeeping. ``_check_single_evidence``
    shows the identifier agent exactly ONE item, so any issue it reports is
    ABOUT that item — and the ``evidence_corrupted`` sentinel acts on an
    Evidence entity. Passing bare strings around lost that link, and the
    destroy then fell on whichever item happened to sort first in
    ``claim.evidence_ids``.
    """

    evidence_id: str
    text: str


@dataclass(frozen=True)
class FoundIssue:
    """One issue the identifier agent reported, with its provenance.

    ``evidence_id`` is ``None`` exactly when the issue came from the
    cross-item contradiction call, which read every representative at once
    and therefore names no single item. That distinction decides whether the
    ``evidence_corrupted`` sentinel has a destroy target at all — see
    ``_handle_issues``.
    """

    description: str
    issue_type: str
    reversal_test: bool
    evidence_id: Optional[str]


@dataclass(frozen=True)
class CorruptionGateOutcome:
    """What the ``evidence_corrupted`` destroy gate did, and why.

    ``note`` is written into whichever record survives the decision — the
    ``Evidence.invalidation_reason`` when the destroy proceeds, the
    ``Uncertainty.description`` when it is declined — so the reasoning is
    never only in a log line.
    """

    destroyed: bool
    note: str


class ScrutiniseClaimOperation(BaseOperation):
    """Run skeptic review on a claim.

    Architecture:
    1. Deterministic evidence weight (``scrutiny_weight.compute_evidence_weight``)
       — strong / moderate / weak / conflicting from the per-evidence judgment
       distributions, quality scores, and corroboration counts. No LLM call.
    2. epistemic_identify_single_issue — identifies issues one at a time (the
       content-reading channel: methodological objections become named issues).
    3. Deterministic combination — blocking issues downgrade the weight, then
       the verdict follows from it.

    Three INDEPENDENT-JUDGE second opinions layer on top (all three are
    ``llm_judge`` FAST, one criterion, one call). Two of them cannot change
    anything; the third can only ever WITHHOLD a destroy:

    * ``_confirm_load_bearing_judgments`` (before the weight is computed) —
      up to ``JUDGMENT_CONFIRMATION_MAX_PER_PASS`` per-item confirmations of
      ONE-HOT directional judgments, whose own entropy is uninformative and
      which the doubt aggregate is therefore structurally blind to. Records
      ``Evidence.direction_opinion`` and counts dissents into
      ``Claim.judgment_dissent_count``. Cannot change the verdict.
    * ``_second_opinion_if_borderline`` (after the issue gather) — at most
      one claim-level read, and only when the deterministic weight sits
      within ``SCRUTINY_BORDERLINE_BAND`` of a cutoff. Records
      ``Claim.scrutiny_opinion``. Cannot change the verdict.
    * ``_destroy_or_retain_corrupted`` (inside ``_handle_issues``) — the gate
      on the ``evidence_corrupted`` sentinel, which is the one place scrutiny
      ERASES evidence. One sampled categorical from a small model is not
      enough to destroy a source, so the destroy proceeds only when
      ``quality_doubt.confirm_corruption`` confidently agrees; otherwise the
      concern is recorded as a NON-BLOCKING ``Uncertainty`` and the evidence
      survives. It fires at most once per reported issue, so at most
      ``MAX_ISSUES`` times per pass, and never on a claim whose issues are
      all of other types.

    All three are strictly sequential and strictly outside
    ``_execute_split``'s ``asyncio.gather``, because ``llm_judge`` does not
    share ``core.AgentRunner``'s semaphore. The first two add at most 8 model
    calls per claim per run; the third adds one per corruption sentinel — in
    every case independent of evidence-pool size.

    Updates scrutiny_verdict to "pass", "fail", or "needs_resolution", and
    records ``scrutiny_doubt`` beside it — the mass-weighted mean entropy of
    the judgments that produced the weight. The verdict says which way; the
    doubt says how sure the judges were. May create Uncertainty entities for
    issues found.
    """

    entity_type = "claim"

    async def _compute_input_fingerprint(self, claim: Claim) -> str:
        """SHA-256 of the claim+evidence inputs that scrutiny depends on.

        Hashes the claim statement, scope, and the sorted set of evidence_ids
        that are extracted and not invalidated. cluster_status is intentionally
        excluded: it's an internal optimization for which subset gets sent to
        the LLM, not an independent input — the same evidence_id set will
        produce the same scrutiny outcome regardless of how clustering chooses
        representatives.
        """
        eligible_ids: list[str] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence):
                continue
            if not ev.extracted_content:
                continue
            if ev.invalidated:
                continue
            eligible_ids.append(eid)
        eligible_ids.sort()

        parts = [claim.statement, claim.scope or "", *eligible_ids]
        blob = "\n".join(parts).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    async def _gather_eligible_evidence(self, claim: Claim) -> list[Evidence]:
        """The claim's eligible representative evidence: extracted, not
        invalidated, and not a corroborative/deferred cluster member.

        This is the set the deterministic evidence-weight is computed over
        (and the candidate pool the LLM issue-finder's representatives are
        drawn from).

        Delegates to ``gates.eligible_representative_evidence`` — the one
        definition of this filter, shared with the cycle-cap terminal
        verdict so the two provably read the same evidence rather than two
        near-copies. See that function for the deliberate divergence from
        ``confidence._claim_directional_masses``.
        """
        from ..gates import eligible_representative_evidence

        return await eligible_representative_evidence(claim, self.repo)

    async def _gather_evidence_summaries(
        self, claim: Claim, candidates: list[Evidence]
    ) -> list[EvidenceSummary]:
        """Format the top representatives for the issue-identification agent.

        Caps at LLM_PANEL_CAP highest-quality representatives so the
        per-evidence issue prompts stay bounded as the evidence base grows.

        Returns the formatted text PAIRED with the entity it came from. The
        ``evidence_corrupted`` sentinel destroys an Evidence, so an issue
        raised about one item has to stay attached to that item all the way
        to the destroy — see ``EvidenceSummary``.
        """
        evidence_summaries: list[EvidenceSummary] = []
        selected = await top_n_representatives(
            candidates,
            LLM_PANEL_CAP,
            claim_text=claim.statement,
            embedding_model=self.embedding_model,
        )
        for ev in selected:
            quality_str = (
                f", quality={ev.quality_score:.2f}" if ev.quality_score else ""
            )
            evidence_summaries.append(
                EvidenceSummary(
                    evidence_id=ev.entity_id,
                    text=(
                        f"[{ev.source_type}{quality_str}] {ev.source_ref}\n"
                        f"{ev.extracted_content}"
                    ),
                )
            )
        return evidence_summaries

    # Blocking types that remain blocking even when scrutiny passes.
    # These represent factual contradictions or logical impossibilities that
    # the scrutiny verdict cannot override.
    _ALWAYS_BLOCKING = frozenset(
        {"contradiction", "logical_inconsistency", "physical_implausibility"}
    )

    #: What a declined ``evidence_corrupted`` destroy becomes. Non-blocking by
    #: construction (``UncertaintyType.EVIDENCE_GAP`` is not in
    #: ``BLOCKING_TYPES``): a source that MIGHT be garbled is a caveat on the
    #: evidence base, not grounds to block the claim — the destroy was already
    #: judged unwarranted, and turning the same unconfirmed suspicion into a
    #: blocking uncertainty would re-impose it through the back door.
    _DECLINED_CORRUPTION_TYPE = UncertaintyType.EVIDENCE_GAP

    async def _destroy_or_retain_corrupted(
        self, claim: Claim, issue: FoundIssue
    ) -> CorruptionGateOutcome:
        """Invalidate the flagged evidence only if a second opinion agrees.

        THE CLIFF THIS GATES. ``evidence_corrupted`` is one sampled
        categorical from ``epistemic_identify_single_issue`` — rule 1 of that
        agent's classification list, on a small model, with no distribution
        and therefore no entropy to read. Acting on it unconditionally is the
        same defect ``quality_doubt`` was written for at the scoring layer:
        an unlucky draw erases a real source, and nothing downstream can tell
        erasure from absence. So the sentinel PROPOSES a destroy and
        ``quality_doubt.confirm_corruption`` disposes.

        WHICH ITEM. ``issue.evidence_id`` — the item the identifier agent was
        actually shown. When it is ``None`` the issue came from the
        cross-item contradiction call, which read the whole body and names no
        victim; there is nothing to gate and nothing to destroy, so the
        concern falls through to an uncertainty. Picking an arbitrary item in
        that case is what the previous implementation did, for EVERY
        corruption issue, and it destroyed whichever evidence sorted first in
        ``claim.evidence_ids`` regardless of which one the agent read.

        SEQUENTIAL, AND OUTSIDE THE GATHER. Reached only from
        ``_handle_issues``, which ``_execute_split`` calls after awaiting its
        ``asyncio.gather`` — ``llm_judge`` does not share
        ``core.AgentRunner``'s semaphore, so a call inside that gather would
        fire concurrent local-model inferences.

        Returns:
            Whether the evidence was destroyed, and the note explaining the
            decision. The caller writes the note onto the surviving record.

        Raises:
            RuntimeError: when a corruption sentinel has a live destroy target
                and no ``judge_model`` was wired. Data-dependent and
                fail-loud, matching ``_confirm_load_bearing_judgments`` and
                ``ExtractEvidenceOperation._confirm_or_retain_low_quality``:
                silently resuming the ungated destroy would hide a wiring bug
                behind lost evidence.
            Whatever ``judge_score`` raises — propagated, so ``_execute_op``
                records a loud quarantine rather than a half-applied destroy.
        """
        from ..quality_doubt import confirm_corruption

        if issue.evidence_id is None:
            return CorruptionGateOutcome(
                destroyed=False,
                note=(
                    "flagged as corrupted by the cross-item contradiction "
                    "check, which reads every source at once and so names no "
                    "single one — recorded rather than acted on"
                ),
            )

        ev = await self.repo.get("evidence", issue.evidence_id)
        if not isinstance(ev, Evidence) or ev.invalidated or not ev.extracted_content:
            # Nothing to destroy and nothing to judge. Not an error: a
            # concurrent issue in the same pass may already have invalidated
            # this item.
            return CorruptionGateOutcome(
                destroyed=False,
                note="flagged as corrupted, but the source is no longer live",
            )

        if not self.judge_model:
            raise RuntimeError(
                f"[_destroy_or_retain_corrupted] no judge_model wired for "
                f"claim {claim.entity_id}: an 'evidence_corrupted' sentinel "
                f"proposes destroying evidence {ev.entity_id}, and that "
                "destroy requires an llm_judge model id so a single sampled "
                "categorical cannot erase a source on its own. An "
                "agent_runner is configured but judge_model is None — thread "
                "EpistemicDeps.judge_model through create_operations / "
                "_make_op rather than letting the destroy run ungated."
            )

        opinion = await confirm_corruption(
            content=ev.extracted_content,
            source_ref=ev.source_ref,
            claim_statement=claim.statement,
            issue_description=issue.description,
            model=self.judge_model,
        )

        if opinion.confidently_fails:
            ev.invalidated = True
            ev.invalidation_reason = (
                f"Scrutiny flagged this source as corrupted: "
                f"{issue.description} — confirmed by an independent judge "
                f"(overall=fails, doubt={opinion.doubt:.2f}): {opinion.summary}"
            )
            await self.repo.save(ev)
            logger.warning(
                "[_handle_issues] TMS: invalidated %s — 'evidence_corrupted' "
                "sentinel, second opinion confidently agreed (doubt=%.2f)",
                ev.entity_id,
                opinion.doubt,
            )
            return CorruptionGateOutcome(
                destroyed=True,
                note=f"evidence {ev.entity_id} invalidated",
            )

        logger.warning(
            "[_handle_issues] TMS: RETAINED %s — 'evidence_corrupted' "
            "sentinel, but the second opinion was overall=%s (doubt=%.2f, "
            "needs_review=%s). The source stands at its full weight and the "
            "concern is recorded as a non-blocking uncertainty — a caveat, "
            "not an attenuation: nothing about the source's contribution to "
            "the verdict changes.",
            ev.entity_id,
            opinion.overall,
            opinion.doubt,
            opinion.needs_review,
        )
        return CorruptionGateOutcome(
            destroyed=False,
            note=(
                f"flagged as corrupted, but an independent judge read source "
                f"{ev.source_ref or ev.entity_id} as intact "
                f"(overall={opinion.overall}, doubt={opinion.doubt:.2f}): "
                f"{opinion.summary} — retained, not destroyed"
            ),
        )

    async def _handle_issues(
        self,
        claim: Claim,
        issues: list[FoundIssue],
        *,
        verdict: str = "pass",
    ) -> None:
        """Create uncertainties from issues, handling evidence_corrupted specially.

        Non-blocking types create non-blocking uncertainties.
        "evidence_corrupted" PROPOSES invalidating the evidence the issue was
        raised about; ``_destroy_or_retain_corrupted`` decides. A declined
        destroy is not dropped — it becomes a non-blocking uncertainty
        carrying the judge's reading, so a flagged concern never disappears
        just because it was not acted on.

        When verdict="pass", blocking types that encode missing information
        ("unknown", "missing_premise") are downgraded to their non-blocking
        equivalents ("evidence_gap", "assumption"). The scrutiny verdict already
        encodes the judgment that the claim is acceptable despite these gaps —
        creating blocking uncertainties on top of a passing verdict would
        override the agent's judgment at the data layer.

        Truly problematic types (contradiction, logical_inconsistency,
        physical_implausibility) remain blocking regardless of verdict.
        """
        for issue in issues:
            issue_type_str = issue.issue_type or "unknown"
            description = issue.description

            # When scrutiny passes, downgrade information-gap blocking types to
            # their non-blocking equivalents. The scrutiny judgment already
            # decided this is acceptable; double-blocking contradicts that.
            if verdict == "pass" and issue_type_str not in self._ALWAYS_BLOCKING:
                if issue_type_str == "unknown":
                    issue_type_str = "evidence_gap"
                elif issue_type_str == "missing_premise":
                    issue_type_str = "assumption"

            # Corrupted evidence: a destroy, and therefore gated. Only a
            # confirmed destroy consumes the issue; a declined one is
            # recorded as a caveat below.
            if issue_type_str == "evidence_corrupted":
                outcome = await self._destroy_or_retain_corrupted(claim, issue)
                if outcome.destroyed:
                    continue
                issue_type_str = self._DECLINED_CORRUPTION_TYPE.value
                description = f"{description} [{outcome.note}]"

            # Map issue_type string to UncertaintyType enum
            try:
                uncertainty_type = UncertaintyType(issue_type_str)
            except ValueError:
                uncertainty_type = UncertaintyType.UNKNOWN

            await raise_uncertainty(
                self.repo,
                objective_id=claim.objective_id,
                uncertainty_type=uncertainty_type,
                description=str(description),
                affected_claim_ids=[claim.entity_id],
                created_by="scrutiny",
                embedding_model=self.embedding_model,
            )

    MAX_ISSUES = 5  # Maximum per-evidence calls (caps cost for large evidence sets)

    @staticmethod
    def _verdict_from_weight(evidence_weight: str) -> str:
        """Map the (possibly blocking-downgraded) evidence weight to a verdict.

        strong / moderate → pass · conflicting → fail · weak → needs_resolution.
        """
        if evidence_weight in ("strong", "moderate"):
            return "pass"
        if evidence_weight == "conflicting":
            return "fail"
        return "needs_resolution"

    async def _execute_split(
        self,
        claim: Claim,
        weight: EvidenceWeight,
        evidence_summaries: list[EvidenceSummary],
    ) -> str:
        """Per-evidence issue identification, combined with a precomputed weight.

        ``weight`` is the deterministic evidence weight over the eligible
        representative set, computed ONCE by ``execute`` (both branches share
        it — this method used to recompute it, and the no-runner branch
        computed it a second time). Per-evidence issue identification then
        checks each representative independently against the claim (Kahneman
        independence), plus one contradiction call over all of them; all run in
        parallel — no shared context, no anchoring. Blocking issues downgrade
        the weight before the verdict is mapped.

        Downstream DeduplicateConcernsOperation handles any duplicate
        uncertainties that result from multiple calls finding the same issue.

        Each per-evidence issue is tagged with the ``evidence_id`` of the ONE
        item that call was shown, so an ``evidence_corrupted`` sentinel can
        be gated against — and can only ever destroy — the source it was
        raised about. The contradiction call is tagged ``None``: it read the
        whole body and names no single item.

        Returns the computed verdict string.
        """
        import asyncio

        # Identify issues — one call per evidence item + one contradiction call.
        # Each per-evidence call sees only ONE evidence item (independent judgment).
        # The contradiction call sees all items but asks only about contradictions.
        async def _check_single_evidence(
            item: EvidenceSummary,
        ) -> FoundIssue | None:
            result = await self.run_agent(
                "epistemic_identify_single_issue",
                claim=claim.statement,
                scope=claim.scope,
                evidence=item.text,
            )
            if result.has_issue:
                return FoundIssue(
                    description=result.description,
                    issue_type=result.issue_type,
                    reversal_test=result.reversal_test,
                    evidence_id=item.evidence_id,
                )
            return None

        async def _check_contradictions(all_evidence: str) -> FoundIssue | None:
            result = await self.run_agent(
                "epistemic_identify_single_issue",
                claim=claim.statement,
                scope=claim.scope,
                evidence=all_evidence,
                focus="Check whether any of the evidence items contradict each other.",
            )
            if result.has_issue:
                return FoundIssue(
                    description=result.description,
                    issue_type=result.issue_type,
                    reversal_test=result.reversal_test,
                    evidence_id=None,
                )
            return None

        # Build tasks: one per evidence item (capped at MAX_ISSUES) + one contradiction check
        tasks: list[asyncio.Task[FoundIssue | None]] = []
        for ev_summary in evidence_summaries[: self.MAX_ISSUES]:
            tasks.append(asyncio.ensure_future(_check_single_evidence(ev_summary)))
        if len(evidence_summaries) >= 2:
            all_evidence_text = "\n\n".join(s.text for s in evidence_summaries)
            tasks.append(asyncio.ensure_future(_check_contradictions(all_evidence_text)))

        results = await asyncio.gather(*tasks)
        found_issues: list[FoundIssue] = [r for r in results if r is not None]

        # ── Deterministic combination ────────────────────────────────────
        evidence_weight: str = weight.label

        # Check for blocking issues via reversal_test
        has_blocking = any(iss.reversal_test for iss in found_issues)

        # If blocking issues exist but evidence was rated strong, downgrade to moderate
        if has_blocking and evidence_weight == "strong":
            evidence_weight = "moderate"

        # Compute verdict deterministically from evidence weight
        verdict = self._verdict_from_weight(evidence_weight)

        # STRICTLY AFTER the gather has been awaited: the corruption destroy
        # gate inside makes an llm_judge call, which must never run inside
        # core.AgentRunner's semaphore-free gather.
        await self._handle_issues(claim, found_issues, verdict=verdict)

        return verdict

    async def _confirm_load_bearing_judgments(
        self, claim: Claim, eligible: list[Evidence]
    ) -> int:
        """Ask an independent judge to confirm up to N one-hot directions.

        THE BLIND SPOT THIS COVERS. ``Claim.scrutiny_doubt`` is a mean of
        judgment ENTROPIES, and a one-hot judgment has none — so on the
        53-82% of small-model judgments that come back one-hot
        (``thresholds.JUDGMENT_ONE_HOT_THRESHOLD``), the doubt machinery is
        structurally silent. A second, independent elicitation is the only
        thing that can speak there. See ``direction_opinion``'s module
        docstring for the full argument.

        Called from ``execute`` BETWEEN the eligible gather and the single
        ``compute_evidence_weight`` call, so the confirmations and the
        verdict come from one reading — the same ordering constraint the
        weight/doubt pair already carries.

        STRICTLY SEQUENTIAL, and deliberately outside ``_execute_split``'s
        ``asyncio.gather``: ``llm_judge`` does not share
        ``core.AgentRunner``'s semaphore (sized 1 for Ollama), so a call
        inside that gather would fire concurrent local-model inferences.

        Returns how many of the claim's CURRENTLY-LINKED evidence items carry
        a confirmation that dissented from the primary judge's direction —
        written by the caller onto ``claim.judgment_dissent_count``.

        COUNTED OVER THE CLAIM'S STANDING EVIDENCE, not over this pass's new
        calls. Both properties the field needs follow from that, and only
        from that:

        * it never goes STALE — an item that has since been unlinked or
          replaced leaves ``claim.evidence_ids``, so its dissent leaves the
          count on the very next pass;
        * it never goes DEAF — a dissent found in an earlier pass is still
          counted while the item that carries it is still load-bearing.

        Counting only this pass's new confirmations would satisfy the first
        and violate the second, and the violation is silent and total: an
        item is confirmed AT MOST ONCE EVER (the ``direction_opinion is
        None`` guard), so a second pass makes no new calls for it and would
        write 0 over a real dissent. Gate 0 reads this field at synthesis
        time, after the last pass — so under a per-pass count SITE A's signal
        would be erased by any rescrutiny, which is precisely the case
        rescrutiny exists to handle.

        COST OF THE TALLY: ``len(claim.evidence_ids)`` repository reads per
        pass, zero tokens and zero model calls. It is a SECOND traversal of
        the same ids in the same ``execute`` — the same walk *shape*
        ``_compute_input_fingerprint`` performs, not shared work with it:
        that method's short-circuit call returns before this code is
        reached, and its end-of-pass call happens after the confirmations.

        Raises:
            RuntimeError: when candidates exist, an ``agent_runner`` is
                configured, and no ``judge_model`` was wired. Data-dependent
                and fail-loud, matching ``ExtractEvidenceOperation.
                _confirm_or_retain_low_quality`` — a construction-time
                assertion would make a judge model mandatory for the many
                callers that never elicit one.
            Whatever ``judge_score`` raises — propagated, so ``_execute_op``
                records a loud quarantine rather than an unconfirmed verdict.
        """
        # One traversal, two tallies. Reads the repository only — no model
        # call, no tokens — so it runs on EVERY construction, online or not.
        #
        # ``already_confirmed`` counts over EVERY evidence id the claim
        # currently links, not just the eligible subset, so evidence ADDED
        # between passes cannot reopen the per-claim budget. An id REMOVED
        # from the claim does release its slot: the tally describes the
        # claim's current books, not a run-lifetime ledger. Nothing in the
        # pipeline removes ids today (``verification`` appends,
        # ``investigation`` extends), and a bound test pins the growth case.
        #
        # ``standing_dissents`` carries forward the dissents already on the
        # claim's books, so the returned count describes the claim's current
        # state rather than this pass's call log. See the docstring.
        already_confirmed = 0
        standing_dissents = 0
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if isinstance(ev, Evidence) and ev.direction_opinion is not None:
                already_confirmed += 1
                if ev.direction_opinion.dissents:
                    standing_dissents += 1

        if self.agent_runner is None:
            # Offline construction: the deterministic reading still happens,
            # there is simply no independent judge to ask. The dissents
            # already on the claim's books STILL STAND — returning 0 here
            # would have ``execute`` write over a real dissent recorded by an
            # earlier live pass, on evidence that still carries it.
            return standing_dissents

        from ..direction_opinion import confirm_direction, select_confirmation_candidates
        from ..thresholds import (
            JUDGMENT_CONFIRMATION_MAX_PER_CLAIM,
            JUDGMENT_CONFIRMATION_MAX_PER_PASS,
        )

        candidates = select_confirmation_candidates(
            eligible,
            per_pass_limit=JUDGMENT_CONFIRMATION_MAX_PER_PASS,
            remaining_budget=JUDGMENT_CONFIRMATION_MAX_PER_CLAIM - already_confirmed,
        )
        if not candidates:
            return standing_dissents

        if not self.judge_model:
            raise RuntimeError(
                f"[_confirm_load_bearing_judgments] no judge_model wired for "
                f"claim {claim.entity_id}: {len(candidates)} one-hot directional "
                "judgment(s) are eligible for an independent confirmation. An "
                "agent_runner is configured but judge_model is None — thread "
                "EpistemicDeps.judge_model through create_operations / _make_op "
                "rather than letting the confirmation silently not run."
            )

        # Seeded with the dissents already on the claim's books. The
        # candidates are disjoint from those by construction (the selector
        # keeps only ``direction_opinion is None`` items), so nothing is
        # double-counted.
        dissents = standing_dissents
        for ev in candidates:  # strictly sequential — never asyncio.gather
            opinion = await confirm_direction(
                content=ev.extracted_content,
                source_ref=ev.source_ref,
                claim_statement=claim.statement,
                claim_scope=claim.scope,
                stance=ev.support_judgment or "",
                model=self.judge_model,
            )
            ev.direction_opinion = opinion
            await self.repo.save(ev)
            if opinion.dissents:
                dissents += 1
        return dissents

    async def _second_opinion_if_borderline(
        self,
        claim: Claim,
        weight: EvidenceWeight,
        evidence_summaries: list[EvidenceSummary],
    ) -> JudgeOpinion | None:
        """One independent read on the claim-level question, when the
        deterministic reading was a coin flip — else ``None``.

        ``is_borderline`` is pure and free, and is evaluated BEFORE any model
        call, so a clearly-strong or clearly-thin reading costs nothing.

        Called from ``execute`` STRICTLY AFTER ``_execute_split`` has been
        awaited: ``llm_judge`` does not share ``core.AgentRunner``'s
        semaphore, so this must never sit inside that gather. It reuses the
        already-computed ``evidence_summaries`` — no second selection pass,
        no second embedding call.

        The opinion NEVER changes ``scrutiny_verdict``. See
        ``scrutiny_opinion``'s module docstring for why that is load-bearing
        rather than cautious.

        Raises:
            RuntimeError: when the reading is borderline, an ``agent_runner``
                is configured and no ``judge_model`` was wired — the same
                data-dependent fail-loud shape as
                ``_confirm_load_bearing_judgments``.
            Whatever ``judge_score`` raises — propagated.
        """
        from ..scrutiny_opinion import second_opinion_on_reading
        from ..scrutiny_weight import is_borderline

        if not is_borderline(weight):
            return None

        if not self.judge_model:
            raise RuntimeError(
                f"[_second_opinion_if_borderline] no judge_model wired for "
                f"claim {claim.entity_id}: the deterministic evidence weight "
                f"({weight.label}, supporting={weight.supporting_mass:.2f}, "
                f"contradicting={weight.contradicting_mass:.2f}) sits inside "
                "the borderline band, so an independent second opinion is "
                "warranted. An agent_runner is configured but judge_model is "
                "None — thread EpistemicDeps.judge_model through "
                "create_operations / _make_op rather than letting the second "
                "opinion silently not run."
            )

        return await second_opinion_on_reading(
            evidence_body="\n\n".join(s.text for s in evidence_summaries),
            claim_statement=claim.statement,
            claim_scope=claim.scope,
            weight_label=weight.label,
            model=self.judge_model,
        )

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
                did_work=False,
            )

        # Idempotence: if the inputs that drive scrutiny are unchanged since
        # the last successful pass, the outcome would be identical. Returning
        # early here prevents the operation from minting fresh Uncertainty
        # entities for the same issues every time the graph re-enters the
        # scrutinise → resolve cycle. The Scrutinize graph node clears the
        # fingerprint when it intentionally wants to force a fresh pass.
        if claim.scrutiny_verdict is not None and claim.scrutiny_fingerprint:
            current_fp = await self._compute_input_fingerprint(claim)
            if current_fp == claim.scrutiny_fingerprint:
                return OperationResult(
                    success=True,
                    entity_id=work.entity_id,
                    message=(
                        f"[{claim.statement[:60]}] verdict: "
                        f"{claim.scrutiny_verdict} (inputs unchanged, no-op)"
                    ),
                    did_work=False,
                )

        if claim.scrutiny_verdict is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already scrutinized",
                did_work=False,
            )

        _deferred = 0  # track deferred cluster count for visibility
        # If this is re-scrutiny after investigation, cluster the claim's
        # evidence first (embedding-based, no LLM). Investigation may have
        # fetched many new items that need reducing to a representative subset
        # before the weight is computed over them.
        if claim.investigation_count > 0:
            all_evidence = []
            for eid in claim.evidence_ids:
                ev = await self.repo.get("evidence", eid)
                if (
                    isinstance(ev, Evidence)
                    and ev.extracted
                    and ev.extracted_content
                    and not ev.invalidated
                ):
                    all_evidence.append(ev)
            if len(all_evidence) >= 2:
                _reps, _total, _deferred = await select_top_k_evidence(
                    self.repo, all_evidence, embedding_model=self.embedding_model
                )

        eligible = await self._gather_eligible_evidence(claim)
        # Independent confirmation of the entropy-blind directions, BEFORE
        # the weight is computed, so the confirmations and the verdict come
        # from one reading. Bounded, sequential, and never verdict-changing.
        n_dissent = await self._confirm_load_bearing_judgments(claim, eligible)
        # ONE reading of the evidence, shared by both branches. The weight
        # carries both the direction of the mass and the doubt of the
        # judgments that produced it, so recomputing it anywhere else would
        # risk persisting a doubt that belongs to a different reading.
        weight = compute_evidence_weight(eligible)
        opinion = None
        if self.agent_runner:
            evidence_summaries = await self._gather_evidence_summaries(claim, eligible)
            verdict = await self._execute_split(claim, weight, evidence_summaries)
            # AFTER the gather has been awaited — llm_judge does not share
            # core.AgentRunner's semaphore, so it must never run inside it.
            opinion = await self._second_opinion_if_borderline(
                claim, weight, evidence_summaries
            )
        else:
            # No agent runner: the evidence weight is still deterministic; only
            # the LLM issue-identification channel is unavailable, so the verdict
            # follows from the weight alone.
            verdict = self._verdict_from_weight(weight.label)
        claim.scrutiny_verdict = verdict
        # A MEASUREMENT recorded alongside the verdict, not a routing signal
        # (P1/P3): this operation writes it and returns; the graph decides
        # whether anything follows from it. Same for the two independent-judge
        # records below — they are written unconditionally each pass, by this
        # one writer, so they can never describe a stale reading.
        claim.scrutiny_doubt = weight.doubt
        claim.scrutiny_opinion = opinion
        claim.judgment_dissent_count = n_dissent

        # Stamp the fingerprint of the inputs we just scrutinised so a
        # subsequent call with identical inputs short-circuits.
        claim.scrutiny_fingerprint = await self._compute_input_fingerprint(claim)
        await self.repo.save(claim)

        deferred_note = (
            f" ({_deferred} evidence clusters deferred)" if _deferred > 0 else ""
        )
        doubt_note = (
            f", judgment doubt: {weight.doubt:.2f}"
            if weight.doubt is not None
            else ", judgment doubt: unmeasured"
        )
        confirm_note = (
            f", {n_dissent} direction(s) dissented from by an independent judge"
            if n_dissent
            else ""
        )
        opinion_note = (
            f", second opinion on a borderline reading: {opinion.overall}"
            f" (doubt {opinion.doubt:.2f}{', dissents' if opinion.dissents else ''})"
            if opinion is not None
            else ""
        )
        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"[{claim.statement[:60]}] verdict: {claim.scrutiny_verdict}"
                f"{doubt_note}{confirm_note}{opinion_note}{deferred_note}"
            ),
        )
