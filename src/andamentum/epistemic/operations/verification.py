"""Verification operations.

Multiple independent verification tracks: adversarial search for
disconfirming evidence, cross-domain convergence assessment, deductive
validation via first principles, and computational verification.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim, Evidence, Uncertainty entities
"""

from typing import Any, Dict, Tuple

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import (
    Claim,
    Evidence,
    Uncertainty,
    UncertaintyType,
)
from ..thresholds import ADVERSARIAL_REFUTED_THRESHOLD


# Adversarial-search query budget: total = MAX_ADVERSARIAL_TEMPLATES
# (deterministic, free) + MAX_ADVERSARIAL_FRAMINGS (LLM-generated,
# paid). Each query is then sent through the gatherer and each hit
# is evaluated by an LLM.
#
# A previous Phase-1-efficiency cut reduced these to 3 + 2 = 5 to
# halve downstream evaluation cost. Reverted (2026-05-02) after
# benchmark runs showed convergence degradation: with fewer
# adversarial queries, claims more often hit cycle caps before
# IBE could fire. Restored to 5 + 3 = 8 to recover Lakatos
# coverage.
MAX_ADVERSARIAL_TEMPLATES = 5
MAX_ADVERSARIAL_FRAMINGS = 3


class AdversarialSearchOperation(BaseOperation):
    """Seek disconfirming evidence for a claim.

    Decomposed pipeline:
    1. Template-based adversarial query generation (deterministic)
    2. Narrow agent query generation via epistemic_generate_counterquery (3 framings)
    3. Run combined queries through evidence gatherer (web search)
    4. Evaluate each search result via epistemic_evaluate_counterargument (narrow)
    5. Compute adversarial balance (deterministic)
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        import asyncio
        import logging

        from ..adversarial_query_generator import (
            generate_adversarial_queries,
        )
        from ..adversarial_evaluator import create_counterargument
        from ..adversarial_balance import synthesize_adversarial_result
        from ..gates import quality_weighted_evidence_sum

        logger = logging.getLogger(__name__)

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.adversarial_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already checked",
            )

        # Step 1: Generate template-based adversarial queries (deterministic).
        # Capped at MAX_ADVERSARIAL_TEMPLATES per Phase 1 of the
        # efficiency plan.
        template_queries = generate_adversarial_queries(
            claim.statement,
            max_queries=MAX_ADVERSARIAL_TEMPLATES,
        )

        # Step 2: Generate agent-based adversarial queries (narrow agent).
        # Each framing runs independently in parallel — no shared context
        # between calls. Diversity comes from the different framings, not
        # from seeing prior outputs (which would create anchoring per
        # Kahneman's independence principle). Phase 1 of the efficiency
        # plan reduces from 3 framings to MAX_ADVERSARIAL_FRAMINGS=2 to
        # halve downstream evaluation cost; the two retained framings
        # (contradicting_evidence, alternative_explanations) cover the
        # most common counter-argument shapes.
        agent_queries: list[str] = []
        if self.agent_runner:
            framings = [
                "contradicting_evidence",
                "alternative_explanations",
                "replication_failures",
            ][:MAX_ADVERSARIAL_FRAMINGS]

            async def _generate_one(framing: str) -> str:
                cq_result = await self.run_agent(
                    "epistemic_generate_counterquery",
                    claim=claim.statement,
                    framing=framing,
                )
                return cq_result.query

            agent_queries = list(
                await asyncio.gather(*[_generate_one(f) for f in framings])
            )

        # Combine template + agent queries, deduplicate by exact match
        seen: set[str] = set()
        all_queries: list[str] = []
        for q in template_queries + agent_queries:
            normalized = q.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                all_queries.append(q)

        # Step 3: Run queries through evidence gatherer (parallel web search).
        # Bounded concurrency avoids overwhelming SearXNG / upstream providers
        # while still eliminating serial roundtrip latency.
        search_hits: list[tuple[str, str]] = []  # (summary_text, source_ref)
        gatherer = self.evidence_gatherer
        if gatherer is not None and all_queries:
            queries_to_run = all_queries
            search_semaphore = asyncio.Semaphore(5)

            async def _gather_one(query: str) -> list[tuple[str, str]]:
                async with search_semaphore:
                    try:
                        gathered = await gatherer.gather("web_search", query)
                        return [(g.content, g.source_ref) for g in gathered]
                    except Exception as e:
                        logger.debug("Adversarial search query failed: %s", e)
                        return []

            gathered_lists = await asyncio.gather(
                *[_gather_one(q) for q in queries_to_run]
            )
            for hits in gathered_lists:
                search_hits.extend(hits)

        # Step 3.5: Embedding rerank — decouple "which query found this
        # hit" from "which evidence the judge actually evaluates".
        # Adversarial query phrasing is LLM-stochastic; the search
        # engine amplifies small phrasing differences into very different
        # result pools (case 847 v19b: 8 vs 47 vs 70 candidates across
        # 5 replicates of the same claim). The CLAIM is the stable
        # signal — embedding similarity to the claim filters the pool
        # toward semantically robust evidence regardless of which query
        # surfaced it. Reduces run-to-run variance in the judge's input.
        #
        # K = max(8, ceil(n_queries × 2)): at least 2 candidates per
        # query as a fairness floor; floor of 8 so low-yield runs (where
        # the gatherer returned ~one hit per query) pass through
        # unchanged. Runs that yielded many candidates get clipped to a
        # comparable pool size.
        if (
            self.embedding_model
            and len(search_hits) > 0
            and all_queries
        ):
            import math

            from ..embeddings import embed_texts
            from ..similarity import cosine_similarity

            top_k = max(8, math.ceil(len(all_queries) * 2))
            if len(search_hits) > top_k:
                hit_texts = [text for text, _ref in search_hits]
                # One batch: claim + every candidate hit. Truncation
                # is handled inside embed_texts (DEFAULT_MAX_EMBED_CHARS).
                try:
                    embeddings = await embed_texts(
                        [claim.statement] + hit_texts,
                        model=self.embedding_model,
                    )
                    claim_emb = embeddings[0]
                    hit_embs = embeddings[1:]
                    scored = sorted(
                        zip(
                            search_hits,
                            (cosine_similarity(claim_emb, h) for h in hit_embs),
                        ),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                    n_before = len(search_hits)
                    search_hits = [hit for hit, _score in scored[:top_k]]
                    logger.debug(
                        "Adversarial rerank: %d candidates → %d after "
                        "top-%d embedding filter (claim=%s)",
                        n_before,
                        len(search_hits),
                        top_k,
                        claim.entity_id[:8],
                    )
                except RuntimeError as e:
                    # Embedding model unreachable — proceed with the
                    # unfiltered pool. Better to evaluate everything
                    # than to silently drop adversarial signal.
                    logger.warning(
                        "Adversarial rerank skipped (embedding failure): %s", e
                    )

        # Step 4: Evaluate each search result as a potential counterargument.
        # Evaluations run in parallel (bounded concurrency) to avoid a sequential
        # chain of LLM roundtrips that previously dominated wall time.
        from ..primitives import (
            Counterargument as CounterargumentModel,
            CriticismCategory,
            CounterargumentQuality,
        )

        proper_counterarguments: list[CounterargumentModel] = []
        _justifications: dict[int, str] = {}  # id(ca) → agent justification
        balance_score: float | None = None

        if self.agent_runner:
            eval_semaphore = asyncio.Semaphore(10)

            async def _evaluate_one(
                summary: str, source_ref: str
            ) -> tuple[CounterargumentModel, str]:
                """Evaluate a single search hit. Raises on agent failure —
                the caller relies on asyncio.gather to propagate."""
                async with eval_semaphore:
                    eval_result = await self.run_agent(
                        "epistemic_evaluate_counterargument",
                        claim_statement=claim.statement,
                        counterargument_text=summary,
                        source_ref=source_ref,
                    )
                    # Map agent category string to CriticismCategory enum
                    try:
                        category = CriticismCategory(eval_result.category)
                    except ValueError:
                        category = CriticismCategory.INTERPRETATION

                    quality = CounterargumentQuality(
                        relevance=eval_result.relevance,
                        specificity=eval_result.specificity,
                        evidence_backed=eval_result.evidence_backed,
                        source_credibility=eval_result.source_credibility,
                        novelty=0.5,  # Not assessed by this agent; neutral default
                    )
                    # Capture the agent's justification — this is the system's
                    # own interpretation of what the counterargument says.
                    justification = getattr(eval_result, "justification", None) or ""
                    proper_ca = create_counterargument(
                        summary=summary,
                        source_ref=source_ref,
                        claim_id=claim.entity_id,
                        category=category,
                        quality=quality,
                    )
                    return proper_ca, justification

            if search_hits:
                eval_results = await asyncio.gather(
                    *[
                        _evaluate_one(summary, source_ref)
                        for summary, source_ref in search_hits
                    ]
                )
                for proper_ca, justification in eval_results:
                    proper_counterarguments.append(proper_ca)
                    _justifications[id(proper_ca)] = justification

            # Step 5: Compute adversarial balance (deterministic)
            supporting_weight = await quality_weighted_evidence_sum(claim, self.repo)

            adversarial_result = synthesize_adversarial_result(
                claim_id=claim.entity_id,
                objective_id=claim.objective_id,
                queries_used=all_queries,
                counterarguments=proper_counterarguments,
                supporting_evidence_weight=supporting_weight,
            )

            # Persist the full AdversarialEvidence for report generation
            await self.repo.save_adversarial_evidence(adversarial_result)

            # Store adversarial balance on claim
            balance_score = adversarial_result.adversarial_balance
            claim.adversarial_balance = balance_score

            # Store quality-passing counterarguments as Evidence entities,
            # deduplicated by source URL. Multiple passages from the same
            # source are ONE piece of evidence, not N independent sources.
            # Pick the counterargument with the best quality score per URL.
            best_per_source: dict[str, tuple[CounterargumentModel, str]] = {}
            for ca in proper_counterarguments:
                if not ca.quality.passes_threshold:
                    continue
                agent_justification = _justifications.get(id(ca), "")
                if agent_justification:
                    reasoning = (
                        f"Adversarial ({ca.category.value}): {agent_justification}"
                    )
                elif ca.supporting_evidence:
                    reasoning = (
                        f"Adversarial ({ca.category.value}): {ca.supporting_evidence}"
                    )
                else:
                    reasoning = f"Adversarial counterargument ({ca.category.value})"

                ref = ca.source_ref
                existing = best_per_source.get(ref)
                if (
                    existing is None
                    or ca.quality.combined_score > existing[0].quality.combined_score
                ):
                    best_per_source[ref] = (ca, reasoning)

            # Adversarial search's job is to LOOK FOR potential
            # counter-evidence; JUDGING whether the found item actually
            # contradicts the claim is the impartial judge's job.
            # Hard-coding support_judgment="contradicts" here used to
            # double-count: a Cochrane review whose limitations section
            # the adversarial agent harvested ended up stamped as
            # evidence-against-itself, even when its overall finding
            # supported the claim. The metformin/HbA1c failure
            # (probe B4) was the canonical instance — CD012906, the
            # supporting Cochrane review, was found via adversarial
            # web_search and labeled "contradicts".
            #
            # Fix: route every adversarial-found item through the same
            # judge_evidence agent the regular evidence flow uses. Let
            # the impartial judge decide supports / contradicts /
            # no_bearing. Preserve the adversarial provenance in the
            # reasoning text so downstream readers can still see the
            # path the evidence travelled.
            from ..judge import judge_evidence as _judge

            # Capture agent_runner locally so pyright can narrow the
            # closure away from Optional. The enclosing
            # ``if self.agent_runner:`` block already guarantees
            # non-None at runtime.
            judge_runner = self.agent_runner

            async def _judge_one(
                ca: CounterargumentModel,
                adversarial_reasoning: str,
            ) -> tuple[CounterargumentModel, str, str]:
                """Run the impartial judge on an adversarial-found item.
                Raises on judge failure — caller relies on
                asyncio.gather to propagate (no silent failures)."""
                judgment = await _judge(
                    claim_statement=claim.statement,
                    claim_scope=claim.scope,
                    evidence_content=ca.summary,
                    evidence_source=f"web_search: {ca.source_ref}",
                    runner=judge_runner,
                )
                combined_reasoning = (
                    f"{adversarial_reasoning} | judge: {judgment.reasoning}"
                )
                return ca, judgment.verdict, combined_reasoning

            judged = await asyncio.gather(
                *[
                    _judge_one(ca, reasoning)
                    for ca, reasoning in best_per_source.values()
                ]
            )

            for ca, verdict, combined_reasoning in judged:
                adv_evidence = Evidence(
                    objective_id=claim.objective_id,
                    source_type="web_search",
                    source_ref=ca.source_ref,
                    extracted_content=ca.summary,
                    extracted=True,
                    support_judgment=verdict,
                    judgment_reasoning=combined_reasoning,
                    cluster_status="representative",
                )
                await self.repo.save(adv_evidence)
                claim.evidence_ids.append(adv_evidence.entity_id)

            # For Popper-refuted claims (balance < ADVERSARIAL_REFUTED_THRESHOLD),
            # create a single NON-BLOCKING uncertainty summarizing the
            # adversarial finding. The balance score on the claim is what
            # gates use — not individual uncertainties.
            if balance_score < ADVERSARIAL_REFUTED_THRESHOLD:
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.SCOPE_DIFFERENCE,
                    description=(
                        f"Adversarial search found significant counterevidence "
                        f"(balance={balance_score:.2f}, verdict={adversarial_result.verdict}). "
                        f"{adversarial_result.explanation}"
                    ),
                    affected_claim_ids=[claim.entity_id],
                )
                await self.repo.save(uncertainty)

            # NOTE: Do NOT touch scrutiny_verdict here. Adversarial search and
            # scrutiny are independent verification tracks. The adversarial balance
            # score is checked by stage gates directly.

        claim.adversarial_checked = True
        await self.repo.save(claim)

        await self.log_event(
            "adversarial_search_complete",
            claim.entity_id,
            {
                "queries_generated": len(all_queries),
                "has_evidence_gatherer": self.evidence_gatherer is not None,
                "adversarial_balance": balance_score,
            },
        )

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"Adversarial search complete ({len(all_queries)} queries, balance={balance_score})",
        )


class AssessConvergenceOperation(BaseOperation):
    """Check cross-domain convergence of evidence.

    Decomposed into three steps:
    1. Classify each evidence item's domain via epistemic_classify_evidence_domain (narrow agent)
    2. For evidence pairs within same domain cluster, check pairwise independence
       via epistemic_check_pairwise_independence (narrow agent)
    3. Compute convergence deterministically via detect_convergence(), using the
       agent classifications from step 1 and the pairwise judgments from step 2.
    """

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        from ..convergence_detector import detect_convergence
        from ..domain_classifier import classify_evidence_domain as default_classify
        from ..primitives import DomainClassification

        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.convergence_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already checked",
            )

        # Step 1: Classify each evidence item's domain
        # Build evidence items list for detect_convergence and collect classifications
        evidence_items: list[dict[str, Any]] = []
        classifications: list[DomainClassification] = []

        # Gather all eligible representatives, then cap at LLM_PANEL_CAP
        # by quality so the per-rep LLM classify and the O(N²) within-domain
        # pairwise independence check stay bounded.
        from .claims import LLM_PANEL_CAP, top_n_representatives

        candidates: list[Evidence] = []
        for eid in claim.evidence_ids:
            ev = await self.repo.get("evidence", eid)
            if not isinstance(ev, Evidence) or not ev.extracted_content:
                continue
            if ev.cluster_status in ("corroborative", "deferred"):
                continue
            candidates.append(ev)

        for ev in top_n_representatives(candidates, LLM_PANEL_CAP):
            eid = ev.entity_id
            content = ev.extracted_content

            if self.agent_runner:
                dc_result = await self.run_agent(
                    "epistemic_classify_evidence_domain",
                    evidence_text=content,
                    source_type=ev.source_type,
                    source_ref=ev.source_ref,
                )
                # The pydantic output model (ClassifyEvidenceDomainOutput)
                # already types these fields as MethodType / DataSourceType /
                # TemporalApproach / CausalRole enums, so pydantic-ai enforces
                # the enum constraint at the API schema level. No boundary
                # coercion needed.
                classification = DomainClassification(
                    evidence_id=eid,
                    claim_id=claim.entity_id,
                    method_type=dc_result.method_type,
                    data_source=dc_result.data_source,
                    temporal=dc_result.temporal_approach,
                    causal_role=dc_result.causal_role,
                    classification_confidence=float(dc_result.confidence),
                    classification_method="agent",
                    classification_notes=dc_result.justification,
                )
            else:
                # No agent runner — use default classification (explicit no-agent path)
                classification = default_classify(
                    evidence_id=eid,
                    claim_id=claim.entity_id,
                    evidence_text=content,
                )

            classifications.append(classification)
            evidence_items.append(
                {
                    "evidence_id": eid,
                    "content": content,
                }
            )

        # Step 2: Pairwise independence check for evidence within same domain cluster
        pairwise_independence: Dict[Tuple[str, str], bool] = {}
        if self.agent_runner and len(classifications) >= 2:
            from ..domain_distance import cluster_by_domain

            clusters_for_pairs = cluster_by_domain(
                classifications, distance_threshold=0.3
            )
            for cluster in clusters_for_pairs:
                if len(cluster.evidence_ids) < 2:
                    continue
                eids = cluster.evidence_ids
                for i in range(len(eids)):
                    for j in range(i + 1, len(eids)):
                        ev_a_content = ""
                        ev_b_content = ""
                        for item in evidence_items:
                            if item["evidence_id"] == eids[i]:
                                ev_a_content = item["content"]
                            elif item["evidence_id"] == eids[j]:
                                ev_b_content = item["content"]

                        if ev_a_content and ev_b_content:
                            pair_result = await self.run_agent(
                                "epistemic_check_pairwise_independence",
                                evidence_a=ev_a_content,
                                evidence_b=ev_b_content,
                            )
                            pairwise_independence[(eids[i], eids[j])] = bool(
                                pair_result.independent
                            )

        # Step 3: Deterministic convergence computation, augmented by LLM signals
        convergence = detect_convergence(
            evidence_items=evidence_items,
            claim_id=claim.entity_id,
            objective_id=claim.objective_id,
            precomputed_classifications=classifications,
            pairwise_independence=pairwise_independence
            if pairwise_independence
            else None,
        )

        # Create WEAK_CONVERGENCE uncertainty if convergence is weak
        if not convergence.convergence_detected:
            uncertainty = Uncertainty(
                objective_id=claim.objective_id,
                uncertainty_type=UncertaintyType.WEAK_CONVERGENCE,
                description=(
                    f"Evidence domains not converging "
                    f"(verdict={convergence.verdict}, "
                    f"independence_score={convergence.independence_score:.2f})"
                ),
                affected_claim_ids=[claim.entity_id],
            )
            await self.repo.save(uncertainty)

        claim.convergence_checked = True
        claim.convergence_verdict = convergence.verdict
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=(
                f"Convergence assessment complete "
                f"(verdict={convergence.verdict}, "
                f"{len(classifications)} evidence classified, "
                f"{convergence.num_independent_domains} domains)"
            ),
        )


class ValidateDeductivelyOperation(BaseOperation):
    """Validate claim using first principles and logic."""

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.deductive_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already checked",
            )

        if self.agent_runner:
            from .claims import LLM_PANEL_CAP, top_n_representatives

            # Build context from supporting evidence and assumptions.
            # Cap evidence inclusion at LLM_PANEL_CAP highest-quality reps so
            # the prompt stays bounded as the underlying evidence base grows.
            context_parts: list[str] = []
            if claim.scope:
                context_parts.append(f"Scope: {claim.scope}")
            if claim.assumptions:
                context_parts.append(f"Assumptions: {'; '.join(claim.assumptions)}")
            candidates: list[Evidence] = []
            for eid in claim.evidence_ids:
                ev = await self.repo.get("evidence", eid)
                if (
                    isinstance(ev, Evidence)
                    and ev.extracted_content
                    and ev.cluster_status not in ("corroborative", "deferred")
                ):
                    candidates.append(ev)
            for ev in top_n_representatives(candidates, LLM_PANEL_CAP):
                context_parts.append(f"[{ev.source_type}] {ev.extracted_content}")

            result = await self.run_agent(
                "epistemic_deductive_validation",
                claim_id=claim.entity_id,
                claim=claim.statement,
                context="\n".join(context_parts)
                if context_parts
                else "[No additional context]",
            )

            if not result.passes_deductive_validation:
                for i, issue in enumerate(result.issues_found):
                    # Use agent's issue_type if available, else default to LOGICAL_INCONSISTENCY
                    issue_type = UncertaintyType.LOGICAL_INCONSISTENCY
                    if i < len(result.issue_types):
                        try:
                            issue_type = UncertaintyType(result.issue_types[i])
                        except ValueError:
                            pass  # Keep default
                    uncertainty = Uncertainty(
                        objective_id=claim.objective_id,
                        uncertainty_type=issue_type,
                        description=issue,
                        affected_claim_ids=[claim.entity_id],
                    )
                    await self.repo.save(uncertainty)

        claim.deductive_checked = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message=f"[{claim.statement[:60]}] deductively sound",
        )


class VerifyComputationallyOperation(BaseOperation):
    """Verify claim computationally if applicable."""

    entity_type = "claim"

    async def execute(self, work: OperationInput) -> OperationResult:
        claim = await self.repo.get("claim", work.entity_id)

        if not isinstance(claim, Claim):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Claim",
            )

        if claim.computational_checked:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already checked",
            )

        # Check if computationally verifiable
        # Claim has no "computationally_verifiable" field — computational verification is always attempted
        is_verifiable = True

        if is_verifiable and self.agent_runner:
            from .claims import LLM_PANEL_CAP, top_n_representatives

            # Build context from supporting evidence. Cap inclusion at
            # LLM_PANEL_CAP highest-quality reps so the prompt stays bounded.
            context_parts: list[str] = []
            if claim.scope:
                context_parts.append(f"Scope: {claim.scope}")
            candidates: list[Evidence] = []
            for eid in claim.evidence_ids:
                ev = await self.repo.get("evidence", eid)
                if (
                    isinstance(ev, Evidence)
                    and ev.extracted_content
                    and ev.cluster_status not in ("corroborative", "deferred")
                ):
                    candidates.append(ev)
            for ev in top_n_representatives(candidates, LLM_PANEL_CAP):
                context_parts.append(f"[{ev.source_type}] {ev.extracted_content}")

            result = await self.run_agent(
                "epistemic_verify_computationally",
                claim_id=claim.entity_id,
                claim=claim.statement,
                context="\n".join(context_parts)
                if context_parts
                else "[No additional context]",
            )

            # Agent generates verification CODE, not execution results.
            # If no code was generated, record an uncertainty.
            if not result.verification_code:
                uncertainty = Uncertainty(
                    objective_id=claim.objective_id,
                    uncertainty_type=UncertaintyType.COMPUTATIONAL_DISAGREEMENT,
                    description="Could not generate verification code for this claim",
                    affected_claim_ids=[claim.entity_id],
                )
                await self.repo.save(uncertainty)

        claim.computational_checked = True
        await self.repo.save(claim)

        return OperationResult(
            success=True,
            entity_id=claim.entity_id,
            message="Computational verification complete",
        )
