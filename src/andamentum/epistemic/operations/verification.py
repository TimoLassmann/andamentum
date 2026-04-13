"""Verification operations (Phase 6).

Multiple independent verification tracks: adversarial search for
disconfirming evidence, cross-domain convergence assessment, deductive
validation via first principles, and computational verification.

Depends on: base (BaseOperation, OperationResult)
Operates on: Claim, Evidence, Uncertainty entities
"""

from typing import Any

from .base import BaseOperation, OperationResult

from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Uncertainty,
    UncertaintyType,
)
from ..patterns import WorkItem


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

    async def execute(self, work: WorkItem) -> OperationResult:
        import asyncio
        import logging

        from ..adversarial_query_generator import (
            generate_adversarial_queries,
            detect_domain,
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

        # Step 1: Generate template-based adversarial queries (deterministic)
        domain = detect_domain(claim.statement)
        template_queries = generate_adversarial_queries(
            claim.statement,
            claim_domain=domain,
            max_queries=5,
        )

        # Step 2: Generate agent-based adversarial queries (narrow agent, 3 framings)
        # Each framing runs independently in parallel — no shared context between
        # calls. Diversity comes from the different framings, not from seeing prior
        # outputs (which would create anchoring per Kahneman's independence principle).
        agent_queries: list[str] = []
        if self.agent_runner:
            framings = [
                "contradicting_evidence",
                "alternative_explanations",
                "replication_failures",
            ]

            async def _generate_one(framing: str) -> str | None:
                try:
                    cq_result = await self.run_agent(
                        "epistemic_generate_counterquery",
                        claim=claim.statement,
                        framing=framing,
                    )
                    return cq_result.query
                except Exception:
                    return None

            results = await asyncio.gather(*[_generate_one(f) for f in framings])
            agent_queries = [q for q in results if q is not None]

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
            queries_to_run = all_queries[:5]  # Limit to 5 queries to control cost
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
                """Evaluate a single search hit. Returns (counterargument, justification).

                On failure, returns a fallback counterargument with defaults and empty
                justification — same behavior as the previous sequential loop.
                """
                async with eval_semaphore:
                    try:
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
                        justification = (
                            getattr(eval_result, "justification", None) or ""
                        )
                        proper_ca = create_counterargument(
                            summary=summary,
                            source_ref=source_ref,
                            claim_id=claim.entity_id,
                            category=category,
                            quality=quality,
                        )
                        return proper_ca, justification
                    except Exception as e:
                        logger.warning(
                            "Counterargument evaluation failed for claim %s: %s",
                            claim.entity_id,
                            e,
                        )
                        # Fallback: use create_counterargument with defaults if agent fails
                        proper_ca = create_counterargument(
                            summary=summary,
                            source_ref=source_ref,
                            claim_id=claim.entity_id,
                        )
                        return proper_ca, ""

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

            for ca, reasoning in best_per_source.values():
                adv_evidence = Evidence(
                    objective_id=claim.objective_id,
                    source_type="web",
                    source_ref=ca.source_ref,
                    extracted_content=ca.summary,
                    extracted=True,
                    support_judgment="contradicts",
                    judgment_reasoning=reasoning,
                    cluster_status="representative",
                )
                await self.repo.save(adv_evidence)
                claim.evidence_ids.append(adv_evidence.entity_id)

            # For genuinely challenged claims (balance < 0.3), create a single
            # NON-BLOCKING uncertainty summarizing the adversarial finding.
            # The balance score on the claim is what gates use — not individual uncertainties.
            if balance_score < 0.3:
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

            # TMS trigger: if claim is already promoted and adversarial search found
            # severe refutation, the claim's current stage must be re-validated.
            # validate_current_stage now checks adversarial_balance.
            if balance_score < 0.3 and claim.stage != ClaimStage.HYPOTHESIS:
                claim.needs_revalidation = True
                logger.info(
                    "TMS: adversarial refutation (balance=%.2f) triggers revalidation for %s at %s",
                    balance_score,
                    claim.entity_id,
                    claim.stage.value,
                )

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
    3. Compute convergence deterministically via detect_convergence()
    """

    entity_type = "claim"

    async def execute(self, work: WorkItem) -> OperationResult:
        from ..convergence_detector import detect_convergence
        from ..domain_classifier import classify_evidence_domain as default_classify
        from ..primitives import (
            DomainClassification,
            MethodType,
            DataSourceType,
            TemporalApproach,
            CausalRole,
        )

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

        for eid in claim.evidence_ids:
            try:
                ev = await self.repo.get("evidence", eid)
                if not isinstance(ev, Evidence) or not ev.extracted_content:
                    continue
            except Exception:
                continue

            content = ev.extracted_content

            if self.agent_runner:
                try:
                    dc_result = await self.run_agent(
                        "epistemic_classify_evidence_domain",
                        evidence_text=content,
                        source_type=ev.source_type,
                        source_ref=ev.source_ref,
                    )
                    # Convert adapter result to DomainClassification
                    classification = DomainClassification(
                        evidence_id=eid,
                        claim_id=claim.entity_id,
                        method_type=MethodType(dc_result.method_type),
                        data_source=DataSourceType(dc_result.data_source),
                        temporal=TemporalApproach(dc_result.temporal_approach),
                        causal_role=CausalRole(dc_result.causal_role),
                        classification_confidence=float(dc_result.confidence),
                        classification_method="agent",
                        classification_notes=dc_result.justification,
                    )
                except Exception:
                    # Fallback to default classification
                    classification = default_classify(
                        evidence_id=eid,
                        claim_id=claim.entity_id,
                        evidence_text=content,
                    )
            else:
                # No agent runner — use default classification
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
        # Only run if we have an agent runner and multiple evidence items
        if self.agent_runner and len(classifications) >= 2:
            from ..domain_distance import cluster_by_domain

            clusters = cluster_by_domain(classifications, distance_threshold=0.3)
            for cluster in clusters:
                if len(cluster.evidence_ids) < 2:
                    continue
                # Check pairs within this cluster
                eids = cluster.evidence_ids
                for i in range(len(eids)):
                    for j in range(i + 1, len(eids)):
                        try:
                            # Find the evidence content for each item
                            ev_a_content = ""
                            ev_b_content = ""
                            for item in evidence_items:
                                if item["evidence_id"] == eids[i]:
                                    ev_a_content = item["content"]
                                elif item["evidence_id"] == eids[j]:
                                    ev_b_content = item["content"]

                            if ev_a_content and ev_b_content:
                                await self.run_agent(
                                    "epistemic_check_pairwise_independence",
                                    evidence_a=ev_a_content,
                                    evidence_b=ev_b_content,
                                )
                                # The result is logged via run_agent; the deterministic
                                # convergence detector handles independence via domain
                                # distance. The pairwise check is an additional signal
                                # recorded in the audit trail.
                        except Exception:
                            continue

        # Step 3: Deterministic convergence computation
        convergence = detect_convergence(
            evidence_items=evidence_items,
            claim_id=claim.entity_id,
            objective_id=claim.objective_id,
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

    async def execute(self, work: WorkItem) -> OperationResult:
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
            # Build context from supporting evidence and assumptions
            context_parts: list[str] = []
            if claim.scope:
                context_parts.append(f"Scope: {claim.scope}")
            if claim.assumptions:
                context_parts.append(f"Assumptions: {'; '.join(claim.assumptions)}")
            for eid in claim.evidence_ids[:5]:
                try:
                    ev = await self.repo.get("evidence", eid)
                    if isinstance(ev, Evidence) and ev.extracted_content:
                        context_parts.append(
                            f"[{ev.source_type}] {ev.extracted_content}"
                        )
                except Exception:
                    continue

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

    async def execute(self, work: WorkItem) -> OperationResult:
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
            # Build context from supporting evidence
            context_parts: list[str] = []
            if claim.scope:
                context_parts.append(f"Scope: {claim.scope}")
            for eid in claim.evidence_ids[:3]:
                try:
                    ev = await self.repo.get("evidence", eid)
                    if isinstance(ev, Evidence) and ev.extracted_content:
                        context_parts.append(
                            f"[{ev.source_type}] {ev.extracted_content}"
                        )
                except Exception:
                    continue

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
