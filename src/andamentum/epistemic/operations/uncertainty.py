"""Uncertainty resolution operations (Phase 8).

Resolves blocking uncertainties with sibling grouping (cosine dedup)
and concern dedup. When resolution succeeds, applies the same answer
to semantically similar siblings.

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD, MAX_UNCERTAINTY_DEPTH)
Operates on: Uncertainty, Claim, Evidence, Objective entities
"""

from .base import BaseOperation, DEDUP_SIMILARITY_THRESHOLD, OperationResult, WorkItem

from ..entities import (
    Claim,
    Evidence,
    Objective,
    Uncertainty,
)


class ResolveUncertaintyOperation(BaseOperation):
    """Attempt to resolve a blocking uncertainty."""

    entity_type = "uncertainty"

    async def execute(self, work: WorkItem) -> OperationResult:
        uncertainty = await self.repo.get("uncertainty", work.entity_id)

        if not isinstance(uncertainty, Uncertainty):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Uncertainty",
            )

        if uncertainty.resolution is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Already resolved",
            )

        # ── Sibling grouping: find similar unresolved uncertainties ────
        # Before making the LLM call, look up all unresolved blocking
        # uncertainties on the same objective.  Embed their descriptions,
        # find near-duplicates (cosine >= DEDUP_SIMILARITY_THRESHOLD), and after resolution,
        # apply the same answer to the whole group.
        similar_siblings: list["Uncertainty"] = []

        all_unresolved = await self.repo.query(
            "uncertainty",
            objective_id=uncertainty.objective_id,
            resolution=None,
            is_blocking=True,
        )
        siblings = [
            u
            for u in all_unresolved
            if isinstance(u, Uncertainty) and u.entity_id != uncertainty.entity_id
        ]

        if siblings:
            from ..embeddings import embed_texts
            from ..similarity import cosine_similarity

            if not self.embedding_model:
                raise RuntimeError(
                    "embedding_model is required for sibling deduplication. Pass embedding_model= to create_operations()."
                )
            all_descriptions = [uncertainty.description] + [
                s.description for s in siblings
            ]
            embeddings = await embed_texts(all_descriptions, model=self.embedding_model)

            target_emb = embeddings[0]
            for i, sibling in enumerate(siblings):
                sim = cosine_similarity(target_emb, embeddings[i + 1])
                if sim >= DEDUP_SIMILARITY_THRESHOLD:
                    similar_siblings.append(sibling)

        if self.agent_runner:
            # Gather context: affected claims and their evidence
            affected_claims_text: list[str] = []
            evidence_text: list[str] = []
            objective_description = ""

            # Load objective description
            if uncertainty.objective_id:
                try:
                    obj = await self.repo.get("objective", uncertainty.objective_id)
                    if isinstance(obj, Objective):
                        objective_description = obj.description
                except Exception:
                    pass

            # Load affected claims and their evidence
            for cid in uncertainty.affected_claim_ids:
                try:
                    c = await self.repo.get("claim", cid)
                    if isinstance(c, Claim):
                        affected_claims_text.append(
                            f"- [{c.stage.value}] {c.statement} (scope: {c.scope})"
                        )
                        # Gather evidence linked to this claim
                        for eid in c.evidence_ids[:5]:  # Limit to 5 per claim
                            try:
                                ev = await self.repo.get("evidence", eid)
                                if isinstance(ev, Evidence) and ev.extracted_content:
                                    evidence_text.append(
                                        f"[{ev.source_type}] {ev.source_ref}\n{ev.extracted_content}"
                                    )
                            except Exception:
                                continue
                except Exception:
                    continue

            result = await self.run_agent(
                "epistemic_resolve_uncertainty",
                uncertainty_id=uncertainty.entity_id,
                uncertainty_type=uncertainty.uncertainty_type.value,
                description=uncertainty.description,
                affected_claims="\n".join(affected_claims_text)
                if affected_claims_text
                else "[No affected claims]",
                new_evidence="\n\n".join(evidence_text)
                if evidence_text
                else "[No evidence available]",
                objective_context=objective_description or "[No objective context]",
            )

            if result.can_resolve:
                uncertainty.resolve(result.resolution)

                # Apply same resolution to similar siblings
                for sibling in similar_siblings:
                    if sibling.resolution is None:
                        sibling.resolve(result.resolution)
                        await self.repo.save(sibling)

                # Buffer remaining concerns on objective for batch dedup.
                # Instead of creating uncertainty entities immediately (where
                # each concern is deduped pairwise against whatever exists at
                # that moment), we collect all concerns from the entire resolution
                # round and dedup them as a batch in DeduplicateConcernsOperation.
                if result.remaining_concerns:
                    # Compute depth for demotion tracking
                    depth = 0
                    if uncertainty.spawned_from_id:
                        current_id = uncertainty.spawned_from_id
                        for _ in range(10):  # hard safety limit
                            try:
                                parent = await self.repo.get("uncertainty", current_id)
                                depth += 1
                                parent_spawned = parent.spawned_from_id
                                if not parent_spawned:
                                    break
                                current_id = parent_spawned
                            except Exception:
                                break

                    obj = await self.repo.get("objective", uncertainty.objective_id)
                    if isinstance(obj, Objective):
                        for concern in result.remaining_concerns:
                            obj.pending_concerns.append(
                                {
                                    "text": str(concern),
                                    "parent_id": uncertainty.entity_id,
                                    "affected_claim_ids": uncertainty.affected_claim_ids,
                                    "depth": depth + 1,
                                }
                            )
                        await self.repo.save(obj)
            else:
                # Agent assessed this uncertainty and determined it can't be resolved.
                # Mark as resolved with "unresolvable" so the pattern (resolution=None)
                # no longer matches. Note: setting is_blocking=False doesn't persist
                # because model_post_init recomputes it from uncertainty_type on reload.
                uncertainty.resolve("Unresolvable: acknowledged limitation")

                # Also mark similar siblings as unresolvable
                for sibling in similar_siblings:
                    if sibling.resolution is None:
                        sibling.resolve("Unresolvable: acknowledged limitation")
                        await self.repo.save(sibling)

        await self.repo.save(uncertainty)

        # Peirce cycling: when a blocking uncertainty is resolved, the
        # epistemic landscape has changed. Reset scrutiny on affected claims
        # so they re-enter the investigation→scrutiny loop with updated
        # information. Loop safety is guaranteed by three independent caps:
        #   - investigation_count (monotonic, max 3) — never reset
        #   - MAX_ENTITY_ATTEMPTS (per entity-op pair, max 3) — per run
        #   - MAX_UNCERTAINTY_DEPTH (chain depth, max 3) — per chain
        if uncertainty.resolution is not None and uncertainty.is_blocking:
            for cid in uncertainty.affected_claim_ids:
                try:
                    claim = await self.repo.get("claim", cid)
                    if isinstance(claim, Claim) and not claim.abandoned:
                        claim.scrutiny_verdict = None
                        await self.repo.save(claim)
                except Exception:
                    continue

        return OperationResult(
            success=True,
            entity_id=uncertainty.entity_id,
            message=f"Resolution: {uncertainty.resolution or 'assessed as unresolvable'}",
        )
