"""Batch deduplication of buffered remaining concerns (Phase 8b).

After all blocking uncertainties are resolved in a round, any remaining
concerns they generated are buffered on the objective (pending_concerns).
This operation deduplicates them as a batch — grouping near-duplicates,
keeping one representative per theme, filtering against existing
uncertainties — and only then creates new Uncertainty entities for the
survivors.

This prevents the cascade where each resolution call generates slightly
different wordings of the same concern, each passing pairwise dedup
because it only sees what existed at the moment of its creation.

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD, MAX_UNCERTAINTY_DEPTH)
Operates on: Objective (reads pending_concerns), creates Uncertainty entities
"""

import logging

from .base import BaseOperation, DEDUP_SIMILARITY_THRESHOLD, MAX_UNCERTAINTY_DEPTH, OperationResult

from ..entities import (
    Objective,
    Uncertainty,
    UncertaintyType,
)
from ..patterns import WorkItem

logger = logging.getLogger(__name__)


class DeduplicateConcernsOperation(BaseOperation):
    """Batch dedup buffered remaining concerns, then create survivors as uncertainties.

    Fires when:
    - objective.pending_concerns is non-empty
    - No more blocking uncertainties need resolution (the round is done)

    Steps:
    1. Load all pending concerns from the objective
    2. Group them by embedding similarity (same-theme concerns collapse)
    3. Pick one representative (medoid) per group
    4. Filter representatives against existing uncertainties
    5. Create Uncertainty entities only for survivors
    6. Clear the buffer
    """

    entity_type = "objective"

    async def execute(self, work: WorkItem) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        pending = objective.pending_concerns
        if not pending:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="No pending concerns to process",
            )

        # Load existing uncertainty descriptions for this objective
        existing_entities = await self.repo.query(
            "uncertainty",
            objective_id=objective.entity_id,
        )
        existing_descriptions = [
            u.description for u in existing_entities
            if isinstance(u, Uncertainty)
        ]

        pending_texts = [p["text"] for p in pending]

        from ..embeddings import embed_texts
        from ..similarity import group_by_similarity, medoid as find_medoid, cosine_similarity

        if not self.embedding_model:
            raise RuntimeError("embedding_model is required for concern deduplication. Pass embedding_model= to create_operations().")
        # Embed pending concerns + existing descriptions together
        all_texts = pending_texts + existing_descriptions
        embeddings = await embed_texts(all_texts, model=self.embedding_model)

        pending_embeddings = embeddings[: len(pending_texts)]
        existing_embeddings = embeddings[len(pending_texts) :]

        # Group pending concerns among themselves
        groups = group_by_similarity(pending_embeddings, DEDUP_SIMILARITY_THRESHOLD)

        # Pick one representative per group
        representatives: list[dict] = []
        for group in groups:
            rep_idx = find_medoid(pending_embeddings, group)
            representatives.append(pending[rep_idx])

        # Filter representatives against existing uncertainties
        survivors: list[dict] = []
        for rep in representatives:
            rep_idx = pending_texts.index(rep["text"])
            rep_emb = embeddings[rep_idx]
            is_dup = False
            for existing_emb in existing_embeddings:
                if cosine_similarity(list(rep_emb), list(existing_emb)) >= DEDUP_SIMILARITY_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                survivors.append(rep)

        # Create uncertainty entities for survivors only
        created_ids: list[str] = []
        for s in survivors:
            new_uncertainty = Uncertainty(
                objective_id=objective.entity_id,
                uncertainty_type=UncertaintyType.UNKNOWN,
                description=s["text"],
                affected_claim_ids=s.get("affected_claim_ids", []),
                spawned_from_id=s.get("parent_id"),
            )
            await self.repo.save(new_uncertainty)

            # Depth-based demotion: deep children become non-blocking.
            # Must change uncertainty_type (not just is_blocking) because
            # model_post_init recomputes is_blocking from type on every load.
            child_depth = s.get("depth", 0)
            if child_depth >= MAX_UNCERTAINTY_DEPTH and new_uncertainty.is_blocking:
                new_uncertainty.uncertainty_type = UncertaintyType.EVIDENCE_GAP
                new_uncertainty.is_blocking = False
                await self.repo.save(new_uncertainty)

            created_ids.append(new_uncertainty.entity_id)

        # Clear the buffer
        objective.pending_concerns = []
        await self.repo.save(objective)

        logger.info(
            "deduplicate_concerns: %d pending → %d groups → %d after existing check → %d created",
            len(pending),
            len(groups),
            len(survivors),
            len(created_ids),
        )

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=(
                f"Batch dedup: {len(pending)} concerns → {len(representatives)} themes "
                f"→ {len(survivors)} new uncertainties (filtered {len(representatives) - len(survivors)} "
                f"existing duplicates)"
            ),
            created_entities=created_ids,
        )
