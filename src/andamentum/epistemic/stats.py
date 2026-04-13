"""Standalone statistics and verification evidence queries.

These functions operate directly on DocumentStore, replacing the
statistics methods that previously lived on WorkItemManager.

Architecture: Layer 4 (Application)
"""

import logging
from typing import Any

from andamentum.document_store import DocumentStore
from .primitives import ClaimStage

logger = logging.getLogger(__name__)


async def get_objective_stats(
    store: DocumentStore, objective_id: str
) -> dict[str, Any]:
    """Get statistics for an objective using metadata queries.

    Args:
        store: DocumentStore for the epistemic database
        objective_id: Objective to get stats for

    Returns:
        Dictionary with counts for evidence, claims, uncertainties, etc.
    """
    stats: dict[str, Any] = {
        "objective_id": objective_id,
        "evidence_count": 0,
        "claims_by_stage": {},
        "uncertainties_unresolved": 0,
        "uncertainties_resolved": 0,
        "decisions_active": 0,
        "decisions_reversed": 0,
        "workitems_queued": 0,
        "workitems_done": 0,
        "workitems_failed": 0,
        "snapshots": 0,
        "artefacts": 0,
    }

    # Count evidence
    evidence_results = await store.find_by_metadata(
        {"objective_id": objective_id, "epistemic_type": "evidence"},
        limit=1000,
    )
    stats["evidence_count"] = len(evidence_results)

    # Count claims by stage
    for stage in ClaimStage:
        claim_results = await store.find_by_metadata(
            {
                "objective_id": objective_id,
                "epistemic_type": "claim",
                "claim_stage": stage.value,
            },
            limit=1000,
        )
        stats["claims_by_stage"][stage.value] = len(claim_results)

    # Count uncertainties
    unresolved = await store.find_by_metadata(
        {
            "objective_id": objective_id,
            "epistemic_type": "uncertainty",
            "is_resolved": False,
        },
        limit=1000,
    )
    resolved = await store.find_by_metadata(
        {
            "objective_id": objective_id,
            "epistemic_type": "uncertainty",
            "is_resolved": True,
        },
        limit=1000,
    )
    stats["uncertainties_unresolved"] = len(unresolved)
    stats["uncertainties_resolved"] = len(resolved)

    # Count decisions
    all_decisions = await store.find_by_metadata(
        {"objective_id": objective_id, "epistemic_type": "decision"},
        limit=1000,
    )
    for d in all_decisions:
        if d.metadata.get("is_reversed"):
            stats["decisions_reversed"] += 1
        else:
            stats["decisions_active"] += 1

    # Count workitems
    for status in ["queued", "done", "failed"]:
        wi_results = await store.find_by_metadata(
            {
                "objective_id": objective_id,
                "epistemic_type": "workitem",
                "workitem_status": status,
            },
            limit=1000,
        )
        stats[f"workitems_{status}"] = len(wi_results)

    # Count snapshots and artefacts
    snap_results = await store.find_by_metadata(
        {"objective_id": objective_id, "epistemic_type": "snapshot"},
        limit=1000,
    )
    stats["snapshots"] = len(snap_results)

    art_results = await store.find_by_metadata(
        {"objective_id": objective_id, "epistemic_type": "artefact"},
        limit=1000,
    )
    stats["artefacts"] = len(art_results)

    return stats


async def get_all_verification_evidence(
    store: DocumentStore, objective_id: str
) -> dict[str, list[Any]]:
    """Get all verification evidence for an objective, grouped by type.

    Queries the repository for adversarial and convergent evidence records
    stored per-claim during verification operations.

    Args:
        store: DocumentStore for the epistemic database
        objective_id: Objective to get verification evidence for

    Returns:
        Dict with keys: adversarial, convergent, computational, temporal, deductive
    """
    from .repository import EpistemicRepository
    from .storage import DocumentStoreAdapter

    repo = EpistemicRepository(DocumentStoreAdapter(store))
    claims = await repo.get_claims_for_objective(objective_id)

    adversarial: list[Any] = []
    convergent: list[Any] = []

    for claim in claims:
        adv = await repo.get_adversarial_evidence_for_claim(claim.entity_id)
        if adv is not None:
            adversarial.append(adv)
        conv = await repo.get_convergent_evidence_for_claim(claim.entity_id)
        if conv is not None:
            convergent.append(conv)

    return {
        "adversarial": adversarial,
        "convergent": convergent,
        "computational": [],
        "temporal": [],
        "deductive": [],
    }
