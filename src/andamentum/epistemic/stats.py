"""Standalone statistics and verification evidence queries.

These functions operate directly on the document store's Store, replacing the
statistics methods that previously lived on WorkItemManager.

Architecture: Layer 4 (Application)
"""

import logging
from typing import Any

from andamentum.document_store import Store
from .primitives import ClaimStage

logger = logging.getLogger(__name__)


async def get_objective_stats(store: Store, objective_id: str) -> dict[str, Any]:
    """Get statistics for an objective using metadata queries.

    Args:
        store: Store for the epistemic database
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
    stats["evidence_count"] = await store.count(
        where={"objective_id": objective_id, "epistemic_type": "evidence"}
    )

    # Count claims by stage
    for stage in ClaimStage:
        stats["claims_by_stage"][stage.value] = await store.count(
            where={
                "objective_id": objective_id,
                "epistemic_type": "claim",
                "claim_stage": stage.value,
            }
        )

    # Count uncertainties
    stats["uncertainties_unresolved"] = await store.count(
        where={
            "objective_id": objective_id,
            "epistemic_type": "uncertainty",
            "is_resolved": False,
        }
    )
    stats["uncertainties_resolved"] = await store.count(
        where={
            "objective_id": objective_id,
            "epistemic_type": "uncertainty",
            "is_resolved": True,
        }
    )

    # Count decisions
    all_decisions = await store.list(
        where={"objective_id": objective_id, "epistemic_type": "decision"},
        limit=None,
    )
    for d in all_decisions:
        if d.metadata.get("is_reversed"):
            stats["decisions_reversed"] += 1
        else:
            stats["decisions_active"] += 1

    # Count workitems
    for status in ["queued", "done", "failed"]:
        stats[f"workitems_{status}"] = await store.count(
            where={
                "objective_id": objective_id,
                "epistemic_type": "workitem",
                "workitem_status": status,
            }
        )

    # Count snapshots and artefacts
    stats["snapshots"] = await store.count(
        where={"objective_id": objective_id, "epistemic_type": "snapshot"}
    )
    stats["artefacts"] = await store.count(
        where={"objective_id": objective_id, "epistemic_type": "artefact"}
    )

    return stats


async def get_all_verification_evidence(
    store: Store, objective_id: str
) -> dict[str, list[Any]]:
    """Get all verification evidence for an objective, grouped by type.

    Queries the repository for adversarial and convergent evidence records
    stored per-claim during verification operations.

    Args:
        store: Store for the epistemic database
        objective_id: Objective to get verification evidence for

    Returns:
        Dict with keys: adversarial, convergent, computational, temporal, deductive
    """
    from .repository import EpistemicRepository

    repo = EpistemicRepository(store)
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
