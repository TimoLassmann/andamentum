"""Idempotent uncertainty raising (self-feeding-loop plan, D1).

One helper is the sole path for creating an ``Uncertainty``: it compares the
candidate against the objective's existing uncertainties of the SAME type
that share at least one affected claim — resolved or unresolved — and, on a
near-duplicate, increments the existing entity's ``raise_count`` instead of
creating a second one. This is what stops a claim that keeps failing the
same scrutiny check from accumulating a fresh blocking uncertainty on every
pass (the mechanism diagnosed in ``docs/epistemic/SELF_FEEDING_LOOP_PLAN.md``
§0): the same contradiction raised twice increments a counter instead of
reopening the gate ``ResolveUncertainties`` already cleared.

Similarity is decided the same way sibling grouping and
``DeduplicateConcernsOperation`` already decide it — cosine similarity over
embedded descriptions, compared against the one shared
``DEDUP_SIMILARITY_THRESHOLD`` — so this introduces no second threshold.
When no embedding model is configured, the helper falls back to an exact
match on normalised (lowercased, whitespace-collapsed) description text. It
never falls back to "always create": with no embedding model AND no exact
match, a new uncertainty is still created, but a near-duplicate that merely
differs by embedding similarity would be missed rather than silently merged
into an unrelated existing entity.

This is a repository-level function, not an ``BaseOperation`` subclass and
not a graph node (P1/P2/P6) — callers (currently ``operations/scrutiny.py``,
``operations/analysis.py``, ``operations/concerns.py``, and every blocking
and non-blocking uncertainty raised in ``operations/verification.py`` —
adversarial search, convergence assessment, deductive validation,
computational verification) already run inside an operation and already
carry ``self.embedding_model``; they call this helper in place of
constructing ``Uncertainty(...)`` directly and call ``self.repo.save`` no
further — the helper does the one save that applies. There is no other
persisted ``Uncertainty(...)`` construction site left in ``src/`` outside
this file.

Depends on: embeddings.embed_texts, similarity.cosine_similarity,
operations.base.DEDUP_SIMILARITY_THRESHOLD, and the repository — the same
three names ``operations/uncertainty.py`` already imports for sibling
grouping.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .embeddings import embed_texts
from .entities import Uncertainty, UncertaintyType
from .operations.base import DEDUP_SIMILARITY_THRESHOLD
from .similarity import cosine_similarity

if TYPE_CHECKING:
    from .repository import EpistemicRepository

__all__ = ["raise_uncertainty"]


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace, for the no-embedding-model fallback."""
    return " ".join(text.lower().split())


async def _find_existing(
    candidates: list[Uncertainty],
    *,
    description: str,
    embedding_model: Optional[str],
) -> Optional[Uncertainty]:
    """Return the best near-duplicate among ``candidates``, or None."""
    if not candidates:
        return None

    if embedding_model:
        all_descriptions = [description] + [c.description for c in candidates]
        embeddings = await embed_texts(all_descriptions, model=embedding_model)
        target = embeddings[0]

        best: Optional[Uncertainty] = None
        best_sim = DEDUP_SIMILARITY_THRESHOLD
        for candidate, emb in zip(candidates, embeddings[1:]):
            sim = cosine_similarity(list(target), list(emb))
            if sim >= best_sim:
                best = candidate
                best_sim = sim
        return best

    # No embedding model: exact match on normalised text only — never
    # "always create a new one", but also never a fuzzy guess without the
    # embedding signal that would justify it.
    norm = _normalise(description)
    for candidate in candidates:
        if _normalise(candidate.description) == norm:
            return candidate
    return None


async def raise_uncertainty(
    repo: "EpistemicRepository",
    *,
    objective_id: str,
    uncertainty_type: UncertaintyType,
    description: str,
    affected_claim_ids: list[str],
    created_by: str,
    embedding_model: Optional[str],
    spawned_from_id: Optional[str] = None,
) -> tuple[Uncertainty, bool]:
    """Raise an uncertainty, deduplicating against existing ones (D1).

    Looks up the objective's uncertainties of ``uncertainty_type`` that
    share at least one id with ``affected_claim_ids`` (resolved OR
    unresolved), and compares ``description`` against them. On a
    near-duplicate, increments and saves the existing uncertainty's
    ``raise_count`` and returns it with ``created=False``. Otherwise
    constructs and saves a new ``Uncertainty`` and returns it with
    ``created=True``.

    Args:
        repo: The epistemic repository.
        objective_id: Owning objective, used to scope the dedup lookup.
        uncertainty_type: Type of the candidate uncertainty.
        description: What is uncertain — compared against existing entities.
        affected_claim_ids: Claims this affects; at least one must overlap
            an existing uncertainty's own list for it to be considered a
            candidate duplicate.
        created_by: Provenance tag for a newly created uncertainty.
        embedding_model: Model id for similarity comparison, or None to fall
            back to exact normalised-text matching.
        spawned_from_id: Parent uncertainty id, for a new entity only.

    Returns:
        (uncertainty, created) — the saved existing or newly created entity,
        and whether it was newly created.

    Note: there is no ``is_blocking`` parameter. ``Uncertainty.model_post_init``
    always recomputes ``is_blocking`` from ``uncertainty_type``, so a caller
    cannot actually override it on construction — the field would be a false
    affordance on this API.
    """
    same_target = set(affected_claim_ids)
    candidates = [
        u
        for u in await repo.query("uncertainty", objective_id=objective_id)
        if isinstance(u, Uncertainty)
        and u.uncertainty_type == uncertainty_type
        and same_target & set(u.affected_claim_ids)
    ]

    existing = await _find_existing(
        candidates, description=description, embedding_model=embedding_model
    )
    if existing is not None:
        existing.raise_count += 1
        await repo.save(existing)
        return existing, False

    new_uncertainty = Uncertainty(
        objective_id=objective_id,
        uncertainty_type=uncertainty_type,
        description=description,
        affected_claim_ids=affected_claim_ids,
        created_by=created_by,
        spawned_from_id=spawned_from_id,
    )
    await repo.save(new_uncertainty)
    return new_uncertainty, True
