"""Epistemic Repository - Unified CRUD interface for all entities.

The repository provides a single interface for all entity operations,
replacing scattered loading code across operations.

Features:
- Type-safe entity loading/saving
- Filter-based queries with special syntax
- Automatic serialization/deserialization

Architecture: Layer 1 (framework-agnostic)
"""

from pathlib import Path
from typing import Any, Optional, TypeVar, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .storage import StorageBackend
    from .primitives import AdversarialEvidence, ConvergentEvidence

from .entities import (
    EpistemicEntity,
    ENTITY_CLASSES,
    Objective,
    Evidence,
    Claim,
    Uncertainty,
    Decision,
    Snapshot,
    Artefact,
)


logger = logging.getLogger(__name__)

# Type variable for entity methods
T = TypeVar("T", bound=EpistemicEntity)


class EntityNotFoundError(Exception):
    """Raised when an entity is not found."""

    def __init__(self, entity_type: str, entity_id: str):
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} not found: {entity_id}")


class EpistemicRepository:
    """Single interface for all entity operations.

    Usage:
        repo = EpistemicRepository(store)

        # Get single entity
        claim = await repo.get("claim", "claim-123")

        # Query entities with filters
        claims = await repo.query("claim", stage="supported", scrutiny_verdict=None)

        # Save entity (create or update)
        await repo.save(claim)

        # Delete entity
        await repo.delete("claim", "claim-123")

    Filter syntax:
        - Exact match: field=value
        - None check: field=None
        - List contains: field__contains=value (in-memory)
        - Comparison: field__gte=value, field__lte=value, etc. (in-memory)
    """

    def __init__(self, store: "StorageBackend"):
        """Initialize repository with a storage backend.

        Args:
            store: Any object implementing the StorageBackend protocol.
                   DocumentStore satisfies this naturally.
        """
        self.store = store
        self._entity_classes = ENTITY_CLASSES

    @classmethod
    async def for_database(cls, name: str, db_dir: Path | None = None) -> "EpistemicRepository":
        """Create a repository backed by a persistent named database.

        Args:
            name: Database name (e.g., "benchmark_q1_brca1").
                  Creates file at db_dir/{name}.db
            db_dir: Custom directory for the database file.
                    Defaults to ~/.config/mosaic/databases/

        Returns:
            Repository ready for use.
        """
        from document_store import DocumentStore

        from .storage import DocumentStoreAdapter

        store = DocumentStore.for_database(name, db_dir=db_dir)
        await store.initialize()
        return cls(DocumentStoreAdapter(store))

    async def get(self, entity_type: str, entity_id: str) -> EpistemicEntity:
        """Load a single entity by ID.

        Args:
            entity_type: Type of entity (e.g., "claim", "evidence")
            entity_id: The entity's ID

        Returns:
            The loaded entity

        Raises:
            EntityNotFoundError: If entity not found
            KeyError: If entity_type is unknown
        """
        if entity_type not in self._entity_classes:
            raise KeyError(f"Unknown entity type: {entity_type}")

        id_field = f"{entity_type}_id"
        docs = await self.store.find_by_metadata({
            "epistemic_type": entity_type,
            id_field: entity_id,
        }, limit=1)

        if not docs:
            raise EntityNotFoundError(entity_type, entity_id)

        doc = await self.store.read(docs[0].doc_id)
        if not doc:
            raise EntityNotFoundError(entity_type, entity_id)

        cls = self._entity_classes[entity_type]
        return cls.from_document(doc.content, doc.metadata.metadata if doc.metadata else {})

    async def query(
        self,
        entity_type: str,
        **filters: Any
    ) -> list[EpistemicEntity]:
        """Find entities matching filters.

        Supports:
        - Exact match: field=value
        - None check: field=None
        - List contains: field__contains=value (in-memory, see plan 14.3)
        - Comparison: field__gte=value, field__lte=value, field__gt=value, field__lt=value (in-memory)

        Args:
            entity_type: Type of entity to query
            **filters: Field=value filters

        Returns:
            List of matching entities
        """
        if entity_type not in self._entity_classes:
            raise KeyError(f"Unknown entity type: {entity_type}")

        # Separate database filters from in-memory filters
        db_filters: dict[str, Any] = {"epistemic_type": entity_type}
        in_memory_filters: dict[str, Any] = {}

        for key, value in filters.items():
            if key.endswith("__contains"):
                in_memory_filters[key] = value
            elif key.endswith("__gte"):
                in_memory_filters[key] = value
            elif key.endswith("__lte"):
                in_memory_filters[key] = value
            elif key.endswith("__gt"):
                in_memory_filters[key] = value
            elif key.endswith("__lt"):
                in_memory_filters[key] = value
            else:
                db_filters[key] = value

        # Query database
        docs = await self.store.find_by_metadata(db_filters)

        # Load and filter entities
        cls = self._entity_classes[entity_type]
        entities: list[EpistemicEntity] = []

        for doc_meta in docs:
            doc = await self.store.read(doc_meta.doc_id)
            if not doc:
                continue

            entity = cls.from_document(doc.content, doc.metadata.metadata if doc.metadata else {})

            # Apply in-memory filters
            if self._passes_in_memory_filters(entity, in_memory_filters):
                entities.append(entity)

        return entities

    def _passes_in_memory_filters(
        self,
        entity: EpistemicEntity,
        filters: dict[str, Any]
    ) -> bool:
        """Check if entity passes all in-memory filters.

        Args:
            entity: Entity to check
            filters: In-memory filter dict

        Returns:
            True if entity passes all filters
        """
        for filter_key, expected in filters.items():
            if filter_key.endswith("__contains"):
                field = filter_key[:-10]
                field_value = getattr(entity, field)
                if expected not in field_value:
                    return False

            elif filter_key.endswith("__gte"):
                field = filter_key[:-5]
                field_value = getattr(entity, field)
                if field_value < expected:
                    return False

            elif filter_key.endswith("__lte"):
                field = filter_key[:-5]
                field_value = getattr(entity, field)
                if field_value > expected:
                    return False

            elif filter_key.endswith("__gt"):
                field = filter_key[:-4]
                field_value = getattr(entity, field)
                if field_value <= expected:
                    return False

            elif filter_key.endswith("__lt"):
                field = filter_key[:-4]
                field_value = getattr(entity, field)
                if field_value >= expected:
                    return False

        return True

    async def save(self, entity: EpistemicEntity) -> str:
        """Save entity (create or update).

        Updates updated_at timestamp and recomputes denormalized fields.

        Args:
            entity: Entity to save

        Returns:
            Document ID
        """
        # Touch timestamp and recompute denormalized fields
        entity.touch()
        entity.model_post_init(None)

        # Convert to document format
        content, metadata = entity.to_document()
        id_field = f"{entity.entity_type}_id"

        # Check if exists
        existing = await self.store.find_by_metadata({
            "epistemic_type": entity.entity_type,
            id_field: entity.entity_id,
        }, limit=1)

        if existing:
            # Update existing
            await self.store.update(
                existing[0].doc_id,
                new_content=content,
                metadata=metadata,
            )
            return existing[0].doc_id
        else:
            # Create new
            return await self.store.add(
                file_path=f"{entity.entity_type}_{entity.entity_id[:8]}.json",
                content=content,
                title=f"{entity.entity_type.title()}: {entity.entity_id[:8]}",
                metadata=metadata,
            )

    async def delete(self, entity_type: str, entity_id: str) -> bool:
        """Delete an entity.

        Args:
            entity_type: Type of entity
            entity_id: The entity's ID

        Returns:
            True if deleted, False if not found
        """
        id_field = f"{entity_type}_id"
        docs = await self.store.find_by_metadata({
            "epistemic_type": entity_type,
            id_field: entity_id,
        }, limit=1)

        if not docs:
            return False

        await self.store.delete(docs[0].doc_id)
        return True

    async def exists(self, entity_type: str, entity_id: str) -> bool:
        """Check if entity exists without loading.

        Args:
            entity_type: Type of entity
            entity_id: The entity's ID

        Returns:
            True if exists
        """
        id_field = f"{entity_type}_id"
        docs = await self.store.find_by_metadata({
            "epistemic_type": entity_type,
            id_field: entity_id,
        }, limit=1)
        return len(docs) > 0

    async def count(self, entity_type: str, **filters: Any) -> int:
        """Count entities matching filters.

        Args:
            entity_type: Type of entity
            **filters: Field=value filters

        Returns:
            Count of matching entities
        """
        entities = await self.query(entity_type, **filters)
        return len(entities)

    # Convenience methods for typed access

    async def get_objective(self, objective_id: str) -> Objective:
        """Get objective by ID."""
        entity = await self.get("objective", objective_id)
        assert isinstance(entity, Objective)
        return entity

    async def get_claim(self, claim_id: str) -> Claim:
        """Get claim by ID."""
        entity = await self.get("claim", claim_id)
        assert isinstance(entity, Claim)
        return entity

    async def get_evidence(self, evidence_id: str) -> Evidence:
        """Get evidence by ID."""
        entity = await self.get("evidence", evidence_id)
        assert isinstance(entity, Evidence)
        return entity

    async def get_uncertainty(self, uncertainty_id: str) -> Uncertainty:
        """Get uncertainty by ID."""
        entity = await self.get("uncertainty", uncertainty_id)
        assert isinstance(entity, Uncertainty)
        return entity

    async def get_decision(self, decision_id: str) -> Decision:
        """Get decision by ID."""
        entity = await self.get("decision", decision_id)
        assert isinstance(entity, Decision)
        return entity

    async def get_snapshot(self, snapshot_id: str) -> Snapshot:
        """Get snapshot by ID."""
        entity = await self.get("snapshot", snapshot_id)
        assert isinstance(entity, Snapshot)
        return entity

    async def get_artefact(self, artefact_id: str) -> Artefact:
        """Get artefact by ID."""
        entity = await self.get("artefact", artefact_id)
        assert isinstance(entity, Artefact)
        return entity

    # Query convenience methods

    async def get_claims_for_objective(
        self,
        objective_id: str,
        **filters: Any
    ) -> list[Claim]:
        """Get all claims for an objective with optional filters."""
        entities = await self.query("claim", objective_id=objective_id, **filters)
        return [e for e in entities if isinstance(e, Claim)]

    async def get_evidence_for_objective(
        self,
        objective_id: str,
        **filters: Any
    ) -> list[Evidence]:
        """Get all evidence for an objective with optional filters."""
        entities = await self.query("evidence", objective_id=objective_id, **filters)
        return [e for e in entities if isinstance(e, Evidence)]

    async def get_uncertainties_for_objective(
        self,
        objective_id: str,
        **filters: Any
    ) -> list[Uncertainty]:
        """Get all uncertainties for an objective with optional filters."""
        entities = await self.query("uncertainty", objective_id=objective_id, **filters)
        return [e for e in entities if isinstance(e, Uncertainty)]

    async def get_blocking_uncertainties_for_claim(
        self,
        claim_id: str
    ) -> list[Uncertainty]:
        """Get unresolved blocking uncertainties affecting a claim."""
        # First get all uncertainties that might affect this claim
        all_uncertainties = await self.query(
            "uncertainty",
            resolution=None,
            is_blocking=True,
        )

        # Filter to those affecting this claim (in-memory, see plan 14.3)
        return [
            u for u in all_uncertainties
            if isinstance(u, Uncertainty) and claim_id in u.affected_claim_ids
        ]

    async def get_decisions_for_objective(
        self,
        objective_id: str,
        include_reversed: bool = False,
    ) -> list[Decision]:
        """Get decisions for an objective, optionally including reversed ones."""
        entities = await self.query("decision", objective_id=objective_id)
        if include_reversed:
            return [e for e in entities if isinstance(e, Decision)]
        return [e for e in entities if isinstance(e, Decision) and not e.is_reversed]

    async def get_artefacts_for_objective(
        self,
        objective_id: str,
    ) -> list[Artefact]:
        """Get all artefacts for an objective."""
        entities = await self.query("artefact", objective_id=objective_id)
        return [e for e in entities if isinstance(e, Artefact)]

    async def get_blocking_uncertainties(
        self,
        objective_id: str,
    ) -> list[Uncertainty]:
        """Get unresolved blocking uncertainties for an objective."""
        entities = await self.query(
            "uncertainty",
            objective_id=objective_id,
            is_blocking=True,
            is_resolved=False,
        )
        return [e for e in entities if isinstance(e, Uncertainty)]

    async def save_adversarial_evidence(self, adv: "AdversarialEvidence") -> str:
        """Persist AdversarialEvidence so report generation can retrieve it."""
        metadata = adv.to_metadata()
        content = adv.explanation or "Adversarial evidence assessment"

        # Upsert: replace if one already exists for this claim
        existing = await self.store.find_by_metadata({
            "epistemic_type": "adversarial_evidence",
            "claim_id": adv.claim_id,
        }, limit=1)

        if existing:
            await self.store.update(
                existing[0].doc_id,
                new_content=content,
                metadata=metadata,
            )
            return existing[0].doc_id

        return await self.store.add(
            file_path=f"adversarial_{adv.evidence_id[:8]}.json",
            content=content,
            title=f"Adversarial: {adv.claim_id[:8]}",
            metadata=metadata,
        )

    async def get_adversarial_evidence_for_claim(
        self,
        claim_id: str,
    ) -> Optional["AdversarialEvidence"]:
        """Get adversarial evidence for a claim.

        Returns the full AdversarialEvidence reconstructed from stored metadata,
        or None if no adversarial search was performed for this claim.
        """
        refs = await self.store.find_by_metadata({
            "epistemic_type": "adversarial_evidence",
            "claim_id": claim_id,
        }, limit=1)

        if not refs:
            return None

        doc = await self.store.read(refs[0].doc_id)
        if not doc or not doc.metadata:
            return None

        meta = doc.metadata.metadata if hasattr(doc.metadata, "metadata") else doc.metadata
        from .primitives import AdversarialEvidence
        return AdversarialEvidence.from_metadata(meta)  # type: ignore[arg-type]

    async def get_convergent_evidence_for_claim(
        self,
        claim_id: str,
    ) -> Optional["ConvergentEvidence"]:
        """Get convergent evidence for a claim.

        Returns the full ConvergentEvidence reconstructed from stored metadata,
        or None if no convergence assessment was performed for this claim.
        """
        refs = await self.store.find_by_metadata({
            "epistemic_type": "convergent_evidence",
            "claim_id": claim_id,
        }, limit=1)

        if not refs:
            return None

        doc = await self.store.read(refs[0].doc_id)
        if not doc or not doc.metadata:
            return None

        meta = doc.metadata.metadata if hasattr(doc.metadata, "metadata") else doc.metadata
        from .primitives import ConvergentEvidence
        return ConvergentEvidence.from_metadata(meta)  # type: ignore[arg-type]
