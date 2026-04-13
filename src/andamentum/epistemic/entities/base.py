"""Base Entity - Foundation for all epistemic entities.

All entities inherit from EpistemicEntity which provides:
- Unique ID generation
- Timestamps
- Serialization to/from DocumentStore format
- Entity type discrimination

Architecture: Layer 1 (framework-agnostic)
"""

from abc import ABC
from datetime import datetime
from typing import Any, Self
from pydantic import BaseModel, Field
import uuid


class EpistemicEntity(BaseModel, ABC):
    """Base class for all epistemic entities.

    All entities share:
    - entity_id: Unique identifier (UUID)
    - entity_type: Discriminator for polymorphism
    - objective_id: Parent objective (required for scoping)
    - created_at/updated_at: Timestamps

    Subclasses must define:
    - entity_type as a Literal field
    - Any additional fields specific to the entity
    """

    # Allow subclasses to override entity_type with Literal types
    model_config = {"extra": "forbid"}

    entity_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # Use string union type to allow Literal overrides in subclasses
    entity_type: str = Field(default="base")
    objective_id: str = Field(
        default="", description="Parent objective this entity belongs to"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: Any) -> None:
        """Hook for subclasses to compute denormalized fields after initialization."""
        pass

    def to_document(self) -> tuple[str, dict[str, Any]]:
        """Convert entity to (content, metadata) for DocumentStore.

        By default:
        - content: JSON serialization of the entity
        - metadata: Key fields for filtering/indexing

        Subclasses can override to customize.

        Returns:
            Tuple of (content_string, metadata_dict)
        """
        import json

        # Full entity as content (JSON)
        content = json.dumps(self.model_dump(mode="json"), indent=2)

        # Core metadata fields for filtering
        metadata = self._build_metadata()

        return content, metadata

    def _build_metadata(self) -> dict[str, Any]:
        """Build metadata dict for storage.

        Returns dict with:
        - epistemic_type: Entity type for filtering
        - {entity_type}_id: The entity's ID
        - objective_id: Parent objective
        - created_at: ISO timestamp
        - Any additional fields from _extra_metadata()
        """
        id_field = f"{self.entity_type}_id"
        metadata = {
            "epistemic_type": self.entity_type,
            id_field: self.entity_id,
            "objective_id": self.objective_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        metadata.update(self._extra_metadata())
        return metadata

    def _extra_metadata(self) -> dict[str, Any]:
        """Additional metadata fields for subclasses to add.

        Override in subclasses to add entity-specific metadata
        that should be indexed/filterable.

        Returns:
            Dict of additional metadata fields
        """
        return {}

    @classmethod
    def from_document(cls, content: str, metadata: dict[str, Any]) -> Self:
        """Reconstruct entity from stored document.

        Primary method: Parse content as JSON.
        Fallback: Use metadata fields (for legacy data).

        Args:
            content: Document content (JSON string)
            metadata: Document metadata dict

        Returns:
            Reconstructed entity instance
        """
        import json

        # Try parsing content as JSON first
        try:
            data = json.loads(content)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: reconstruct from metadata
            return cls._from_metadata(content, metadata)

    @classmethod
    def _from_metadata(cls, content: str, metadata: dict[str, Any]) -> Self:
        """Reconstruct from metadata (fallback for legacy data).

        Override in subclasses to handle entity-specific reconstruction.

        Args:
            content: Document content (may not be JSON)
            metadata: Document metadata dict

        Returns:
            Reconstructed entity instance
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _from_metadata for legacy data support"
        )

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dict for storage (legacy API compatibility)."""
        return self._build_metadata()

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
