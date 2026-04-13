"""Epistemic Entities - Pattern-Driven Architecture.

This package contains all epistemic entity classes with:
- Unified base class with serialization
- Denormalized state fields for pattern matching
- Proper validation via Pydantic

Architecture: Layer 1 (framework-agnostic)
"""

from .base import EpistemicEntity
from .objective import Objective
from .evidence import Evidence
from .claim import Claim, ClaimStage
from .uncertainty import Uncertainty, UncertaintyType, UncertaintyScope, BLOCKING_TYPES
from .decision import Decision
from .snapshot import Snapshot
from .artefact import Artefact

# Entity registry for polymorphic deserialization
ENTITY_CLASSES: dict[str, type[EpistemicEntity]] = {
    "objective": Objective,
    "evidence": Evidence,
    "claim": Claim,
    "uncertainty": Uncertainty,
    "decision": Decision,
    "snapshot": Snapshot,
    "artefact": Artefact,
}

__all__ = [
    # Base
    "EpistemicEntity",
    "ENTITY_CLASSES",
    # Entities
    "Objective",
    "Evidence",
    "Claim",
    "ClaimStage",
    "Uncertainty",
    "UncertaintyType",
    "UncertaintyScope",
    "BLOCKING_TYPES",
    "Decision",
    "Snapshot",
    "Artefact",
]
