"""Result type for the epistemic graph End node.

This is the value inside End(EpistemicResult(...)) — what the graph
returns when it terminates.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EpistemicResult:
    """Final output of an epistemic graph run."""

    objective_id: str
    status: str  # "complete", "partial", "no_claims"
    successful: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    operations_log: list[dict[str, Any]] = field(default_factory=list)

    # Termination reason: "complete", "no_claims", "partial"
    termination_reason: str = ""

    # Posterior (computed after graph completes, before returning)
    posterior: Optional[Any] = None  # PosteriorReport
