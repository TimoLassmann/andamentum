"""Immutable dependencies for the epistemic graph.

Passed to every node via ctx.deps. Not modified during execution.
Carries infrastructure that operations need: repo, agent runner,
evidence gatherer, quality scorer.

Architecture: Layer 1 (framework-agnostic, pure dataclass)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..repository import EpistemicRepository

from ..operations_runner import ProgressCallback


@dataclass(frozen=True)
class EpistemicDeps:
    """Infrastructure dependencies for graph execution.

    Frozen — nodes cannot modify these. They're shared configuration
    and connections, not mutable state.
    """

    repo: "EpistemicRepository"
    agent_runner: Any  # AgentRunner or None (for no-LLM mode)
    store: Any = None  # DocumentStore for execution trace recording
    evidence_gatherer: Any | None = None
    quality_scorer: Any | None = None
    embedding_model: str | None = None
    provider: str = "all"
    verbose: bool = False
    progress_callback: Optional[ProgressCallback] = None
