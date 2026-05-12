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
    evidence_gatherer: Any | None = None
    quality_scorer: Any | None = None
    embedding_model: str | None = None
    provider: str = "all"
    verbose: bool = False
    progress_callback: Optional[ProgressCallback] = None
    # Provider registry (``{name: instance}``) — only consulted by the
    # ``dispatch_mode="new"`` path in DispatchGatherOperation. The legacy
    # path accesses providers via ``evidence_gatherer`` (CompositeGatherer
    # already wraps the same dict). Both paths therefore work whether or
    # not this field is set, but new-mode requires it to be populated.
    providers: dict[str, Any] | None = None
