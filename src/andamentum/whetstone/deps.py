"""Run-level dependencies passed to whetstone v2 graph nodes.

Five fields, none required for normal use beyond ``model``. Same shape
as deep_research's ``NodeDeps``: deps are immutable per-run; runtime
state lives in ``ReviewState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from andamentum.core import DEFAULT_EMBEDDING_MODEL

# Re-using the chunker's embedding callable type — same signature.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class ReviewDeps:
    """Per-run config + injected services."""

    model: Any = None  # pydantic-ai model instance; None for Phase 1 (no LLM)
    embedding_fn: EmbeddingFn | None = None
    # Local embedding model for Consolidate's similarity substrate. Used when
    # embedding_fn is not injected; overridable, sensible default.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    correlation_id: str = ""
    target_min_chars: int = 2_000  # passed to chunker.extract_units
    target_max_chars: int = 10_000
    proofread: bool = True  # run proofread.analyze() and append as deterministic findings
