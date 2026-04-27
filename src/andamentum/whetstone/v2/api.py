"""Public entry point: ``review_document``.

Single async function. Two required arguments (``source`` and ``model``)
and four optional ones — kept tight so any agent can call it without
guessing.

In Phase 1 the ``model`` argument is accepted but unused — the
deterministic substrate runs without LLM calls. Later phases will use it
to drive the skim / investigate / challenge / synthesise agents.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Sequence

from .deps import EmbeddingFn, ReviewDeps
from .graph import review_graph
from .nodes import HarvestSource
from .schemas import ReviewResult
from .state import ReviewState


async def review_document(
    source: str | Path,
    *,
    model: str | None = None,
    perspectives: Sequence[str] = ("rigorous",),
    hypothesis_budget: int = 30,
    challenge: bool = True,
    editor: bool = False,
    editor_criteria: Sequence[str] = ("clarity", "concision", "grammar"),
    embedding_fn: EmbeddingFn | None = None,
    target_min_chars: int = 2_000,
    target_max_chars: int = 10_000,
) -> ReviewResult:
    """Review a document. Returns confidence-tagged findings + a synthesis.

    Parameters
    ----------
    source:
        URL string ("http(s)://..."), file path (str or pathlib.Path),
        OR raw markdown (caller already has the text).
    model:
        pydantic-ai model string (e.g. "openai:gpt-5.4-nano",
        "ollama:gemma4:31b-nvfp4"). Optional in Phase 1 because no LLM
        calls happen yet; required from Phase 2 onwards.
    perspectives:
        Reviewer personas. Default is one ("rigorous"). Pass multiple
        (e.g. ("rigorous", "statistician", "writer")) for panel mode.
    hypothesis_budget:
        Max LLM-investigated hypotheses. Caps cost per review.
    challenge:
        Whether to run the Challenge phase (refute high-severity
        findings). On by default.
    embedding_fn:
        Custom embedding function for the chunker. Defaults to local
        Ollama (``embeddinggemma:latest``) inside the chunker module.
    target_min_chars / target_max_chars:
        Section size band, passed to chunker.extract_units.

    Returns
    -------
    ReviewResult
        Findings, document map, metrics. ``deterministic_findings`` is
        always populated; ``findings`` and ``summary`` only after later
        phases ship.
    """
    state = ReviewState(
        source=source,
        perspectives=list(perspectives),
        hypothesis_budget=hypothesis_budget,
        challenge_enabled=challenge,
        editor_enabled=editor,
        editor_criteria=list(editor_criteria),
    )
    deps = ReviewDeps(
        model=_resolve_model(model) if model else None,
        embedding_fn=embedding_fn,
        correlation_id=uuid.uuid4().hex[:8],
        target_min_chars=target_min_chars,
        target_max_chars=target_max_chars,
    )

    started = time.monotonic()
    result = await review_graph.run(HarvestSource(), state=state, deps=deps)
    elapsed = time.monotonic() - started

    output: ReviewResult = result.output
    output.metrics.wall_seconds = elapsed
    return output


def _resolve_model(model_string: str):
    """Convert a model string into something pydantic-ai's Agent will accept.

    pydantic-ai accepts model strings directly as of recent versions
    (e.g. ``Agent("openai:gpt-5.4-nano", ...)``). We pass the string
    through unchanged. Future enhancement: route through
    ``core.models.resolve_model_from_args`` for ollama/bedrock setup
    helpers if needed.
    """
    return model_string
