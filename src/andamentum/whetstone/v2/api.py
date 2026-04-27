"""Public entry point: ``review_document``.

Single async function. Two arguments matter (``source`` and ``model``);
the rest are tuning knobs with sensible defaults.

Without a model the deterministic-only path runs (chunking + structural
extractors). With a model the full critical-review pipeline runs:
lens reading → bounded reflection loop → optional editor → challenge
→ author questions → synthesis.
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
    reflection_round_cap: int = 3,
    challenge: bool = True,
    editor: bool = False,
    editor_criteria: Sequence[str] = ("clarity", "concision", "grammar"),
    embedding_fn: EmbeddingFn | None = None,
    target_min_chars: int = 2_000,
    target_max_chars: int = 10_000,
) -> ReviewResult:
    """Review a document. Returns critical-review findings + a synthesis.

    Parameters
    ----------
    source:
        URL string ("http(s)://..."), file path (str or pathlib.Path),
        OR raw markdown (caller already has the text).
    model:
        pydantic-ai model id (e.g. "openai:gpt-5.4-nano",
        "ollama:gemma4:31b-nvfp4"). Optional — without it, only the
        deterministic substrate runs (no LLM calls).
    perspectives:
        Lens names. Each lens is one configured reviewer personality.
        Available: rigorous, writer, methodology, statistician.
        Default is one ("rigorous"). Multiple lenses run in parallel.
    reflection_round_cap:
        Hard upper bound on rounds of the reflection–investigation loop.
        Default 3. The loop typically exits earlier when the senior
        reviewer says "nothing more to do".
    challenge:
        Whether to run the Challenge phase (refute high-severity
        findings). On by default.
    editor:
        Whether to run the Editor phase, generating concrete edits.
        Off by default — adds one LLM call per section.
    editor_criteria:
        Editorial criteria for the Editor phase.
    embedding_fn:
        Custom embedding function for the chunker. Defaults to local
        Ollama (``embeddinggemma:latest``) inside the chunker module.
    target_min_chars / target_max_chars:
        Section size band, passed to chunker.extract_units.

    Returns
    -------
    ReviewResult
        Findings, edits, author questions, document map, metrics.
        ``deterministic_findings`` is always populated; the other LLM-
        driven fields populate when ``model`` is set.
    """
    state = ReviewState(
        source=source,
        perspectives=list(perspectives),
        reflection_round_cap=reflection_round_cap,
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
    """Convert a model string into a pydantic-ai-ready model object.

    Delegates to ``core.models.resolve_model`` which:
      • constructs an ``OllamaModel`` with the right ``OllamaProvider``
        (honours ``$OLLAMA_BASE_URL``, defaults to localhost) for
        ``ollama:...`` strings;
      • constructs a ``BedrockConverseModel`` with regional inference
        profile for ``bedrock:...`` strings;
      • passes anything else through (``openai:``, ``anthropic:``, etc.) —
        pydantic-ai's own ``infer_model`` handles those natively.
    """
    from andamentum.core import resolve_model

    return resolve_model(model_string)
