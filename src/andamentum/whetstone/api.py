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
from typing import Literal, Sequence

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
    mode: Literal["review", "panel", "guidelines", "custom"] = "review",
    n_experts: int = 4,
    panel_disciplines: Sequence[str] | None = None,
    guidelines: str = "",
    custom_criteria: Sequence[str] | None = None,
    check_novelty: bool = False,
    novelty_search_depth: int = 2,
    novelty_cache_dir: Path | None = None,
) -> ReviewResult:
    """Review a document. Returns critical-review findings + a synthesis.

    Two pipelines:

    * ``mode="review"`` (default) — the lens-based critical-review
      pipeline: ``HarvestSource → ChunkAndScan → CriticalRead →
      ReflectAndInvestigate → (EditSections) → Challenge →
      AuthorQuestions → Synthesise``. Cost is one LLM call per
      lens × section + reflection-loop rounds + a few terminal calls.

    * ``mode="panel"`` — simulate a multi-expert review panel.
      ``HarvestSource → ChunkAndScan → ExtractKeywords →
      GenerateExpertPanel → ExpertReview → PanelSynthesise``. Cost is
      ``2N + 2`` LLM calls where ``N`` is the panel size (default 4):
      one keyword extraction + N expert biosketch generations + N
      expert reviews + one panel synthesis. At ``n_experts=4`` that is
      10 calls. Panel mode is intentionally heavier — it is not a
      drop-in replacement for review mode.

    * ``mode="guidelines"`` — evaluate the document against a journal's
      free-text author guidelines. ``HarvestSource → ChunkAndScan →
      ExtractCheckableItems → EvaluateGuidelineItems``. Cost is
      ``1 + N`` LLM calls where ``N`` is the number of rules the
      extractor finds (10-30 typical). Requires ``guidelines`` to be a
      non-empty string. Outputs ``checkable_items`` and
      ``guideline_evaluations``.

    * ``mode="custom"`` — evaluate the document against caller-supplied
      criteria. ``HarvestSource → ChunkAndScan → CustomReviewer``. Cost
      is one LLM call regardless of criterion count (the reviewer fills
      a runtime-built schema with all verdicts at once). Requires
      ``custom_criteria`` to be non-empty. Outputs ``custom_evaluations``.

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
        Ignored in ``mode="panel"``.
    reflection_round_cap:
        Hard upper bound on rounds of the reflection–investigation loop.
        Default 3. The loop typically exits earlier when the senior
        reviewer says "nothing more to do". Ignored in ``mode="panel"``.
    challenge:
        Whether to run the Challenge phase (refute high-severity
        findings). On by default. Ignored in ``mode="panel"``.
    editor:
        Whether to run the Editor phase, generating concrete edits.
        Off by default — adds one LLM call per section. Ignored in
        ``mode="panel"``.
    editor_criteria:
        Editorial criteria for the Editor phase.
    embedding_fn:
        Custom embedding function for the chunker. Defaults to local
        Ollama (``embeddinggemma:latest``) inside the chunker module.
    target_min_chars / target_max_chars:
        Section size band, passed to chunker.extract_units.
    mode:
        Pipeline selector — ``"review"`` (default) or ``"panel"``.
    n_experts:
        In ``mode="panel"``, the cap on how many experts to generate.
        Default 4. If ``ExtractKeywords`` returns more disciplines
        than this, only the first ``n_experts`` are kept.
    panel_disciplines:
        In ``mode="panel"``, an explicit list of disciplines to use
        for the panel. When supplied, the keyword-extraction LLM call
        is skipped and these are used directly. Useful for tests and
        for callers who want to control the panel composition.
    guidelines:
        In ``mode="guidelines"``, the journal's free-text author
        guidelines. Required and must be non-empty when this mode is
        selected. Ignored otherwise.
    custom_criteria:
        In ``mode="custom"``, a sequence of free-text criteria to
        evaluate the document against (e.g.
        ``("originality", "depth of literature", "clarity of methods")``).
        Required and must be non-empty when this mode is selected.
        Each criterion becomes a status + notes field in the runtime
        schema; up to 30 criteria supported. Ignored otherwise.

    Returns
    -------
    ReviewResult
        Findings, edits, author questions, document map, metrics.
        ``deterministic_findings`` is always populated; the other LLM-
        driven fields populate when ``model`` is set. In ``mode="panel"``
        runs, ``expert_profiles``, ``expert_reviews``, and
        ``panel_synthesis`` are populated instead of the lens-driven
        findings.
    """
    _validate_mode_args(mode, model, guidelines, custom_criteria)

    state = ReviewState(
        source=source,
        perspectives=list(perspectives),
        reflection_round_cap=reflection_round_cap,
        challenge_enabled=challenge,
        editor_enabled=editor,
        editor_criteria=list(editor_criteria),
        mode=mode,
        n_experts=n_experts,
        panel_disciplines=list(panel_disciplines) if panel_disciplines else [],
        guidelines_text=guidelines,
        check_novelty=check_novelty,
        novelty_search_depth=novelty_search_depth,
        novelty_cache_dir=novelty_cache_dir,
        custom_criteria=list(custom_criteria) if custom_criteria else [],
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


def _validate_mode_args(
    mode: str,
    model: str | None,
    guidelines: str,
    custom_criteria: Sequence[str] | None,
) -> None:
    """Argument validation for mode-specific requirements.

    Raises
    ------
    ValueError
        With a clear message naming the offending argument when a mode
        is selected without its required input, or when a non-LLM mode
        receives mode-specific data.
    """
    if mode == "guidelines":
        if model is None:
            raise ValueError(
                "mode='guidelines' requires a model — every checkable "
                "item is an LLM call. Pass model=... or pick mode='review' "
                "with model=None for the deterministic-only path."
            )
        if not guidelines.strip():
            raise ValueError(
                "mode='guidelines' requires non-empty guidelines text. "
                "Pass guidelines=<journal author guidelines string>."
            )
    elif mode == "custom":
        if model is None:
            raise ValueError(
                "mode='custom' requires a model — the custom reviewer is "
                "an LLM call. Pass model=... or pick mode='review' with "
                "model=None for the deterministic-only path."
            )
        if not custom_criteria:
            raise ValueError(
                "mode='custom' requires custom_criteria. Pass "
                "custom_criteria=('...', '...') with at least one entry."
            )
        # Reject empty / whitespace-only criterion strings.
        cleaned = [c for c in custom_criteria if c and c.strip()]
        if not cleaned:
            raise ValueError(
                "mode='custom': custom_criteria contained only empty / "
                "whitespace strings."
            )
    else:
        # Mode is review or panel — guard against mode-mismatched extras.
        if guidelines.strip():
            raise ValueError(
                f"guidelines was supplied but mode={mode!r}. "
                "Use mode='guidelines' to evaluate against journal guidelines."
            )
        if custom_criteria:
            raise ValueError(
                f"custom_criteria was supplied but mode={mode!r}. "
                "Use mode='custom' to evaluate against custom criteria."
            )


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
