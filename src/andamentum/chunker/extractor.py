"""Structural-first chunker orchestrator.

Three stages, in order, each cheaper and more reliable than the next:

  1. STRUCTURE — split at markdown headings (deterministic, free).
     Most academic papers and clean web articles finish here.
  2. SEMANTIC — for sections that exceed `target_max`, split at paragraph
     boundaries chosen by largest cosine drops between adjacent paragraph
     embeddings.
  3. JUDGE   — optional. For each cut whose semantic-drop percentile is in
     the grey zone (60–90th), ask a small LLM `keep | merge`. Cuts the
     judge says to merge are removed.

Public entry point: ``extract_units``. The output is a ``ChunkingResult``
with ``Unit`` objects whose ``text`` is byte-identical to a source span —
exactly what the editor and benchmark already consume.
"""

from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from .embeddings import EmbeddingFn, make_ollama_embedder
from .judge import judge_cut
from .prompts import TARGET_MAX_CHARS, TARGET_MIN_CHARS
from .semantic_split import semantic_split_section
from .structural import build_section_tree, find_headings, split_section_recursively
from .types import ChunkingResult, Gap, Unit

# Same shape as before — kept for backward compatibility with existing callers
# (the editor app, the benchmark CLI). The judge stage uses an executor of
# this signature; the judge-disabled path doesn't need it at all.
ExecutorFn = Callable[..., Awaitable[object]]

# Anchor extraction defaults — these are stored on Unit objects so the editor
# can display them and the benchmark can verify against truth files.
_ANCHOR_WORDS = 8


async def extract_units(
    source: str,
    *,
    target_min_chars: int = TARGET_MIN_CHARS,
    target_max_chars: int = TARGET_MAX_CHARS,
    embedding_fn: EmbeddingFn | None = None,
    judge_executor: ExecutorFn | None = None,
    judge_low_pct: float = 0.60,
    judge_high_pct: float = 0.90,
    domain: str = "general",
    # ----- Backward-compat aliases (ignored or remapped) -------------------
    primary_executor: ExecutorFn | None = None,
    backup_executors: list[ExecutorFn] | None = None,
    window_size: int | None = None,
    extension_chars: int | None = None,
    max_iterations: int | None = None,
    lookahead: int | None = None,
) -> ChunkingResult:
    """Chunk `source` into self-contained units sized within the target band.

    Parameters
    ----------
    source:
        The full document text (markdown for papers/web, plain text for
        transcripts). The chunker uses leading `#` headings as primary
        structural cues.
    target_min_chars / target_max_chars:
        Soft size band for units. Sections smaller than `target_min_chars`
        are kept (we don't merge across topics). Sections larger than
        `target_max_chars` are split semantically.
    embedding_fn:
        Async function that returns one embedding per input string. If None,
        defaults to a local Ollama call (``embeddinggemma:latest``).
    judge_executor:
        Optional. If supplied, the LLM judge is consulted for grey-zone
        boundaries. If None, stage 3 is skipped.
    judge_low_pct / judge_high_pct:
        The grey-zone band (cosine-drop percentile range) for which the
        judge is consulted. Defaults: 60–90th percentile.
    domain:
        Currently unused — accepted for API compatibility with the old
        chunker. Future hook for domain-specific rules.

    Returns
    -------
    A ``ChunkingResult`` whose units' ``text`` is byte-identical to a
    contiguous span of ``source``. ``model_calls`` is the count of judge
    calls (0 if no judge_executor was provided).
    """
    # --- Backward-compat: silently accept legacy params -----------------
    _ = (
        primary_executor,
        backup_executors,
        window_size,
        extension_chars,
        max_iterations,
        lookahead,
        domain,
    )
    # If no explicit judge_executor but a primary_executor was passed in
    # (legacy callers), use it as the judge — they already wired up an LLM.
    if judge_executor is None and primary_executor is not None:
        judge_executor = primary_executor

    if not source.strip():
        return ChunkingResult(
            units=[],
            gaps=[],
            total_chars=len(source),
            model_calls=0,
            retries_used=0,
            windows_processed=0,
        )

    # ===== Stage 1: structural split ======================================
    headings = find_headings(source)
    sections = build_section_tree(source, headings)

    # Pieces are (start, end) spans into source — units to materialise.
    pieces: list[tuple[int, int]] = []

    # Preamble: text BEFORE the first heading (if any). Treat as its own
    # piece — typically a paper title + abstract block.
    first_section_start = sections[0].start if sections else len(source)
    if first_section_start > 0 and source[:first_section_start].strip():
        pieces.append((0, first_section_start))

    # Walk top-level sections and split-recursively into structural pieces.
    flat_pieces: list[tuple[int, int]] = []
    for sec in sections:
        for p in split_section_recursively(sec, target_max_chars):
            flat_pieces.append((p.start, p.end))

    pieces.extend(flat_pieces)

    # ===== Stage 2: semantic split for over-budget pieces =================
    judge_calls = 0
    candidate_grey_zone_cuts: list[tuple[int, float, float]] = []
    # Lazy-init embedder only if we have over-budget pieces.
    embedder: EmbeddingFn | None = None
    refined_pieces: list[tuple[int, int]] = []

    for start, end in pieces:
        length = end - start
        if length <= target_max_chars:
            refined_pieces.append((start, end))
            continue

        if embedder is None:
            embedder = embedding_fn or make_ollama_embedder()
        sub_spans, candidates = await semantic_split_section(
            source=source,
            section_start=start,
            section_end=end,
            target_max=target_max_chars,
            target_min=target_min_chars,
            embedding_fn=embedder,
        )
        # Track grey-zone cuts among the chosen ones for stage 3.
        chosen_offsets = {e for s, e in sub_spans[:-1]}  # last span has no cut
        for cand in candidates:
            if cand.cut_offset in chosen_offsets:
                if judge_low_pct <= cand.percentile <= judge_high_pct:
                    candidate_grey_zone_cuts.append(
                        (cand.cut_offset, cand.drop, cand.percentile)
                    )
        refined_pieces.extend(sub_spans)

    # ===== Stage 3: LLM judge on grey-zone cuts ==========================
    if judge_executor is not None and candidate_grey_zone_cuts:
        # Sort by offset so we process spans in order
        cuts_to_judge = sorted({c[0] for c in candidate_grey_zone_cuts})
        cuts_to_remove: set[int] = set()
        for cut in cuts_to_judge:
            verdict = await judge_cut(
                executor=judge_executor,
                source=source,
                cut_offset=cut,
            )
            judge_calls += 1
            if verdict.decision == "merge":
                cuts_to_remove.add(cut)
        if cuts_to_remove:
            refined_pieces = _remove_cuts(refined_pieces, cuts_to_remove)

    # ===== Materialise units + gaps ======================================
    units: list[Unit] = []
    gaps: list[Gap] = []
    cursor = 0
    for start, end in refined_pieces:
        if start < cursor:
            # Should not happen if pieces are sorted/non-overlapping, but
            # guard against it.
            continue
        if start > cursor:
            gap_text = source[cursor:start]
            if gap_text.strip():
                gaps.append(Gap(source_start=cursor, source_end=start, text=gap_text))
        unit_text = source[start:end]
        if not unit_text.strip():
            cursor = end
            continue
        title = _infer_title(unit_text)
        start_anchor = _first_words(unit_text, _ANCHOR_WORDS)
        end_anchor = _last_words(unit_text, _ANCHOR_WORDS)
        units.append(
            Unit(
                id=uuid.uuid4().hex[:12],
                title=title,
                text=unit_text,
                kind="prose",
                source_start=start,
                source_end=end,
                complete=True,
                anchor_match_method="exact",
                metadata={
                    "start_anchor": start_anchor,
                    "end_anchor": end_anchor,
                },
            )
        )
        cursor = end

    if cursor < len(source):
        tail = source[cursor:]
        if tail.strip():
            gaps.append(Gap(source_start=cursor, source_end=len(source), text=tail))

    return ChunkingResult(
        units=units,
        gaps=gaps,
        total_chars=len(source),
        model_calls=judge_calls,
        retries_used=0,
        windows_processed=len(refined_pieces),
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _remove_cuts(
    spans: list[tuple[int, int]], cuts_to_remove: set[int]
) -> list[tuple[int, int]]:
    """Merge adjacent spans whose boundary is in `cuts_to_remove`."""
    if not spans:
        return spans
    out: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        prev_s, prev_e = out[-1]
        if prev_e == s and prev_e in cuts_to_remove:
            out[-1] = (prev_s, e)
        else:
            out.append((s, e))
    return out


def _first_words(text: str, n: int) -> str:
    return " ".join(text.strip().split()[:n])


def _last_words(text: str, n: int) -> str:
    return " ".join(text.strip().split()[-n:])


def _infer_title(text: str) -> str:
    """Pick a title from the first non-empty line, stripping markdown markers."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading `#` and any trailing punctuation common in headings
        line = line.lstrip("#").strip()
        if line:
            return line[:80]
    return _first_words(text, 5) or "(unnamed)"


# ----------------------------------------------------------------------------
# Compatibility shim for existing callers that still construct an executor
# from an AgentRunner. The chunker no longer NEEDS an executor for the main
# path, but this lets the editor app keep its current wiring.
# ----------------------------------------------------------------------------


def make_runner_executor(runner: object) -> ExecutorFn:
    """Build a judge-stage executor from an AgentRunner.

    Wraps `core.run_agent_with_fallback` so per-call validators register
    correctly. The returned executor matches the legacy ExecutorFn signature
    (kwargs: instructions, user_message, output_type, validators).
    """
    from andamentum.core.agents import run_agent_with_fallback

    async def executor(*, instructions, user_message, output_type, validators):
        return await run_agent_with_fallback(
            model=runner.model,  # type: ignore[attr-defined]
            instructions=instructions,
            user_message=user_message,
            output_type=output_type,
            validators=validators,
        )

    return executor
