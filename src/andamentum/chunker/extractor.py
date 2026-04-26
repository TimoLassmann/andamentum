"""The extractor: per-window calls and the main loop.

Per-window flow:
  1. Build window with lookahead
  2. Build prompt
  3. Call executor (fresh Agent per call so validators register correctly)
  4. Return ExtractionAttempt (success, skip, or error)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .types import ChunkingResult, Gap, NextUnitResult, Unit
from .validation import find_anchor, make_validator
from .windowing import Window, make_window


ExecutorFn = Callable[..., Awaitable[Any]]
"""Async signature: fn(*, instructions, user_message, output_type, validators) -> Any"""


@dataclass
class ExtractionAttempt:
    """Outcome of a single _extract_one call."""

    result: Optional[NextUnitResult]
    calls: int
    error: Optional[Exception]


async def _extract_one(
    *,
    executor: ExecutorFn,
    window: Window,
    domain: str,
    prior_unit_titles: list[str],
) -> ExtractionAttempt:
    """One LLM call to extract the next unit (or report 'nothing here')."""
    user_prompt = build_user_prompt(
        window_text=window.text,
        domain=domain,
        window_size=window.window_end_offset - window.cursor,
        prior_unit_titles=prior_unit_titles,
    )
    validator = make_validator(window)

    try:
        result = await executor(
            instructions=SYSTEM_PROMPT,
            user_message=user_prompt,
            output_type=NextUnitResult,
            validators=[validator],
        )
        return ExtractionAttempt(result=result, calls=1, error=None)
    except Exception as exc:
        return ExtractionAttempt(result=None, calls=1, error=exc)


async def extract_units(
    source: str,
    *,
    primary_executor: ExecutorFn,
    backup_executors: list[ExecutorFn] | None = None,
    window_size: int = 10_000,
    lookahead: int = 4_000,
    domain: str = "general",
) -> ChunkingResult:
    """Chunk `source` into a list of self-contained units.

    The LLM only points at boundaries; extracted text is byte-identical to
    a contiguous span of the source. Validation drives ModelRetry inside
    each call. On exhausted retries, the system halves the window and/or
    escalates to a backup executor. If all executors fail at any cursor
    position, raises ``ChunkingFailedError``.
    """
    # Deferred import to break the circular dependency:
    # extractor → refinement → extractor (_extract_one, ExecutorFn, ExtractionAttempt)
    from .refinement import escalate

    backup_executors = backup_executors or []

    units: list[Unit] = []
    gaps: list[Gap] = []
    cursor = 0
    total_calls = 0
    windows_processed = 0
    prior_titles: list[str] = []

    while cursor < len(source):
        outcome = await escalate(
            primary_executor=primary_executor,
            backup_executors=backup_executors,
            source=source,
            cursor=cursor,
            window_size=window_size,
            lookahead=lookahead,
            domain=domain,
            prior_unit_titles=prior_titles,
        )
        windows_processed += 1
        total_calls += outcome.total_calls

        result = outcome.attempt.result
        assert result is not None  # escalate raises on total failure

        # Re-build the window the outcome used (for anchor resolution)
        window = make_window(
            source,
            cursor=cursor,
            window_size=outcome.window_size_used,
            lookahead=lookahead,
        )

        if result.found:
            unit, new_cursor = _materialise_unit(source, cursor, result, window)
            if unit is None:
                raise RuntimeError(
                    f"Internal error: validator passed but anchor lookup "
                    f"failed at cursor={cursor}"
                )
            # Account for any gap between previous cursor and this unit's start.
            # Merge with the preceding gap when contiguous (e.g. skip gap
            # followed by a short whitespace-only gap before the next unit).
            if unit.source_start > cursor:
                gap_text = source[cursor : unit.source_start]
                if gaps and gaps[-1].source_end == cursor:
                    prev = gaps[-1]
                    gaps[-1] = Gap(
                        source_start=prev.source_start,
                        source_end=unit.source_start,
                        text=prev.text + gap_text,
                    )
                else:
                    gaps.append(
                        Gap(
                            source_start=cursor,
                            source_end=unit.source_start,
                            text=gap_text,
                        )
                    )
            units.append(unit)
            prior_titles.append(unit.title)
            cursor = new_cursor
        else:
            # Skip — model said this region has no extractable content
            new_cursor = _advance_past_skip(source, cursor, result.skip_to, window)
            if new_cursor <= cursor:
                raise RuntimeError(
                    f"Internal error: skip_to {result.skip_to!r} did not advance "
                    f"cursor at {cursor}"
                )
            gaps.append(
                Gap(
                    source_start=cursor,
                    source_end=new_cursor,
                    text=source[cursor:new_cursor],
                )
            )
            cursor = new_cursor

    return ChunkingResult(
        units=units,
        gaps=gaps,
        total_chars=len(source),
        model_calls=total_calls,
        retries_used=0,  # not tracked granularly yet
        windows_processed=windows_processed,
    )


def _materialise_unit(
    source: str, cursor: int, result: NextUnitResult, window: Window
) -> tuple[Unit | None, int]:
    """Convert a validated NextUnitResult into a Unit by locating anchors."""
    start = find_anchor(result.start_anchor, window.text, search_from=0)
    if start is None:
        return None, cursor
    end = find_anchor(result.end_anchor, window.text, search_from=start.end)
    if end is None:
        return None, cursor

    abs_start = cursor + start.start
    abs_end = cursor + end.end
    method = end.method if end.method != "exact" else start.method

    unit = Unit(
        id=uuid.uuid4().hex[:12],
        title=result.title,
        text=source[abs_start:abs_end],
        kind=result.kind,
        source_start=abs_start,
        source_end=abs_end,
        complete=result.complete,
        anchor_match_method=method,
    )
    return unit, abs_end


def _advance_past_skip(source: str, cursor: int, skip_to: str, window: Window) -> int:
    """Find skip_to in the visible window and advance cursor past it."""
    m = find_anchor(skip_to, window.text, search_from=0)
    if m is None:
        # Fallback: advance to the end of the window so we don't loop
        return min(window.window_end_offset, len(source))
    return cursor + m.end


def make_runner_executor(runner: Any) -> ExecutorFn:
    """Build a production executor from an AgentRunner (uses its model).

    Wraps `core.run_agent_with_fallback` to create a fresh Agent each call,
    so per-call validators register correctly (AgentRunner's caching would
    skip them).
    """
    from andamentum.core.agents import run_agent_with_fallback

    async def executor(*, instructions, user_message, output_type, validators):
        return await run_agent_with_fallback(
            model=runner.model,
            instructions=instructions,
            user_message=user_message,
            output_type=output_type,
            validators=validators,
        )

    return executor
