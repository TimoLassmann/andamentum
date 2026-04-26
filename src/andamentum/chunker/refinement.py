"""Escalation chain: window halving → next executor → loud failure.

NO heuristic fallbacks. Every recovery path is agentic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .extractor import ExecutorFn, ExtractionAttempt, _extract_one
from .types import ChunkingFailedError
from .windowing import make_window


@dataclass
class EscalationOutcome:
    """Result of one cursor position's escalation attempt."""

    attempt: ExtractionAttempt
    executor_used: ExecutorFn
    window_size_used: int
    total_calls: int


def _executor_label(executor: Any) -> str:
    """Best-effort label for an executor, for diagnostics."""
    return getattr(executor, "label", None) or repr(executor)[:60]


async def escalate(
    *,
    primary_executor: ExecutorFn,
    backup_executors: list[ExecutorFn],
    source: str,
    cursor: int,
    window_size: int,
    lookahead: int,
    domain: str,
    prior_unit_titles: list[str],
) -> EscalationOutcome:
    """Run extraction at cursor with full escalation: primary → halved → backups.

    Each executor gets one full-size attempt and one halved-size attempt.
    On success, returns immediately. On total failure, raises
    ``ChunkingFailedError`` with diagnostic info.
    """
    attempted_models: list[str] = []
    error_messages: list[str] = []
    total_calls = 0

    executors = [primary_executor, *backup_executors]
    for executor in executors:
        attempted_models.append(_executor_label(executor))

        # Tier 1: full-size window
        window = make_window(
            source, cursor=cursor, window_size=window_size, lookahead=lookahead
        )
        attempt = await _extract_one(
            executor=executor,
            window=window,
            domain=domain,
            prior_unit_titles=prior_unit_titles,
        )
        total_calls += attempt.calls
        if attempt.error is None and attempt.result is not None:
            return EscalationOutcome(
                attempt=attempt,
                executor_used=executor,
                window_size_used=window_size,
                total_calls=total_calls,
            )
        error_messages.append(f"[{_executor_label(executor)} full]: {attempt.error}")

        # Tier 2: halved window
        halved = window_size // 2
        window2 = make_window(
            source, cursor=cursor, window_size=halved, lookahead=lookahead // 2
        )
        attempt2 = await _extract_one(
            executor=executor,
            window=window2,
            domain=domain,
            prior_unit_titles=prior_unit_titles,
        )
        total_calls += attempt2.calls
        if attempt2.error is None and attempt2.result is not None:
            return EscalationOutcome(
                attempt=attempt2,
                executor_used=executor,
                window_size_used=halved,
                total_calls=total_calls,
            )
        error_messages.append(f"[{_executor_label(executor)} halved]: {attempt2.error}")

    # All executors exhausted
    raise ChunkingFailedError(
        cursor=cursor,
        attempted_models=attempted_models,
        last_validator_messages=error_messages,
        message="extraction failed across all configured executors",
    )
