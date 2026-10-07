"""The run log: what happened at each step of a run, recorded as it happens.

``RunLogger`` is the recorder a run carries (in its ``Asker`` and the worker
functions); ``RunLog`` / ``LogEvent`` (``types.py``) are the record, with
``to_jsonl`` and ``to_markdown``. Every run fills a log; nothing turns it on.
``write_log`` writes the two files (the CLI and the bench call it).

Every event names its step explicitly (``event(..., step=...)``): the loop runs
sub-questions and pages in parallel, so there is no ambient "current step".
The event catalogue (names and data keys) is in the spec section 1.10 and in
``v2/CLAUDE.md``. Text a later stage saw (snippets, fetched text, the window
given to the notes, the answer, the report) is logged whole, because the bench's
loss attribution reads it; only a fetched page is clipped, at ``TEXT_CHARS``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, JsonValue

from .types import LogEvent, LogStep, RunLog

# Characters of a model's output kept in an ``llm_call`` event.
OUTPUT_CHARS = 500
# Characters of one fetched page kept whole in a ``fetched`` event.
TEXT_CHARS = 200_000
# Titles shown per source and query, and works in the final ranking.
TOP_TITLES = 5
FINAL_TOP = 20


def _jsonable(value: object) -> JsonValue:
    """*value* as plain JSON (tuples become lists, unknown objects strings)."""
    loaded: JsonValue = json.loads(json.dumps(value, default=str))
    return loaded


def clipped(text: str, *, limit: int = TEXT_CHARS) -> tuple[str, bool]:
    """*text* cut to *limit* characters, and whether it was cut."""
    return (text, False) if len(text) <= limit else (text[:limit], True)


class RunLogger:
    """Appends events to one ``RunLog`` on the run's clock."""

    def __init__(self, *, clock: Callable[[], float], run_id: str = "") -> None:
        self.clock = clock
        self.started = clock()
        self.log = RunLog(run_id=run_id)

    def event(
        self,
        event: str,
        message: str,
        *,
        step: LogStep,
        **data: object,
    ) -> None:
        payload = _jsonable(data)
        assert isinstance(payload, dict)
        self.log.events.append(
            LogEvent(
                seq=len(self.log.events),
                t=max(0.0, self.clock() - self.started),
                step=step,
                event=event,
                message=message,
                data=payload,
            )
        )

    def snapshot(self) -> RunLog:
        return self.log.model_copy(deep=True)


def write_log(log: RunLog, *, md_path: Path) -> tuple[Path, Path]:
    """Write *log* as ``md_path`` (Markdown) and the ``.jsonl`` beside it
    (``x.log.md`` -> ``x.log.jsonl``); returns both paths."""
    name = md_path.name
    stem = name[: -len(".md")] if name.endswith(".md") else name
    jsonl_path = md_path.with_name(stem + ".jsonl")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(log.to_markdown(), encoding="utf-8")
    jsonl_path.write_text(log.to_jsonl(), encoding="utf-8")
    return md_path, jsonl_path


def llm_call(
    logger: RunLogger | None,
    *,
    purpose: str,
    step: str,
    model: str,
    prompt: str,
    output: BaseModel | None,
    seconds: float,
    error: str = "",
) -> None:
    """One ``llm_call`` event; the step it is filed under is the call's own."""
    if logger is None:
        return
    text = output.model_dump_json() if output is not None else ""
    logger.event(
        "llm_call",
        f"{purpose}: {model}, {len(prompt)} prompt characters, {seconds:.2f} s"
        + (f", failed: {error}" if error else ""),
        step=_step_of(step),
        purpose=purpose,
        model=model,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_chars=len(prompt),
        output=text[:OUTPUT_CHARS],
        seconds=round(seconds, 3),
        error=error,
    )


_LOG_STEP_NAMES: tuple[LogStep, ...] = (
    "Setup",
    "Plan",
    "Search",
    "Expand",
    "Choose",
    "Read",
    "Notes",
    "Reflect",
    "Write",
    "Verify",
    "Finish",
)


def _step_of(step: str) -> LogStep:
    """*step* as a ``LogStep`` (an unknown name is filed under Setup)."""
    for name in _LOG_STEP_NAMES:
        if name == step:
            return name
    return "Setup"


def records(
    logger: RunLogger | None,
    *,
    step: LogStep,
    degraded: Sequence[str] = (),
    notes: Sequence[str] = (),
    skipped: Sequence[str] = (),
) -> None:
    """``degraded`` / ``note`` / ``skipped`` events for the given lines."""
    if logger is None:
        return
    for line in degraded:
        logger.event("degraded", f"degraded: {line}", step=step, text=line)
    for line in notes:
        logger.event("note", f"note: {line}", step=step, text=line)
    for line in skipped:
        logger.event("skipped", f"skipped: {line}", step=step, text=line)
