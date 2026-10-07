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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, JsonValue

from andamentum.references import Work

from .origin import registrable_domain
from .types import Candidate, LogEvent, LogStep, RunLog, SubQuestion, VerifyFlag

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


# -- the events of the loop's steps (spec 1.10) --------------------------------


def subquestions(logger: RunLogger, planned: Sequence[SubQuestion]) -> None:
    for subquestion in planned:
        logger.event(
            "subquestion",
            f"{subquestion.id} ({subquestion.kind}): {subquestion.text}",
            step="Plan",
            id=subquestion.id,
            kind=subquestion.kind,
            text=subquestion.text,
        )


def query_hits(logger: RunLogger, hits: Sequence[Mapping[str, object]]) -> None:
    """One ``query`` row per query and source."""
    for hit in hits:
        error = hit.get("error")
        logger.event(
            "query",
            f"{hit.get('subquestion')}: {hit.get('query')!r} on {hit.get('source')}: "
            + (f"failed ({error})" if error else f"{hit.get('count')} results"),
            step="Search",
            subquestion=hit.get("subquestion"),
            query=hit.get("query"),
            source=hit.get("source"),
            count=hit.get("count"),
            top=hit.get("top"),
            error=error or "",
        )


def search_results(
    logger: RunLogger, results: Sequence[Mapping[str, object]], *, round_number: int
) -> None:
    """One ``search_results`` event per query and source, every result whole."""
    for entry in results:
        rows = entry.get("results")
        logger.event(
            "search_results",
            f"{entry.get('subquestion')}: {entry.get('query')!r} on "
            f"{entry.get('source')}: {len(rows) if isinstance(rows, list) else 0} results",
            step="Search",
            subquestion=entry.get("subquestion"),
            round=round_number,
            query=entry.get("query"),
            source=entry.get("source"),
            results=rows,
        )


def admitted(
    logger: RunLogger, *, before: int, after: int, step: LogStep, via: str
) -> None:
    logger.event(
        "admitted",
        f"{via}: {after - before} new works ({after} held)",
        step=step,
        via=via,
        new_works=after - before,
        works_held=after,
    )


def candidate_row(candidate: Candidate) -> dict[str, object]:
    url = candidate.work.url or (
        f"https://doi.org/{candidate.work.doi}" if candidate.work.doi else ""
    )
    return {
        "work_id": candidate.work.work_id,
        "url": url,
        "title": candidate.work.title,
        "host": registrable_domain(url),
    }


def choice(
    logger: RunLogger,
    *,
    subquestion: str,
    round_number: int,
    shown: Sequence[Candidate],
    picks: Sequence[Candidate],
    reserves: Sequence[Candidate],
    fallback: bool,
) -> None:
    """The ``choice`` event and one ``chosen_work`` row per result shown."""
    chosen = {c.work.work_id for c in picks}
    held = {c.work.work_id for c in reserves}
    logger.event(
        "choice",
        f"{subquestion}: {len(picks)} of {len(shown)} results chosen to read"
        + (" (fallback: top by rank)" if fallback else ""),
        step="Choose",
        subquestion=subquestion,
        round=round_number,
        shown=len(shown),
        picks=[candidate_row(c) for c in picks],
        reserves=[candidate_row(c) for c in reserves],
        not_chosen=[
            candidate_row(c)
            for c in shown
            if c.work.work_id not in chosen and c.work.work_id not in held
        ],
        fallback=fallback,
    )
    for candidate in shown:
        row = candidate_row(candidate)
        logger.event(
            "chosen_work",
            f"{subquestion}: {row['title']!r}",
            step="Choose",
            subquestion=subquestion,
            title=row["title"],
            host=row["host"],
            chosen=candidate.work.work_id in chosen,
        )


def expanded(
    logger: RunLogger,
    *,
    subquestion: str,
    seeds: Sequence[Work],
    found: Sequence[Candidate],
) -> None:
    logger.event(
        "expanded",
        f"{subquestion}: citations of {len(seeds)} works brought {len(found)} candidates",
        step="Expand",
        subquestion=subquestion,
        seeds=[{"work_id": w.work_id, "title": w.title} for w in seeds],
        count=len(found),
        top=[
            {"rank": place, "title": c.work.title}
            for place, c in enumerate(found[:TOP_TITLES], start=1)
        ],
    )


def reflected(
    logger: RunLogger,
    *,
    subquestion: str,
    round_number: int,
    answered: bool,
    answer: str,
    confidence: str,
    gaps: Sequence[str],
    next_queries: Sequence[str],
) -> None:
    logger.event(
        "reflect",
        f"{subquestion} round {round_number}: "
        + ("answered" if answered else "not answered")
        + (f" ({confidence}): {answer}" if answer else ""),
        step="Reflect",
        subquestion=subquestion,
        round=round_number,
        answered=answered,
        answer=answer,
        confidence=confidence,
        gaps=list(gaps),
        next_queries=list(next_queries),
    )


def report_written(
    logger: RunLogger, *, report: str, cited_works: int, findings: int
) -> None:
    logger.event(
        "report",
        f"report written: {len(report)} characters citing {cited_works} works",
        step="Write",
        chars=len(report),
        cited_works=cited_works,
        findings=findings,
        text=report,
    )


def flags(logger: RunLogger, raised: Sequence[VerifyFlag]) -> None:
    for flag in raised:
        logger.event(
            "verify_flag",
            f"{flag.kind}: {flag.ref}",
            step="Verify",
            kind=flag.kind,
            ref=flag.ref,
            detail=flag.detail,
        )


def final_ranking(
    logger: RunLogger,
    *,
    works: Sequence[Work],
    basis: Mapping[str, str],
    source_ranks: Mapping[str, Mapping[str, int]],
    stop_reason: str,
) -> None:
    """The stop reason and the result's first ``FINAL_TOP`` works with why each
    is where it is (``basis``: cited, read or fused) and its source ranks."""
    entries: list[dict[str, object]] = [
        {
            "rank": place,
            "work_id": work.work_id,
            "title": work.title,
            "basis": basis.get(work.work_id, "fused"),
            "source_ranks": dict(source_ranks.get(work.work_id, {})),
        }
        for place, work in enumerate(works[:FINAL_TOP], start=1)
    ]
    logger.event(
        "final_ranking",
        f"stopped: {stop_reason}; the first {len(entries)} of {len(works)} ranked works",
        step="Finish",
        stop_reason=stop_reason,
        works_ranked=len(works),
        works=entries,
    )
