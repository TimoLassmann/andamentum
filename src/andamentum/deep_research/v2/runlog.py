"""The run log: what happened at each step of a run, recorded as it happens.

``RunLogger`` is the recorder a run carries in its deps (and its ``Asker``);
``RunLog`` / ``LogEvent`` (``types.py``) are the record, with ``to_jsonl`` and
``to_markdown``. The functions below turn a step's outcome into events, so a
graph node stays one line per thing it logs. Every run fills a log; nothing
turns it on. ``write_log`` writes the two files (the CLI and the bench call it).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue

from .types import (
    Candidate,
    CandidateScore,
    Coverage,
    LogEvent,
    LogStep,
    RunLog,
    RunStats,
    SubQuestion,
    VerifyFlag,
)

if TYPE_CHECKING:
    from .strategies import FusionRow

# Titles shown per source and query.
TOP_TITLES = 5
# Works shown in the final ranking.
FINAL_TOP = 20
# Characters of a model's output and of a passage kept in an event.
OUTPUT_CHARS = 500
PASSAGE_CHARS = 160

Pair = tuple[str, str]


def _jsonable(value: object) -> JsonValue:
    """*value* as plain JSON (tuples become lists, unknown objects strings)."""
    loaded: JsonValue = json.loads(json.dumps(value, default=str))
    return loaded


class RunLogger:
    """Appends events to one ``RunLog`` on the run's clock. ``step`` is the
    step the next event belongs to (the graph sets it per node)."""

    def __init__(self, *, clock: Callable[[], float], run_id: str = "") -> None:
        self.clock = clock
        self.started = clock()
        self.step: LogStep = "Setup"
        self.log = RunLog(run_id=run_id)

    def event(
        self,
        event: str,
        message: str,
        *,
        step: LogStep | None = None,
        **data: object,
    ) -> None:
        payload = _jsonable(data)
        assert isinstance(payload, dict)
        self.log.events.append(
            LogEvent(
                seq=len(self.log.events),
                t=max(0.0, self.clock() - self.started),
                step=step or self.step,
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


# -- model calls ----------------------------------------------------------------


def llm_call(
    logger: RunLogger | None,
    *,
    purpose: str,
    model: str,
    prompt: str,
    output: BaseModel | None,
    seconds: float,
    error: str = "",
) -> None:
    if logger is None:
        return
    text = output.model_dump_json() if output is not None else ""
    logger.event(
        "llm_call",
        f"{purpose}: {model}, {len(prompt)} prompt characters, {seconds:.2f} s"
        + (f", failed: {error}" if error else ""),
        purpose=purpose,
        model=model,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        prompt_chars=len(prompt),
        output=text[:OUTPUT_CHARS],
        seconds=round(seconds, 3),
        error=error,
    )


def decision_batch(
    logger: RunLogger | None, *, step: str, instruction: str, probabilities: list[float]
) -> None:
    if logger is None:
        return
    logger.event(
        "decision_batch",
        f"{step}: {len(probabilities)} states asked {instruction!r}",
        purpose=step,
        question=instruction,
        states=len(probabilities),
        p=[round(p, 4) for p in probabilities],
    )


def step_done(
    logger: RunLogger, *, step: str, seconds: float, before: RunStats, after: RunStats
) -> None:
    llm = after.llm_calls - before.llm_calls
    decisions = after.decision_calls - before.decision_calls
    embeddings = after.embedding_calls - before.embedding_calls
    logger.event(
        "step_done",
        f"{step} took {seconds:.2f} s: {llm} language-model, {decisions} decision "
        f"and {embeddings} embedding calls",
        seconds=round(seconds, 3),
        llm_calls=llm,
        decision_calls=decisions,
        embedding_calls=embeddings,
    )


# -- the steps ------------------------------------------------------------------


def records(
    logger: RunLogger,
    *,
    degraded: Sequence[str] = (),
    notes: Sequence[str] = (),
    skipped: Sequence[str] = (),
) -> None:
    for line in degraded:
        logger.event("degraded", f"degraded: {line}", text=line)
    for line in notes:
        logger.event("note", f"note: {line}", text=line)
    for line in skipped:
        logger.event("skipped", f"skipped: {line}", text=line)


def subquestions(logger: RunLogger, planned: Sequence[SubQuestion]) -> None:
    for subquestion in planned:
        logger.event(
            "subquestion",
            f"{subquestion.id} ({subquestion.kind}): {subquestion.text}",
            id=subquestion.id,
            kind=subquestion.kind,
            text=subquestion.text,
        )


def query_hits(
    logger: RunLogger, hits: Sequence[Mapping[str, object]], *, raw_question: bool
) -> None:
    for hit in hits:
        count = hit.get("count")
        error = hit.get("error")
        what = "the question" if raw_question else str(hit.get("subquestion"))
        logger.event(
            "query",
            f"{what}: {hit.get('query')!r} on {hit.get('source')}: "
            + (f"failed ({error})" if error else f"{count} results"),
            raw_question=raw_question,
            subquestion=hit.get("subquestion"),
            query=hit.get("query"),
            source=hit.get("source"),
            count=count,
            top=hit.get("top"),
            error=error,
        )


def admitted(
    logger: RunLogger, *, before: int, after: int, pairs: int, via: str
) -> None:
    logger.event(
        "admitted",
        f"{via}: {after - before} new works ({after} held), {pairs} new "
        "(work, sub-question) pairs waiting",
        via=via,
        new_works=after - before,
        works_held=after,
        new_pairs=pairs,
    )


def screened(
    logger: RunLogger,
    scores: Sequence[CandidateScore],
    *,
    candidates: Mapping[str, Candidate],
    subquestions: Sequence[SubQuestion],
    before: Mapping[Pair, int],
    after: Mapping[str, int],
) -> None:
    """One event per screened candidate, and per sub-question the screen's
    question with every p (the decision requests, summarised)."""
    by_id = {s.id: s for s in subquestions}
    batches: dict[str, list[float]] = {}
    for score in scores:
        pair = (score.work_id, score.subquestion_id)
        candidate = candidates.get(score.work_id)
        title = candidate.work.title if candidate is not None else ""
        batches.setdefault(score.subquestion_id, []).append(score.score)
        logger.event(
            "screened",
            f"{score.subquestion_id}: {title!r} p={score.score:.2f} "
            f"{'kept' if score.kept else 'dropped'}",
            work_id=score.work_id,
            title=title,
            subquestion=score.subquestion_id,
            p=round(score.score, 4),
            kept=score.kept,
            reason=score.reason,
            rank_before=before.get(pair),
            rank_after=after.get(score.work_id) if score.kept else None,
        )
    for subquestion_id, values in batches.items():
        subquestion = by_id.get(subquestion_id)
        logger.event(
            "screen_batch",
            f"{subquestion_id}: {len(values)} candidates screened",
            subquestion=subquestion_id,
            question=subquestion.text if subquestion else "",
            states=len(values),
            p=[round(v, 4) for v in values],
        )


def kept_places(kept: Mapping[str, tuple[float, list[str]]]) -> dict[str, int]:
    """Each kept work's 1-based place by its ranking value (ties by work id)."""
    ordered = sorted(kept.items(), key=lambda item: (-item[1][0], item[0]))
    return {work_id: place for place, (work_id, _) in enumerate(ordered, start=1)}


def final_ranking(
    logger: RunLogger,
    *,
    works: Sequence[str],
    titles: Mapping[str, str],
    rows: Mapping[Pair, FusionRow],
    stop_reason: str,
) -> None:
    """The stop reason and the result's first ``FINAL_TOP`` works with the
    place they reached in every list (per source, dense, screen p) for their
    best sub-question."""
    best: dict[str, tuple[str, FusionRow]] = {}
    for (work_id, subquestion_id), row in rows.items():
        held = best.get(work_id)
        if held is None or row.fused > held[1].fused:
            best[work_id] = (subquestion_id, row)
    entries: list[dict[str, object]] = []
    for place, work_id in enumerate(works[:FINAL_TOP], start=1):
        entry: dict[str, object] = {
            "rank": place,
            "work_id": work_id,
            "title": titles.get(work_id, ""),
        }
        if work_id in best:
            subquestion_id, row = best[work_id]
            entry.update(
                subquestion=subquestion_id,
                fused=round(row.fused, 6),
                ranks=row.ranks,
                p=row.p,
            )
        entries.append(entry)
    logger.event(
        "final_ranking",
        f"stopped: {stop_reason}; the first {len(entries)} of {len(works)} ranked works",
        stop_reason=stop_reason,
        works_ranked=len(works),
        works=entries,
    )


def coverage(logger: RunLogger, verdicts: Sequence[Coverage]) -> None:
    for entry in verdicts:
        state = (
            "covered"
            if entry.covered
            else "unresolvable"
            if entry.unresolvable
            else "open"
        )
        logger.event(
            "coverage",
            f"{entry.subquestion_id}: {state}, {entry.independent_sources} "
            "independent source(s)" + (f"; {entry.note}" if entry.note else ""),
            subquestion=entry.subquestion_id,
            state=state,
            independent_sources=entry.independent_sources,
            note=entry.note,
        )


def flags(logger: RunLogger, raised: Sequence[VerifyFlag]) -> None:
    for flag in raised:
        logger.event(
            "verify_flag",
            f"{flag.kind}: {flag.ref}",
            kind=flag.kind,
            ref=flag.ref,
            detail=flag.detail,
        )
