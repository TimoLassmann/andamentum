"""The data a v2 research run reads and produces. Flat, JSON-serialisable.

A run is a table of evidence, not a piece of prose: sub-questions, the criteria
compiled for each, the works found and how each scored, verbatim passages,
findings that quote a passage, coverage per sub-question, conflicts, and the
flags verification raised. The report cites findings by id (``[F12]``);
:class:`ResearchResult` carries all of it, so a consumer can check a sentence
against the quote behind it instead of trusting the prose.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    model_validator,
)

from andamentum.references import Work

Mode = Literal["web", "scholarly", "both"]
SubQuestionKind = Literal["topical", "navigational", "metadata"]
CriterionKind = Literal["must", "should", "exclude"]
CriterionScope = Literal["candidate", "passage"]
SourceType = Literal[
    "primary_study",
    "review",
    "guideline",
    "news",
    "vendor",
    "forum",
    "dataset",
    "other",
]
SOURCE_TYPES: tuple[SourceType, ...] = (
    "primary_study",
    "review",
    "guideline",
    "news",
    "vendor",
    "forum",
    "dataset",
    "other",
)
VerifyKind = Literal[
    "unsupported",  # the cited quotes do not state what the sentence claims
    "unknown_finding",  # the report cites a finding id that does not exist
    "quote_missing",  # a finding's quote is not in its passage
    "dead_link",  # a URL in the report does not resolve
    "doi_unresolved",  # a DOI in the report does not resolve
    "retracted",  # a retracted work is cited
    "single_source",  # a claim rests on fewer than MIN_INDEPENDENT_SOURCES sources
]

# A sub-question is covered when this many independent sources support it.
MIN_INDEPENDENT_SOURCES = 2
# The steps of the graph a run's time and calls are counted under, in order.
STEPS = ("Plan", "Search", "Screen", "Expand", "Read", "Coverage", "Write", "Verify")

DEFAULT_MAX_ROUNDS = 3
DEFAULT_QUERIES_PER_SUBQ = 3
DEFAULT_CANDIDATES_PER_ROUND = 60
DEFAULT_EXPAND_PER_ROUND = 40
DEFAULT_PASSAGES_PER_WORK = 6
DEFAULT_READ_PER_ROUND = 20
DEFAULT_MAX_LLM_CALLS = 200
DEFAULT_MAX_DECISION_CALLS = 5000
DEFAULT_MAX_WALL_S = 3600.0
DEFAULT_MAX_FETCH_BYTES = 50_000_000

_FINDING_REFERENCE = re.compile(r"\[\s*(F\d+(?:\s*[,;]\s*F\d+)*)\s*\]")


class SubQuestion(BaseModel):
    """One part of the question; the unit of coverage."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: SubQuestionKind = "topical"


class Criterion(BaseModel):
    """An atomic yes/no question about ONE property of a candidate or passage."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: CriterionKind
    weight: int = Field(default=1, ge=1, le=3)
    applies_to: CriterionScope = "candidate"


class Candidate(BaseModel):
    """A work a source returned, with where it came from.

    ``work`` holds whatever identifiers the source had. After
    ``identity.merge`` one candidate stands for one work and the provenance
    lists are the union of every record that was merged into it.
    """

    work: Work
    sources: list[str] = Field(min_length=1)
    snippet: str = ""
    queries: list[str] = Field(default_factory=list)
    subquestion_ids: list[str] = Field(default_factory=list)
    rank: int | None = None
    # The best (smallest, 1-based) position this work reached in each ranked
    # list it came from, by list name: a source's search results (``openalex``)
    # or a citation hop (``openalex:citations``). Read by rank fusion.
    source_ranks: dict[str, int] = Field(default_factory=dict)

    @field_serializer("snippet")
    def _licensed_snippet(self, value: str) -> str:
        """Every dump follows the licence, as ``Work.abstract`` does: the snippet
        is the first characters of the abstract, so a candidate of a
        ``no_redistribute`` work never exports it. A frozen run reads the text
        back from the snapshot store (``steps.restore_licensed_text``)."""
        return value if self.work.redistributable else ""


class CandidateScore(BaseModel):
    """How one work fared against the criteria of one sub-question, and why."""

    work_id: str = Field(min_length=1)
    subquestion_id: str = Field(min_length=1)
    criterion_p: dict[str, float] = Field(default_factory=dict)
    source_type: SourceType | None = None
    score: float = Field(ge=0.0, le=1.0)
    kept: bool
    reason: str = ""

    @model_validator(mode="after")
    def _probabilities(self) -> CandidateScore:
        for criterion_id, p in self.criterion_p.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"criterion {criterion_id!r}: p={p} is not in [0, 1]")
        return self


def passage_id_for(work_id: str, text: str) -> str:
    """The content address of a passage: sha256 of its work id and its text."""
    return hashlib.sha256((work_id + text).encode("utf-8")).hexdigest()


class Passage(BaseModel):
    """A verbatim span of a work's text. ``text`` is exactly ``[start:end]``."""

    passage_id: str
    work_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int
    text: str = Field(min_length=1)
    retrieved_at: float = 0.0

    @model_validator(mode="after")
    def _addressed_and_sliced(self) -> Passage:
        if self.end - self.start != len(self.text):
            raise ValueError(
                f"passage offsets [{self.start}:{self.end}] do not span its "
                f"{len(self.text)} characters"
            )
        if self.passage_id != passage_id_for(self.work_id, self.text):
            raise ValueError("passage_id is not sha256(work_id + text)")
        return self

    @classmethod
    def make(
        cls, *, work_id: str, start: int, text: str, retrieved_at: float = 0.0
    ) -> Passage:
        """A passage at *start* in its work's text, with its id and end computed."""
        return cls(
            passage_id=passage_id_for(work_id, text),
            work_id=work_id,
            start=start,
            end=start + len(text),
            text=text,
            retrieved_at=retrieved_at,
        )


class Finding(BaseModel):
    """A claim and the verbatim quote that supports it."""

    finding_id: str = Field(pattern=r"^F\d+$")
    subquestion_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    passage_id: str = Field(min_length=1)
    work_id: str = Field(min_length=1)
    criterion_ids: list[str] = Field(default_factory=list)


class Conflict(BaseModel):
    """Two findings of one sub-question that contradict each other."""

    subquestion_id: str = Field(min_length=1)
    finding_ids: list[str] = Field(min_length=2, max_length=2)
    note: str = ""


class Coverage(BaseModel):
    """How well one sub-question is supported, by independent sources."""

    subquestion_id: str = Field(min_length=1)
    independent_sources: int = Field(ge=0)
    per_criterion: dict[str, int] = Field(default_factory=dict)
    covered: bool
    unresolvable: bool = False
    note: str = ""


class VerifyFlag(BaseModel):
    """One thing verification found wrong. ``ref`` names it: a finding id, a
    work id, a URL, or the report sentence."""

    kind: VerifyKind
    ref: str
    detail: str = ""


class Budget(BaseModel):
    """Every bound on a run, explicit with a named default.

    Exceeding any bound stops the run with ``stop_reason='budget:<name>'``; it
    is a result, not an exception.
    """

    model_config = ConfigDict(frozen=True)

    max_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1)
    queries_per_subq: int = Field(default=DEFAULT_QUERIES_PER_SUBQ, ge=1)
    candidates_per_round: int = Field(default=DEFAULT_CANDIDATES_PER_ROUND, ge=1)
    expand_per_round: int = Field(default=DEFAULT_EXPAND_PER_ROUND, ge=0)
    passages_per_work: int = Field(default=DEFAULT_PASSAGES_PER_WORK, ge=1)
    read_per_round: int = Field(default=DEFAULT_READ_PER_ROUND, ge=1)
    max_llm_calls: int = Field(default=DEFAULT_MAX_LLM_CALLS, ge=1)
    max_decision_calls: int = Field(default=DEFAULT_MAX_DECISION_CALLS, ge=1)
    max_wall_s: float = Field(default=DEFAULT_MAX_WALL_S, gt=0)
    max_fetch_bytes: int = Field(default=DEFAULT_MAX_FETCH_BYTES, ge=1)


class RunStats(BaseModel):
    """What a run cost."""

    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    decision_calls: int = Field(default=0, ge=0)
    wall_s: float = Field(default=0.0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    fetched_bytes: int = Field(default=0, ge=0)
    per_source: dict[str, int] = Field(default_factory=dict)
    # Per step of the graph (``STEPS``): seconds on the run's clock, and the
    # language-model and decision-model requests made while it ran. A step the
    # run never reached has no entry.
    step_seconds: dict[str, float] = Field(default_factory=dict)
    step_llm_calls: dict[str, int] = Field(default_factory=dict)
    step_decision_calls: dict[str, int] = Field(default_factory=dict)
    # Report sentences whose claim support was settled by the token-overlap
    # rule in code, and by the model (the ``code_first`` verification only
    # settles any by code).
    verify_by_code: int = Field(default=0, ge=0)
    verify_by_model: int = Field(default=0, ge=0)
    # The decision cache: its state for this run ("off", "enabled", or
    # "disabled: <why>"), the answers it served instead of a request, and the
    # requests of the determinism check that decided it (not in
    # ``decision_calls``).
    decision_cache: str = "off"
    decision_cache_hits: int = Field(default=0, ge=0)
    decision_probe_calls: int = Field(default=0, ge=0)
    # Of ``decision_cache_hits``, the answers a store shared between runs held
    # (``research(shared_decisions=)``); the rest were the run's own.
    decision_shared_hits: int = Field(default=0, ge=0)
    # The embedding model (``rank: rrf_dense``): requests sent (one batch of
    # texts each), per step, and texts whose vector the store already held.
    embedding_calls: int = Field(default=0, ge=0)
    step_embedding_calls: dict[str, int] = Field(default_factory=dict)
    embedding_cache_hits: int = Field(default=0, ge=0)


LogStep = Literal[
    "Setup",
    "Plan",
    "Search",
    "Screen",
    "Expand",
    "Read",
    "Coverage",
    "Write",
    "Verify",
    "Finish",
]
LOG_STEPS: tuple[LogStep, ...] = (
    "Setup",
    "Plan",
    "Search",
    "Screen",
    "Expand",
    "Read",
    "Coverage",
    "Write",
    "Verify",
    "Finish",
)


class LogEvent(BaseModel):
    """One thing that happened in a run: ``t`` is seconds since the run began
    (on the run's clock), ``event`` a short snake_case name, ``message`` one
    sentence for a reader, ``data`` the details as JSON."""

    seq: int = Field(ge=0)
    t: float = Field(ge=0)
    step: LogStep
    event: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str
    data: dict[str, JsonValue] = Field(default_factory=dict)


# Events the markdown rendering shows as table rows (consecutive events of one
# kind form one table): event -> (column header, data key) pairs.
_LOG_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "query": (
        ("sub-question", "subquestion"),
        ("query", "query"),
        ("source", "source"),
        ("results", "count"),
        ("top titles", "top"),
    ),
    "screened": (
        ("sub-question", "subquestion"),
        ("title", "title"),
        ("p", "p"),
        ("kept", "kept"),
        ("rank before", "rank_before"),
        ("rank after", "rank_after"),
        ("reason", "reason"),
    ),
    "llm_call": (
        ("purpose", "purpose"),
        ("model", "model"),
        ("prompt chars", "prompt_chars"),
        ("seconds", "seconds"),
        ("output", "output"),
    ),
    "passage": (
        ("work", "title"),
        ("sub-question", "subquestion"),
        ("passage", "passage_id"),
        ("score", "score"),
        ("text", "text"),
    ),
    "verify_flag": (("kind", "kind"), ("ref", "ref"), ("detail", "detail")),
}
_CELL_CHARS = 160


def _cell(value: JsonValue) -> str:
    if isinstance(value, float):
        text = f"{value:.3f}"
    elif isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                rank = item.get("rank")
                parts.append(f"{rank}. {item.get('title', '')}" if rank else str(item))
            else:
                parts.append(str(item))
        text = "; ".join(parts)
    elif value is None:
        text = ""
    else:
        text = str(value)
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= _CELL_CHARS else text[: _CELL_CHARS - 3] + "..."


def _table(columns: tuple[tuple[str, str], ...], rows: list[LogEvent]) -> list[str]:
    lines = [
        "| t (s) | " + " | ".join(h for h, _ in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for event in rows:
        cells = [_cell(event.data.get(key)) for _, key in columns]
        lines.append(f"| {event.t:.2f} | " + " | ".join(cells) + " |")
    return lines


def _final_ranking(event: LogEvent) -> list[str]:
    works = event.data.get("works")
    if not isinstance(works, list) or not works:
        return []
    lines = [
        "",
        "| rank | title | fused | signal ranks | p |",
        "|---|---|---|---|---|",
    ]
    for entry in works:
        if not isinstance(entry, dict):
            continue
        ranks = entry.get("ranks")
        shown = (
            ", ".join(f"{k} {v}" for k, v in ranks.items())
            if isinstance(ranks, dict)
            else ""
        )
        lines.append(
            f"| {_cell(entry.get('rank'))} | {_cell(entry.get('title'))} | "
            f"{_cell(entry.get('fused'))} | {_cell(shown)} | {_cell(entry.get('p'))} |"
        )
    return lines


class RunLog(BaseModel):
    """What a run did, step by step, in order (``LogEvent``). Every run fills
    one (``ResearchResult.log``); ``to_jsonl`` and ``to_markdown`` write it."""

    run_id: str = ""
    events: list[LogEvent] = Field(default_factory=list)

    def to_jsonl(self) -> str:
        """One event per line, as JSON; ``from_jsonl`` reads it back."""
        header = json.dumps({"run_id": self.run_id})
        return "\n".join([header, *(e.model_dump_json() for e in self.events)]) + "\n"

    @classmethod
    def from_jsonl(cls, text: str) -> RunLog:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return cls()
        header = json.loads(lines[0])
        return cls(
            run_id=str(header.get("run_id", "")),
            events=[LogEvent.model_validate_json(line) for line in lines[1:]],
        )

    def to_markdown(self) -> str:
        """A readable timeline: a heading each time the step changes, tables
        for queries, screening, model calls, passages and verification flags,
        the final ranking as a table, every other event as a line."""
        total = self.events[-1].t if self.events else 0.0
        out = [
            f"# Run log {self.run_id}".rstrip(),
            "",
            f"{len(self.events)} events over {total:.1f} s.",
        ]
        step: str | None = None
        index = 0
        events = self.events
        while index < len(events):
            event = events[index]
            if event.step != step:
                step = event.step
                out += ["", f"## {step}", ""]
            columns = _LOG_TABLES.get(event.event)
            if columns is not None:
                rows = [event]
                while (
                    index + 1 < len(events)
                    and events[index + 1].event == event.event
                    and events[index + 1].step == step
                ):
                    index += 1
                    rows.append(events[index])
                if out[-1] != "":
                    out.append("")
                out += _table(columns, rows)
                out.append("")
            else:
                out.append(f"- [{event.t:.2f} s] {event.message}")
                if event.event == "final_ranking":
                    out += _final_ranking(event)
                    out.append("")
            index += 1
        return "\n".join(out).rstrip() + "\n"


class ScreeningEntry(BaseModel):
    """One (work, sub-question) pair a run held, and what became of it.

    Compact on purpose: the identifiers, title and year are enough to decide
    whether the work is a given one (``bench.metrics.work_matches`` through
    :meth:`identity`); abstracts and passages are not kept. ``arrival`` is the
    pair's place (0-based) in the order the run found its pairs. ``p`` is the
    screen's score (``None``: the pair was never screened, it waited over a
    cap or the run ended first), ``kept`` the screen's decision, ``rank`` the
    work's 1-based place among the result's ``works`` when this pair was kept
    and the work is there (``None`` otherwise).
    """

    work_id: str = Field(min_length=1)
    subquestion_id: str = Field(min_length=1)
    arrival: int = Field(ge=0)
    source_ranks: dict[str, int] = Field(default_factory=dict)
    p: float | None = None
    kept: bool = False
    rank: int | None = Field(default=None, ge=1)
    title: str = ""
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    local_id: str | None = None
    url: str | None = None

    def identity(self) -> Work:
        """The work as far as matching needs it (identifiers, title, year)."""
        return Work(
            work_id=self.work_id,
            title=self.title,
            year=self.year,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            pmid=self.pmid,
            pmcid=self.pmcid,
            openalex_id=self.openalex_id,
            s2_id=self.s2_id,
            local_id=self.local_id,
            url=self.url,
        )


class ResearchResult(BaseModel):
    """The outcome of one run: the evidence table, the report, and its checks.

    ``works`` are the cited works, in order of first citation, then the other
    kept works by score; ``excluded`` maps a kept work that was left out (a
    retracted one) to the reason, and such a work is never in ``works``.
    ``queries`` are the searches that were run, in order. ``verification``
    lists every flag raised; ``degraded`` gives the reason for every source or
    model failure (empty means none occurred). ``stop_reason``
    is why the run ended (``covered``, ``flat``, ``unresolvable``, or
    ``budget:<name>``). ``config_name`` and ``config_hash`` name the
    ``ResearchConfig`` the run used (``config.config_hash``); ``frozen`` is the
    digest of the frozen inputs it ran on (empty for a live run), and
    ``skipped`` lists what such a run did not fetch because the snapshot did
    not hold it. ``notes`` records what is worth knowing without being a
    failure, such as a web search that answered with results while some
    engines did not.
    """

    run_id: str
    question: str
    mode: Mode
    config_name: str = ""
    config_hash: str = ""
    frozen: str = ""
    subquestions: list[SubQuestion] = Field(default_factory=list)
    criteria: dict[str, list[Criterion]] = Field(default_factory=dict)
    works: list[Work] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    coverage: list[Coverage] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    report_markdown: str = ""
    verification: list[VerifyFlag] = Field(default_factory=list)
    stop_reason: str = ""
    degraded: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)
    # Every (work, sub-question) pair the run held and what became of it, in
    # the order the run found them (``ScreeningEntry``).
    screening: list[ScreeningEntry] = Field(default_factory=list)
    # What happened at each step, in order (always filled; ``RunLog``).
    log: RunLog = Field(default_factory=RunLog)

    def cited_work_ids(self) -> list[str]:
        """Work ids behind the findings the report cites, in order of first citation.

        With no report, the works of every finding.
        """
        by_id = {f.finding_id: f for f in self.findings}
        if self.report_markdown.strip():
            ordered = [
                token
                for group in _FINDING_REFERENCE.findall(self.report_markdown)
                for token in re.findall(r"F\d+", group)
            ]
        else:
            ordered = [f.finding_id for f in self.findings]
        work_ids: list[str] = []
        for finding_id in ordered:
            finding = by_id.get(finding_id)
            if finding is not None and finding.work_id not in work_ids:
                work_ids.append(finding.work_id)
        return work_ids

    def to_csl(self) -> list[dict[str, JsonValue]]:
        """CSL-JSON for the cited works (abstracts of ``no_redistribute`` works omitted)."""
        by_id = {w.work_id: w for w in self.works}
        return [
            by_id[work_id].to_csl()
            for work_id in self.cited_work_ids()
            if work_id in by_id
        ]
