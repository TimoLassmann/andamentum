"""The data a v2 research run reads and produces. Flat, JSON-serialisable.

A run is a table of evidence, not a piece of prose: sub-questions, the works
found, verbatim passages, findings that quote a passage (or say, honestly, that
the quote could not be found), the answer the research reached for each
sub-question, and the flags verification raised. The report cites findings by
id (``[F12]``); :class:`ResearchResult` carries all of it, so a consumer can
check a sentence against the quote behind it instead of trusting the prose.
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
VerifyKind = Literal[
    "unknown_finding",  # the report cites a finding id that does not exist
    "quote_missing",  # a verified finding's quote is not in its passage (a bug)
    "unverified_quote",  # a cited finding's quote could not be found in its page
    "dead_link",  # a URL in the report does not resolve
    "doi_unresolved",  # a DOI in the report does not resolve
    "retracted",  # a retracted work is cited
]
# How a page's text was obtained: a text source (plain fetch or the record's
# full text), the same Chrome that searches (when the plain fetch failed), the
# record's abstract, the search engine's snippet, or not at all.
PageVia = Literal["fetched", "chrome", "abstract", "snippet", "none"]
# The steps a run's time and calls are counted under, in order.
STEPS = (
    "Plan",
    "Search",
    "Expand",
    "Choose",
    "Read",
    "Notes",
    "Reflect",
    "Write",
    "Verify",
)

DEFAULT_MAX_ROUNDS = 3
DEFAULT_QUERIES_PER_ROUND = 3
DEFAULT_READS_PER_ROUND = 5
DEFAULT_WINDOW_CHARS = 12_000
DEFAULT_EXPAND_PER_ROUND = 20
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_LLM_CALLS = 200
DEFAULT_MAX_DECISION_CALLS = 5000
DEFAULT_MAX_WALL_S = 3600.0
DEFAULT_MAX_FETCH_BYTES = 50_000_000

_FINDING_REFERENCE = re.compile(r"\[\s*(F\d+(?:\s*[,;]\s*F\d+)*)\s*\]")


class SubQuestion(BaseModel):
    """One part of the question; the unit of the research loop.

    ``kind`` only decides whether a scholarly search reads a year range or an
    open-access wish out of the text (``plan.filters_for``).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: SubQuestionKind = "topical"


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
        ``no_redistribute`` work never exports it."""
        return value if self.work.redistributable else ""


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


class RawFinding(BaseModel):
    """A finding before the engine numbers it (``read.read_page`` returns these).

    ``quote_verified`` is true when code found ``quote`` in a passage of the
    page: then ``quote`` is the passage's own words at that span and
    ``passage_id`` names that passage. When false, ``quote`` is what the model
    wrote (whitespace squashed) and ``passage_id`` the passage it named (else
    the first passage of the window). A finding is never dropped for it.
    """

    subquestion_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    quote_verified: bool
    passage_id: str = Field(min_length=1)
    work_id: str = Field(min_length=1)


class Finding(BaseModel):
    """A claim and the quote behind it; ``F<n>`` is the id the report cites."""

    finding_id: str = Field(pattern=r"^F\d+$")
    subquestion_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    # False: the model's quote could not be found in the page text. The finding
    # stays (visible in the report and the log); compose does not use it.
    quote_verified: bool = True
    passage_id: str = Field(min_length=1)
    work_id: str = Field(min_length=1)

    @classmethod
    def from_raw(cls, raw: RawFinding, *, number: int) -> Finding:
        return cls(
            finding_id=f"F{number}",
            subquestion_id=raw.subquestion_id,
            claim=raw.claim,
            quote=raw.quote,
            quote_verified=raw.quote_verified,
            passage_id=raw.passage_id,
            work_id=raw.work_id,
        )


class PageReading(BaseModel):
    """What reading one page produced (``read.load_page`` / ``read.read_page``).

    ``passages`` is the window the notes were taken from: exact slices of the
    page text, stored, in document order. ``via`` says where the text came
    from. ``excluded`` is set for a retracted work (nothing read). The notes
    lists follow the old rules: ``failed_notes`` are failures of one text
    source that make the page unreadable but not the run degraded, ``degraded``
    are failures that do (a model that failed).
    """

    work: Work
    subquestion_id: str
    via: PageVia = "none"
    passages: list[Passage] = Field(default_factory=list)
    findings: list[RawFinding] = Field(default_factory=list)
    excluded: str = ""
    status: int | None = None
    error: str = ""
    notes: list[str] = Field(default_factory=list)
    failed_notes: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)

    @property
    def readable(self) -> bool:
        """The page gave text to take notes from."""
        return bool(self.passages) and not self.excluded


class SubQuestionResult(BaseModel):
    """How the research of one sub-question ended: the reflect step's last
    judgement (``answer`` is the best-supported answer so far, ``""`` when the
    findings give none), the rounds run and why it stopped (``answered``,
    ``rounds`` or ``budget:<name>``)."""

    subquestion_id: str = Field(min_length=1)
    answered: bool = False
    answer: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    gaps: list[str] = Field(default_factory=list)
    rounds: int = Field(default=0, ge=0)
    stop_reason: str = ""


class VerifyFlag(BaseModel):
    """One thing verification found wrong. ``ref`` names it: a finding id, a
    work id, a URL, or the report sentence."""

    kind: VerifyKind
    ref: str
    detail: str = ""


class Budget(BaseModel):
    """Every bound on a run, explicit with a named default.

    Exceeding a bound that ends the search (``max_wall_s``, ``max_llm_calls``,
    ``max_fetch_bytes``) gives ``stop_reason='budget:<name>'``; the report is
    still written from what was found. It is a result, not an exception.
    """

    model_config = ConfigDict(frozen=True)

    # Rounds of search-choose-read-reflect per sub-question.
    max_rounds: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1)
    # The most queries the model may write for a round.
    queries_per_round: int = Field(default=DEFAULT_QUERIES_PER_ROUND, ge=1)
    # K: pages chosen to be read in full each round (and as many reserves).
    reads_per_round: int = Field(default=DEFAULT_READS_PER_ROUND, ge=1)
    # The most characters of one page the notes are taken from.
    window_chars: int = Field(default=DEFAULT_WINDOW_CHARS, ge=1000)
    # Scholarly: new candidates one round's citation expansion may add.
    expand_per_round: int = Field(default=DEFAULT_EXPAND_PER_ROUND, ge=0)
    # Model calls in flight together (a local model is held to one).
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1)
    max_llm_calls: int = Field(default=DEFAULT_MAX_LLM_CALLS, ge=1)
    # Only the novelty check's local decision model counts against this.
    max_decision_calls: int = Field(default=DEFAULT_MAX_DECISION_CALLS, ge=1)
    max_wall_s: float = Field(default=DEFAULT_MAX_WALL_S, gt=0)
    max_fetch_bytes: int = Field(default=DEFAULT_MAX_FETCH_BYTES, ge=1)


class RunStats(BaseModel):
    """What a run cost."""

    llm_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    # The novelty check's decision model (0 for the engine itself).
    decision_calls: int = Field(default=0, ge=0)
    wall_s: float = Field(default=0.0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    fetched_bytes: int = Field(default=0, ge=0)
    per_source: dict[str, int] = Field(default_factory=dict)
    # Per step (``STEPS``): busy seconds on the run's clock, summed over the
    # tasks that ran in parallel (so the sum can exceed ``wall_s``), and the
    # language-model and decision-model requests made under it.
    step_seconds: dict[str, float] = Field(default_factory=dict)
    step_llm_calls: dict[str, int] = Field(default_factory=dict)
    step_decision_calls: dict[str, int] = Field(default_factory=dict)
    # Pages the loop tried to read, those that gave no text at all, and
    # findings whose quote could not be found in the page.
    pages_read: int = Field(default=0, ge=0)
    pages_unreadable: int = Field(default=0, ge=0)
    quotes_unverified: int = Field(default=0, ge=0)


LogStep = Literal[
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
]
LOG_STEPS: tuple[LogStep, ...] = (
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
    "chosen_work": (
        ("sub-question", "subquestion"),
        ("title", "title"),
        ("site", "host"),
        ("chosen", "chosen"),
    ),
    "finding_noted": (
        ("sub-question", "subquestion"),
        ("work", "title"),
        ("claim", "claim"),
        ("verified", "verified"),
        ("quote", "quote"),
    ),
    "llm_call": (
        ("purpose", "purpose"),
        ("model", "model"),
        ("prompt chars", "prompt_chars"),
        ("seconds", "seconds"),
        ("output", "output"),
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
    lines = ["", "| rank | title | basis | source ranks |", "|---|---|---|---|"]
    for entry in works:
        if not isinstance(entry, dict):
            continue
        ranks = entry.get("source_ranks")
        shown = (
            ", ".join(f"{k} {v}" for k, v in ranks.items())
            if isinstance(ranks, dict)
            else ""
        )
        lines.append(
            f"| {_cell(entry.get('rank'))} | {_cell(entry.get('title'))} | "
            f"{_cell(entry.get('basis'))} | {_cell(shown)} |"
        )
    return lines


class RunLog(BaseModel):
    """What a run did, step by step, in order (``LogEvent``). Every run fills
    one (``ResearchResult.log``); ``to_jsonl`` and ``to_markdown`` write it.

    The events carry the full text a stage saw (search snippets, fetched page
    text, the window given to the notes, the answer, the report), because the
    log is also what a loss-attribution tool reads to say at which stage a
    gold answer disappeared (see the bench's ``attribution`` module)."""

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
        for queries, choices, findings, model calls and verification flags, the
        final ranking as a table, every other event as a line."""
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


class ResearchResult(BaseModel):
    """The outcome of one run: the evidence table, the report, and its checks.

    ``works`` are the ranked works (``fusion.rank_works``: cited works first,
    in order of first citation, then the other works that were read, then the
    rest by reciprocal rank fusion of their source ranks); ``excluded`` maps a
    work that was left out (a retracted one) to the reason, and such a work is
    never in ``works``. ``queries`` are the searches that were run, in order.
    ``subanswers`` holds one :class:`SubQuestionResult` per sub-question.
    ``verification`` lists every flag raised; ``degraded`` gives the reason for
    every source or model failure (empty means none occurred). ``stop_reason``
    is why the run ended (``answered`` when every sub-question was, else
    ``budget:<name>`` or ``rounds``). ``notes`` records what is worth knowing
    without being a failure, such as a web search that answered with results
    while some engines did not.
    """

    run_id: str
    question: str
    mode: Mode
    subquestions: list[SubQuestion] = Field(default_factory=list)
    subanswers: list[SubQuestionResult] = Field(default_factory=list)
    works: list[Work] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    passages: list[Passage] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    report_markdown: str = ""
    verification: list[VerifyFlag] = Field(default_factory=list)
    stop_reason: str = ""
    degraded: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stats: RunStats = Field(default_factory=RunStats)
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
