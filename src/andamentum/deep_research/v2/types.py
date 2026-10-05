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
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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


class ResearchResult(BaseModel):
    """The outcome of one run: the evidence table, the report, and its checks.

    ``works`` are the cited works, in order of first citation, then the other
    kept works by score; ``excluded`` maps a kept work that was left out (a
    retracted one) to the reason, and such a work is never in ``works``.
    ``queries`` are the searches that were run, in order. ``verification``
    lists every flag raised; ``degraded`` gives the reason for every source or
    model failure (empty means none occurred). ``stop_reason``
    is why the run ended (``covered``, ``flat``, ``unresolvable``, or
    ``budget:<name>``).
    """

    run_id: str
    question: str
    mode: Mode
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
    stats: RunStats = Field(default_factory=RunStats)

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
