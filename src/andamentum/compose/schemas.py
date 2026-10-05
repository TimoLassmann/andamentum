"""Data models for compose. Flat on purpose: small models fill them."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["drafter", "critic", "judge", "reader"]
ROLE_NAMES: tuple[Role, ...] = ("drafter", "critic", "judge", "reader")
StopReason = Literal["goal_met", "no_progress", "cap_reached", "unresolvable"]


class Roles(BaseModel):
    """The model id used for each job. No defaults: every role is named."""

    model_config = ConfigDict(frozen=True)

    drafter: str
    critic: str
    judge: str
    reader: str

    @classmethod
    def all(cls, model: str) -> Roles:
        return cls(drafter=model, critic=model, judge=model, reader=model)


class ReaderQuestion(BaseModel):
    question: str
    expected_answer: str


class Brief(BaseModel):
    """What the document must do. Compiled once, then frozen."""

    model_config = ConfigDict(frozen=True)

    task: str
    genre: str
    title: str
    audience: str
    purpose: str
    key_messages: list[str]
    must_include: list[str]
    constraints: list[str]
    word_range: tuple[int, int]
    reader_questions: list[ReaderQuestion]
    guidelines: str = ""
    email_subject: str = ""


class Source(BaseModel):
    id: str
    path: str
    title: str
    sha256: str
    chars: int
    kind: Literal["file", "url", "research"]
    label: str = ""
    """What a reader of the finished document sees: the file name, the URL, or
    a description of the research report. ``path`` stays machine-specific
    (resume needs it) and never reaches a document."""


class Fact(BaseModel):
    id: str
    source_id: str
    statement: str
    quote: str
    start: int
    end: int


class FigurePlan(BaseModel):
    source_id: str
    kind: str
    x: str
    y: str
    caption: str
    label: str


class SectionPlan(BaseModel):
    title: str
    purpose: str
    key_points: list[str]
    fact_ids: list[str]
    target_words: int
    figure: FigurePlan | None = None


class Outline(BaseModel):
    sections: list[SectionPlan]


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: list[str]
    section_id: str | None = None


class WorkItem(BaseModel):
    level: Literal["section", "outline"]
    section_id: str | None
    reason: str
    source: str  # the name of the check that raised it


class RoundRecord(BaseModel):
    round: int
    checks: list[CheckResult]
    work: list[WorkItem]
    changed_sections: list[str]
    goal_met: bool


class ComposeResult(BaseModel):
    bundle: Path
    outputs: list[Path]
    goal_met: bool
    stop_reason: StopReason
    rounds: list[RoundRecord]
    open_questions: list[str]
    brief: Brief
    usage: dict[str, int]  # tokens per role

    def summary_lines(self) -> list[str]:
        """What the CLI prints after a bundle run."""
        lines = [
            f"bundle: {self.bundle}",
            f"goal met: {'yes' if self.goal_met else 'no'} ({self.stop_reason})",
        ]
        lines += [f"output: {path}" for path in self.outputs]
        lines.append(f"report: {self.bundle / 'report.md'}")
        lines += [f"open: {question}" for question in self.open_questions]
        return lines
