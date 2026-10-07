"""WRITE: the report, from the findings table and nothing else.

Package c implements this module (spec section 1.6): ONE model call over all
sub-questions, best-supported answer first, conflicts weighed by source quality,
agreement and the question's time scope; everything else built in code.
Old implementation: ``git show bbdb010:src/andamentum/deep_research/v2/write.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, Field

from andamentum.references import Work

from .ask import Asker
from .types import Finding, SubQuestion, SubQuestionResult

NO_EVIDENCE_LINE = "No evidence could be read for this question."
SOURCES_HEADING = "## Sources"
# Lines written by code that carry no finding id; verification does not expect one.
CODE_LINE_PREFIXES = (NO_EVIDENCE_LINE,)
# Findings put to the model for one sub-question: one origin at a time.
MAX_FINDINGS_PER_SECTION = 30


class Report(BaseModel):
    """The report and what went into it."""

    markdown: str
    cited_finding_ids: list[str] = Field(default_factory=list)
    excluded_finding_ids: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)


def citation(finding_ids: Sequence[str]) -> str:
    """``[F3, F7]``: the citation group the report puts at the end of a sentence."""
    return f"[{', '.join(finding_ids)}]"


async def write_report(
    question: str,
    *,
    subquestions: Sequence[SubQuestion],
    answers: Mapping[str, SubQuestionResult],
    findings: Sequence[Finding],
    works: Mapping[str, Work],
    asker: Asker,
) -> Report:
    """The report for *question*, written only from *findings*, in one model call.

    *answers* maps a sub-question id to the research's last judgement (the draft
    answer the writer may keep where findings support it). *works* maps work id
    to work (titles, sites, retraction state, the Sources section). A section the
    writer cannot write (spent budget, failed model) lists its findings' claims
    instead, and ``degraded`` names it. Findings of retracted works are never
    offered to the model."""
    raise NotImplementedError("package c")
