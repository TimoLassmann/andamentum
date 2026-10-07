"""VERIFY: every report passes through these checks, in code, before it is returned.

Package c implements this module (spec section 1.6): cited finding ids exist,
verified quotes are in their passages, unverified quotes are flagged and marked,
retractions, links and DOIs. No model. Old implementation:
``git show bbdb010:src/andamentum/deep_research/v2/verify.py`` (``report_units``,
``links_in``, ``check_links`` and the retraction code carry over; the claim-support
model path, ``code_first`` and ``single_source`` do not).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from andamentum.references import ReferenceHttp, Work

from .types import Finding, Passage, VerifyFlag

UNVERIFIED_MARK = " [quote not verified]"


@dataclass(frozen=True)
class ReportUnit:
    """A sentence of the report: its text, the finding ids it cites, where it ends."""

    text: str
    finding_ids: tuple[str, ...]
    end: int


@dataclass
class VerifyOutcome:
    report_markdown: str
    flags: list[VerifyFlag] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    # Links and DOIs that could not be checked (a timeout, a bot block): listed, not flagged.
    unchecked: list[str] = field(default_factory=list)


def report_units(report: str) -> list[ReportUnit]:
    """The sentences of the report body, each ending at its citation."""
    raise NotImplementedError("package c")


def check_quotes(
    findings: Sequence[Finding], passages: Mapping[str, Passage]
) -> list[VerifyFlag]:
    """A ``quote_missing`` flag for each VERIFIED finding whose quote is not in its
    passage (it should never fire); unverified findings are skipped here."""
    raise NotImplementedError("package c")


async def verify_report(
    report_markdown: str,
    *,
    findings: Sequence[Finding],
    passages: Mapping[str, Passage],
    works: Mapping[str, Work],
    http: ReferenceHttp,
) -> VerifyOutcome:
    """Run every check on *report_markdown*: cited ids exist, verified quotes are in
    their passages, cited unverified findings are flagged (``unverified_quote``, and
    ``UNVERIFIED_MARK`` on a sentence whose cited findings are all unverified),
    retracted works and DOIs, dead links and unresolved DOIs. Returns the report
    (marked) and the flags."""
    raise NotImplementedError("package c")
