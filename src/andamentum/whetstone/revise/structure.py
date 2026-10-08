"""Structure/flow check — reuses the review's own findings, nothing new.

"Clean" here means the active review criteria raised nothing blocking.
Which criteria are active is entirely the caller's choice (``--reviewer`` /
``--brief`` / ``--document-type``); this module never names one, so pairing
``--until-clean structure`` with ``--reviewer reader`` is what makes the
condition "the flow is correct" rather than anything else.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas import Finding, ReviewResult

_BLOCKING_SEVERITIES = frozenset({"major", "moderate"})


class StructureReport(BaseModel):
    blocking_count: int
    blocking_titles: list[str] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return self.blocking_count == 0


def structure_clean(result: ReviewResult) -> StructureReport:
    """Zero major/moderate findings, under whatever criteria produced them."""
    blocking: list[Finding] = [
        finding
        for finding in result.findings
        if finding.severity in _BLOCKING_SEVERITIES
    ]
    return StructureReport(
        blocking_count=len(blocking),
        blocking_titles=[finding.title for finding in blocking[:5]],
    )
