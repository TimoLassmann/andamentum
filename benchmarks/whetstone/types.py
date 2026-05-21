"""Data shapes for the whetstone evaluation, persisted as JSON.

Pydantic throughout so every artefact round-trips to ``runs/`` and back.
Paths are stored as strings for clean serialisation.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Source = Literal["biorxiv", "arxiv"]
Arm = Literal["A", "B"]
Bucket = Literal["both", "a_only", "b_only"]
Severity = Literal["critical", "minor"]
Locality = Literal["cross_section", "local"]


class PaperRef(BaseModel):
    """One corpus paper at a specific version."""

    source: Source
    id: str  # bioRxiv DOI or arXiv id
    version: int = 1
    title: str = ""
    subfield: str = ""
    pdf_path: Optional[str] = None
    markdown_path: Optional[str] = None

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for this paper."""
        return f"{self.source}_{self.id.replace('/', '_')}_v{self.version}"


class ArmFinding(BaseModel):
    """One issue as reported by an arm (before adjudication)."""

    title: str
    detail: str = ""


class ArmOutput(BaseModel):
    """One arm's full review of one paper."""

    arm: Arm
    findings: list[ArmFinding] = Field(default_factory=list)
    verdict: str = ""  # synthesis (A) / top central weaknesses (B)


class AdjudicatedFinding(BaseModel):
    """One issue after the judge aligned the two arms.

    ``bucket`` says which arm(s) raised it; ``severity`` and ``locality`` are
    the judge's rubric tags. The headline metric counts findings that are
    ``bucket=="b_only"`` AND ``severity=="critical"`` AND
    ``locality=="cross_section"`` — issues the whole-document read caught that
    whetstone missed and that genuinely need cross-section reasoning.
    """

    text: str
    bucket: Bucket
    severity: Severity
    locality: Locality
    note: str = ""


class PaperResult(BaseModel):
    """Everything produced for one paper: both arms + the adjudication."""

    paper: PaperRef
    arm_a: ArmOutput
    arm_b: ArmOutput
    adjudications: list[AdjudicatedFinding] = Field(default_factory=list)
    verdict_match: Optional[bool] = None  # did A's synthesis match B's top weaknesses?
