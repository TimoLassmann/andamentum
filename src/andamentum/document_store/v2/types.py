"""Flat output schemas for the agentic-retrieval loop (small models fill them)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Sufficiency(BaseModel):
    """The stop decision after round 0: is the retrieved evidence enough to answer?"""

    enough: bool
    missing: str | None = None  # what the evidence lacks; seeds round 1


class Citation(BaseModel):
    path: str
    line_from: int
    line_to: int


class Answer(BaseModel):
    text: str  # short answer, a few words where possible
    citations: list[Citation] = []
    confidence: Literal["high", "medium", "low"]
