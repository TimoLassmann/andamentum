"""Flat output schemas for the v2 model calls (small models fill them)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Sufficiency(BaseModel):
    """The stop decision after round 0: do the passages cover the request?"""

    enough: bool
    missing: str | None = None  # what the passages lack; seeds the search loop


class PassageRef(BaseModel):
    """One passage, by document path and line range."""

    path: str
    line_from: int
    line_to: int


class Selection(BaseModel):
    """The search agent's last step: the passages the request needs, most important first.

    A list of references and nothing else: the search agent never writes prose.
    """

    passages: list[PassageRef] = []


class Citation(BaseModel):
    path: str
    line_from: int
    line_to: int


class Answer(BaseModel):
    """A short answer with citations; written only by ``v2.answering``, never by search."""

    text: str  # short answer, a few words where possible
    citations: list[Citation] = []
    confidence: Literal["high", "medium", "low"]
