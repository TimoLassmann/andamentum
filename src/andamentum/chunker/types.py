"""Pydantic types for the chunker module.

Schemas are kept FLAT and SIMPLE — only what's strictly required —
because small local models fill them (used by the optional judge stage).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# System-facing types: richer, because they carry provenance and metadata.
class Unit(BaseModel):
    """A successfully extracted unit, byte-identical to a source span."""

    id: str
    title: str
    text: str  # source[source_start:source_end], verbatim
    kind: str
    source_start: int  # inclusive char offset
    source_end: int  # exclusive char offset
    complete: bool  # False if this unit was truncated at a window boundary
    anchor_match_method: Literal[
        "exact", "whitespace_normalised", "fuzzy", "best_effort"
    ]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Gap(BaseModel):
    """A region of source that no unit claimed."""

    source_start: int
    source_end: int
    text: str

    @property
    def length(self) -> int:
        return self.source_end - self.source_start


class ChunkingResult(BaseModel):
    """Final result returned by extract_units()."""

    units: list[Unit]
    gaps: list[Gap]
    total_chars: int
    model_calls: int
    retries_used: int
    windows_processed: int

    @property
    def extracted_chars(self) -> int:
        return sum(u.source_end - u.source_start for u in self.units)

    @property
    def gap_chars(self) -> int:
        return sum(g.length for g in self.gaps)

    @property
    def coverage(self) -> float:
        return self.extracted_chars / self.total_chars if self.total_chars else 0.0

    @property
    def gap_fraction(self) -> float:
        return self.gap_chars / self.total_chars if self.total_chars else 0.0
