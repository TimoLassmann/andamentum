"""Pydantic types for the chunker module.

Schemas are kept FLAT and SIMPLE — only what's strictly required —
because small local models fill them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# Model-facing schema: kept tiny on purpose.
class NextUnitResult(BaseModel):
    """What the LLM returns per call: at most one unit (or 'nothing here')."""

    found: bool = Field(
        description=(
            "True if a coherent unit starting at the beginning of the "
            "visible text was identified. False if the visible text is "
            "navigation, ads, repeated headers, or otherwise has no "
            "extractable content."
        )
    )
    title: str = Field(
        default="",
        description="Short descriptive title, 3-8 words. Empty when found=False.",
    )
    start_anchor: str = Field(
        default="",
        description=(
            "First 8-12 words of the unit, copied VERBATIM from the source. "
            "Must appear exactly in the visible text."
        ),
    )
    end_anchor: str = Field(
        default="",
        description=(
            "Last 8-12 words of the unit, copied VERBATIM. Must appear "
            "AFTER start_anchor in the visible text."
        ),
    )
    kind: str = Field(
        default="prose",
        description=(
            "Free-text label, e.g. prose / list / table / code / quote / "
            "heading / definition. Closed sets confuse small models — keep open."
        ),
    )
    complete: bool = Field(
        default=True,
        description=(
            "False if the unit clearly continues past the visible text "
            "(no natural ending was reached)."
        ),
    )
    skip_to: str = Field(
        default="",
        description=(
            "Used only when found=False: a short verbatim phrase from "
            "near the end of the visible text. The system advances the "
            "cursor past this phrase to skip the junk region."
        ),
    )


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


class ChunkingFailedError(Exception):
    """Raised when the entire escalation chain has been exhausted."""

    def __init__(
        self,
        *,
        cursor: int,
        attempted_models: list[str],
        last_validator_messages: list[str],
        message: str,
    ):
        self.cursor = cursor
        self.attempted_models = attempted_models
        self.last_validator_messages = last_validator_messages
        super().__init__(
            f"{message} at cursor={cursor}; "
            f"attempted models: {attempted_models}; "
            f"last validator complaints: {last_validator_messages}"
        )
