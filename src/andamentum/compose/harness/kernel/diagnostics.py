"""Diagnostics a check reports, and the actions that would resolve them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .ops import Transaction

Severity = Literal["error", "warning", "info"]
Tier = Literal[0, 1, 2]


class Diagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    tier: Tier
    severity: Severity
    paragraph_id: str | None
    start: int | None
    end: int | None
    message: str
    evidence: list[str]
    data: dict[str, object] = Field(default_factory=dict)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostic_index: int
    transaction: Transaction
