"""The adapter protocol every document format implements."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .ops import Transaction
from .view import DocumentView


class UntouchedReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    changed_outside: list[str]  # paragraph ids that changed though no op targeted them


class ApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output: Path
    applied: int
    refused: list[str]
    untouched: UntouchedReport


class DocumentAdapter(Protocol):
    def open(self, path: Path) -> DocumentView: ...

    def apply(
        self, view: DocumentView, txn: Transaction, *, output: Path
    ) -> ApplyResult:
        """Write the transaction to ``output``; never edit the input in place.

        Implementations re-open the output and verify that every targeted range
        reads as intended, that untargeted paragraphs are unchanged and that
        the file opens. Any failure raises.
        """
        ...
