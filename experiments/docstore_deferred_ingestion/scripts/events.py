"""The queue-transition event log: every headline count, re-derivable offline.

One append-only JSONL per rule, one record per OBSERVED transition. Sources are
the ``on_progress(done, total, title)`` callback, a ``count_by_status()`` read
taken at the same instant, and the terminal ``ProcessReport``.

IMPLEMENTATION TRAP, recorded here because it is easy to get wrong:
``process_pending`` increments ``documents_skipped`` with a bare ``continue``
and WITHOUT incrementing ``done``, so ``on_progress`` never fires for a skipped
source and ``done`` can end below ``total``. Any consumer that assumes ``done``
reaches ``total`` is wrong.

A second scoping note: ``total`` is the size of the pending snapshot taken once
at drain entry, and that snapshot loads every row's full ``markdown_content``
into memory. Harmless at four documents; do not generalise a memory or
progress-reporting conclusion beyond small queues.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .instrument import append_jsonl, truncate_ledger

#: Field order for the flat CSV sibling produced by ``merge_events``.
EVENT_FIELDS = (
    "ts_utc",
    "monotonic_s",
    "run_id",
    "rule",
    "db_name",
    "doc_id",
    "arxiv_id",
    "from_status",
    "to_status",
    "observer",
    "stage_seconds",
    "markdown_sha256",
    "error_type",
    "error_message",
    "note",
)


class EventLog:
    """Append-only observer log for one rule."""

    def __init__(self, path: str | Path, *, rule: str, run_id: str, truncate: bool = True):
        self.path = Path(path)
        self.rule = rule
        self.run_id = run_id
        self._t0 = time.monotonic()
        if truncate:
            truncate_ledger(self.path)

    def emit(
        self,
        *,
        db_name: str,
        observer: str,
        doc_id: str | None = None,
        arxiv_id: str | None = None,
        from_status: str | None = None,
        to_status: str | None = None,
        stage_seconds: float | None = None,
        markdown_sha256: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        note: str | None = None,
    ) -> None:
        """Record one observation. Never raises — an event log must not break a rule."""
        record: dict[str, Any] = {
            "ts_utc": time.time(),
            "monotonic_s": time.monotonic() - self._t0,
            "run_id": self.run_id,
            "rule": self.rule,
            "db_name": db_name,
            "doc_id": doc_id,
            "arxiv_id": arxiv_id,
            "from_status": from_status,
            "to_status": to_status,
            "observer": observer,
            "stage_seconds": stage_seconds,
            "markdown_sha256": markdown_sha256,
            "error_type": error_type,
            "error_message": error_message,
            "note": note,
        }
        append_jsonl(self.path, record, fsync=False)
