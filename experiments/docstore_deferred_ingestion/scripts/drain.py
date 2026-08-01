"""Instrumented drain: one call, fully observed, reconciled against the database.

Wraps ``process_pending`` with the four instruments and returns everything the
hypotheses need:

  * the ``ProcessReport`` as a plain dict
  * pre- and post- database snapshots (logical fingerprint + full row dump)
  * the converter-ledger delta for this drain
  * the report-vs-database reconciliation (H-x)

H-x MATTERS OPERATIONALLY: the report is what a cron job logs and a UI shows. If
it disagrees with the database, every operational claim built on it is unsafe.

The ``on_progress`` observer reads a READ-ONLY sqlite connection at the same
instant — sync, so it can run inside the sync callback, and read-only so it
cannot contend on the writer.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .events import EventLog
from .fingerprint import poll_all_rows, snapshot, status_counts
from .instrument import HttpLedger, read_jsonl


def report_to_dict(report: Any) -> dict[str, Any]:
    """ProcessReport -> plain dict (it is a dataclass; keep JSON interoperable)."""
    return asdict(report)


def reconcile(
    report: dict[str, Any],
    pre: dict[str, Any],
    post: dict[str, Any],
    *,
    ledger_delta: list[dict[str, Any]],
    convert_fn_supplied: bool,
) -> dict[str, Any]:
    """Field-by-field agreement between the report, the database and the ledger.

    Covers ALL SIX ``ProcessReport`` fields H-x names. An earlier version checked
    four and left ``documents_skipped`` and ``stopped_early`` unchecked while the
    hypothesis claimed all six — a statement outrunning its metric turns a PASS
    into a claim nobody tested.

    Deliberately does NOT try to reconcile ``pending_source`` deltas
    arithmetically: a document that fails DURING conversion also leaves
    pending_source, and the report does not say which failures happened at which
    stage. The converter ledger settles that instead — successful (error-free)
    ledger entries in this drain are exactly ``documents_converted``.

    ``stopped_early`` is reconciled as an IMPLICATION, not an equality: the drain
    breaks at the top of an iteration whose document is therefore still queued, so
    ``stopped_early`` requires at least one pending row. The converse does not
    hold (a skipped source also leaves work queued), so asserting equality would
    manufacture a mismatch out of correct behaviour.
    """
    pre_counts = pre["status_counts"]
    post_counts = post["status_counts"]
    successful_conversions = len([r for r in ledger_delta if not r.get("error_type")])
    post_pending = post_counts["pending_source"] + post_counts["pending_enrich"]

    checks: list[dict[str, Any]] = [
        {
            "field": "documents_enriched",
            "report": report["documents_enriched"],
            "database": post_counts["complete"] - pre_counts["complete"],
            "source": "delta in complete count",
        },
        {
            "field": "documents_failed",
            "report": report["documents_failed"],
            "database": post_counts["failed"] - pre_counts["failed"],
            "source": "delta in failed count",
        },
        {
            "field": "remaining",
            "report": report["remaining"],
            "database": post_pending,
            "source": "post-drain pending_source + pending_enrich",
        },
        {
            "field": "documents_converted",
            "report": report["documents_converted"],
            "database": successful_conversions,
            "source": "error-free converter-ledger entries appended by this drain",
        },
        {
            # `documents_skipped` counts pending_source rows the loop walked past
            # because no converter was supplied. With a converter it must be 0;
            # without one the loop reaches every pending_source row (the skip
            # branch `continue`s WITHOUT incrementing `done`, so max_docs cannot
            # cut it short), and all of them stay queued.
            "field": "documents_skipped",
            "report": report["documents_skipped"],
            "database": 0 if convert_fn_supplied else pre_counts["pending_source"],
            "source": (
                "0 when a convert_fn was supplied; otherwise every pre-drain "
                "pending_source row, all of which remain queued"
            ),
        },
    ]
    for check in checks:
        check["agrees"] = check["report"] == check["database"]

    checks.append(
        {
            "field": "stopped_early",
            "report": report["stopped_early"],
            "database": post_pending,
            "source": "stopped_early implies >= 1 document still queued (implication only)",
            "agrees": (not report["stopped_early"]) or post_pending >= 1,
        }
    )

    mismatches = [c for c in checks if not c["agrees"]]
    return {
        "fields_checked": [c["field"] for c in checks],
        "checks": checks,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "pre_status_counts": pre_counts,
        "post_status_counts": post_counts,
    }


def attribute_document_seconds(
    drains: list[dict[str, Any]], doc_uuid: str
) -> dict[str, Any]:
    """Seconds a specific document occupied inside a drain, by IDENTITY.

    Returns the interval between the tick at which ``doc_uuid`` settled and the
    tick before it. Index 0 is REFUSED: its predecessor is the drain's ``t0``, so
    the interval would silently include ``process_pending``'s per-process
    preflight and — for a ``pending_source`` row — the whole Docling conversion,
    under a label that says "enrichment".
    """
    for drain in drains:
        ticks = drain.get("progress") or []
        for index, tick in enumerate(ticks):
            if tick.get("doc_uuid") != doc_uuid:
                continue
            if index == 0:
                return {
                    "seconds": None,
                    "drain": drain.get("label"),
                    "tick_index": 0,
                    "note": (
                        "refused: the first tick's interval starts at the drain's t0 and "
                        "therefore contains the preflight tax and any conversion, which "
                        "is not this document's enrichment cost"
                    ),
                }
            return {
                "seconds": tick["monotonic_s"] - ticks[index - 1]["monotonic_s"],
                "drain": drain.get("label"),
                "tick_index": index,
                "note": "interval between the preceding commit and this document's commit",
            }
    return {
        "seconds": None,
        "drain": None,
        "tick_index": None,
        "note": f"doc_uuid {doc_uuid} settled in no supplied drain",
    }


def mid_stage_documents(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Documents a cooperative pause must never produce.

    Two shapes, both structural and both re-checkable offline from the snapshot:

    * ``failed`` — a pause turned a document into an error, which it must not;
    * PARTIALLY ENRICHED — chunk rows or chunk-embedding rows persisted for a row
      that is not ``complete``. That is enrichment work committed and then
      stranded: exactly the discarded work claim (d) says cannot exist.
    """
    stranded: list[dict[str, Any]] = []
    for row in snapshot["documents"]:
        status = row["ingest_status"]
        if status == "failed":
            stranded.append({"doc_uuid": row["doc_uuid"], "reason": "failed", "status": status})
        elif status != "complete" and (
            row.get("n_chunks", 0) or row.get("n_chunk_embeddings", 0)
        ):
            stranded.append(
                {
                    "doc_uuid": row["doc_uuid"],
                    "reason": "partial enrichment persisted on a non-complete row",
                    "status": status,
                    "n_chunks": row.get("n_chunks"),
                    "n_chunk_embeddings": row.get("n_chunk_embeddings"),
                }
            )
    return stranded


async def instrumented_drain(
    *,
    database: str,
    db_path: Path,
    label: str,
    model: str,
    embedding_model: str,
    convert_fn: Callable[[str], Any] | None = None,
    convert_ledger: Path | None = None,
    should_continue: Callable[[], bool] | None = None,
    progress_hook: Callable[[int, int, str], None] | None = None,
    max_docs: int | None = None,
    max_seconds: float | None = None,
    event_log: EventLog | None = None,
    http_ledger: HttpLedger | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one fully-observed drain and return the complete observation record."""
    from andamentum.document_store import process_pending

    pre = snapshot(db_path, label=f"{label}_pre")
    ledger_before = len(read_jsonl(convert_ledger)) if convert_ledger else 0
    http_before = len(http_ledger.records) if http_ledger is not None else 0

    progress: list[dict[str, Any]] = []
    t0 = time.monotonic()
    # Identity, not title. Two rows can carry the SAME title — claim (a) arm 2
    # queues one paper as a pending_source alongside the markdown ingest of that
    # same paper — so attributing a progress tick by title prefix can silently
    # price the wrong document (and, at index 0, fold the drain's preflight and a
    # Docling conversion into what is labelled "enrichment seconds"). Diffing the
    # terminal-status row set between ticks names the document that just settled.
    settled_before: set[str] = {
        r["doc_uuid"]
        for r in poll_all_rows(db_path)
        if r["ingest_status"] in ("complete", "failed")
    }

    def on_progress(done: int, total: int, title: str) -> None:
        """Observer. Reads a read-only connection at the same instant.

        NOTE the shipped behaviour this must tolerate: ``documents_skipped`` uses
        ``continue`` WITHOUT incrementing ``done``, so this never fires for a
        skipped source and ``done`` can end below ``total``.
        """
        nonlocal settled_before
        rows = poll_all_rows(db_path)
        settled_now = {
            r["doc_uuid"] for r in rows if r["ingest_status"] in ("complete", "failed")
        }
        newly_settled = sorted(settled_now - settled_before)
        settled_before = settled_now
        by_uuid = {r["doc_uuid"]: r for r in rows}
        entry = {
            "done": done,
            "total": total,
            "title": title,
            "monotonic_s": time.monotonic() - t0,
            "settled_doc_uuids": newly_settled,
            # Exactly one document settles per tick in the shipped loop; the list
            # form is kept so a future change that batches cannot silently produce
            # a wrong scalar.
            "doc_uuid": newly_settled[0] if len(newly_settled) == 1 else None,
            "settled_status": (
                by_uuid[newly_settled[0]]["ingest_status"]
                if len(newly_settled) == 1
                else None
            ),
            "markdown_chars": (
                by_uuid[newly_settled[0]]["markdown_chars"]
                if len(newly_settled) == 1
                else None
            ),
            "status_counts_at_instant": status_counts(rows),
        }
        progress.append(entry)
        # The caller's own observer runs LAST, so a should_continue() built on it
        # sees exactly the state the drain has committed.
        if progress_hook is not None:
            progress_hook(done, total, title)
        if event_log is not None:
            event_log.emit(
                db_name=database,
                observer="on_progress",
                to_status="processed",
                stage_seconds=entry["monotonic_s"],
                note=f"[{done}/{total}] {title}",
            )

    started = time.monotonic()
    report = await process_pending(
        database,
        model=model,
        embedding_model=embedding_model,
        convert_fn=convert_fn,
        should_continue=should_continue,
        on_progress=on_progress,
        max_docs=max_docs,
        max_seconds=max_seconds,
    )
    elapsed = time.monotonic() - started

    post = snapshot(db_path, label=f"{label}_post")
    ledger_delta = (
        read_jsonl(convert_ledger)[ledger_before:] if convert_ledger else []
    )
    http_delta = (
        http_ledger.records[http_before:] if http_ledger is not None else []
    )
    report_dict = report_to_dict(report)

    if event_log is not None:
        event_log.emit(
            db_name=database,
            observer="report",
            stage_seconds=elapsed,
            note=(
                f"{label}: converted={report_dict['documents_converted']} "
                f"enriched={report_dict['documents_enriched']} "
                f"failed={report_dict['documents_failed']} "
                f"skipped={report_dict['documents_skipped']} "
                f"remaining={report_dict['remaining']} "
                f"stopped_early={report_dict['stopped_early']}"
            ),
        )

    if snapshot_dir is not None:
        from . import _common as C

        snapshot_dir.mkdir(parents=True, exist_ok=True)
        C.write_json(
            snapshot_dir / f"{label}_pre.json",
            pre,
            schema="andamentum.experiment.docstore_deferred.snapshot/1",
        )
        C.write_json(
            snapshot_dir / f"{label}_post.json",
            post,
            schema="andamentum.experiment.docstore_deferred.snapshot/1",
        )

    # WALL-CLOCK AFTER THE LAST COMMIT. For a cooperative stop this is the loop's
    # own exit path (a status count and a return) and is expected to be a small
    # fraction of a document. For a mid-document stop it would be the remainder of
    # that document's processing. Measuring it is what makes "stops BETWEEN
    # documents" falsifiable instead of asserted by construction.
    post_commit_seconds = (
        elapsed - progress[-1]["monotonic_s"] if progress else None
    )

    return {
        "label": label,
        "database": database,
        "elapsed_seconds": elapsed,
        "post_commit_seconds": post_commit_seconds,
        "report": report_dict,
        "progress": progress,
        "snapshot_pre": pre,
        "snapshot_post": post,
        "fingerprint_pre": pre["logical_fingerprint"],
        "fingerprint_post": post["logical_fingerprint"],
        "fingerprint_moved": pre["logical_fingerprint"] != post["logical_fingerprint"],
        "converter_ledger_delta": ledger_delta,
        "converter_calls": len(ledger_delta),
        "http_delta_requests": len(http_delta),
        "http_delta_ollama_requests": len(
            [r for r in http_delta if "11434" in (r.get("host") or "")]
        ),
        "reconciliation": reconcile(
            report_dict,
            pre,
            post,
            ledger_delta=ledger_delta,
            convert_fn_supplied=convert_fn is not None,
        ),
        "params": {
            "max_docs": max_docs,
            "max_seconds": max_seconds,
            "should_continue": should_continue is not None,
            "convert_fn": convert_fn is not None,
        },
    }
