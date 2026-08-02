"""``andamentum-docstore`` — capture now, enrich later.

The deferred-ingestion queue lives in the database; this CLI is how you fill and
drain it from a shell, a cron entry or a launchd timer. The store owns a
*drainable queue*, not a scheduler — the clock stays with the caller.

    # capture markdown: fast, no LLM, keyword-searchable immediately
    andamentum-docstore ingest brain notes.md --defer
    # capture a source: only a reference is stored — NOT searchable until the
    # drain converts it (there is no text to index yet)
    andamentum-docstore ingest-source brain ~/papers/big.pdf

    # inspect
    andamentum-docstore status brain

    # drain (e.g. from cron at 02:00), self-capping before morning
    andamentum-docstore process-pending brain \
        --embedding-model embeddinggemma:latest \
        --max-seconds 21600

Ctrl-C (or SIGTERM from a job runner) pauses cleanly: the in-flight document
finishes and commits, then the run exits. Re-run the same command to resume —
work already done is never repeated.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from .pipeline import harvest_convert_fn
from .public import (
    ingest as _ingest,
    ingest_source as _ingest_source,
    pending_status as _pending_status,
    process_pending as _process_pending,
    retry_failed as _retry_failed,
)


def _install_pause_handler() -> "callable[[], bool]":  # type: ignore[valid-type]
    """Wire SIGINT/SIGTERM to a cooperative stop; return a should_continue().

    The signal only flips a flag — the drain checks it *between* documents, so a
    pause never interrupts a commit and never loses a converted document.
    """
    stopping = {"flag": False}

    def _handle(signum, _frame):  # type: ignore[no-untyped-def]
        if stopping["flag"]:  # second signal: user means it
            print("\nForced exit — in-flight document not committed.", file=sys.stderr)
            raise SystemExit(130)
        stopping["flag"] = True
        name = signal.Signals(signum).name
        print(
            f"\n{name} received — finishing the current document, then pausing. "
            "(Press again to force-quit.)",
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    return lambda: not stopping["flag"]


async def _cmd_ingest(args: argparse.Namespace) -> int:
    content = (
        sys.stdin.read()
        if args.path == "-"
        else Path(args.path).expanduser().read_text()
    )
    doc_id = await _ingest(
        args.database,
        content,
        title=args.title,
        embedding_model=args.embedding_model,
        process="defer" if args.defer else "now",
    )
    print(f"{'queued' if args.defer else 'ingested'}: {doc_id}")
    return 0


async def _cmd_ingest_source(args: argparse.Namespace) -> int:
    doc_id = await _ingest_source(
        args.database,
        args.source,
        title=args.title,
        convert_fn=harvest_convert_fn() if not args.defer else None,
        embedding_model=args.embedding_model,
        process="defer" if args.defer else "now",
    )
    print(f"{'queued' if args.defer else 'ingested'}: {doc_id}")
    return 0


async def _cmd_status(args: argparse.Namespace) -> int:
    st = await _pending_status(args.database)
    print(f"database        : {args.database}")
    print(f"pending_source  : {st.pending_source}")
    print(f"pending_enrich  : {st.pending_enrich}")
    print(f"complete        : {st.complete}")
    print(f"failed          : {st.failed}")
    print(f"total pending   : {st.pending}")
    return 0


async def _cmd_process_pending(args: argparse.Namespace) -> int:
    should_continue = _install_pause_handler()

    def on_progress(done: int, total: int, title: str) -> None:
        print(f"[{done}/{total}] {title}", flush=True)

    report = await _process_pending(
        args.database,
        embedding_model=args.embedding_model,
        convert_fn=harvest_convert_fn(),
        should_continue=should_continue,
        on_progress=on_progress,
        max_docs=args.max_docs,
        max_seconds=args.max_seconds,
    )

    print()
    print(f"converted : {report.documents_converted}")
    print(f"enriched  : {report.documents_enriched}")
    print(f"failed    : {report.documents_failed}")
    if report.documents_skipped:
        print(f"skipped   : {report.documents_skipped} (no converter available)")
    print(f"remaining : {report.remaining}")
    if report.stopped_early:
        print("\nPaused before the queue was empty — re-run to resume.")
    for f in report.failures:
        print(f"  ! {f}", file=sys.stderr)
    # Failures are reported but do not fail the run: a paused/partial drain is a
    # normal outcome, and the queue still holds the work.
    return 0


async def _cmd_retry_failed(args: argparse.Namespace) -> int:
    n = await _retry_failed(args.database)
    print(f"requeued {n} failed document(s)")
    return 0


def _add_model_args(p: argparse.ArgumentParser, *, required: bool) -> None:
    """Embedding model only — the store never calls an LLM to ingest."""
    p.add_argument(
        "--embedding-model",
        required=required,
        default=None,
        help="Embedding model (e.g. embeddinggemma:latest)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-docstore",
        description="Document store: capture now, enrich later.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest markdown/text from a file or '-'")
    p_ing.add_argument("database")
    p_ing.add_argument("path", help="File to ingest, or '-' for stdin")
    p_ing.add_argument("--title", default=None)
    p_ing.add_argument(
        "--defer",
        action="store_true",
        help="Register + index only; queue enrichment for process-pending (no LLM)",
    )
    _add_model_args(p_ing, required=False)
    p_ing.set_defaults(func=_cmd_ingest)

    p_src = sub.add_parser(
        "ingest-source", help="Queue a file/URL for conversion (PDF, DOCX, HTML, …)"
    )
    p_src.add_argument("database")
    p_src.add_argument("source", help="Path or URL")
    p_src.add_argument("--title", default=None)
    p_src.add_argument(
        "--now",
        dest="defer",
        action="store_false",
        help="Convert and enrich immediately instead of queueing",
    )
    p_src.set_defaults(defer=True)
    _add_model_args(p_src, required=False)
    p_src.set_defaults(func=_cmd_ingest_source)

    p_st = sub.add_parser("status", help="Show queue depth")
    p_st.add_argument("database")
    p_st.set_defaults(func=_cmd_status)

    p_pp = sub.add_parser(
        "process-pending", help="Drain the queue (resumable; Ctrl-C pauses cleanly)"
    )
    p_pp.add_argument("database")
    _add_model_args(p_pp, required=True)
    p_pp.add_argument("--max-docs", type=int, default=None)
    p_pp.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop after this much wall-clock (checked between documents)",
    )
    p_pp.set_defaults(func=_cmd_process_pending)

    p_rf = sub.add_parser("retry-failed", help="Requeue documents marked failed")
    p_rf.add_argument("database")
    p_rf.set_defaults(func=_cmd_retry_failed)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        sys.exit(asyncio.run(args.func(args)))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
