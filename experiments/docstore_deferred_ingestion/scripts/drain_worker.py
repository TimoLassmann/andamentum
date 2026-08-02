"""The killable drain subprocess. Deliberately boring.

argparse -> ``process_pending`` with the counting convert_fn wrapping
``pipeline.harvest_convert_fn()``. No interrupt logic of its own, no cleverness,
no cleanup handlers — so what the supervisor kills IS the real drain rather than
a harness impersonating one.

Runs in its own process group (the supervisor calls ``setsid`` via
``start_new_session``) so a single ``killpg`` takes down the worker and any child
Docling processes together.

Run (normally spawned, but runnable by hand):
    uv run python -m experiments.docstore_deferred_ingestion.scripts.drain_worker \
        --db dfr_kill --model ollama:gemma4:26b-nvfp4 \
        --embedding-model embeddinggemma:latest \
        --convert-ledger results/ledgers/kill_convert.jsonl \
        --http-ledger results/ledgers/c1_http.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import _common as C
from .instrument import counting_convert_fn, http_recorder


async def drain(args: argparse.Namespace) -> int:
    from andamentum.document_store import process_pending
    from andamentum.document_store.pipeline import harvest_convert_fn

    database = C.require_db_name(args.db)
    convert_fn = counting_convert_fn(
        Path(args.convert_ledger), harvest_convert_fn()
    )

    def on_progress(done: int, total: int, title: str) -> None:
        # The same line shape the real CLI prints, so log-scrapers written
        # against one work against the other.
        print(f"[{done}/{total}] {title}", flush=True)

    # truncate=False: the http ledger for a resume run must not erase the kill
    # run's record of what the killed process did.
    with http_recorder(Path(args.http_ledger), truncate=args.truncate_http):
        report = await process_pending(
            database,
            model=args.model,
            embedding_model=args.embedding_model,
            convert_fn=convert_fn,
            on_progress=on_progress,
            max_docs=args.max_docs,
            max_seconds=args.max_seconds,
        )

    print("REPORT " + json.dumps(asdict(report)), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--convert-ledger", required=True)
    parser.add_argument("--http-ledger", required=True)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--truncate-http",
        action="store_true",
        help="Truncate the http ledger at start (the FIRST run only)",
    )
    args = parser.parse_args()
    print(f"worker pid={__import__('os').getpid()} db={args.db}", flush=True)
    try:
        return asyncio.run(drain(args))
    except Exception as exc:  # noqa: BLE001 — re-raised after a visible marker
        print(f"WORKER-ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
