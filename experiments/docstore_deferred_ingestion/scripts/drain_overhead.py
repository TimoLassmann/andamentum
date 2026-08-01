"""The preflight tax: what an empty drain costs a cron job on every wake-up.

THREE FRESH PROCESSES against an EMPTY queue. Fresh is MANDATORY: ``_stores`` and
``_preflight_done`` are module-level per-PROCESS caches, so a repeat inside one
process would measure zero and report a tax of nothing.

This is the correction term that makes the first document's attributed time
honest, and it is the reason H-b is stated as "<= 2 requests" rather than "zero".
If the tax is large relative to one document, frequent small drains are a bad
operational strategy and the README says so.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.drain_overhead
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any

from . import _common as C
from .instrument import load_http_ledger
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.drain_overhead/1"

N_REPEATS = 3


async def _one_empty_drain(database: str, ledger_path: str) -> dict[str, Any]:
    """One drain against an empty queue, in THIS (fresh) process."""
    from andamentum.document_store import pending_status, process_pending

    from .instrument import http_recorder

    with http_recorder(ledger_path) as ledger:
        started = time.monotonic()
        status = await pending_status(database)
        report = await process_pending(
            database,
            model=C.LLM_MODEL,
            embedding_model=C.EMBEDDING_MODEL,
            convert_fn=None,
        )
        elapsed = time.monotonic() - started
        summary = ledger.summary()
    return {
        "elapsed_seconds": elapsed,
        "queue": asdict(status),
        "report": asdict(report),
        "http": summary,
        "ollama_requests": len(
            [r for r in ledger.records if "11434" in (r.get("host") or "")]
        ),
    }


def child() -> int:
    """Subprocess entry point: run one empty drain and print the record as JSON."""
    database = C.require_db_name(C.CONFIG["databases"]["empty"])
    index = int(sys.argv[2])
    ledger = str(C.LEDGERS / f"overhead_{index}_http.jsonl")
    record = asyncio.run(_one_empty_drain(database, ledger))
    print("RECORD " + json.dumps(record, default=str))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child()

    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    database = C.require_db_name(cfg["databases"]["empty"])
    root_url = cfg["models"]["ollama_root_url"]
    C.drop_database(database)

    runs: list[dict[str, Any]] = []
    for index in range(N_REPEATS):
        proc = subprocess.run(
            [
                "uv", "run", "python", "-m",
                "experiments.docstore_deferred_ingestion.scripts.drain_overhead",
                "--child", str(index),
            ],
            cwd=str(C.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"empty-drain child {index} exited {proc.returncode}:\n{proc.stderr[-3000:]}"
            )
        record = None
        for line in proc.stdout.splitlines():
            if line.startswith("RECORD "):
                record = json.loads(line[len("RECORD ") :])
        if record is None:
            raise RuntimeError(f"child {index} printed no RECORD line:\n{proc.stdout[-2000:]}")
        record["index"] = index
        record["ledger"] = load_http_ledger(
            C.LEDGERS / f"overhead_{index}_http.jsonl"
        ).summary()
        runs.append(record)
        print(
            f"empty drain {index}: {record['elapsed_seconds']:.2f}s, "
            f"{record['ollama_requests']} ollama requests",
            flush=True,
        )

    seconds = [r["elapsed_seconds"] for r in runs]
    requests = [r["ollama_requests"] for r in runs]
    payload: dict[str, Any] = {
        "database": database,
        "n_processes": N_REPEATS,
        "fresh_process_rationale": (
            "_stores and _preflight_done are module-level per-PROCESS caches; a repeat "
            "inside one process would measure zero"
        ),
        "runs": runs,
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.mean(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "median_ollama_requests": statistics.median(requests),
        "ollama_ps": ollama_ps(root_url),
    }
    rec.check(
        "preflight_tax/requests",
        statistics.median(requests) <= 2,
        measured=statistics.median(requests),
        expected="<= 2 (1 embed + 1 chat)",
        detail="the fixed per-wake-up cost a cron-driven drain pays",
    )
    rec.observe(
        "preflight_tax_seconds",
        payload["median_seconds"],
        detail=(
            "if this is large relative to one document's processing time, frequent small "
            "drains are a bad operational strategy"
        ),
    )
    C.write_json(
        C.RESULTS / "drain_overhead.json", {**payload, **rec.payload()}, schema=SCHEMA
    )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
