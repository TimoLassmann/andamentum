"""Claim (c), part 1: SIGKILL a real drain the instant the checkpoint should hold.

WHY A HARD KILL AND NOT A COOPERATIVE STOP
------------------------------------------
``process_pending`` converts AND enriches a ``pending_source`` inside the SAME
loop iteration, and ``should_continue`` / ``max_docs`` / ``max_seconds`` are all
checked at the TOP of the iteration. A cooperative pause therefore STRUCTURALLY
CANNOT leave a converted-but-unenriched document. Only a hard kill tests the
checkpoint against the filesystem and sqlite. (Cooperative stops are claim (d);
they are different mechanisms, not variants of one.)

THE KILL GATE IS A CONJUNCTION
-------------------------------
The supervisor waits until BOTH hold:
  * the converter ledger has an entry WITH an end timestamp, and
  * document #1's row reads ``pending_enrich``.

A kill delivered before the conversion has committed would make H-c1 vacuous, so
a timeout FAILS LOUD rather than killing at an arbitrary moment.

While polling it also watches for the H-c4 window: ``_convert_document`` commits
markdown via ``store.update(...)`` and only THEN calls
``set_ingest_status(PENDING_ENRICH)`` — two separate transactions. A kill landing
between them would leave markdown on disk with the row still ``pending_source``,
and the next drain WOULD re-convert. Observing that window is a reportable
finding; observing none bounds it below the poll interval.

Polling uses a READ-ONLY URI connection at >= 250 ms so it cannot contend on the
write lock and perturb the thing being measured.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_c_kill
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

from . import _common as C
from .events import EventLog
from .fingerprint import integrity_check, poll_all_rows, poll_row, snapshot
from .instrument import read_jsonl, truncate_ledger
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_c_kill/1"

KILL_LEDGER = C.LEDGERS / "kill_convert.jsonl"
HTTP_LEDGER = C.LEDGERS / "c1_http.jsonl"
WORKER_LOG = C.LOGS / "kill_run1.log"


async def queue_sources(database: str, sources: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Queue each (arxiv_id, path) as pending_source. No conversion happens here."""
    from andamentum.document_store.pipeline import ingest_source

    queued = []
    for arxiv_id, path in sources:
        doc_id = await ingest_source(
            database, path, title=None, metadata={"arxiv_id": arxiv_id}, process="defer"
        )
        queued.append({"arxiv_id": arxiv_id, "source": path, "doc_id": doc_id})
    return queued


def spawn_worker(database: str) -> subprocess.Popen[str]:
    """Start the drain in its OWN process group so killpg reaches Docling too."""
    C.LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.docstore_deferred_ingestion.scripts.drain_worker",
        "--db",
        database,
        "--model",
        C.LLM_MODEL,
        "--embedding-model",
        C.EMBEDDING_MODEL,
        "--convert-ledger",
        str(KILL_LEDGER),
        "--http-ledger",
        str(HTTP_LEDGER),
        "--truncate-http",
    ]
    log = WORKER_LOG.open("w")
    return subprocess.Popen(
        cmd,
        cwd=str(C.REPO_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=os.environ.copy(),
    )


@contextmanager
def supervised_worker(database: str) -> Iterator[subprocess.Popen[str]]:
    """Spawn the drain worker and GUARANTEE its process group is dead on exit.

    WHY THIS EXISTS (a defect this experiment actually hit).
    ``run_kill`` spawns a long-running drain and only kills it at the planned
    moment. Every ``raise`` between the spawn and that moment — a failed gate,
    a missing column, an assertion — used to leak a LIVE drain worker: it kept
    converting into the same database and kept calling Ollama. The next run then
    measured a database mutated by a process it did not know about, and two
    inference processes ran at once, which this project forbids outright. The
    symptom was a convert-ledger showing each source converted twice under two
    different pids, and a 1800 s resume timeout caused by contention.

    A leaked worker is worse than a crash, because it corrupts the NEXT run
    silently. So cleanup is structural (``finally``), not a step someone
    remembers to write on each error path. Killing an already-dead group is a
    no-op, so the deliberate SIGKILL inside the body stays correct.
    """
    proc = spawn_worker(database)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print(
                f"WARNING: drain worker pid={proc.pid} survived cleanup SIGKILL",
                file=sys.stderr,
                flush=True,
            )


def run_kill() -> tuple[dict[str, Any], C.ClaimRecorder]:
    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    kill_db = C.require_db_name(cfg["databases"]["kill"])
    db_path = C.db_file(kill_db)
    root_url = cfg["models"]["ollama_root_url"]
    log = EventLog(C.EVENTS / "claim_c_kill.jsonl", rule="claim_c_kill", run_id=C.run_id())

    # Clean slate + a fresh converter ledger. claim_c_resume APPENDS to this same
    # file and therefore declares it as both an input and an output.
    C.drop_database(kill_db)
    truncate_ledger(KILL_LEDGER)

    registry = {p["arxiv_id"]: p for p in C.read_json(C.REGISTRY_PATH)["papers"]}
    n_sources = int(cfg["lineages"]["kill_sources"])
    chosen = list(registry.values())[:n_sources]
    sources = [(p["arxiv_id"], str(C.EXP_DIR / p["path"])) for p in chosen]

    queued = asyncio.run(queue_sources(kill_db, sources))
    for item in queued:
        log.emit(
            db_name=kill_db,
            observer="report",
            doc_id=item["doc_id"],
            arxiv_id=item["arxiv_id"],
            to_status="pending_source",
            note="queued for the kill lineage",
        )

    # list_pending orders pending_source first, then created_date ASC — so
    # document #1 is the first one queued.
    first = queued[0]
    payload: dict[str, Any] = {
        "database": kill_db,
        "queued": queued,
        "target_doc_id": first["doc_id"],
        "target_arxiv_id": first["arxiv_id"],
        "ollama_ps_before": ollama_ps(root_url),
        "snapshot_before": snapshot(db_path, label="claim_c_before"),
    }

    with supervised_worker(kill_db) as proc:
        print(f"worker pid={proc.pid} (own process group)", flush=True)

        # --- the kill gate ----------------------------------------------------
        timeout = float(cfg["timeouts"]["kill_gate_seconds"])
        interval = float(cfg["polling"]["interval_seconds"])
        deadline = time.monotonic() + timeout
        window_observations: list[dict[str, Any]] = []
        n_polls = 0
        gate_state: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            n_polls += 1
            rows = {r["doc_uuid"]: r for r in poll_all_rows(db_path)}

            # H-c4: markdown present while the row still says pending_source.
            for uuid, row in rows.items():
                if row["ingest_status"] == "pending_source" and row["markdown_chars"] > 0:
                    window_observations.append(
                        {"observed_at": C.utc_now(), "doc_uuid": uuid, **row}
                    )

            entries = read_jsonl(KILL_LEDGER)
            finished = [e for e in entries if e.get("ts_end")]
            target = rows.get(first["doc_id"])
            if finished and target is not None and target["ingest_status"] == "pending_enrich":
                # The scan above is deliberately CHEAP — it reads
                # length(markdown_content), not the content — because it runs every
                # ``polling.interval_seconds`` and must not contend on the writer.
                # H-c2 needs the actual pre-kill sha256, so read the target row ONCE,
                # here, with the single-row helper written for the kill gate. Doing
                # this inside the poll loop would hash every document's markdown four
                # times a second and perturb the thing being measured.
                authoritative = poll_row(db_path, first["doc_id"])
                if authoritative is None:
                    raise RuntimeError(
                        f"Kill gate opened for doc {first['doc_id']} but the "
                        "authoritative single-row re-read returned nothing. Refusing "
                        "to record a checkpoint hash that was never observed."
                    )
                gate_state = {
                    "n_polls": n_polls,
                    "ledger_entries_at_gate": len(entries),
                    "finished_conversions_at_gate": len(finished),
                    "target_row": authoritative,
                }
                break

            if proc.poll() is not None:
                raise RuntimeError(
                    f"The drain worker exited (code {proc.returncode}) BEFORE the kill gate "
                    f"opened. See {WORKER_LOG}. Killing nothing would make H-c1 vacuous, so "
                    "this is a hard failure rather than a skipped arm."
                )
            time.sleep(interval)

        if gate_state is None:
            proc.kill()
            raise TimeoutError(
                f"Kill gate never opened within {timeout:.0f}s (ledger entry with an end "
                "timestamp AND document #1 reading pending_enrich). Failing loud rather than "
                f"killing at an arbitrary moment. Worker log: {WORKER_LOG}"
            )

        pre_kill_row = gate_state["target_row"]
        payload["gate"] = gate_state
        payload["pre_kill_markdown_sha256"] = pre_kill_row["markdown_sha256"]
        log.emit(
            db_name=kill_db,
            observer="poll",
            doc_id=first["doc_id"],
            arxiv_id=first["arxiv_id"],
            from_status="pending_source",
            to_status="pending_enrich",
            markdown_sha256=pre_kill_row["markdown_sha256"],
            note="kill gate opened",
        )

        # --- the kill ---------------------------------------------------------
        kill_at = time.time()
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            raise RuntimeError("worker survived SIGKILL for 60s — refusing to continue")
        payload["kill"] = {
            "signal": "SIGKILL",
            "why_not_sigterm": (
                "SIGTERM exercises the CLI's cooperative handler, which is claim (d) arm 4. "
                "A cooperative stop structurally cannot leave a converted-but-unenriched "
                "document, so only a hard kill tests the checkpoint"
            ),
            "killed_at": kill_at,
            "worker_returncode": proc.returncode,
            "worker_log": str(WORKER_LOG.relative_to(C.EXP_DIR)),
        }

    # --- post-kill truth --------------------------------------------------
    post = snapshot(db_path, label="claim_c_post_kill")
    integrity = integrity_check(db_path)
    post_kill_rows = {r["doc_uuid"]: r for r in post["documents"]}
    target_row = post_kill_rows[first["doc_id"]]

    ledger_after_kill = read_jsonl(KILL_LEDGER)
    # THE WORK THE SIGKILL ACTUALLY DISCARDED, measured rather than borrowed.
    #
    # The target's conversion committed at its ledger entry's ts_end; the row then
    # flipped to pending_enrich and enrichment began. The kill landed at
    # killed_at. The difference is the enrichment seconds that must be redone on
    # resume — an upper bound on it, since it also contains the status flip.
    #
    # An earlier version reported `micro_stages.sequential_total_seconds` here
    # instead: a sequential micro-benchmark of a DIFFERENT paper in a DIFFERENT
    # database, plotted as the measured cost of THIS kill. It overstated the
    # quantity by ~4800x under an axis labelled "seconds of repeated work".
    last_conversion = (
        max(
            (e for e in ledger_after_kill if e.get("ts_end")),
            key=lambda e: e["ts_end"],
            default=None,
        )
    )
    discarded_enrichment = (
        kill_at - last_conversion["ts_end"] if last_conversion else None
    )
    payload.update(
        {
            "discarded_enrichment_seconds": discarded_enrichment,
            "discarded_enrichment_basis": {
                "killed_at": kill_at,
                "last_conversion_ts_end": (
                    last_conversion["ts_end"] if last_conversion else None
                ),
                "last_conversion_source": (
                    last_conversion["source"] if last_conversion else None
                ),
                "meaning": (
                    "wall-clock between the conversion committing and the SIGKILL: the "
                    "enrichment work the resume must redo. Upper bound — it also "
                    "contains the pending_source -> pending_enrich status flip and the "
                    "supervisor's poll latency"
                ),
            },
            "snapshot_post_kill": post,
            "sqlite_integrity": integrity,
            "post_kill_markdown_sha256": target_row["markdown_sha256"],
            "post_kill_target_status": target_row["ingest_status"],
            "ledger_entries_after_kill": len(ledger_after_kill),
            "ledger_after_kill": ledger_after_kill,
            "hc4_window": {
                "observed": bool(window_observations),
                "n_observations": len(window_observations),
                "observations": window_observations[:20],
                "poll_interval_seconds": interval,
                "n_polls": n_polls,
                "interpretation": (
                    "observing the window is a reportable finding (such a document WOULD "
                    "be re-converted); observing none bounds the two-transaction window "
                    "below the poll interval"
                ),
            },
            "ollama_ps_after": ollama_ps(root_url),
        }
    )

    # --- what must hold at this instant -----------------------------------
    rec.check(
        "H-c/gate",
        target_row["ingest_status"] in ("pending_enrich", "complete"),
        measured=target_row["ingest_status"],
        expected="pending_enrich (or complete if enrichment beat the kill)",
        detail="the kill must land AFTER the conversion checkpoint, or H-c1 is vacuous",
    )
    rec.check(
        "H-c/markdown_durable",
        target_row["markdown_chars"] > 0,
        measured=target_row["markdown_chars"],
        expected="> 0",
        detail="the converted markdown must have survived the hard kill on disk",
    )
    rec.check(
        "H-c3/integrity",
        integrity["integrity_ok"] and integrity["quick_ok"],
        measured={
            "integrity_check": integrity["integrity_check"],
            "quick_check": integrity["quick_check"],
        },
        expected="both 'ok'",
        detail=(
            "if a hard kill can wedge the database, resumability is theoretical. A "
            "failure here is a genuine negative result and must NOT be worked around by "
            "deleting a -wal file"
        ),
    )
    rec.observe(
        "H-c4",
        payload["hc4_window"]["observed"],
        detail=(
            f"{len(window_observations)} observations of non-empty markdown while the "
            f"row still read pending_source, across {n_polls} polls at {interval}s"
        ),
    )
    rec.observe(
        "discarded_enrichment_seconds",
        discarded_enrichment,
        detail=(
            "killed_at minus the converter ledger's last ts_end — the enrichment "
            "seconds this hard kill threw away, measured on THIS lineage"
        ),
    )
    rec.observe(
        "sidecar_files_after_kill",
        integrity["sidecar_files"],
        detail="-wal/-shm files present after the kill; the resume must not delete them",
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = run_kill()
    finally:
        C.write_json(
            C.RESULTS / "claim_c_kill.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}", file=sys.stderr)
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
