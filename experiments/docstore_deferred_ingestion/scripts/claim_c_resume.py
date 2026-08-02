"""Claim (c), part 2: resume in a FRESH process and prove nothing was re-converted.

Asserts, against the same database and the SAME converter ledger opened in append
mode:

  H-c1  exactly 3 ledger entries for 3 sources; no duplicate source string;
        report.documents_converted == 2 in this run
  H-c2  document #1's markdown sha256 is byte-identical to the pre-kill hash
  H-c3  the database opened through the library's ordinary path — no manual
        repair, no -wal/-shm deletion
  H-d4  the only work repeated is the enrichment of the single in-flight document

H-c2 is strictly stronger than H-c1: a call counter cannot distinguish "not
re-converted" from "re-converted to the same string". Comparison is WITHIN one
run only — Docling/RapidOCR output is not byte-stable across versions or
machines, so the analyzer refuses a cross-run markdown comparison.

Also computes ``checkpoint_savings_seconds``: the per-PDF conversion wall-time
measured by ``convert_reference`` times the number of already-converted documents
this drain did NOT re-convert.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_c_resume
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from typing import Any

from . import _common as C
from .claim_c_kill import KILL_LEDGER
from .drain import reconcile as drain_reconcile
from .events import EventLog
from .fingerprint import integrity_check, snapshot
from .instrument import duplicate_sources, ledger_sources, load_http_ledger, read_jsonl
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_c_resume/1"

HTTP_LEDGER = C.LEDGERS / "c2_http.jsonl"
WORKER_LOG = C.LOGS / "kill_run2.log"


def run_resume() -> tuple[dict[str, Any], C.ClaimRecorder]:
    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    kill_db = C.require_db_name(cfg["databases"]["kill"])
    db_path = C.db_file(kill_db)
    root_url = cfg["models"]["ollama_root_url"]
    log = EventLog(
        C.EVENTS / "claim_c_resume.jsonl", rule="claim_c_resume", run_id=C.run_id()
    )

    kill_result = C.read_json(C.RESULTS / "claim_c_kill.json")
    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    conversion_seconds = {
        c["arxiv_id"]: c["conversion_seconds"] for c in baseline["conversions"]
    }

    entries_before = read_jsonl(KILL_LEDGER)
    n_sources_expected = int(cfg["lineages"]["kill_sources"])
    # RE-RUN GUARD. Snakemake cannot own this file (two rules may not declare one
    # output), so the protection against a double-append lives here instead. A
    # second append would silently corrupt the very count H-c1 rests on.
    if len(entries_before) >= n_sources_expected:
        raise RuntimeError(
            f"{KILL_LEDGER} already holds {len(entries_before)} entries, which is the "
            f"full expected count ({n_sources_expected}). This rule APPENDS, so running "
            "it again would double-count and make H-c1 meaningless. Re-run the kill "
            "lineage from the start:\n"
            "    uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile "
            "--cores 1 --resources ollama=1 --forcerun claim_c_kill"
        )

    integrity_before = integrity_check(db_path)
    pre = snapshot(db_path, label="claim_c_resume_pre")

    payload: dict[str, Any] = {
        "database": kill_db,
        "target_doc_id": kill_result["target_doc_id"],
        "target_arxiv_id": kill_result["target_arxiv_id"],
        "pre_kill_markdown_sha256": kill_result["pre_kill_markdown_sha256"],
        "post_kill_markdown_sha256": kill_result["post_kill_markdown_sha256"],
        "ledger_entries_before_resume": len(entries_before),
        "integrity_before_resume": integrity_before,
        "sidecars_preserved": True,
        "sidecars_note": (
            "no -wal/-shm file is deleted and no repair is run: the resume must succeed "
            "through the library's ordinary path or the durability claim fails honestly"
        ),
        "snapshot_pre": pre,
        "ollama_ps_before": ollama_ps(root_url),
    }

    # --- the resume drain, in a FRESH process ------------------------------
    C.LOGS.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "-m",
        "experiments.docstore_deferred_ingestion.scripts.drain_worker",
        "--db", kill_db,
        "--model", C.LLM_MODEL,
        "--embedding-model", C.EMBEDDING_MODEL,
        "--convert-ledger", str(KILL_LEDGER),   # SAME file, append mode
        "--http-ledger", str(HTTP_LEDGER),
        "--truncate-http",
    ]
    started = time.monotonic()
    # start_new_session + killpg, NOT subprocess.run(timeout=...).
    #
    # `uv run python -m ...` is a two-process chain: subprocess.run's timeout
    # handler kills only the direct child (`uv`), leaving the python grandchild
    # ALIVE. This experiment hit exactly that: after a resume timeout the worker
    # kept draining the same database and kept calling Ollama, so the convert
    # ledger recorded each source converted twice under two different pids and a
    # later run contended for the model. Putting the worker in its own process
    # group and killing the GROUP is the only way to take the whole chain (plus
    # any Docling children) down together.
    proc = subprocess.Popen(
        cmd,
        cwd=str(C.REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=os.environ.copy(),
    )
    try:
        stdout, _ = proc.communicate(
            timeout=float(cfg["timeouts"]["drain_worker_seconds"])
        )
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        stdout, _ = proc.communicate()
        WORKER_LOG.write_text(stdout or "")
        raise
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)
    WORKER_LOG.write_text(stdout or "")
    elapsed = time.monotonic() - started

    payload["resume_returncode"] = proc.returncode
    payload["resume_seconds"] = elapsed
    payload["resume_log"] = str(WORKER_LOG.relative_to(C.EXP_DIR))

    report: dict[str, Any] | None = None
    for line in (stdout or "").splitlines():
        if line.startswith("REPORT "):
            report = json.loads(line[len("REPORT ") :])
    payload["resume_report"] = report

    rec.check(
        "H-c3/resume_opens",
        proc.returncode == 0 and report is not None,
        measured={"returncode": proc.returncode, "report": report is not None},
        expected="exit 0 with a report",
        detail=(
            "the resume opened the post-SIGKILL database through the library's normal "
            "path with no manual intervention"
        ),
    )
    if report is None:
        payload["resume_stdout_tail"] = (stdout or "")[-4000:]
        return payload, rec

    # --- H-c1: the converter ledger ---------------------------------------
    entries_after = read_jsonl(KILL_LEDGER)
    sources = ledger_sources(KILL_LEDGER)
    duplicates = duplicate_sources(KILL_LEDGER)
    n_sources = int(cfg["lineages"]["kill_sources"])

    payload.update(
        {
            "ledger_entries": len(entries_after),
            "ledger_sources": sources,
            "ledger_duplicates": duplicates,
            "ledger_entries_added_by_resume": len(entries_after) - len(entries_before),
            "ledger": entries_after,
        }
    )
    rec.check(
        "H-c1/entries",
        len(entries_after) == n_sources,
        measured=len(entries_after),
        expected=n_sources,
        detail=f"{n_sources} sources must produce exactly {n_sources} conversions in total",
    )
    rec.check(
        "H-c1/no_duplicates",
        not duplicates,
        measured=duplicates,
        expected=[],
        detail="a repeated source string is a re-conversion",
    )
    rec.check(
        "H-c1/resume_converted",
        report["documents_converted"] == n_sources - 1,
        measured=report["documents_converted"],
        expected=n_sources - 1,
        detail="the already-converted document #1 must NOT be converted again",
    )

    # --- H-c2: byte identity ----------------------------------------------
    post = snapshot(db_path, label="claim_c_final")
    rows = {r["doc_uuid"]: r for r in post["documents"]}
    target = rows[kill_result["target_doc_id"]]
    payload["final_markdown_sha256"] = target["markdown_sha256"]
    payload["snapshot_final"] = post
    stable = target["markdown_sha256"] == kill_result["post_kill_markdown_sha256"]
    payload["markdown_sha_stable"] = stable
    rec.check(
        "H-c2",
        stable,
        measured=target["markdown_sha256"],
        expected=kill_result["post_kill_markdown_sha256"],
        detail=(
            "byte identity within ONE run. A call counter cannot distinguish "
            "'not re-converted' from 're-converted to the same string'"
        ),
    )

    # --- final queue state -------------------------------------------------
    counts = post["status_counts"]
    payload["final_status_counts"] = counts
    rec.check(
        "H-c/complete",
        counts["complete"] == n_sources,
        measured=counts["complete"],
        expected=n_sources,
    )
    rec.check(
        "H-c/no_failures",
        counts["failed"] == 0,
        measured=counts["failed"],
        expected=0,
        detail="; ".join(report.get("failures", [])),
    )

    integrity_after = integrity_check(db_path)
    payload["integrity_after_resume"] = integrity_after
    payload["sqlite_ok"] = (
        integrity_before["integrity_ok"]
        and integrity_before["quick_ok"]
        and integrity_after["integrity_ok"]
        and integrity_after["quick_ok"]
    )

    # --- H-d4 + the checkpoint's price tag ---------------------------------
    #
    # `saved_documents == 1` EXACTLY, not `<= 1`. Three sources were queued; the
    # kill landed after #1's conversion committed, so the resume must convert
    # exactly the other two. The earlier `<= 1` could not fail in the direction it
    # named: a completely broken checkpoint that re-converted all three would give
    # documents_converted == 3, saved_documents == 0, and `0 <= 1` would PASS.
    saved_documents = n_sources - report["documents_converted"]
    rec.check(
        "H-d4/documents_not_reconverted",
        saved_documents == 1,
        measured=saved_documents,
        expected=1,
        detail=(
            "exactly the one document converted before the kill; 0 would mean the "
            "checkpoint was ignored and everything was re-converted, 2+ would mean the "
            "resume did not finish"
        ),
    )

    # THE SAVING, from the MEASURED in-drain conversion of that source — not from
    # `conversion_baseline`'s first call, which on this host carries the one-time
    # Docling/RapidOCR initialisation (14.98 / 13.47 / 9.64 / 3.86 s in call order,
    # uncorrelated with document size). Both numbers are published, plus the
    # measured warm/cold split, so the reader can see which is which.
    target_source = kill_result["queued"][0]["source"]
    target_ledger = next(
        (e for e in entries_before if e.get("source") == target_source and e.get("ts_end")),
        None,
    )
    baseline_first_call = conversion_seconds.get(kill_result["target_arxiv_id"])
    warm_reference = baseline.get("docling_warm_conversion_seconds")
    payload["checkpoint_savings_seconds"] = (
        target_ledger["seconds"] if target_ledger else None
    )
    payload["checkpoint_savings_documents"] = saved_documents
    payload["checkpoint_savings_basis"] = {
        "source": "the kill lineage's own converter-ledger entry for this document",
        "in_drain_conversion_seconds": target_ledger["seconds"] if target_ledger else None,
        "in_drain_was_first_in_its_process": True,
        "includes_docling_initialisation": True,
        "standalone_first_call_seconds": baseline_first_call,
        "docling_warm_conversion_seconds": warm_reference,
        "docling_init_seconds": baseline.get("docling_init_seconds"),
        "marginal_saving_note": (
            "The resume process converts two more PDFs regardless, so it pays Docling "
            "initialisation either way. The MARGINAL saving is therefore the warm "
            "conversion cost, not the first-call cost — both are recorded so the "
            "distinction is on the artefact's face rather than in a reader's head."
        ),
    }

    # The repeated ENRICHMENT, measured on this lineage by claim_c_kill: killed_at
    # minus the converter ledger's last ts_end. Previously this field copied
    # micro_stages' sequential total — a different paper, a different database, and
    # a deliberately sequential path the experiment itself measures as slower.
    repeated_enrichment = kill_result.get("discarded_enrichment_seconds")
    payload["repeated_enrichment_seconds"] = repeated_enrichment
    payload["repeated_enrichment_basis"] = kill_result.get("discarded_enrichment_basis")
    payload["repeated_enrichment_over_conversion"] = (
        repeated_enrichment / target_ledger["seconds"]
        if (repeated_enrichment is not None and target_ledger and target_ledger["seconds"])
        else None
    )
    payload["checkpoint_economics_note"] = (
        "The saving is the SMALLER of the two stages: conversion is tens of seconds "
        "while enrichment is minutes at the measured per-chunk LLM latency. That is a "
        "deflating but useful result — it tells a future optimiser to attack the LLM "
        "loop, not the converter."
    )
    rec.check(
        "H-d4/repeated_under_conversion",
        payload["repeated_enrichment_over_conversion"] is not None
        and payload["repeated_enrichment_over_conversion"] < 1.0,
        measured=payload["repeated_enrichment_over_conversion"],
        expected="< 1 (repeated enrichment cheaper than the conversion it preserved)",
        detail=(
            "the pre-registered statement's second clause, which previously had no "
            "metric attached to it at all"
        ),
    )
    rec.observe(
        "checkpoint_savings_seconds",
        payload["checkpoint_savings_seconds"],
        detail=(
            f"{saved_documents} document(s) not re-converted; measured in-drain, "
            "includes one-time Docling initialisation"
        ),
    )

    # H-x for the kill lineage. The resume drain runs in a subprocess and so does
    # not pass through `drain.instrumented_drain`; reconciling it here is what
    # makes H-x's "for every drain" true rather than "for every drain that
    # happened to use the wrapper".
    payload["reconciliation"] = drain_reconcile(
        report,
        pre,
        post,
        ledger_delta=entries_after[len(entries_before) :],
        convert_fn_supplied=True,
    )
    payload["report_db_mismatches"] = payload["reconciliation"]["n_mismatches"]
    rec.check(
        "H-x/claim_c_resume",
        payload["report_db_mismatches"] == 0,
        measured=payload["report_db_mismatches"],
        expected=0,
        detail="; ".join(
            f"{m['field']}: report={m['report']} db={m['database']}"
            for m in payload["reconciliation"]["mismatches"]
        ),
    )

    http = load_http_ledger(HTTP_LEDGER)
    payload["http"] = http.summary()
    rec.observe(
        "resume_max_in_flight",
        http.max_in_flight,
        detail=(
            "GLOBAL peak concurrent Ollama requests — an upper bound over all "
            "endpoints, NOT an attribution. The per-endpoint breakdown is the next "
            "observation; attributing the global peak to a single semaphore is how "
            "an earlier report ended up blaming Semaphore(5) for a peak of 8"
        ),
    )
    rec.observe(
        "resume_max_in_flight_by_path",
        http.max_in_flight_by_path(),
        detail="peak concurrency counted per endpoint, from a per-path in-flight counter",
    )
    payload["ollama_ps_after"] = ollama_ps(root_url)

    log.emit(
        db_name=kill_db,
        observer="report",
        note=(
            f"resume: converted={report['documents_converted']} "
            f"enriched={report['documents_enriched']} "
            f"failed={report['documents_failed']}"
        ),
        stage_seconds=elapsed,
    )
    return payload, rec


FINAL_LEDGER = C.LEDGERS / "kill_convert.final.jsonl"


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = run_resume()
    finally:
        C.write_json(
            C.RESULTS / "claim_c_resume.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
        # The DAG-owned, immutable copy of the ledger's end state. Written even on
        # failure so a failed claim still leaves the evidence a reader would open.
        FINAL_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        FINAL_LEDGER.write_text(
            KILL_LEDGER.read_text() if KILL_LEDGER.exists() else ""
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
