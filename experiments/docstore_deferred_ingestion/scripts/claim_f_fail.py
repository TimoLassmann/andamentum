"""Claim (f): failures are recorded, diagnosable, and do not abort the drain.

Queue: 1 good PDF + 4 REAL failure modes. No monkeypatching, no fault injection —
the failures are genuine, because a fabricated exception proves nothing about how
the shipped code behaves on a real one.

    (i)   a path that does not exist
    (ii)  400 bytes of garbage saved with a .pdf extension
    (iii) https://arxiv.org/pdf/0000.00000 — a real 404 over the real network.
          NOT a localhost URL: core.url_safety's SSRF guard would reject that for
          a DIFFERENT reason than the one under test and muddy the result.
    (iv)  a .md file containing only whitespace, which drives the
          ValueError("Conversion produced no content") branch inside
          _convert_document — which raises BEFORE the write and therefore
          correctly routes back to pending_source on retry.

Sequence:
    drain      -> complete 1, failed 4, ingest_error names the exception TYPE on
                  every failed row, 4 entries in ProcessReport.failures  (H-f1)
    retry      -> retry_failed() == 4, all back to pending_source (markdown empty)
    drain      -> they fail AGAIN. The systematic problem SURFACES rather than
                  looping silently — that is the designed behaviour, not a defect.
    fix one    -> copy a real PDF to the missing path, retry, drain -> 1 completes
    stage test -> take the completed good document, mark it FAILED via the module's
                  own public queue.set_ingest_status (state CONSTRUCTION with the
                  public API, never a patch of the logic under test), retry_failed()
                  -> must land in pending_enrich because markdown is present, and
                  the following drain must NOT invoke the converter.  (H-f2)

The stage test is the in-process cross-check of the same checkpoint guarantee
claim (c) tests against a process kill.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_f_fail
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any

from . import _common as C
from .corpus import BAD_SOURCES
from .drain import instrumented_drain
from .events import EventLog
from .instrument import counting_convert_fn, http_recorder, read_jsonl, truncate_ledger
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_f/1"

BAD_DIR = C.EXP_DIR / "data" / "bad_sources"

#: A genuinely non-existent arXiv id. One request only, with a polite sleep.
MISSING_URL = "https://arxiv.org/pdf/0000.00000"


def build_bad_sources() -> dict[str, str]:
    """Materialise the four broken sources on disk. Returns key -> source string."""
    BAD_DIR.mkdir(parents=True, exist_ok=True)

    missing = BAD_DIR / "does_not_exist.pdf"
    if missing.exists():
        missing.unlink()

    garbage = BAD_DIR / "garbage.pdf"
    garbage.write_bytes(bytes(range(256)) * 2 + b"not a pdf at all, 400 bytes of noise\n" * 3)

    whitespace = BAD_DIR / "whitespace_only.md"
    whitespace.write_text("   \n\n\t  \n   \n")

    return {
        "missing_path": str(missing),
        "garbage_pdf": str(garbage),
        "missing_url": MISSING_URL,
        "whitespace_markdown": str(whitespace),
    }


def error_names_type(error: str | None) -> bool:
    """An error column reading 'Exception: ' is a silent failure wearing a label."""
    if not error or not error.strip():
        return False
    head = error.split(":", 1)[0].strip()
    return bool(head) and head[0].isupper() and head not in ("Exception",)


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.document_store import retry_failed
    from andamentum.document_store.pipeline import harvest_convert_fn, ingest_source
    from andamentum.document_store.queue import FAILED, set_ingest_status

    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    fail_db = C.require_db_name(cfg["databases"]["fail"])
    db_path = C.db_file(fail_db)
    root_url = cfg["models"]["ollama_root_url"]
    log = EventLog(C.EVENTS / "claim_f.jsonl", rule="claim_f_fail", run_id=C.run_id())

    C.drop_database(fail_db)
    convert_ledger = truncate_ledger(C.LEDGERS / "f_convert.jsonl")
    convert_fn = counting_convert_fn(convert_ledger, harvest_convert_fn())

    registry = {p["arxiv_id"]: p for p in C.read_json(C.REGISTRY_PATH)["papers"]}
    good_pdf = C.EXP_DIR / registry[cfg["small_paper"]]["path"]
    bad = build_bad_sources()

    good_doc_id = await ingest_source(
        fail_db, str(good_pdf), metadata={"arm": "good"}, process="defer"
    )
    queued_bad: dict[str, str] = {}
    for key, source in bad.items():
        queued_bad[key] = await ingest_source(
            fail_db, source, metadata={"arm": "bad", "mode": key}, process="defer"
        )
        if source.startswith("http"):
            time.sleep(3.0)  # polite: one real request to arxiv.org, spaced out

    payload: dict[str, Any] = {
        "database": fail_db,
        "good_doc_id": good_doc_id,
        "good_source": str(good_pdf),
        "bad_sources": [
            {
                "key": b.key,
                "description": b.description,
                "why": b.why,
                "source": bad[b.key],
                "doc_id": queued_bad[b.key],
            }
            for b in BAD_SOURCES
        ],
        "ollama_ps_before": ollama_ps(root_url),
    }

    stages: dict[str, Any] = {}
    with http_recorder(C.LEDGERS / "f_http.jsonl") as ledger:
        common = dict(
            database=fail_db, db_path=db_path, model=C.LLM_MODEL,
            embedding_model=C.EMBEDDING_MODEL, convert_fn=convert_fn,
            convert_ledger=convert_ledger, event_log=log, http_ledger=ledger,
            snapshot_dir=C.SNAPSHOTS,
        )

        # --- H-f1: the first drain ---------------------------------------
        drain1 = await instrumented_drain(label="f_drain1", **common)
        stages["drain1"] = drain1
        counts = drain1["snapshot_post"]["status_counts"]
        payload["failed_count"] = counts["failed"]
        rec.check(
            "H-f1/complete", counts["complete"] == 1, measured=counts["complete"], expected=1,
            detail="the good PDF must succeed even though four siblings failed",
        )
        rec.check(
            "H-f1/failed", counts["failed"] == 4, measured=counts["failed"], expected=4,
        )
        rec.check(
            "H-f1/report_failures",
            len(drain1["report"]["failures"]) == 4,
            measured=len(drain1["report"]["failures"]), expected=4,
        )
        rec.check(
            "H-f1/not_aborted",
            drain1["report"]["stopped_early"] is False,
            measured=drain1["report"]["stopped_early"], expected=False,
            detail="a failing document must not abort the drain",
        )

        failed_rows = [
            r for r in drain1["snapshot_post"]["documents"] if r["ingest_status"] == "failed"
        ]
        untyped = [r["doc_uuid"] for r in failed_rows if not error_names_type(r["ingest_error"])]
        payload["errors_without_type_name"] = untyped
        payload["failure_fidelity"] = [
            {
                "doc_uuid": r["doc_uuid"],
                "dc_title": r["dc_title"],
                "ingest_error": r["ingest_error"],
                "exception_type": (r["ingest_error"] or "").split(":", 1)[0].strip() or None,
                "markdown_chars": r["markdown_chars"],
                "mode": next(
                    (k for k, v in queued_bad.items() if v == r["doc_uuid"]), None
                ),
            }
            for r in failed_rows
        ]
        payload["report_failures"] = drain1["report"]["failures"]
        rec.check(
            "H-f1/error_fidelity",
            not untyped,
            measured=untyped,
            expected=[],
            detail=(
                "claim (f) is about diagnosability: an error column reading "
                "'Exception: ' is a silent failure wearing a label"
            ),
        )
        rec.observe(
            "real_exception_classes",
            sorted({f["exception_type"] for f in payload["failure_fidelity"] if f["exception_type"]}),
            detail="what a downstream UI can actually pattern-match on",
        )

        # --- H-f2 part 1: retry routes by markdown presence ---------------
        from .fingerprint import snapshot as _snapshot

        n_retried = await retry_failed(fail_db)
        after_retry = _snapshot(db_path, label="f_retry1")
        stages["after_retry1"] = after_retry
        payload["retry_returned"] = n_retried
        wrong_stage = [
            r["doc_uuid"]
            for r in after_retry["documents"]
            if r["doc_uuid"] in queued_bad.values()
            and r["ingest_status"] != "pending_source"
        ]
        payload["retry_wrong_stage"] = wrong_stage
        rec.check(
            "H-f2/retry_count", n_retried == 4, measured=n_retried, expected=4,
        )
        rec.check(
            "H-f2/retry_stage",
            not wrong_stage,
            measured=wrong_stage,
            expected=[],
            detail="markdown is empty for all four, so retry_failed must route to pending_source",
        )

        # --- the systematic problem must SURFACE, not loop ----------------
        time.sleep(3.0)  # polite before the second real 404
        drain2 = await instrumented_drain(label="f_drain2", **common)
        stages["drain2"] = drain2
        counts2 = drain2["snapshot_post"]["status_counts"]
        payload["second_drain_failed"] = counts2["failed"]
        rec.check(
            "H-f/surfaces_again",
            counts2["failed"] == 4,
            measured=counts2["failed"],
            expected=4,
            detail=(
                "the drain never silently re-attempts failures; a re-drain after an "
                "explicit retry re-attempts and the systematic problem surfaces again"
            ),
        )

        # --- fix one, retry, drain ---------------------------------------
        shutil.copyfile(good_pdf, C.EXP_DIR / "data" / "bad_sources" / "does_not_exist.pdf")
        n_retried_2 = await retry_failed(fail_db)
        time.sleep(3.0)
        drain3 = await instrumented_drain(label="f_drain3_fixed", **common)
        stages["drain3_fixed"] = drain3
        counts3 = drain3["snapshot_post"]["status_counts"]
        payload["after_fix"] = {
            "retry_returned": n_retried_2,
            "complete": counts3["complete"],
            "failed": counts3["failed"],
        }
        rec.check(
            "H-f/fix_one_completes",
            counts3["complete"] == 2,
            measured=counts3["complete"],
            expected=2,
            detail="the good PDF plus the newly-fixed path",
        )

        # --- H-f2 part 2: the stage test ---------------------------------
        # State CONSTRUCTION through the module's own public API — never a patch
        # of the logic under test.
        ledger_before_stage = len(read_jsonl(convert_ledger))
        await set_ingest_status(
            str(db_path), good_doc_id, FAILED, error="ExperimentInduced: stage-routing test"
        )
        n_retried_3 = await retry_failed(fail_db)
        staged = _snapshot(db_path, label="f_stage")
        stages["after_stage_retry"] = staged
        good_row = next(r for r in staged["documents"] if r["doc_uuid"] == good_doc_id)
        payload["stage_test"] = {
            "retry_returned": n_retried_3,
            "status_after_retry": good_row["ingest_status"],
            "markdown_chars": good_row["markdown_chars"],
        }
        rec.check(
            "H-f2/stage_routing",
            good_row["ingest_status"] == "pending_enrich",
            measured=good_row["ingest_status"],
            expected="pending_enrich",
            detail=(
                "retry_failed routes purely by markdown emptiness, not by a remembered "
                "stage — markdown is present, so the conversion must not be redone"
            ),
        )

        drain4 = await instrumented_drain(label="f_drain4_stage", **common)
        stages["drain4_stage"] = drain4

        # ATTRIBUTE THE CONVERSIONS, DON'T JUST COUNT THEM.
        #
        # ``retry_failed`` requeues EVERY failed row, not only the one this stage
        # test marked. Three sources are still genuinely broken at this point
        # (garbage.pdf, the 404, whitespace_only.md); they hold no markdown, so
        # routing them back to pending_source and re-invoking the converter on
        # them is CORRECT — it is the same markdown-presence rule the hypothesis
        # relies on, applied to documents that legitimately have none.
        #
        # H-f2 is about the document that DOES have markdown: the good PDF, marked
        # failed by hand. So the check counts conversions of THAT source. Counting
        # every conversion in the drain measured the broken sources' correct
        # behaviour and reported it as a checkpoint violation (measured 3,
        # expected 0) — a false alarm about a real guarantee.
        new_entries = read_jsonl(convert_ledger)[ledger_before_stage:]
        good_source = str(good_pdf)
        good_reconversions = [e for e in new_entries if e.get("source") == good_source]
        payload["stage_test_converter_calls"] = len(new_entries)
        payload["stage_test_converter_calls_good_doc"] = len(good_reconversions)
        payload["stage_test_converter_sources"] = [e.get("source") for e in new_entries]
        rec.check(
            "H-f2/no_reconversion",
            not good_reconversions,
            measured=len(good_reconversions),
            expected=0,
            detail=(
                "the markdown-bearing document must NOT be re-converted — the "
                "in-process cross-check of the same checkpoint guarantee claim (c) "
                "tests against a process kill. Conversions of the still-broken "
                "sources are counted separately and are expected"
            ),
        )
        rec.observe(
            "stage_test_broken_source_reconversions",
            len(new_entries) - len(good_reconversions),
            detail=(
                "still-broken sources re-converted by the same retry — expected and "
                "correct: they carry no markdown, so pending_source is the right stage"
            ),
        )
        payload["http"] = ledger.summary()

    mismatches = sum(
        stage["reconciliation"]["n_mismatches"]
        for stage in stages.values()
        if isinstance(stage, dict) and "reconciliation" in stage
    )
    payload["report_db_mismatches"] = mismatches
    rec.check("H-x/claim_f", mismatches == 0, measured=mismatches, expected=0)

    payload["stages"] = stages
    payload["ollama_ps_after"] = ollama_ps(root_url)
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = asyncio.run(run())
    finally:
        C.write_json(
            C.RESULTS / "claim_f_fail.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
