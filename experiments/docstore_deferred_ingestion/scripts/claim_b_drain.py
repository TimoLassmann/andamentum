"""Claim (b): the drain is resumable, and a no-op drain is genuinely a no-op.

Three drains against dfr_main:

  drain #1  max_docs=2      — a partial run
  drain #2  unrestricted    — finishes the rest
  drain #3  unrestricted    — must be a GENUINE no-op:
                                documents_enriched == 0
                                logical fingerprint UNCHANGED
                                <= 2 Ollama requests

The fingerprint clause is what distinguishes "the report says it did nothing"
from "it did nothing". The <= 2 (rather than 0) is not a fudge: ``_preflight_done``
is a module-level per-PROCESS cache, and this rule runs all three drains in ONE
process, so drain #3 pays no preflight at all — but a fresh-process drain would
pay exactly 1 embed + 1 chat. The bound is stated for the general case and the
actual per-drain count is recorded.

dfr_main also holds the arm-2 ``pending_source`` document from claim (a), so the
queue is 5 documents (4 markdown + 1 source) — hence a converter IS supplied and
the conversion is counted. That accounting is explicit in claim_a_defer.json.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_b_drain
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from . import _common as C
from .drain import instrumented_drain
from .events import EventLog
from .instrument import counting_convert_fn, http_recorder, truncate_ledger
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_b/1"


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.document_store import pending_status
    from andamentum.document_store.pipeline import harvest_convert_fn

    rec = C.ClaimRecorder()
    main_db = C.require_db_name(C.CONFIG["databases"]["main"])
    db_path = C.db_file(main_db)
    root_url = C.CONFIG["models"]["ollama_root_url"]

    log = EventLog(C.EVENTS / "claim_b.jsonl", rule="claim_b_drain", run_id=C.run_id())
    convert_ledger = truncate_ledger(C.LEDGERS / "b_convert.jsonl")
    convert_fn = counting_convert_fn(convert_ledger, harvest_convert_fn())

    payload: dict[str, Any] = {
        "database": main_db,
        "ollama_ps_before": ollama_ps(root_url),
        "queue_before": asdict(await pending_status(main_db)),
    }

    drains: list[dict[str, Any]] = []
    with http_recorder(C.LEDGERS / "b_http.jsonl") as ledger:
        for label, kwargs in (
            ("drain1_max_docs_2", {"max_docs": 2}),
            ("drain2_unrestricted", {}),
            ("drain3_noop", {}),
        ):
            print(f"--- {label} ---", flush=True)
            result = await instrumented_drain(
                database=main_db,
                db_path=db_path,
                label=label,
                model=C.LLM_MODEL,
                embedding_model=C.EMBEDDING_MODEL,
                convert_fn=convert_fn,
                convert_ledger=convert_ledger,
                event_log=log,
                http_ledger=ledger,
                snapshot_dir=C.SNAPSHOTS,
                **kwargs,
            )
            drains.append(result)
            print(
                f"  report: {result['report']}  ollama_requests="
                f"{result['http_delta_ollama_requests']}",
                flush=True,
            )
        payload["http"] = ledger.summary()

    payload["drains"] = drains
    payload["ollama_ps_after"] = ollama_ps(root_url)
    final = await pending_status(main_db)
    payload["queue_after"] = asdict(final) | {"pending": final.pending}

    d1, d2, d3 = drains

    # --- resumability -----------------------------------------------------
    rec.check(
        "H-b/partial",
        d1["report"]["stopped_early"] is True,
        measured=d1["report"]["stopped_early"],
        expected=True,
        detail="max_docs=2 must report a deliberate early stop",
    )
    processed_1 = (
        d1["report"]["documents_enriched"]
        + d1["report"]["documents_failed"]
    )
    rec.check(
        "H-b/max_docs_honoured",
        processed_1 <= 2,
        measured=processed_1,
        expected="<= 2",
        detail="max_docs is checked at the TOP of each iteration",
    )

    total_enriched = sum(d["report"]["documents_enriched"] for d in drains)
    expected_documents = (
        C.CONFIG["lineages"]["main_documents"] + 1  # + the arm-2 source probe
    )
    rec.check(
        "H-b/all_completed",
        total_enriched == expected_documents,
        measured=total_enriched,
        expected=expected_documents,
        detail="4 markdown documents + the 1 queued source from claim (a) arm 2",
    )
    rec.check(
        "H-b/queue_empty",
        final.pending == 0,
        measured=final.pending,
        expected=0,
    )
    rec.check(
        "H-b/no_failures",
        final.failed == 0,
        measured=final.failed,
        expected=0,
        detail="; ".join(sum((d["report"]["failures"] for d in drains), [])),
    )

    # --- the no-op drain --------------------------------------------------
    payload["drain3_enriched"] = d3["report"]["documents_enriched"]
    payload["drain3_ollama_requests"] = d3["http_delta_ollama_requests"]
    payload["drain3_fingerprint_moved"] = d3["fingerprint_moved"]

    rec.check(
        "H-b/noop_enriched",
        d3["report"]["documents_enriched"] == 0,
        measured=d3["report"]["documents_enriched"],
        expected=0,
    )
    rec.check(
        "H-b/noop_fingerprint",
        not d3["fingerprint_moved"],
        measured=d3["fingerprint_post"],
        expected=d3["fingerprint_pre"],
        detail="a no-op drain must not move the logical fingerprint by one bit",
    )
    rec.check(
        "H-b/noop_requests",
        d3["http_delta_ollama_requests"] <= 2,
        measured=d3["http_delta_ollama_requests"],
        expected="<= 2 (the per-process preflight: 1 embed + 1 chat)",
        detail=(
            "this process already ran preflight in drain #1, so 0 is expected here; "
            "the bound covers the fresh-process case"
        ),
    )

    # --- H-x reconciliation ----------------------------------------------
    mismatches = sum(d["reconciliation"]["n_mismatches"] for d in drains)
    payload["report_db_mismatches"] = mismatches
    rec.check(
        "H-x/claim_b",
        mismatches == 0,
        measured=mismatches,
        expected=0,
        detail="ProcessReport must agree field-by-field with the database and the ledger",
    )
    rec.observe(
        "converter_calls_total",
        sum(d["converter_calls"] for d in drains),
        detail="exactly one conversion is expected: the arm-2 source probe",
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = asyncio.run(run())
    finally:
        C.write_json(
            C.RESULTS / "claim_b_drain.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
