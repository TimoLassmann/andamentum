"""Claim (e), control arm: the SAME probes against the SAME database, BEFORE the drain.

Without this arm, post-drain recall@3 == 1.0 proves nothing about the drain — the
FTS index alone might already have answered every probe. Running the identical
probes first is what turns claim (e) into "the drain CREATED this capability".

A second job, and the reason this rule exists as its own step rather than being
folded into claim_e_post: it is a TRIPWIRE. ``search()``-adjacent code calls
``_preflight`` with ``auto_repair=True`` (unlike ``process_pending``, which
disables it). A regression there would silently drain the backlog and destroy
claim (b)'s setup before it ran. So pending counts are snapshotted immediately
before and after the probes and asserted unchanged.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_e_pre
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import _common as C
from .fingerprint import snapshot
from .instrument import http_recorder
from .retrieval import acceptable_doc_ids, run_probes

SCHEMA = "andamentum.experiment.docstore_deferred.claim_e_pre/1"


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    rec = C.ClaimRecorder()
    main_db = C.require_db_name(C.CONFIG["databases"]["main"])
    db_path = C.db_file(main_db)

    claim_a = C.read_json(C.RESULTS / "claim_a_defer.json")
    doc_id_by_short = acceptable_doc_ids(claim_a)

    before = snapshot(db_path, label="claim_e_pre_before")
    with http_recorder(C.LEDGERS / "e_pre_http.jsonl") as ledger:
        probe_results = await run_probes(
            db_path, doc_id_by_short, embedding_model=C.EMBEDDING_MODEL
        )
    after = snapshot(db_path, label="claim_e_pre_after")

    payload: dict[str, Any] = {
        "database": main_db,
        "phase": "pre_drain",
        "doc_id_by_short": doc_id_by_short,
        "probes": probe_results,
        "http": ledger.summary(),
        "snapshot_before": before,
        "snapshot_after": after,
    }

    chunk_recall_3 = probe_results["aggregate"]["chunk_embeddings"]["recall_at_3"]
    payload["pre_chunk_recall_at_3"] = chunk_recall_3
    payload["pre_chunk_recall_at_1"] = probe_results["aggregate"]["chunk_embeddings"][
        "recall_at_1"
    ]
    # THE INFORMATIVE CONTROL. `pre_chunk_recall_at_3 == 0` is definitional — H-a2
    # already asserts every deferred document has zero chunks, and semantic_search
    # cannot return a row that does not exist, so this measures that a table is
    # empty. The number that shows the drain CREATED a capability rather than
    # merely populating a table nothing reads is the full RRF stack: FTS5 alone
    # already answers some probes, so unified recall is non-zero before the drain
    # and must rise afterwards.
    payload["pre_unified_recall_at_3"] = probe_results["aggregate"]["unified_rrf"][
        "recall_at_3"
    ]
    payload["pre_fts5_recall_at_3"] = probe_results["aggregate"]["fts5"]["recall_at_3"]
    payload["retrieval_power"] = probe_results["power"]
    rec.observe(
        "pre_chunk_recall_at_3",
        chunk_recall_3,
        detail=(
            "SANITY NOTE, not evidence: the chunk table is empty (H-a2), so this is 0 "
            "by definition. The scored control is pre_unified_recall_at_3"
        ),
    )
    rec.check(
        "H-e1/pre_chunk_table_empty",
        chunk_recall_3 == 0.0,
        measured=chunk_recall_3,
        expected=0.0,
        detail="there are no chunks yet, so the chunk-embedding signal must be empty",
    )

    fingerprint_moved = before["logical_fingerprint"] != after["logical_fingerprint"]
    payload["fingerprint_moved"] = fingerprint_moved
    rec.check(
        "tripwire/auto_repair",
        not fingerprint_moved,
        measured=after["status_counts"],
        expected=before["status_counts"],
        detail=(
            "search()'s preflight runs with auto_repair=True; if it drained the "
            "deliberate backlog it would destroy claim (b)'s setup before it ran"
        ),
    )
    rec.observe(
        "pre_drain_signal_recall",
        probe_results["aggregate"],
        detail="FTS5 may legitimately score above zero here — the keyword index exists",
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = asyncio.run(run())
    finally:
        C.write_json(
            C.RESULTS / "claim_e_pre.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
