"""Claim (a): defer is fast, LLM-free, and immediately findable — but only for ingest().

Three arms, in one process, against a clean slate:

  Arm 1  ingest(process="defer", model=None, embedding_model=None) x4 into dfr_main,
         timed under the httpx recorder.
           H-a1  the ledger must contain ZERO requests to the Ollama host
           H-a2  persisted state: 0 chunks, 0 chunk embeddings, no doc embedding,
                 status pending_enrich, metadata keys exactly {source, title}
           H-a3  median per-document seconds, and the ratio against arm 3
           H-a4  FTS5-retrievable by a rare token from its own body, and the stored
                 title equals the first non-empty line stripped of heading marks

  Arm 2  pipeline.ingest_source(pdf, process="defer") x1 into dfr_main, then an
         immediate FTS probe that must return 0 hits.
           H-a5  register_pending_source writes markdown_content='', so the
                 docstring's "keyword-searchable the moment this returns" holds for
                 ingest() and is OVER-GENERAL for ingest_source().

         ACCOUNTING DECISION (stated rather than hidden): the arm-2 document is KEPT
         in dfr_main and accounted for explicitly. Every downstream count in lineage
         MAIN is therefore stated over the 4 markdown documents, and this source row
         is tracked separately as ``source_probe_doc_id``. It is NOT deleted, because
         deleting it would exercise soft-delete machinery that has nothing to do with
         the queue and would perturb the very fingerprint claim (b) rests on.
         Consequence: dfr_main's queue holds 5 pending documents, of which 1 is
         pending_source; lineage MAIN's drain converts it too.

  Arm 3  ingest(process="now") on ONE of the same documents into dfr_now, with the
         REAL models — the control proving the deferred path SKIPPED real work rather
         than that the work happened to be cheap.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_a_defer
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time
from typing import Any

from . import _common as C
from .events import EventLog
from .fingerprint import snapshot
from .instrument import http_recorder
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.claim_a/1"

#: Metadata keys ``ingest(process="defer")`` is allowed to write. Anything else is
#: evidence that an LLM ran.
ALLOWED_DEFER_KEYS = {"source", "title"}

#: A caller-supplied key, so H-a2's "union the caller's own keys" clause is
#: actually exercised rather than assumed.
CALLER_METADATA = {"experiment_arm": "defer"}


def rare_token(markdown: str) -> str | None:
    """Pick a long, alphabetic, low-frequency token from a document's own body.

    Used for the H-a4 FTS probe. Long-and-rare keeps the probe specific without
    hand-picking terms per paper (which would make the probe a curated fixture
    rather than a property of the document).
    """
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z]{9,}", markdown):
        low = token.lower()
        counts[low] = counts.get(low, 0) + 1
    singletons = sorted(t for t, n in counts.items() if n == 1)
    if singletons:
        # Deterministic pick: longest, then alphabetical.
        return sorted(singletons, key=lambda t: (-len(t), t))[0]
    return sorted(counts, key=lambda t: (counts[t], -len(t)))[0] if counts else None


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.document_store import ingest
    from andamentum.document_store.pipeline import ingest_source
    from andamentum.document_store.public import _fallback_title
    from andamentum.document_store.search import search_fts5

    rec = C.ClaimRecorder()
    cfg = C.CONFIG
    main_db = C.require_db_name(cfg["databases"]["main"])
    now_db = C.require_db_name(cfg["databases"]["now"])
    root_url = cfg["models"]["ollama_root_url"]
    log = EventLog(C.EVENTS / "claim_a.jsonl", rule="claim_a_defer", run_id=C.run_id())

    # Clean slate. dfr_main is the substrate for claims (b) and (e) too.
    C.drop_database(main_db)
    C.drop_database(now_db)

    baseline = C.read_json(C.RESULTS / "conversion_baseline.json")
    conversions = {c["arxiv_id"]: c for c in baseline["conversions"]}
    registry = {p["arxiv_id"]: p for p in C.read_json(C.REGISTRY_PATH)["papers"]}

    payload: dict[str, Any] = {
        "database_main": main_db,
        "database_now": now_db,
        "ollama_ps_before": ollama_ps(root_url),
        "accounting_decision": (
            "the arm-2 pending_source document is KEPT in dfr_main and accounted for "
            "explicitly (source_probe_doc_id); lineage MAIN therefore drains 5 documents, "
            "4 markdown + 1 source"
        ),
    }

    # ------------------------------------------------------------------
    # Arm 1 — four deferred markdown ingests, under the transport recorder
    # ------------------------------------------------------------------
    defer_records: list[dict[str, Any]] = []
    with http_recorder(C.LEDGERS / "a_defer_http.jsonl") as defer_ledger:
        for arxiv_id, conv in conversions.items():
            markdown = (C.EXP_DIR / conv["markdown_path"]).read_text()
            started = time.monotonic()
            doc_id = await ingest(
                main_db,
                markdown,
                source=f"arxiv:{arxiv_id}",
                metadata=dict(CALLER_METADATA),
                model=None,
                embedding_model=None,
                process="defer",
            )
            seconds = time.monotonic() - started
            defer_records.append(
                {
                    "arxiv_id": arxiv_id,
                    "short": conv["short"],
                    "doc_id": doc_id,
                    "seconds": seconds,
                    "markdown_chars": len(markdown),
                    "expected_title": _fallback_title(markdown),
                    "probe_token": rare_token(markdown),
                }
            )
            log.emit(
                db_name=main_db,
                observer="report",
                doc_id=doc_id,
                arxiv_id=arxiv_id,
                from_status=None,
                to_status="pending_enrich",
                stage_seconds=seconds,
                note="ingest(process='defer')",
            )
            print(f"defer {arxiv_id}: {seconds:.3f}s -> {doc_id}", flush=True)

    payload["defer_http"] = defer_ledger.summary()
    ollama_requests = len(
        [r for r in defer_ledger.records if "11434" in (r.get("host") or "")]
    )
    payload["defer_ollama_requests"] = ollama_requests
    rec.check(
        "H-a1",
        ollama_requests == 0,
        measured=ollama_requests,
        expected=0,
        detail=(
            "the defer branch of ingest() returns before _preflight is ever called, so "
            "any request at all is a defect"
        ),
    )

    # ------------------------------------------------------------------
    # H-a2 — persisted state (host-independent, re-checkable offline)
    # ------------------------------------------------------------------
    db_path = C.db_file(main_db)
    post_defer = snapshot(db_path, label="claim_a_after_defer")
    rows_by_id = {r["doc_uuid"]: r for r in post_defer["documents"]}
    state_violations: list[str] = []
    for record in defer_records:
        row = rows_by_id.get(record["doc_id"])
        if row is None:
            state_violations.append(f"{record['doc_id']}: row absent after ingest")
            continue
        if row["ingest_status"] != "pending_enrich":
            state_violations.append(
                f"{record['doc_id']}: status {row['ingest_status']} != pending_enrich"
            )
        if row["n_chunks"] != 0:
            state_violations.append(f"{record['doc_id']}: {row['n_chunks']} chunks written")
        if row["n_chunk_embeddings"] != 0:
            state_violations.append(
                f"{record['doc_id']}: {row['n_chunk_embeddings']} chunk embeddings written"
            )
        if row["has_doc_embedding"]:
            state_violations.append(f"{record['doc_id']}: doc embedding written")
        expected_keys = ALLOWED_DEFER_KEYS | set(CALLER_METADATA)
        actual_keys = set(row["metadata_keys"])
        if actual_keys != expected_keys:
            state_violations.append(
                f"{record['doc_id']}: metadata keys {sorted(actual_keys)} != "
                f"{sorted(expected_keys)}"
            )
        record["metadata_keys"] = row["metadata_keys"]
        record["stored_title"] = row["dc_title"]

    payload["defer_state_violations"] = state_violations
    rec.check(
        "H-a2",
        not state_violations,
        measured=len(state_violations),
        expected=0,
        detail="; ".join(state_violations[:5]),
    )

    # ------------------------------------------------------------------
    # H-a4 — findable immediately, and titled deterministically
    # ------------------------------------------------------------------
    fts_misses: list[str] = []
    for record in defer_records:
        token = record["probe_token"]
        hits = await search_fts5(str(db_path), token, 20) if token else []
        hit_ids = {doc_id for doc_id, _ in hits}
        record["fts_hit"] = record["doc_id"] in hit_ids
        record["fts_n_hits"] = len(hits)
        if not record["fts_hit"]:
            fts_misses.append(f"{record['arxiv_id']}: token {token!r} did not retrieve it")
        if record.get("stored_title") != record["expected_title"]:
            fts_misses.append(
                f"{record['arxiv_id']}: title {record.get('stored_title')!r} != "
                f"{record['expected_title']!r}"
            )
    payload["defer_fts_misses"] = fts_misses
    rec.check(
        "H-a4",
        not fts_misses,
        measured=len(fts_misses),
        expected=0,
        detail="; ".join(fts_misses[:5]),
    )

    # ------------------------------------------------------------------
    # Arm 2 — a queued SOURCE is NOT searchable (H-a5)
    # ------------------------------------------------------------------
    small_id = cfg["small_paper"]
    small_pdf = C.EXP_DIR / registry[small_id]["path"]
    source_doc_id = await ingest_source(
        main_db, str(small_pdf), title=None, metadata={"experiment_arm": "source_probe"},
        process="defer",
    )
    log.emit(
        db_name=main_db,
        observer="report",
        doc_id=source_doc_id,
        arxiv_id=small_id,
        to_status="pending_source",
        note="pipeline.ingest_source(process='defer')",
    )
    # A term unambiguously present in the PDF body but absent from its file name.
    source_probe_token = "stochastic"
    source_hits = await search_fts5(str(db_path), source_probe_token, 20)
    source_hit_ids = {doc_id for doc_id, _ in source_hits}
    source_self_hits = 1 if source_doc_id in source_hit_ids else 0
    payload["source_probe"] = {
        "doc_id": source_doc_id,
        "arxiv_id": small_id,
        "probe_token": source_probe_token,
        "self_hits": source_self_hits,
        "total_hits_in_db": len(source_hits),
        "note": (
            "other documents may legitimately match the token; only hits on the queued "
            "source row falsify H-a5"
        ),
    }
    payload["source_probe_doc_id"] = source_doc_id
    rec.check(
        "H-a5",
        source_self_hits == 0,
        measured=source_self_hits,
        expected=0,
        detail=(
            "register_pending_source writes markdown_content='' — the 'searchable "
            "immediately' property does not transfer to the source path"
        ),
    )

    # ------------------------------------------------------------------
    # Arm 3 — the process="now" control, with the REAL models
    # ------------------------------------------------------------------
    control = defer_records[0]
    control_markdown = (
        C.EXP_DIR / conversions[control["arxiv_id"]]["markdown_path"]
    ).read_text()
    with http_recorder(C.LEDGERS / "a_now_http.jsonl") as now_ledger:
        started = time.monotonic()
        now_doc_id = await ingest(
            now_db,
            control_markdown,
            source=f"arxiv:{control['arxiv_id']}",
            model=C.LLM_MODEL,
            embedding_model=C.EMBEDDING_MODEL,
            process="now",
        )
        now_seconds = time.monotonic() - started
    log.emit(
        db_name=now_db,
        observer="report",
        doc_id=now_doc_id,
        arxiv_id=control["arxiv_id"],
        to_status="complete",
        stage_seconds=now_seconds,
        note="ingest(process='now') control",
    )

    defer_seconds = [r["seconds"] for r in defer_records]
    median_defer = statistics.median(defer_seconds)
    # SAME DOCUMENT ON BOTH SIDES. The control ingested `control`'s markdown, so
    # the ratio must use `control`'s own defer time. Dividing the MEDIAN over four
    # documents of differing sizes by a single run on one of them is a mixed
    # quantity — the same-document number is available for free and is what H-a3
    # actually states. The median stays published as the separate absolute-latency
    # metric H-a3's first threshold scores.
    ratio = control["seconds"] / now_seconds if now_seconds > 0 else float("inf")

    payload.update(
        {
            "defer_records": defer_records,
            "defer_seconds": defer_seconds,
            "defer_median_seconds": median_defer,
            "now_control": {
                "doc_id": now_doc_id,
                "arxiv_id": control["arxiv_id"],
                "seconds": now_seconds,
                "http": now_ledger.summary(),
                "ollama_requests": len(
                    [r for r in now_ledger.records if "11434" in (r.get("host") or "")]
                ),
            },
            "defer_now_ratio": ratio,
            "defer_now_ratio_basis": {
                "arxiv_id": control["arxiv_id"],
                "doc_id": control["doc_id"],
                "defer_seconds": control["seconds"],
                "now_seconds": now_seconds,
                "why": "same document, same content, both sides — not a median over four",
            },
            "snapshot_after_defer": post_defer,
            "snapshot_after_all": snapshot(db_path, label="claim_a_post"),
            # The now-control lives in its OWN database (dfr_now), so without this
            # its doc_id is referenced by the artefact but grounded in no
            # fingerprint at all — validate_results flags it as an orphan, and
            # rightly: an id no snapshot can resolve is an unverifiable claim.
            # Snapshotting is read-only SQL and also evidences that the control
            # really did complete.
            "snapshot_now_control": snapshot(
                C.db_file(now_db), label="claim_a_now_control"
            ),
            "ollama_ps_after": ollama_ps(root_url),
        }
    )

    rec.check(
        "H-a3/median",
        median_defer < 1.0,
        measured=median_defer,
        expected="< 1.0 s",
        detail="absolute seconds are not portable; the ratio below is the transferable claim",
    )
    rec.check(
        "H-a3/ratio",
        ratio < 0.02,
        measured=ratio,
        expected="< 0.02",
        detail=(
            f"same document ({control['arxiv_id']}): t_defer {control['seconds']:.4f}s "
            f"vs t_now {now_seconds:.1f}s"
        ),
    )
    return payload, rec


def main() -> int:
    payload: dict[str, Any] = {}
    rec = C.ClaimRecorder()
    try:
        payload, rec = asyncio.run(run())
    finally:
        C.write_json(
            C.RESULTS / "claim_a_defer.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
