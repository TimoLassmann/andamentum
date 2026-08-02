"""Claim (e): the drain CREATED semantic reachability, and enrichment is structurally complete.

Same 8 paraphrase probes, same database, same three signals — now after the drain.
H-e1 is the pre/post contrast; H-e2 is the structural check on what enrichment
actually wrote.

WHAT IS DELIBERATELY *NOT* ASSERTED: doc-level embeddings.
``_run_phase2._embed_doc_level`` catches an oversized-input failure with an INFO
log, and a ~50k-char paper very likely exceeds embeddinggemma's input budget. So
claim (e) is stated over CHUNK embeddings, and the doc-embedding skip rate is
reported as a measured fact about the 4-signal RRF stack — one of whose signals
therefore goes dark for long documents. Asserting the opposite would fail claim
(e) for a reason unrelated to the queue.

``public.search()`` is called ONCE and RECORDED but never scored: it runs an LLM
query planner whose output varies run to run, and a non-reproducible metric is
not a metric.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.claim_e_post
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from . import _common as C
from .fingerprint import snapshot
from .instrument import http_recorder
from .retrieval import acceptable_doc_ids, run_probes

SCHEMA = "andamentum.experiment.docstore_deferred.claim_e_post/1"


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.document_store import search as public_search

    rec = C.ClaimRecorder()
    main_db = C.require_db_name(C.CONFIG["databases"]["main"])
    db_path = C.db_file(main_db)

    claim_a = C.read_json(C.RESULTS / "claim_a_defer.json")
    claim_e_pre = C.read_json(C.RESULTS / "claim_e_pre.json")
    doc_id_by_short = acceptable_doc_ids(claim_a)

    with http_recorder(C.LEDGERS / "e_post_http.jsonl") as ledger:
        probe_results = await run_probes(
            db_path, doc_id_by_short, embedding_model=C.EMBEDDING_MODEL
        )

        # ONE smoke call through the natural-language surface. Recorded, never scored.
        smoke_query = "how did researchers make very deep networks trainable"
        try:
            smoke_hits = await public_search(
                main_db,
                smoke_query,
                limit=5,
                model=C.LLM_MODEL,
                embedding_model=C.EMBEDDING_MODEL,
            )
            smoke = {
                "query": smoke_query,
                "n_results": len(smoke_hits),
                "results": [asdict(h) for h in smoke_hits],
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 — recorded, not swallowed silently
            smoke = {
                "query": smoke_query,
                "n_results": 0,
                "results": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    post = snapshot(db_path, label="claim_e_post")
    payload: dict[str, Any] = {
        "database": main_db,
        "phase": "post_drain",
        "doc_id_by_short": doc_id_by_short,
        "probes": probe_results,
        "public_search_smoke": smoke,
        "public_search_smoke_note": (
            "NOT SCORED — public.search() runs an LLM query planner whose output varies "
            "run to run; scoring it would make claim (e) non-reproducible"
        ),
        "http": ledger.summary(),
        "snapshot": post,
    }

    # --- H-e1: the pre/post contrast --------------------------------------
    pre_aggregate = claim_e_pre["probes"]["aggregate"]
    post_aggregate = probe_results["aggregate"]
    pre_recall = pre_aggregate["chunk_embeddings"]["recall_at_3"]
    post_recall = post_aggregate["chunk_embeddings"]["recall_at_3"]
    payload["pre_chunk_recall_at_3"] = pre_recall
    payload["post_chunk_recall_at_3"] = post_recall
    # recall@1 and MRR are the informative headline: with a 5-document candidate
    # pool, chance recall@3 is 3/5, so recall@3 == 1.0 is a weak statement while
    # recall@1 == 1.0 against a chance level of 1/5 is a real one. Both are scored.
    payload["post_chunk_recall_at_1"] = post_aggregate["chunk_embeddings"]["recall_at_1"]
    payload["post_chunk_mrr"] = post_aggregate["chunk_embeddings"]["mrr"]
    # THE INFORMATIVE CONTROL: FTS5 alone already answers some probes, so the full
    # RRF stack scores above zero BEFORE the drain. Its rise is evidence the drain
    # created a capability; the chunk-embedding 0 -> 1 transition is merely evidence
    # that an empty table became a populated one.
    payload["pre_unified_recall_at_3"] = pre_aggregate["unified_rrf"]["recall_at_3"]
    payload["post_unified_recall_at_3"] = post_aggregate["unified_rrf"]["recall_at_3"]
    payload["retrieval_power"] = probe_results["power"]
    payload["recall_delta"] = {
        signal: {
            "pre": pre_aggregate[signal],
            "post": post_aggregate[signal],
        }
        for signal in ("fts5", "chunk_embeddings", "unified_rrf")
    }

    rec.check(
        "H-e1/post_recall_at_1",
        payload["post_chunk_recall_at_1"] >= 1.0,
        measured=payload["post_chunk_recall_at_1"],
        expected=">= 1.0",
        detail=(
            "chunk-embedding recall@1 over 8 paraphrase probes; chance level is "
            f"{probe_results['power']['chance_recall_at_1']}"
        ),
    )
    rec.check(
        "H-e1/post",
        post_recall >= 1.0,
        measured=post_recall,
        expected=">= 1.0",
        detail="chunk-embedding recall@3 over 8 paraphrase probes after the drain",
    )
    rec.check(
        "H-e1/unified_rose",
        payload["post_unified_recall_at_3"] > payload["pre_unified_recall_at_3"],
        measured={
            "pre": payload["pre_unified_recall_at_3"],
            "post": payload["post_unified_recall_at_3"],
        },
        expected="post > pre on the full 4-signal RRF stack",
        detail=(
            "the non-trivial contrast: the keyword index already existed, so this "
            "measures a capability the drain created rather than a table it filled"
        ),
    )
    rec.observe(
        "pre_chunk_recall_at_3",
        pre_recall,
        detail=(
            "0 by definition — H-a2 asserts every deferred document has zero chunks, "
            "so semantic_search has no row it could return. Kept as a sanity note"
        ),
    )
    rec.observe(
        "retrieval_power",
        probe_results["power"],
        detail=(
            "STATED LIMITATION: 4 topically disjoint papers, no hard negatives, a "
            "5-document candidate pool. This rules out an empty or scrambled index; "
            "it does not measure retrieval quality"
        ),
    )

    # --- H-e2: structural completeness of enrichment ----------------------
    completed = [r for r in post["documents"] if r["ingest_status"] == "complete"]
    violations: list[str] = []
    doc_embedding_present = 0
    for row in completed:
        if row["n_chunks"] == 0:
            violations.append(f"{row['doc_uuid']}: 0 chunks")
        if row["n_chunk_embeddings"] != row["n_chunks"]:
            violations.append(
                f"{row['doc_uuid']}: {row['n_chunk_embeddings']} chunk embeddings "
                f"for {row['n_chunks']} chunks"
            )
        if not row["llm_metadata_populated"]:
            violations.append(
                f"{row['doc_uuid']}: no LLM metadata — the signature of the "
                "swallowed-provider-error failure mode (see H-G)"
            )
        if row["has_doc_embedding"]:
            doc_embedding_present += 1

    payload["n_completed"] = len(completed)
    payload["enrichment_structure_violations"] = violations
    payload["doc_embedding_present"] = doc_embedding_present
    payload["doc_embedding_skip_rate"] = (
        1.0 - doc_embedding_present / len(completed) if completed else None
    )
    payload["per_document"] = [
        {
            "doc_uuid": r["doc_uuid"],
            "dc_title": r["dc_title"],
            "markdown_chars": r["markdown_chars"],
            "n_chunks": r["n_chunks"],
            "n_chunk_embeddings": r["n_chunk_embeddings"],
            "has_doc_embedding": r["has_doc_embedding"],
            "metadata_keys": r["metadata_keys"],
            "llm_metadata_populated": r["llm_metadata_populated"],
        }
        for r in completed
    ]

    rec.check(
        "H-e2",
        not violations,
        measured=len(violations),
        expected=0,
        detail="; ".join(violations[:5]),
    )
    rec.observe(
        "doc_embedding_skip_rate",
        payload["doc_embedding_skip_rate"],
        detail=(
            "measured, not asserted: _embed_doc_level swallows an oversized-input "
            "failure with an INFO log, so one of the four RRF signals can go dark for "
            "long documents without anything reporting an error"
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
            C.RESULTS / "claim_e_post.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
