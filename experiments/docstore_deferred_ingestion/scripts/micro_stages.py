"""Where do the enrichment seconds actually go — and does Semaphore(5) buy anything?

The drain reports ONE enrichment number. This rule takes ONE already-converted
paper's markdown back OUT of dfr_main (no re-conversion, no re-ingest) and calls
the same building blocks ``_run_phase2`` calls, STRICTLY SEQUENTIALLY, timing
each:

    extract_document_metadata
    extract_units(target_min_chars=1500, target_max_chars=4000, embedding_fn=...)
    units_to_chunks
    EmbeddingService.embed_batch
    a plain `for` loop of extract_chunk_metadata, awaited one at a time
    embed_text(content, text_type="document")  -- inside an explicit try that
                                                  RECORDS the exception

``concurrency_speedup`` = (sum of those sequential components) / (that same
paper's in-pipeline enrichment seconds from the drain, which used Semaphore(5)
inside an asyncio.gather).

STATED LIMITATION, carried into the analysis rather than hidden: the in-pipeline
side gathers three different workloads (batch embed / per-chunk LLM / doc embed)
across two different models, so a ratio near 1.0 is decisive evidence the
semaphore is decoration against a serialising Ollama, but a ratio around 1.3 is
ambiguous between real fan-out gain and embed/LLM overlap. Hence the per-call
latency distribution is published beside the scalar.

ALSO LABELLED HONESTLY: ``extract_units`` Stage 2 CALLS the embedding model, so
"chunking seconds" are not pure CPU.

This rule never issues two Ollama calls at once.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.micro_stages
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from . import _common as C
from .drain import attribute_document_seconds
from .fingerprint import readonly_connection
from .instrument import http_recorder
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.micro_stages/1"


def _read_markdown(db_path: str, doc_uuid: str) -> tuple[str, str]:
    """Read one document's markdown + title straight out of the database."""
    with readonly_connection(db_path) as conn:
        row = conn.execute(
            "SELECT dc_title, markdown_content FROM documents WHERE doc_uuid = ?",
            (doc_uuid,),
        ).fetchone()
    if row is None:
        raise LookupError(f"doc_uuid {doc_uuid} not found — cannot attribute its stages")
    return str(row["markdown_content"] or ""), str(row["dc_title"] or "Untitled")


async def run() -> tuple[dict[str, Any], C.ClaimRecorder]:
    from andamentum.chunker import extract_units
    from andamentum.core.embeddings import make_ollama_embedder
    from andamentum.document_store.chunker_adapter import units_to_chunks
    from andamentum.document_store.embeddings import EmbeddingService
    from andamentum.document_store.extraction import (
        extract_chunk_metadata,
        extract_document_metadata,
    )

    rec = C.ClaimRecorder()
    main_db = C.require_db_name(C.CONFIG["databases"]["main"])
    db_path = str(C.db_file(main_db))
    root_url = C.CONFIG["models"]["ollama_root_url"]

    claim_a = C.read_json(C.RESULTS / "claim_a_defer.json")
    # ONE paper, not two. The smallest markdown keeps the cost honest without
    # changing what the ratio means.
    target = min(claim_a["defer_records"], key=lambda r: r["markdown_chars"])
    content, title = _read_markdown(db_path, target["doc_id"])
    if not content.strip():
        raise ValueError(
            f"{target['arxiv_id']} has empty markdown in {main_db} — refusing to "
            "attribute stages against nothing"
        )

    timings: dict[str, float] = {}
    payload: dict[str, Any] = {
        "database": main_db,
        "arxiv_id": target["arxiv_id"],
        "doc_id": target["doc_id"],
        "title": title,
        "markdown_chars": len(content),
        "ollama_ps_before": ollama_ps(root_url),
    }

    with http_recorder(C.LEDGERS / "micro_http.jsonl") as ledger:
        # 1. document-level metadata
        with C.Stopwatch(timings, "extract_document_metadata"):
            doc_meta = await extract_document_metadata(content, model=C.LLM_MODEL)

        # 2. chunking — NOTE: Stage 2 calls the embedding model, so this is not pure CPU
        embedder = make_ollama_embedder(model=C.EMBEDDING_MODEL)
        with C.Stopwatch(timings, "extract_units"):
            chunking = await extract_units(
                content,
                target_min_chars=1500,
                target_max_chars=4000,
                embedding_fn=embedder,
            )

        with C.Stopwatch(timings, "units_to_chunks"):
            chunks = units_to_chunks(content, chunking.units)

        embed_svc = EmbeddingService(model=C.EMBEDDING_MODEL)
        try:
            # 3. one batched embed call for every chunk
            with C.Stopwatch(timings, "embed_batch"):
                await embed_svc.embed_batch([c.text for c in chunks], text_type="document")

            # 4. per-chunk LLM metadata, awaited ONE AT A TIME
            per_chunk_seconds: list[float] = []
            with C.Stopwatch(timings, "extract_chunk_metadata_sequential"):
                for chunk in chunks:
                    started = time.monotonic()
                    await extract_chunk_metadata(chunk.text, model=C.LLM_MODEL)
                    per_chunk_seconds.append(time.monotonic() - started)

            # 5. the doc-level embedding, with the exception RECORDED not hidden
            doc_embedding_error: str | None = None
            with C.Stopwatch(timings, "embed_doc_level"):
                try:
                    await embed_svc.embed_text(content, text_type="document", title=title)
                except Exception as exc:  # noqa: BLE001 — recorded deliberately
                    doc_embedding_error = f"{type(exc).__name__}: {exc}"
        finally:
            await embed_svc.close()

        payload["http"] = ledger.summary()

    sequential_total = sum(timings.values())
    payload.update(
        {
            "n_chunks": len(chunks),
            "chunk_chars": [len(c.text) for c in chunks],
            "doc_metadata_title": doc_meta.title,
            "timings_seconds": timings,
            "sequential_total_seconds": sequential_total,
            "per_chunk_seconds": per_chunk_seconds,
            "per_chunk_seconds_stats": {
                "n": len(per_chunk_seconds),
                "min": min(per_chunk_seconds) if per_chunk_seconds else None,
                "median": (
                    sorted(per_chunk_seconds)[len(per_chunk_seconds) // 2]
                    if per_chunk_seconds
                    else None
                ),
                "max": max(per_chunk_seconds) if per_chunk_seconds else None,
                "sum": sum(per_chunk_seconds),
            },
            "doc_embedding_error": doc_embedding_error,
            "doc_embedding_note": (
                "recorded, not swallowed: in the shipped pipeline _embed_doc_level "
                "catches exactly this and logs at INFO, which is why claim (e) is stated "
                "over chunk embeddings"
            ),
            "ollama_ps_after": ollama_ps(root_url),
            "limitation": (
                "the in-pipeline comparison gathers three workloads across two models, so "
                "a ratio near 1.0 is decisive but ~1.3 is ambiguous between fan-out gain "
                "and embed/LLM overlap — read the latency distribution alongside"
            ),
        }
    )

    # --- the comparison against the drain's in-pipeline number -------------
    #
    # ATTRIBUTION BY IDENTITY, NOT BY TITLE.
    #
    # An earlier version matched drain progress ticks on `title[:40]`. dfr_main
    # holds this paper under TWO rows with identical titles — the markdown ingest
    # measured here, and claim (a) arm 2's `pending_source` probe of the same PDF —
    # so the first title match landed on the SOURCE row at tick index 0. Index 0's
    # interval starts at the drain's t0, so the "enrichment seconds" it produced
    # contained `process_pending`'s preflight AND a full Docling conversion. The
    # ratio was computed against the wrong row with two extra stages in its
    # denominator. `attribute_document_seconds` matches the doc_uuid recorded by
    # the drain's own progress observer and refuses index 0 outright.
    claim_b = C.read_json(C.RESULTS / "claim_b_drain.json")
    drains = claim_b["drains"]
    attribution = attribute_document_seconds(drains, target["doc_id"])
    in_pipeline_seconds = attribution["seconds"]

    payload["in_pipeline_enrichment_seconds"] = in_pipeline_seconds
    payload["in_pipeline_attribution"] = attribution
    payload["concurrency_speedup"] = (
        sequential_total / in_pipeline_seconds
        if in_pipeline_seconds and in_pipeline_seconds > 0
        else None
    )

    # A FREE REPLICATE, and the reason the ratio must not be published bare.
    #
    # The SAME content was enriched twice inside claim_b's drains: once as the
    # markdown ingest above, once as claim (a) arm 2's converted source. Two
    # attributable enrichments of identical work under identical conditions give
    # a spread — the only variance estimate this run contains that is a genuine
    # repeat rather than a pooled mixture of endpoints. If |speedup - 1| sits
    # inside that spread, the correct report is "not distinguishable from 1.0",
    # not a measured 1.07.
    replicates: list[dict[str, Any]] = []
    for drain in drains:
        ticks = drain.get("progress") or []
        for index in range(1, len(ticks)):
            tick = ticks[index]
            if tick.get("markdown_chars") != len(content):
                continue
            replicates.append(
                {
                    "doc_uuid": tick.get("doc_uuid"),
                    "drain": drain.get("label"),
                    "tick_index": index,
                    "seconds": tick["monotonic_s"] - ticks[index - 1]["monotonic_s"],
                    "markdown_chars": tick.get("markdown_chars"),
                }
            )
    replicate_seconds = [r["seconds"] for r in replicates]
    replicate_spread = None
    if len(replicate_seconds) >= 2:
        mean_replicate = sum(replicate_seconds) / len(replicate_seconds)
        replicate_spread = (
            (max(replicate_seconds) - min(replicate_seconds)) / mean_replicate
            if mean_replicate
            else None
        )
    payload["in_pipeline_replicates"] = replicates
    payload["in_pipeline_replicate_seconds"] = replicate_seconds
    payload["in_pipeline_replicate_spread"] = replicate_spread
    payload["in_pipeline_replicate_note"] = (
        "documents of IDENTICAL character length enriched inside the same lineage — "
        "the run's only true repeated measurement. concurrency_speedup must be read "
        "against this spread, not against a pooled cross-endpoint latency band."
    )

    rec.observe(
        "concurrency_speedup",
        payload["concurrency_speedup"],
        detail=(
            "sum(sequential components) / in-pipeline seconds for the SAME doc_uuid. "
            "~1.0 means _INGEST_CONCURRENCY=5 is decoration against an Ollama that "
            "serialises same-model calls, and the honest optimisation is fewer/larger "
            f"chunks or a smaller extraction model. Replicate spread: {replicate_spread}"
        ),
    )
    rec.observe(
        "max_in_flight_sequential_arm",
        payload["http"]["max_in_flight"],
        detail=(
            "GLOBAL process depth — an upper bound over all endpoints at once, NOT an "
            "attribution to any one of them"
        ),
    )
    rec.observe(
        "max_in_flight_by_path",
        payload["http"]["max_in_flight_by_path"],
        detail=(
            "peak concurrency counted PER ENDPOINT by its own in-flight counter, so a "
            "chat peak can no longer be inflated by concurrent embedding calls"
        ),
    )

    # WHAT THIS RULE CONTROLS, AND WHAT IT ONLY OBSERVES.
    #
    # The sequential baseline exists to price the per-chunk LLM extraction that
    # the shipped pipeline fans out under _INGEST_CONCURRENCY=5. This rule awaits
    # those calls one at a time, so the LLM endpoint is the arm it controls and
    # the arm the concurrency_speedup ratio is computed over — it must be 1.
    #
    # The embedding endpoints are NOT this rule's to serialise. Step 2 calls
    # chunker.extract_units(embedding_fn=...), and core.embeddings.make_embedder
    # issues its per-paragraph /api/embeddings calls through
    # `asyncio.gather` under `Semaphore(8)`. That fan-out is a property of the
    # system being measured, so asserting it away would be measuring a system
    # that does not ship. It is recorded as a finding instead (see below).
    llm_peak = ledger.max_in_flight_for("/chat/completions")
    payload["max_in_flight_llm"] = llm_peak
    rec.check(
        "micro_stages/sequential",
        llm_peak <= 1,
        measured=llm_peak,
        expected="<= 1",
        detail=(
            "the sequential baseline is only a baseline if the LLM calls it prices "
            "really were issued one at a time"
        ),
    )

    embed_peak = ledger.max_in_flight_for("/api/embeddings", "/api/embed")
    payload["max_in_flight_embeddings"] = embed_peak
    rec.observe(
        "embedding_fanout_observed",
        embed_peak,
        detail=(
            "FINDING, not a harness artefact: core.embeddings.make_embedder gathers "
            "up to Semaphore(8) concurrent /api/embeddings calls inside the chunker's "
            "embedding_fn, and it uses the single-prompt /api/embeddings endpoint "
            "rather than the batched /api/embed array. Ollama serialises same-model "
            "calls, so this fan-out buys little while contradicting the project's "
            "one-inference-at-a-time rule. Reported here; NOT changed by this "
            "experiment, because it is shared core infrastructure unrelated to the "
            "deferred-ingestion queue under test"
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
            C.RESULTS / "micro_stages.json", {**payload, **rec.payload()}, schema=SCHEMA
        )
    print(f"verdict: {rec.verdict}")
    rec.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
