"""H-G: the BLOCKING enrichment gate. Proves the LLM really enriches.

THE FAILURE MODE THIS EXISTS TO CATCH
-------------------------------------
Measured on this machine during design:

    WITHOUT OLLAMA_BASE_URL:  extract_chunk_metadata -> 0.47 s, topics == []
                              (pydantic-ai's provider error is CAUGHT inside
                              extraction.py, a warning is logged, DEFAULTS are
                              returned, and NOTHING is raised)
    WITH    OLLAMA_BASE_URL:  extract_chunk_metadata -> 10.85 s, topics populated

A drain in the first state reports ``documents_enriched`` for every document
while writing no LLM metadata at all. Every claim about enrichment would then be
measured against a silently degraded path — a whole experiment green and wrong.

So this rule runs FIRST and every measurement rule declares
``results/gate_llm.json`` as an input: it is structurally impossible to measure a
drain through a mis-wired provider.

Exactly two Ollama calls, strictly sequential (never two at once).

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.gate_llm
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from . import _common as C
from .instrument import http_recorder
from .provenance import ollama_ps

SCHEMA = "andamentum.experiment.docstore_deferred.gate_llm/1"

#: A fixed ~4.3k-char probe. Fixed so the latency number means something across
#: runs, and prose-like so the extractor has real topics to find.
GATE_TEXT = (
    "# Deferred ingestion in a personal knowledge base\n\n"
    "A knowledge base has two expensive stages when it takes in a document: turning "
    "the original file into text, and then understanding that text well enough to "
    "search it later. The first stage is a parsing problem; the second is a modelling "
    "problem. Both are slow, and neither needs to happen while the person who dropped "
    "the file in is still waiting.\n\n"
    "The design separates capture from enrichment. Capture writes the document to "
    "durable storage and indexes it for keyword search, which is fast and involves no "
    "model at all. Enrichment splits the document into passages, embeds each passage "
    "into a vector space, asks a language model for a short list of topics, and writes "
    "all of that back. Enrichment can therefore be postponed to a quiet hour and "
    "resumed after an interruption, because each stage commits its own transition "
    "before the next stage begins.\n\n"
    "The property that makes this safe is not cleverness in the scheduler. It is that "
    "the state on disk is always truthful: a document is either waiting for "
    "conversion, waiting for enrichment, finished, or failed, and the row says which. "
    "There is no in-memory cursor to lose, so a machine that loses power mid-run wakes "
    "up knowing exactly what remains. The worst case is repeating the work of the one "
    "document that was in flight.\n\n"
    "Failure is recorded rather than retried. A document whose conversion raised is "
    "marked failed with the exception type and message on the row, and the drain moves "
    "on to the next document rather than aborting the run. A systematic problem — a "
    "missing file, an unreachable model, a corrupted archive — then surfaces as a "
    "visible count instead of an invisible loop. Requeuing failed work is an explicit "
    "operator action, and it routes by evidence: if the markdown is already present, "
    "the retry resumes at enrichment; if it is empty, the conversion never landed and "
    "the retry starts from the source.\n\n"
    "The operational question this leaves open is how long a drain should be allowed "
    "to run. A background job that wakes at two in the morning wants a wall-clock "
    "budget it will respect, and it wants to stop between documents rather than "
    "halfway through one. Checking the budget at the top of each iteration gives "
    "exactly that: the run may overshoot by at most the cost of a single document, and "
    "it never leaves half-written state behind. The cost of the guarantee is that a "
    "budget smaller than one document still processes one document.\n\n"
    "Measurement matters more than argument here. Counting the calls a converter "
    "receives shows whether conversion is repeated. Hashing the stored text before and "
    "after an interruption shows whether it was regenerated. Running the same "
    "paraphrased queries before and after enrichment shows whether the expensive stage "
    "actually bought any retrieval quality, or merely wrote rows into tables that "
    "nothing reads. Each of those is a number a reader can check, which is a different "
    "kind of claim from a design that merely sounds correct.\n"
)


async def run_gate() -> dict[str, Any]:
    from andamentum.document_store.embeddings import EmbeddingService
    from andamentum.document_store.extraction import extract_chunk_metadata

    # PREFLIGHT THE CLI BINARY HERE, in the cheapest blocking rule, rather than
    # letting cli_smoke and claim (d)'s SIGTERM arms discover it missing 35
    # minutes into a serialised lineage chain. Every measurement rule depends on
    # this rule's output, so this check gates the whole run for the price of a
    # stat() call.
    cli_bin = C.require_cli_binary()

    root_url = C.CONFIG["models"]["ollama_root_url"]
    ledger_path = C.LEDGERS / "gate_http.jsonl"

    result: dict[str, Any] = {
        "cli_binary": str(cli_bin),
        "gate_text_chars": len(GATE_TEXT),
        "gate_text_sha256": C.sha256_text(GATE_TEXT),
        "resolved_ollama_base_url": os.environ.get("OLLAMA_BASE_URL"),
        "llm_model": C.LLM_MODEL,
        "embedding_model": C.EMBEDDING_MODEL,
        "ollama_ps_before": ollama_ps(root_url),
    }

    with http_recorder(ledger_path) as ledger:
        # Call 1 — the LLM. This is the one that silently degrades.
        t0 = time.monotonic()
        chunk_meta = await extract_chunk_metadata(GATE_TEXT, model=C.LLM_MODEL)
        result["extract_chunk_metadata_seconds"] = time.monotonic() - t0

        # Call 2 — the embedding backend. Sequential: never two model calls at once.
        embed_svc = EmbeddingService(model=C.EMBEDDING_MODEL)
        try:
            t1 = time.monotonic()
            vectors = await embed_svc.embed_batch([GATE_TEXT[:2000]], text_type="document")
            result["embed_batch_seconds"] = time.monotonic() - t1
            result["embedding_dimensions"] = len(vectors[0]) if vectors else 0
        finally:
            await embed_svc.close()

        result["http"] = ledger.summary()

    result["topics"] = list(chunk_meta.topics)
    result["topics_count"] = len(chunk_meta.topics)
    result["people"] = list(chunk_meta.people)
    result["has_decision"] = chunk_meta.has_decision
    result["has_action_item"] = chunk_meta.has_action_item
    result["ollama_ps_after"] = ollama_ps(root_url)
    return result


def main() -> int:
    recorder = C.ClaimRecorder()
    payload = asyncio.run(run_gate())

    recorder.check(
        "H-G",
        payload["topics_count"] >= 1,
        measured=payload["topics_count"],
        expected=">= 1 topic",
        detail=(
            "Empty topics here means extraction.py caught a provider error and returned "
            "defaults. Every downstream enrichment measurement would be of a degraded path."
        ),
    )
    recorder.check(
        "H-G/embedding",
        payload.get("embedding_dimensions", 0) > 0,
        measured=payload.get("embedding_dimensions"),
        expected="> 0 dimensions",
        detail="the embedding backend must actually return a vector",
    )
    recorder.observe(
        "gate_latency_seconds",
        {
            "extract_chunk_metadata": payload["extract_chunk_metadata_seconds"],
            "embed_batch": payload["embed_batch_seconds"],
        },
        detail=(
            "a sub-second extract_chunk_metadata is itself the signature of the "
            "swallowed-provider-error failure mode"
        ),
    )

    C.write_json(
        C.RESULTS / "gate_llm.json", {**payload, **recorder.payload()}, schema=SCHEMA
    )
    print(f"verdict : {recorder.verdict}")
    print(f"topics  : {payload['topics']}")
    print(f"llm     : {payload['extract_chunk_metadata_seconds']:.2f}s")
    print(f"embed   : {payload['embed_batch_seconds']:.2f}s")
    recorder.raise_if_failed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
