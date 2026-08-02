**Sequential attribution, and the concurrency question.**

One already-converted paper (25 chunks, 38,978 chars) is enriched with strictly
one model call in flight at a time, so each stage's cost can be attributed:
301.9 s sequential against 287.1 s in-pipeline for the same document identity.

The per-endpoint in-flight counters live here too, and they are the reason this
rule matters beyond timing. Measured peaks: **8** concurrent requests on
``/api/embeddings`` and **5** on ``/v1/chat/completions`` — exactly the
``Semaphore(8)`` in ``core.embeddings.make_embedder`` and the
``Semaphore(_INGEST_CONCURRENCY=5)`` in ``_run_phase2``. Both are **pre-existing
library behaviour**, reported and not changed, and both contradict this
project's standing one-inference-at-a-time rule. The measured speedup of 1.052
is what that fan-out buys against an Ollama that serialises same-model calls.

Attribution is by ``doc_uuid``, never by title, and index 0 of a drain's
progress stream is refused outright — its interval starts at the drain's t0 and
therefore contains the preflight tax and a full Docling conversion. An earlier
edition matched on a 40-character title prefix, landed on exactly that row, and
published a number contaminated by both.
