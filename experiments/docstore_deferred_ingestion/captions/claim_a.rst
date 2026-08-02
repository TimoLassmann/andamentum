**Claim (a): defer is LLM-free and fast** (H-a1 … H-a5).

Four papers are ingested with ``process="defer"`` while an HTTP recorder is
attached to this process's ``httpx`` transport, plus one control document
ingested with ``process="now"`` in a separate database.

Measured: **zero** requests to the Ollama host on the defer path; per-document
0.017 / 0.009 / 0.009 / 0.007 s (median 0.0092 s); the same-document ratio of
defer to ``process="now"`` is 4.9e-05 (0.017 s against 352.8 s). Zero state
violations — each deferred document has 0 chunks, 0 chunk embeddings, no
document embedding, status ``pending_enrich`` and metadata keys exactly
``{source, title}``.

The "LLM-free" claim is checked **twice, in different currencies**: at the
transport (no request was made) and in the persisted state (no artefact of a
request exists). Either alone is weaker than it looks.

H-a4/H-a5 also check *findability*: every deferred document is FTS5-retrievable
by a rare token from its own body **before** any drain, and a queued
``ingest_source`` — which has no text yet — returns **0** FTS hits, as it must.
