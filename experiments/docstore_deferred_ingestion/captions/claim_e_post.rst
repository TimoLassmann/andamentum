**Claim (e): after a drain, documents are semantically searchable** (H-e1,
H-e2).

The same eight paraphrase probes as the control arm, re-run after the drain.
Chunk-embedding recall@1 = 1.0 (chance 0.2), MRR = 1.0, unified RRF recall@3
0.25 to 1.0. Zero enrichment-structure violations: every completed document has
chunks > 0, exactly one chunk embedding per chunk, and non-empty LLM metadata.

Also measured rather than asserted: the **document-level embedding skip rate**
(0.0 on this corpus). ``_run_phase2._embed_doc_level`` swallows an
oversized-input failure with an INFO log, so one of the four RRF signals can go
dark on long documents with nothing reporting an error. Asserting it away would
have failed claim (e) for a reason unrelated to the queue; measuring it says
what actually happened here.
