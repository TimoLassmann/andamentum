**The blocking gate (H-G).** Exactly two sequential Ollama calls — one
``extract_chunk_metadata``, one ``embed_batch`` — asserting that the chunk
metadata comes back with a **non-empty** topics list.

This is the single most important rule in the workflow and it runs before every
measurement. Without ``OLLAMA_BASE_URL`` set, the same call returns in 0.47 s
with empty topics, having caught the provider error and returned defaults
without raising; a drain in that state reports ``documents_enriched`` for every
document while writing no LLM metadata whatsoever.

Measured here: 21.8 s, 3 topics, 768-dimension embeddings. The LLM really
enriched. Every measurement rule declares this file as an input, so a
mis-wired provider cannot be measured through.
