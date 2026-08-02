**Sequential component breakdown of one document's enrichment**, from
``rule micro_stages``: the same paper enriched with strictly one model call at
a time, so the cost of each stage can be attributed.

Note that ``extract_units`` Stage 2 calls the embedding model, so the
"chunking" component is **not** pure CPU.

This is the baseline that makes ``concurrency_speedup`` interpretable: 301.9 s
sequential against 287.1 s in-pipeline for the same document identity, a ratio
of 1.052. That ratio has **n=1 and no variance estimate** and must be read as a
single observation — consistent with the library's fan-out buying essentially
nothing against an Ollama that serialises same-model calls, but not evidence of
a 5% gain.
