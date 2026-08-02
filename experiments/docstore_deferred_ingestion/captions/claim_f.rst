**Claim (f): failures are recorded, and retried at the right stage** (H-f1,
H-f2).

Four **real** failure modes, no monkeypatching: a file path that does not
exist; 400 bytes of random data with a ``.pdf`` extension; a genuine 404 from
the real ``arxiv.org`` (not a localhost URL — the SSRF guard would reject that
for a different reason than the one under test); and a markdown file containing
only whitespace.

Measured: 1 good document completes, **4** fail, the drain does not abort, all
4 appear in ``ProcessReport.failures``, and **every** ``ingest_error`` names the
exception **type** — real classes, ``FetchError`` and ``ExtractionError``, not
a swallowed string.

``retry_failed()`` then routes each failure by **markdown presence**: a
document that already holds markdown goes back to ``pending_enrich``, one that
does not goes back to ``pending_source``. Zero wrong-stage requeues, and zero
re-conversions of the markdown-bearing document. The three broken sources *are*
re-converted, which is correct — they hold no markdown — and that count is
published separately as an observation rather than folded into the hypothesis.
