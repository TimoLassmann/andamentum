**Claim (b): a drain is resumable, and idempotent once done** (H-b).

Three drains in sequence against the same database: ``max_docs=2``, then
unrestricted, then a third that must be a **genuine no-op**.

Measured: the cap is honoured; the unrestricted drain finishes the rest (all 5
documents complete, queue empty, 0 failures); the third drain enriches **0**
documents, makes **0** Ollama requests and moves the logical fingerprint by
**0** bits.

The fingerprint is the load-bearing instrument: it is a hash over *meaning*
(status counts, chunk counts, embedding counts, metadata keys) read through a
read-only ``mode=ro`` connection, not over bytes — sqlite's WAL moves bytes
freely without changing anything a user would notice.
