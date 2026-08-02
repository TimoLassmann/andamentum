**Claim (e), control arm.** The retrieval probes are run *before* the drain, so
the post-drain numbers have something to be compared against.

Chunk-embedding recall@3 is 0.0 here — by construction, since claim (a) has
just asserted that deferred documents have zero chunks, which is why that
particular figure is published as an **observation** rather than scored as a
hypothesis. The informative control is the unified four-signal RRF stack, which
is non-zero (0.25) before the drain because FTS5 alone already works, and which
rises to 1.0 after it.

This rule also carries the ``search()`` auto-repair tripwire: if the public
search path were quietly repairing the store on read, the pre-drain measurement
would be measuring a drain.
