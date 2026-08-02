**Claim (d): a pause stops between documents** (H-d1, H-d2, H-d3, H-d5).

Five arms, five different stop mechanisms: ``max_docs``, ``max_seconds``,
``should_continue``, a SIGTERM to the CLI on a truncated document, and a SIGTERM
to the CLI with a **full-size** document in flight.

Measured: **0** mid-stage documents across every arm, checked on each arm's
**own** post-drain snapshot (a row that is ``failed``, or that carries
chunk/chunk-embedding rows while not ``complete``, is enrichment committed and
stranded). Post-commit wall clock totals 3.3 ms, 1.6e-05 of that drain's own
longest document (72.4 s). In a mixed queue, a ``max_docs=1`` drain processes
the **source** first, as designed. The ``should_continue`` arm reports
``stopped_early=True`` with work still queued.

That last clause is not decoration. In an earlier edition the arms consumed the
queue between them, so the ``should_continue`` drain ran out of work and ended
naturally while an assertion of the form ``processed <= 1`` passed anyway — an
exhausted queue measured and reported as a pause. The check is now
``stopped_early == 1 AND remaining >= 1``, which fails on that earlier data.

Arm 5 exists because the one number an operator acts on — SIGTERM-to-exit,
which sizes a launchd/cron shutdown grace period — does not transfer from a
2-chunk truncation (61.2 s) to a 26-chunk paper (**274.2 s**).
