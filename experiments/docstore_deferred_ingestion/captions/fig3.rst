**Discarded work per stop mechanism.**

**Every bar is measured.** In an earlier edition three of five bars were ``0.0``
literals in the plotting script and the SIGKILL bar was borrowed from a
sequential benchmark of a *different* paper in a *different* database. This
version derives each library-arm bar from that arm's own post-commit
wall-clock, and the SIGKILL bar from ``killed_at`` minus the converter ledger's
last ``ts_end`` (0.187 s). The plotting script now **raises** rather than
drawing a bar the harness did not measure.

The CLI SIGTERM arms are deliberately absent: their in-flight document is
allowed to finish, so "discarded seconds" is not the quantity that describes
them. What describes them is time-to-exit, which is reported separately —
274.2 s with a full-size 26-chunk paper in flight, and that is the number an
operator should size a shutdown grace period with.
