**Claim (c), part 1: the SIGKILL.** The centrepiece of the experiment.

A real drain is started in a real subprocess against three queued sources; a
poller watches the database until the moment the conversion checkpoint should
hold, then the worker is **SIGKILL**ed — not SIGTERM. The distinction is
structural: ``process_pending`` converts *and* enriches a source inside the
**same** loop iteration, and all three cooperative stop conditions are checked
at the **top** of that iteration, so a cooperative stop *cannot* leave a
converted-but-unenriched document. Only a hard kill can.

Measured: the target document's markdown sha256 is captured before the kill
(``ad22e4e8…``, 48,959 chars) and the converter ledger — one fsynced JSON line
per invocation, valid under a hard kill — holds one entry.

The work the kill discarded is measured too, as ``killed_at`` minus the
ledger's last ``ts_end``: **0.187 s**. An earlier edition drew 283 s there,
borrowed from a sequential benchmark of a different paper in a different
database.
