**Claim (c), part 2: the resume, in a fresh process** (H-c1, H-c2, H-c3, H-d4).

The killed database is drained again from a cold process. The result is the
single most load-bearing set of numbers in the experiment:

* the converter ledger holds **3** entries for 3 sources with **0** duplicates
  — the already-converted document was **not** re-converted;
* the resume reports ``documents_converted=2``, not 3;
* the markdown sha256 is **byte-identical** before the kill, after the kill and
  after the resume;
* ``PRAGMA integrity_check`` and ``quick_check`` both return ``ok``, the WAL is
  0 bytes, and the resume opened the database normally — no repair, no manual
  ``-wal`` deletion, no wedged lock;
* exactly **one** row was ``pending_enrich`` at resume, and the enrichment work
  the checkpoint made us repeat (0.187 s) is 1.2% of the conversion it
  preserved (15.71 s).

Ledger ownership is a documented deviation: Snakemake refuses to let two rules
produce one file, so the ledger stays ``claim_c_kill``'s output and this rule
declares it as an input it **appends to**, emitting a verbatim copy as its own
declared output. Because Snakemake can no longer protect against a
double-append, the script fails loud if the ledger already holds the full
expected count.
