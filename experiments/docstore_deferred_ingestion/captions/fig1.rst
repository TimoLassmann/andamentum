**Process wall time per rule**, taken from Snakemake's own ``benchmark:``
column — one quantity, populated for every rule.

Why it is drawn this way: an earlier edition plotted a sum-of-in-script
components for two rules on the same axis as a process wall time for the
others, which is two different quantities under one label. The in-script
component seconds still exist in ``results/timings.csv``, under a column that
now says what they are.

Caveat stated on the artefact: Snakemake's memory and IO columns are literally
**0** on this macOS host. Wall-clock is used for cross-checking only; in-script
``time.monotonic()`` is authoritative.
