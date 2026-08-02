**The cost roll-up.** Per rule: Snakemake's process wall time
(``benchmark_wall_seconds``, one quantity, populated everywhere) and — in a
separately named column that says what it is — the sum of the in-script
components that rule measured itself.

The two are deliberately not merged. They answer different questions, and an
earlier edition plotted one against the other under a single axis label.
