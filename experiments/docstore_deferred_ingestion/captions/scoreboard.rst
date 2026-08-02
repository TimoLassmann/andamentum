**The scoreboard.** One row per pre-registered threshold: hypothesis id, metric
name, the threshold as registered *before* measurement, the measured value and
the verdict. ``analyze.py`` reads both the thresholds **and** the scoring rules
from ``results/preregistration.json`` and raises if that block is absent, so it
can no more invent a scoring rule than it can invent a threshold.

Result: **33 PASS, 0 FAIL, 0 INCONCLUSIVE, 0 NOT MEASURED, 6 OBSERVATION.**

Also carried here: the noise band (relative spread 0.4922 over 512 raw
per-request latencies on ``/v1/chat/completions`` — one endpoint, one
distribution, read from the JSONL ledgers rather than from pooled order
statistics), and the cross-cutting observations, including the two *pre-existing*
library fan-outs this run measured but did not change.
