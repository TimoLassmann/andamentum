**The merged queue-transition timeline**, in JSONL and flat CSV: every
``on_progress`` callback paired with a read-only status read taken at the same
instant, across every drain in the run.

This is the file that makes the headline counts re-derivable offline. If you
distrust a number in ``claims.json``, the transitions that produced it are here
in plain text.
