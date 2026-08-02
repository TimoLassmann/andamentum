**The data dictionary** — field, type, units and meaning for every artefact,
generated from the same single definition that drives the validator, so the
human table and the machine check cannot drift apart.

It covers the glob-keyed classes too (converter ledgers, HTTP ledgers,
snapshots, event logs), which is how the count of artefacts carrying
``schema: null`` went from 101 of 122 to zero.
