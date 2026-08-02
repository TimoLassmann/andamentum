**Schema and referential integrity over every results artefact**, checked after
the fact and exiting non-zero on any violation: every JSON carries
``schema``/``schema_version``/``written_at``/``provenance_ref``; every
``provenance_ref`` hashes to the provenance file actually on disk; every
document id referenced by an artefact is grounded in a snapshot;
``working_tree.patch`` exists and hashes correctly whenever the run was dirty.

Verdict: **PASS, 0 violations**, over 46 named artefacts plus 7 glob classes.
