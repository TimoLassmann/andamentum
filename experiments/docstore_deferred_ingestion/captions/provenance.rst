**The provenance manifest.** ``run_id``, git commit and branch, dirty flag,
Python/Snakemake/package versions, ``uv.lock`` sha256, the allowlisted
environment, and **both pinned Ollama models recorded by digest rather than by
tag** — the rule fails loud if either is absent from ``/api/tags``, because an
experiment that silently runs against a model it did not intend has no
measurement at all.

When the working tree is dirty it also writes ``results/working_tree.patch``
(``git diff HEAD``) and records its sha256, and ``validate_results`` then
**requires** that patch to exist and to hash correctly. Without it a run can
record a commit sha that names code nobody can recover — which is exactly what
happened in the first edition, whose measurements depended on an uncommitted
``fts_query.py`` fix.

Also carried: ``harness_sha256``, one digest over the Snakefile, ``config.yaml``
and every ``scripts/*.py``, computed independently of the manifest's.
