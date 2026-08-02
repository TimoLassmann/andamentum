**sha256, producing rule and schema for every artefact — and for every piece of
the apparatus.** ``scripts/`` and the ``Snakefile`` are hashed on purpose:
hashing 122 outputs and none of the code that produced them leaves no route
from a published number back to the machinery that produced it, which is the
Reproducible leg of FAIR.

Re-check it at any time with::

    uv run python -m experiments.docstore_deferred_ingestion.scripts.manifest --verify

Two files are self-excluded and say so on the artefact: ``MANIFEST.json`` would
have to contain its own hash, and Snakemake writes ``bench/manifest.tsv`` only
after the rule body returns.
