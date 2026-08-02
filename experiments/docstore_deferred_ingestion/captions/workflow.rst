Deferred ingestion on the real path
===================================

**What this workflow is.** A pre-registered, ledger-instrumented validation of
``andamentum.document_store``'s *deferred ingestion* queue, driven through the
**real public API**, against **real arXiv PDFs**, a **real Docling converter**
and a **local Ollama**. Nothing under ``src/`` is patched, stubbed or copied —
every instrument attaches at a parameter the shipped API already exposes, at
the HTTP transport boundary inside this experiment's own process, or at
read-only SQL.

**Why it exists.** The feature already had unit tests
(``src/andamentum/document_store/tests/test_deferred_ingestion.py``) that cover
the *mechanics* in-process with stubs. Those tests cannot reach four things,
and those four things are the whole point of this experiment:

1. the **real converter** (Docling on real PDFs) and **real models**;
2. **process-level interruption by SIGKILL**, so durability is tested against
   the filesystem and sqlite rather than against Python control flow;
3. **behavioural retrieval quality created by the drain**, which no unit test
   asserts;
4. **byte-level identity** of the checkpointed markdown, which a stub-call
   counter structurally cannot reach.

The six claims under test
-------------------------

======  ============================================================
claim   statement
======  ============================================================
**a**   ``process="defer"`` is fast and makes **no** LLM call
**b**   a drain is **resumable**; re-running finishes the rest and a
        third drain is a genuine no-op
**c**   **conversion is checkpointed** — a converted source whose
        enrichment failed is not re-converted
**d**   a **pause** stops *between* documents, losing at most one
        document's work
**e**   after a drain, documents are **semantically searchable**
**f**   failures are recorded as ``failed`` and ``retry_failed``
        requeues them at the right stage
======  ============================================================

Each is decomposed into numbered hypotheses (``H-a1`` … ``H-x``) with a numeric
threshold and a falsifier, all written to ``results/preregistration.json``
**before any measurement exists**. ``analyze.py`` may only compare against that
file — it may not invent a threshold, and it raises on a pre-registration with
no ``scoring_rules`` block.

The gate that makes the whole run meaningful
--------------------------------------------

``OLLAMA_BASE_URL`` is not optional. Measured on this host during design:
without it, ``extract_chunk_metadata`` returns in 0.47 s with an **empty**
topics list — the pydantic-ai provider error is caught inside ``extraction.py``,
a warning is logged, defaults are returned and **nothing raises**. A drain in
that state reports ``documents_enriched`` for every document while writing no
LLM metadata at all, and the entire experiment could run green while measuring
an LLM-free path.

``rule gate_llm`` therefore runs before every measurement, and **every**
measurement rule declares its output as an input. It is structurally impossible
to measure a drain through a mis-wired provider.

How the DAG is shaped, and why it is linear
-------------------------------------------

The five measurement lineages (MAIN, KILL, CLI, PAUSE, FAIL) are logically
independent but **must not** run concurrently: Ollama serialises same-model
calls and this project forbids concurrent local inference. They are chained
into one linear order by explicit **resource edges** — artificial input
dependencies that exist only to force sequencing through the DAG rather than
through operator discipline. ``--cores 1`` is a correctness constraint, not an
optimisation hint; every model-driving rule additionally declares ``threads: 1``
and ``resources: ollama=1``, and ``retries: 0`` is set everywhere because a
silent Snakemake retry would append to a ledger twice and corrupt a count.

Isolation
---------

Every store- or model-touching rule exports
``DOCUMENT_STORE_DIR=<experiment>/dbs``, and ``scripts/_common.py`` **raises at
import** unless that path resolves inside this directory. The user's real
document store is unreachable by construction, not by remembering. Every
database is named ``dfr_*`` because ``lifecycle.EPHEMERAL_PREFIXES`` silently
redirects ``test_*``/``tmp*`` databases into a ``.ephemeral/`` subdirectory,
where a carelessly written fingerprint would find nothing and report a clean
state.

Headline result
---------------

**33 PASS · 0 FAIL · 0 INCONCLUSIVE · 0 NOT MEASURED · 6 OBSERVATION.**
All six claims held on the real path. Integrity: ``results/validation.json``
PASS with 0 violations over 46 artefacts and 7 glob classes; ``manifest
--verify`` reports 0 discrepancies over 163 artefacts.

One genuine bug in shipped code was found and fixed as a consequence:
``document_store/fts_query.py`` classified ordinary prose containing a
lowercase ``and`` as a deliberate FTS5 power-query and returned it unescaped,
so a hyphenated token in the same sentence reached FTS5 raw and raised
``OperationalError: no such column: training``. See ``FINDINGS.md``.

What this workflow does **not** establish
-----------------------------------------

Stated here rather than buried, because it is part of the claim:

* **Every timing is n=1.** One kill, one resume, one pause per mechanism, one
  drain sequence. The count- and state-based hypotheses are discrete facts that
  need no n; the derived timings carry no uncertainty except
  ``concurrency_speedup``, which has a single replicate and no variance
  estimate, and says so on its own artefact.
* **Retrieval is a weak discrimination.** Four topically disjoint famous papers,
  no hard negatives, a five-document candidate pool. Chance recall@1 is 0.2.
  This rules out an empty or scrambled index; it does not measure retrieval
  quality.
* **Scale.** Four papers, tens of documents, hours. Nothing here speaks to a
  store with 10^5 documents, to concurrent writers, or to weeks of uptime.
* **LLM metadata content, wall-clock seconds and Docling bytes across versions
  are not reproducible**, and no schema, caption or figure implies otherwise.

Where to look
-------------

``report.md`` is the narrative version of these results; ``README.md`` carries
the full hypothesis table, the FAIR posture and the scope decisions;
``FINDINGS.md`` is the plain-language critical summary for the repo owner.
Every number in all three is re-derivable from the artefacts collected below.
