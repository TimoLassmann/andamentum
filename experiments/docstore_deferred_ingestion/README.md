# Deferred ingestion on the real path

A pre-registered, ledger-instrumented validation of `andamentum.document_store`'s
deferred-ingestion queue, driven through the **real public API** against **real
arXiv PDFs** and a **local Ollama**.

## Why this exists

`src/andamentum/document_store/tests/test_deferred_ingestion.py` already covers
the *mechanics* with stubs, in-process: defer skips the LLM, the drain completes,
the drain is idempotent, `max_docs` / `should_continue` pause, failures are
recorded, `retry_failed` requeues, sources are queued without converting,
conversion is checkpointed (by counting stub calls), sources drain before
markdown, repair leaves deliberate pendings alone.

This experiment differs on exactly **four axes** and only those:

1. the **real converter** (Docling on real arXiv PDFs) and **real models**;
2. **process-level interruption by SIGKILL**, so durability is tested against the
   filesystem and sqlite rather than against Python control flow;
3. **behavioural retrieval quality created by the drain**, which no unit test
   asserts;
4. **byte-level identity** of the checkpointed markdown, which a stub-call counter
   structurally cannot reach.

## The one thing that nearly invalidated everything

`OLLAMA_BASE_URL` is not optional. Measured on this machine during design:

| environment | `extract_chunk_metadata` | topics |
|---|---|---|
| **without** `OLLAMA_BASE_URL` | 0.47 s | `[]` — provider error CAUGHT, warning logged, defaults returned, **nothing raised** |
| **with** `OLLAMA_BASE_URL=http://localhost:11434/v1` | 10.85 s | populated |

A drain in the first state reports `documents_enriched` for every document while
writing no LLM metadata whatsoever. The whole experiment could run green and be
measuring an LLM-free path.

Three independent mitigations, none relied on alone:

- **`rule gate_llm`** runs first and **every** measurement rule declares its output
  as an input — it is structurally impossible to measure a drain through a
  mis-wired provider;
- **`scripts/_common.py` raises at import** unless both environment variables are
  set and `DOCUMENT_STORE_DIR` resolves *inside* this directory;
- **`llm_metadata_populated`** is asserted per completed document (H-e2), so the
  degraded state leaves a persistent trace even if the gate were bypassed.

## Prerequisites

The one command below is not sufficient on a fresh checkout. In order:

```bash
# 1. snakemake is NOT a default dependency — it lives in the `benchmark` extra
uv sync --extra dev --extra benchmark

# 2. both pinned models, ~18 GB of weights. `rule provenance` FAILS LOUD if
#    either is absent from /api/tags, and records the DIGEST, not the tag.
ollama pull gemma4:26b-nvfp4
ollama pull embeddinggemma:latest
```

Also assumed, and checked where checking is cheap:

- **macOS or Linux** with the venv at the repo root. The SIGTERM arms signal
  `.venv/bin/andamentum-docstore` **directly** — going through `uv run` puts a
  signal-forwarding wrapper in the process group, which turns one operator
  Ctrl-C into two deliveries and sends the CLI down its force-exit branch. Its
  existence is checked by `rule gate_llm`, seconds into the run, so a missing
  console script fails immediately rather than 35 minutes into a serialised
  lineage chain.
- **~1.2 h of exclusive Ollama use.** Nothing else may drive the model.
- **Network access to `arxiv.org`** for the first run only (`data/pdfs/` is
  checksum-guarded afterwards; the arXiv *API* is blocked here and is never used).

## Run it

One command, from the **repo root**:

```bash
uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile \
    --cores 1 --resources ollama=1 --printshellcmds
```

**A re-run does nothing unless you clean first.** `results/`, `figures/` and
`report.*` are committed evidence, so on a fresh clone Snakemake reports
*"Nothing to be done"* and you are looking at the shipped answer, not one you
produced. To genuinely re-run:

```bash
# full reset: results/ figures/ bench/ dbs/ logs/ report.* — data/ is KEPT
# (the PDFs are checksum-guarded; re-downloading them every run would be rude)
uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile clean --cores 1

# then the run command above, then:
uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile archive --cores 1
```

To re-run only the analysis chain against existing measurements, delete
`results/claims.json`. `results/` is flat (house style), so a re-run **overwrites**
— `rule archive` is how a previous run survives, and it is part of the documented
command sequence rather than an optional extra.

`--cores 1` is a **correctness constraint, not an optimisation hint.** Ollama
serialises same-model calls and this project forbids concurrent local inference.
Every model-driving rule also declares `threads: 1` and `resources: ollama=1`, so
even a mistyped `--cores 4` cannot co-schedule two inference rules. `retries: 0`
everywhere — a silent Snakemake retry would append to a ledger twice and corrupt
a count.

Useful extras:

```bash
# what would run, without running it
uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile -n --cores 1

# databases only (data/ and results/ kept) — the partial reset
uv run snakemake -s experiments/docstore_deferred_ingestion/Snakefile clean_dbs --cores 1

# offline harness tests (no Ollama, no network, no store)
uv run pytest experiments/docstore_deferred_ingestion/tests -q

# re-check every recorded sha256 against what is on disk
uv run python -m experiments.docstore_deferred_ingestion.scripts.manifest --verify
```

Expected wall-clock: **~1.2 hours**. Nothing else may use Ollama for the duration.
Every rule caches its output, so a partial run resumes rather than restarting.

## Isolation

Every store- or model-touching rule exports:

```
DOCUMENT_STORE_DIR = experiments/docstore_deferred_ingestion/dbs
OLLAMA_BASE_URL    = http://localhost:11434/v1
```

**Database naming gotcha.** `lifecycle.EPHEMERAL_PREFIXES = ("ask_", "test_",
"varfolders", "tmp")` silently redirects such databases into
`<DOCUMENT_STORE_DIR>/.ephemeral/`. A fingerprint script looking for
`dbs/test_drain.db` would find nothing and — written carelessly — report a clean
state. Every experiment database is therefore named `dfr_*` (`dfr_main`,
`dfr_now`, `dfr_kill`, `dfr_pause`, `dfr_pause_full`, `dfr_fail`, `dfr_empty`,
`dfr_cli`), `require_db_name()` refuses anything else, and `fingerprint.py`
**fails loud** on a missing database file rather than emitting zeros.

## FAIR posture

Stated so it can be checked rather than asserted.

- **Findable** — a stable directory *inside the repository*. The repo-root
  `.gitignore` previously excluded `experiments/` wholesale, and git never
  descends into an excluded directory, so this experiment's own `.gitignore`
  (which opens *"anything that is EVIDENCE stays in git"*) could not take effect
  and `git ls-files experiments/` returned zero. The pattern is now
  `experiments/*` plus `!experiments/docstore_deferred_ingestion/`, so the
  directory is un-excluded and the child rules apply. Verify with
  `git ls-files experiments/docstore_deferred_ingestion | wc -l`.
- **Accessible** — inputs by versioned URL + sha256 + ETag in
  `data/REGISTRY.json`; outputs as committed JSON/CSV/JSONL. `data/pdfs/` and
  `data/markdown/` are deliberately not committed (large, and Docling output is
  explicitly not byte-stable across versions — committing it would imply
  otherwise); both are exactly re-derivable from the registry.
- **Interoperable** — plain JSON, JSONL and CSV. No pickles. Every JSON artefact
  carries `schema` / `schema_version` / `written_at` / `provenance_ref`, and
  `results/SCHEMAS.md` documents field, type, units and meaning for every
  artefact **including** the glob-keyed ledgers, snapshots and event logs.
- **Reproducible** — pinned models by **digest** not tag, `uv.lock` sha256, a
  provenance manifest, a one-command re-run, `results/working_tree.patch` whenever
  the tree is dirty, and a `harness_sha256` over the Snakefile plus every
  `scripts/*.py` so the measuring apparatus is identified by one value.

## Hypotheses

Every one is registered with its numeric threshold and its falsifier in
`results/preregistration.json` **before** any measurement exists. `analyze.py`
may only compare against that file; it may not invent a threshold.

| id | claim | statement (abridged) | falsified by |
|---|---|---|---|
| **H-G** | gate | one `extract_chunk_metadata` call returns a non-empty `topics` list | empty metadata with `OLLAMA_BASE_URL` set |
| **H-a1** | a | the defer arm makes **zero** requests to the Ollama host (transport ledger) | ≥ 1 request |
| **H-a2** | a | every deferred doc: 0 chunks, 0 chunk embeddings, no doc embedding, `pending_enrich`, metadata keys exactly `{source, title}` ∪ caller's | any extra key or any chunk/embedding row |
| **H-a3** | a | median defer < 1.0 s, and `t_defer / t_now` < 0.02 | median ≥ 1.0 s, or ratio ≥ 0.02 |
| **H-a4** | a | every deferred doc is FTS5-retrievable by a rare token from its own body **before** any drain; title = first non-empty line | any miss |
| **H-a5** | a | after `ingest_source(process="defer")` an FTS5 probe returns **0** hits | > 0 hits |
| **H-b** | b | `max_docs=2` → unrestricted → a third drain that enriches 0, moves the fingerprint 0 bits, and costs ≤ 2 Ollama requests | remaining > 0, failed > 0, moved fingerprint, or > 2 requests |
| **H-c1** | c | after SIGKILL + fresh-process resume: exactly 3 ledger entries for 3 sources, no duplicate source, `documents_converted == 2` | any duplicate, or a converted count of 3 |
| **H-c2** | c | doc #1's markdown sha256 is byte-identical before and after the resume | any difference |
| **H-c3** | c | `PRAGMA integrity_check` / `quick_check` both `ok`; the resume opens the DB normally, no repair, no `-wal` deletion | corruption, a wedged lock, or manual intervention |
| **H-c4** | c | *(observation)* was the two-transaction window ever observed? | — |
| **H-d1** | d | five pauses (`max_docs`, `max_seconds`, `should_continue`, SIGTERM truncated, SIGTERM full-size) leave nothing mid-stage **on each arm's own snapshot**, and post-commit wall-clock is < ½ of one document | a failed or partially-enriched row, or a large post-commit remainder |
| **H-d2** | d | `T ≤ elapsed ≤ T + longest single-document time` | an overrun exceeding one document |
| **H-d3** | d | in a mixed queue, a `max_docs=1` drain processes the **source** | the first item processed is `pending_enrich` |
| **H-d4** | d | exactly one row is `pending_enrich` at resume, and the repeated enrichment costs **less than** the conversion the checkpoint preserved | 0 or ≥2 rows mid-stage, or repeated work exceeding the conversion |
| **H-d5** | d | the `should_continue` arm reports `stopped_early=True` with work still queued | `stopped_early=False`, or `remaining==0` — an exhausted queue, not a pause |
| **H-e1** | e | chunk-embedding **recall@1** == 1.0 after the drain (chance 1/N), and the unified RRF stack **rises** from a non-zero pre-drain score | post-drain recall@1 < 1.0, or a unified stack that did not rise |
| **H-e2** | e | every completed doc: chunks > 0, one chunk-embedding per chunk, non-empty LLM metadata | zero chunks, a count mismatch, or empty LLM metadata |
| **H-f1** | f | 1 good + 4 broken → complete 1, failed 4, every `ingest_error` names the exception **type**, 4 strings in `ProcessReport.failures` | a swallowed failure, an aborted drain, or an untyped error |
| **H-f2** | f | `retry_failed()` routes by markdown presence, without re-invoking the converter **on the markdown-bearing document** | a wrong-stage requeue, or a re-conversion of that document |
| **H-m1** | all | strictly-sequential enrichment costs ≤ 1.3× the in-pipeline cost — the `Semaphore(5)` fan-out buys little | a speedup above 1.3 |
| **H-x** | all | all **six** `ProcessReport` fields reconcile with the database and the ledger, for every drain including the subprocess resume | any mismatch |

### Two things the hypothesis table cannot say for itself

**Scoring rules are pre-registered too.** Which metrics may be softened to
INCONCLUSIVE, what each is compared against, and where its band comes from live
in `preregistration.json` under `scoring_rules`, and `analyze.py` reads them from
there — it can no more invent a scoring rule than it can invent a threshold. It
**raises** on a pre-registration with no such block.

**Post-hoc changes are itemised.** `results/deviations.json` carries one row per
change made after a number was seen: what moved, why, which hypothesis ids it
could have shifted, and the honest direction. A pre-registration editable after
the fact with no amendment trail provides the appearance of the guarantee rather
than the guarantee.

### Why claim (c) needs a SIGKILL and claim (d) needs cooperative stops

`process_pending` converts **and** enriches a `pending_source` inside the **same**
loop iteration, and `should_continue` / `max_docs` / `max_seconds` are all checked
at the **top** of the iteration. A cooperative pause therefore *structurally
cannot* leave a converted-but-unenriched document. They are different mechanisms,
not variants of one.

### Why doc-level embeddings are measured, not asserted

`_run_phase2._embed_doc_level` catches an oversized-input failure with an INFO
log, and a ~50k-char paper very likely exceeds `embeddinggemma`'s budget. So claim
(e) is stated over **chunk** embeddings and the doc-embedding **skip rate** is
reported as a measured fact about the four-signal RRF stack — one of whose signals
therefore goes dark for long documents without anything reporting an error.
Asserting the opposite would fail claim (e) for a reason unrelated to the queue.

## The instruments

Every one attaches at a parameter the shipped API already exposes, at the
transport boundary inside this experiment's own process, or at read-only SQL.
**Nothing under `src/` is patched, stubbed, copied or reconfigured** — otherwise
the experiment would be testing a modified system.

| instrument | where it attaches | what it settles |
|---|---|---|
| converter ledger | the injected `convert_fn` parameter | H-c1, H-f2 — one fsynced JSON line per invocation, valid under a hard kill |
| httpx recorder | `httpx.AsyncClient.send`, this process only | H-a1, H-b, max in-flight, per-call latency |
| logical fingerprint | read-only `mode=ro` URI connection | H-b, H-a2 — meaning, not bytes (WAL moves bytes freely) |
| event log | `on_progress` + a read-only status read at the same instant | every headline count, re-derivable offline |
| sqlite integrity | `PRAGMA integrity_check` / `quick_check` | H-c3 |

The one place this experiment writes state through the library — marking a
complete document `failed` for the H-f2 stage test — uses the module's own public
`queue.set_ingest_status`. That is state **construction**, not modification of the
logic under test.

## Scope decisions (deliberate, not accidental)

The union of the source designs was roughly 35 full-paper enrichments and 3–4
hours. Every cut below is recorded with its reason. **Do not quietly re-expand the
corpus** — the point is a sound validation of specific claims, not a benchmark.

| decision | reason |
|---|---|
| **four** papers, not five (GPT-3 `2005.14165` dropped) | ~75 pages → ~30 chunks → ~6 min of enrichment plus a slow Docling pass, and it buys nothing for any hypothesis |
| **one** `process="now"` control document, not a parallel corpus | both paths call the identical `_convert_document` + `_enrich_document`; the only delta is DB round-trips, which code reading already settles |
| **three** PDFs in the kill lineage, not five | the checkpoint claim needs one converted document and one interruption |
| **truncated** markdown (~6000 chars) in the pause lineage — but **one full-size SIGTERM arm** | pause semantics do not depend on document size, and the `max_seconds` overrun bound is expressed *relative to* the longest document in that same drain. The one number an operator acts on — SIGTERM-to-exit, which sizes a shutdown grace period — does not transfer from a 2-chunk truncation to a 37-chunk paper, so arm 5 measures it on an untruncated document and arm 4's figure is published only as contrast |
| variance from **one endpoint's** raw per-request latencies plus **one true replicate** | pooling embedding calls (tens of ms) with chat calls (tens of s) measures workload heterogeneity, not run-to-run variance. The replicate is free: the same content is enriched twice inside claim (b), which is what `concurrency_speedup` is judged against |
| `micro_stages` on **one** paper | it uniquely produces the sequential baseline needed to interpret `Semaphore(5)` against a serialising Ollama |
| `snakemake --report` HTML **demoted** out of `all` | ~3 MB per run, its memory/IO columns are literally 0 on macOS, and `report.md` + `MANIFEST.json` + `provenance.json` + `SCHEMAS.md` already carry the FAIR load |

## What is NOT reproducible here

Stating the limits is part of the FAIR claim, not a caveat to it.

- **LLM-generated metadata content** — nondeterministic even at temperature 0 on a
  local model. No schema, caption or figure implies otherwise.
- **Wall-clock timings** — machine- and load-dependent. Absolute seconds are never
  published as reproducible constants; H-a3 is a *ratio* for exactly this reason.
- **Docling markdown bytes across versions or machines** — so H-c2 compares hashes
  only *within* a single run, and the analyzer refuses a cross-run comparison when
  `provenance.packages.docling` differs.

Exactly three things are claimed deterministic: **queue state transitions**,
**conversion bytes within a fixed docling version within one run**, and **chunk
boundaries** for a fixed input and chunker configuration.

### Statistical power, stated rather than implied

- **Every timing claim is n=1**: one kill, one resume, one pause per mechanism,
  one drain sequence. The exact count- and state-based hypotheses (H-a1, H-a2,
  H-b, H-c1, H-c2, H-c3, H-f1, H-x) are discrete facts that need no n. The
  derived **timings** carry no uncertainty except `concurrency_speedup`, which has
  a genuine replicate and is reported INCONCLUSIVE inside it.
- **Retrieval is a weak discrimination.** Four topically disjoint famous papers,
  no hard negatives, a five-document candidate pool: chance recall@3 is 3/5 and
  chance recall@1 is 1/5, both published beside the score. This rules out an
  empty or scrambled index; it does **not** measure retrieval quality, and
  `claim_e_post.retrieval_power.limitation` says so on the artefact's face.
- **H-c4 bounds nothing.** The two commits inside `_convert_document` are
  microseconds apart and the poller samples every 0.25 s, so a null observation is
  what a non-existent window and a sub-millisecond window both look like. It is
  recorded as an observation precisely because it supports no inference either way.

Snakemake's `benchmark:` **memory and IO columns are literally 0** on this macOS
host. They are used for wall-clock cross-checking only; in-script
`time.monotonic()` is authoritative, and `results/SCHEMAS.md` states that split.

## Files

### Workflow and configuration

| file | what it is |
|---|---|
| `Snakefile` | the DAG, the ENV prefix, and the RESOURCE edges that serialise Ollama |
| `config.yaml` | pinned model ids, versioned arXiv ids, per-lineage document counts, timeouts |
| `README.md` | this file |
| `report.md` / `report.html` | the rendered findings (produced by `rule report`) |

### Harness (`scripts/`)

| file | what it does |
|---|---|
| `_common.py` | **raises at import** unless isolated; paths, config, JSON envelope, `ClaimRecorder` |
| `instrument.py` | `counting_convert_fn`, `http_recorder`, JSONL ledgers, `poll_until` |
| `fingerprint.py` | read-only snapshots, the logical fingerprint, `integrity_check` |
| `corpus.py` | the four papers, the 8 paraphrase probes, the 4 broken sources |
| `events.py` | the queue-transition event log |
| `retrieval.py` | the three-signal probe runner and the recall metric |
| `drain.py` | the instrumented `process_pending` wrapper + the H-x reconciliation |
| `schemas.py` | ONE definition → `SCHEMAS.json` (validator) + `SCHEMAS.md` (human table) |
| `drain_worker.py` | the killable subprocess — deliberately boring |

### Rules (`scripts/`, one module per rule)

`prereg` · `provenance` · `deviations` · `fetch_pdfs` · `verify_registry` ·
`convert_reference` · `gate_llm` · `claim_a_defer` · `claim_e_pre` ·
`claim_b_drain` · `claim_e_post` · `micro_stages` · `claim_c_kill` ·
`claim_c_resume` · `cli_smoke` · `claim_d_pause` · `claim_f_fail` ·
`drain_overhead` · `merge_events` · `analyze` · `validate_results` ·
`make_figures` · `build_report` · `manifest`

### Outputs

| path | what it holds |
|---|---|
| `data/REGISTRY.json` | per-paper sha256, URL, ETag, bytes (committed; `data/pdfs/` is gitignored) |
| `data/markdown/` | the Docling-converted corpus |
| `results/preregistration.json` | thresholds, falsifiers **and scoring rules**, written first |
| `results/deviations.json` | every post-hoc change, with the direction it could have cut |
| `results/provenance.json` | run_id, git, versions, **model digests**, allowlisted env, **harness sha256** |
| `results/working_tree.patch` | `git diff HEAD` — written **only** when the tree is dirty, and then **required** by `validate_results` |
| `results/cli_smoke.json` | every `andamentum-docstore` subcommand: exit code, stdout shape, `status` output reconciled against sqlite |
| `results/claim_*.json` | per-claim evidence, each with an explicit `verdict` |
| `results/claims.json` | the scoreboard: one row per threshold |
| `results/validation.json` | schema + referential integrity (exits 1 on violation) |
| `results/MANIFEST.json` | sha256 + producing rule + schema per artefact — **including the Snakefile and every `scripts/*.py`**, so the apparatus is pinned too. `--verify` re-checks it |
| `results/SCHEMAS.md` | field / type / units / meaning for every artefact |
| `results/ledgers/*.jsonl` | converter and HTTP ledgers |
| `results/snapshots/*.json` | pre/post database snapshots per drain |
| `results/events.jsonl` `.csv` | the merged transition timeline |
| `results/timings.csv` `per_document.csv` | cost roll-up and per-document structure |
| `figures/fig1..fig4.png` | timeline, recall pre/post, stop cost, enrichment attribution |
| `runs/<run_id>/` | verbatim archive (produced by `rule archive`, not by `all`) |
| `dbs/` | the isolated `dfr_*` databases (gitignored; `clean_dbs` removes them) |

## Known deviations from the design sketch

- **The converter ledger cannot be declared as both input and output of
  `claim_c_resume`.** Snakemake refuses (two rules may not produce one file), so
  `results/ledgers/kill_convert.jsonl` stays `claim_c_kill`'s output and
  `claim_c_resume` declares it as an **input it appends to**, plus a verbatim copy
  `kill_convert.final.jsonl` as its own declared output. Because Snakemake can no
  longer protect against a double-append, `claim_c_resume` **fails loud** if the
  ledger already holds the full expected count and tells you to re-run from
  `claim_c_kill`.
- **`claim_a_defer` arm 2 keeps its `pending_source` document** in `dfr_main`
  rather than deleting it, and accounts for it explicitly as
  `source_probe_doc_id`. Deleting it would exercise soft-delete machinery
  unrelated to the queue and perturb the very fingerprint claim (b) rests on.
  Consequence: lineage MAIN drains **5** documents (4 markdown + 1 source).
- **`documents_skipped` never fires `on_progress`.** `process_pending` uses
  `continue` without incrementing `done`, so `done` can end below `total`. Any
  consumer assuming otherwise is wrong; the event-log schema records this.
