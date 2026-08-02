# andamentum.document_store

Personal knowledge base with 4-signal search. Store anything — thoughts, papers, meeting notes, decisions, tasks — and find it again with natural language.

## What it provides

The core async functions that form the public surface:

| Function | Purpose |
|----------|---------|
| `ingest(database, content)` | Store content with auto-chunking and embedding (no LLM) |
| `search(database, query)` | Natural language search with LLM query planning |
| `find_by_metadata(database, filters)` | Structured query by metadata fields — exact match or set-membership |
| `describe_metadata(database)` | Discover the metadata schema — fields present + value distributions |
| `update_metadata(database, doc_id, metadata)` | Update fields on a document |
| `delete(database, doc_id)` | Remove a document and all its chunks |
| `repair(database)` | Fix incomplete ingestions after crashes |
| `find_duplicates(database)` | Detect near-duplicate documents via embeddings |
| `ingest_source(database, path_or_url)` | Queue a PDF/DOCX/HTML source for later conversion + enrichment |
| `process_pending(database)` | Drain the deferred-ingestion queue — resumable and pausable |
| `pending_status(database)` | Queue depth per stage (no model needed) |
| `retry_failed(database)` | Requeue documents whose processing failed |

### The three read functions

The store is **schema-less** — metadata is arbitrary JSON, and consumers define
their own vocabulary (`record_type`, `status`, …). Three uniform functions cover
reading, identically usable by humans and agents:

- **`search`** — unstructured recall over *content* (NL query, RRF ranking). It
  does **not** rank or filter on arbitrary metadata; only the LLM query planner's
  small built-in field whitelist narrows by metadata.
- **`find_by_metadata`** — deterministic structured query over *any* metadata
  field. A scalar value matches by equality; a **list value matches by
  set-membership** (`field IN (...)`), so "any of several statuses" is one call.
- **`describe_metadata`** — reports the vocabulary actually present, so a caller
  can fill `find_by_metadata` filters without prior knowledge of the schema.

Domain-specific querying (a task layer's "open" / "overdue" / "high priority")
belongs to the **consumer**, built on top of these primitives — the store does
not grow a function per consumer.

## Querying metadata

```python
from andamentum.document_store import describe_metadata, find_by_metadata

# 1. Discover what exists — no prior knowledge of the schema.
schema = await describe_metadata("brain")
#   {"record_type": FieldProfile(present_in=62, distinct=3,
#                                values={"task": 42, "idea": 13, "decision": 7}),
#    "title":       FieldProfile(present_in=62, distinct=62, values=None)}  # high-cardinality

# 2. Drill into a subset to see its fields.
await describe_metadata("brain", filters={"record_type": "task"})
#   {"status": FieldProfile(present_in=42, distinct=4,
#                           values={"todo": 20, "in_progress": 5, "blocked": 3, "done": 14}), ...}

# 3. Now the filter is grounded. Set-membership in a single query:
open_tasks = await find_by_metadata("brain", {
    "record_type": "task",
    "status": ["todo", "in_progress", "blocked"],
})

# Cheap overview — metadata only, no per-document content reads:
rows = await find_by_metadata("brain", {"record_type": "task"},
                              limit=10_000, include_content=False)
```

A `FieldProfile` carries `present_in` (documents with the field), `distinct`
(number of distinct values), and `values` (the value→count breakdown, populated
only for low-cardinality fields so output stays bounded — `None` otherwise; tune
the cut-off with `describe_metadata(..., max_values=...)`).

## How search works

`search()` fuses four signals via Reciprocal Rank Fusion (RRF):

1. **FTS5 keyword matching** — fast, available immediately after ingest
2. **Chunk-level semantic search** — embedding similarity on individual chunks
3. **Document-level semantic search** — embedding similarity on whole documents
4. **DHP temporal clustering** — recently active and relevant topics rank higher

An LLM decomposes the query into a search plan (semantic query + optional metadata filter) before running the four signals.

## How ingest works

Two-phase design:
- **Phase 1** (atomic): document stored in SQLite, FTS5 keyword-searchable immediately
- **Phase 2** (repairable): content chunked and each chunk embedded; if interrupted, `repair()` re-runs it

**Ingest never calls an LLM.** The store does not invent a metadata vocabulary —
that belongs to whoever uses it. Pass your own fields with `metadata=`, or set
them later with `update_metadata()`, and query them deterministically with
`find_by_metadata()`. If you want an LLM-generated title, call
`extract_document_metadata()` yourself and pass the result as `title=`.

> Per-chunk LLM tagging (`topics`, `people`, `has_decision`, `has_action_item`)
> was removed. Nothing read those fields, and producing them cost ~93% of ingest
> wall-time — one LLM call per chunk. Measured on one arXiv paper: **283 s → 3.5 s
> (81×)**, with retrieval unchanged. Only `search()` still uses an LLM, for query
> planning.

## Deferred ingestion — capture now, enrich later

The expensive work (converting a source to markdown, and chunking/embedding/LLM
metadata) can be queued instead of run inline, then drained on demand or on a
schedule. The queue is three columns on `documents`, because the unit of work
*is* a document:

```
pending_source ──convert──► pending_enrich ──enrich──► complete
       └──────────────── failed ◄────────────────┘
```

```python
from andamentum.document_store import ingest, ingest_source, process_pending, pending_status

# Capture fast: register + FTS index only. No LLM call, no embeddings.
await ingest("research", note_text, process="defer")

# Queue a slow source (PDF/DOCX/HTML). Nothing is converted yet.
await ingest_source("research", "~/papers/big.pdf")

print((await pending_status("research")).pending)   # -> 2

# Drain later — e.g. overnight, capped so it stops before morning.
report = await process_pending(
    "research", model=..., embedding_model=...,
    convert_fn=...,                    # see "Conversion" below
    max_seconds=6 * 3600,
    should_continue=lambda: not stop_button_pressed,
    on_progress=lambda done, total, title: print(f"[{done}/{total}] {title}"),
)
```

**Resumable and pausable.** Each stage transition commits as it completes, so
all state lives in the database — there is no in-memory cursor to lose.
"Resume" is simply calling `process_pending` again; it picks up whatever is
still pending. Pausing (via `should_continue`, `max_docs`, or `max_seconds`) is
checked *between* documents, so a pause never interrupts a commit. A hard kill
mid-document loses **at most that one document's** work — everything already
committed is kept, and phase 2 is idempotent.

Because conversion is checkpointed (`pending_source → pending_enrich` persists
the markdown), an interrupted run never re-parses a PDF it already converted.

**Conversion is injected, not imported.** `process_pending` takes a
`convert_fn: (source) -> markdown` so this module never depends on
`andamentum.harvest`. `document_store.pipeline` is the *only* place harvest is
imported and simply pre-fills that parameter — it is a convenience, not a
gateway; an app may call the core directly with its own converter:

```python
from andamentum.document_store.pipeline import ingest_source, drain   # harvest-backed
await drain("research", model=..., embedding_model=...)
```

> **Note.** Converting *outside* the drain (calling harvest yourself, then
> `ingest()`) works, but is **not** checkpointed: an interrupted run re-parses
> every in-flight document. For resumable bulk work, enqueue sources and let
> `process_pending` convert them.

**What is searchable, and when** — the two entry points differ, and the
distinction matters:

| queued via | keyword (FTS5) | semantic | LLM metadata |
|---|---|---|---|
| `ingest(..., process="defer")` | **immediately** | after drain | after drain |
| `ingest_source(...)` | **only after conversion** | after drain | after drain |

`ingest_source` records a *reference* and writes `markdown_content=''`, so there
is no text to index until `process_pending` converts it — a queued source
returns zero FTS hits (measured, not assumed: see the validation experiment's
`H-a5`). Only `ingest` gives you the immediate keyword-searchability property.

`repair()` deliberately ignores `pending_*` documents, so an interactive
`search()` never turns into a synchronous drain of the backlog.

### CLI

The store owns a drainable queue, not a scheduler — cron/launchd supplies the clock.

```bash
andamentum-docstore ingest research notes.md --defer
andamentum-docstore ingest-source research ~/papers/big.pdf
andamentum-docstore status research
andamentum-docstore process-pending research \
    --model ollama:gemma4:26b-nvfp4 --embedding-model embeddinggemma:latest \
    --max-seconds 21600
andamentum-docstore retry-failed research
```

`Ctrl-C` (or `SIGTERM`) pauses cleanly: the in-flight document finishes and
commits, then the run exits. Re-run the same command to resume.

## Installation

```bash
pip install andamentum
```

## Quick start

```python
from andamentum.document_store import ingest, search

doc_id = await ingest("research", "MAP-Elites could work for antibody optimization")
results = await search("research", "What do I know about quality-diversity optimization?")
for r in results:
    print(r.title, r.score)
```

Databases are stored at `~/.config/andamentum/databases/{name}.db`. Override with `$DOCUMENT_STORE_DIR` or `$ANDAMENTUM_DATABASES_DIR`.
